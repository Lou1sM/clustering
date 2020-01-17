import warnings
warnings.filterwarnings('ignore')
import gc
from fastai import datasets,layers
from fastai.vision import ImageList
from fastai.basic_train import Learner
from fastai.basic_data import DataBunch
from fastai.callback import Callback
from pdb import set_trace
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import torchvision.datasets as tdatasets
import torch.nn as nn
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from time import time
import os
import psutil
from shutil import rmtree
import math
import random
import torch.nn.functional as F
import utils
import hdbscan
import umap.umap_ as umap


NZ=50

def reload():
    import importlib, utils
    importlib.reload(utils)

class Concepts():
    def __init__(self,dim,opt,thresh,num_starting_concepts=10): 
        self.dim,self.opt,self.thresh=dim,opt,thresh
        self.exemplars = []; params = []
        self.dist_func = nn.MSELoss()
        self.exemplars = [Exemplar(normalize(torch.randn(self.dim,device='cuda'))) for _ in range(num_starting_concepts)]
        params = [ex.tensor for ex in self.exemplars]

        assert len(self) == num_starting_concepts == len(params)
        self.opt.param_groups = self.opt.param_groups[:2]
        self.opt.add_param_group({'params': params, 'lr':1e-5})
        
    def add_exemplar(self,tensor):
        new_exemplar = Exemplar(tensor)
        self.exemplars.append(new_exemplar)
        return new_exemplar
 
    @property
    def exemplars_as_tensor(self): return torch.stack([e.tensor for e in self.exemplars])
    def __len__(self): return len(self.exemplars)
        
    def interpret_wo_grad(self,inp_,eps):
        for e in self.exemplars:
            e.count+=1
            e.age+=1
            if e.count - math.log(e.age) > 20: 
                self.exemplars.remove(e)
        try: dist,idx = torch.min((self.exemplars_as_tensor-inp_).norm(dim=-1),0)
        except RuntimeError: pass # No exemplars defined yet
        if len(self) == 0 or dist > self.thresh:
            best_match = self.add_exemplar(inp_.data.clone().squeeze(0))
            dist = torch.tensor(0.,device='cuda')
        else:
            best_match = self.exemplars[idx]
            best_match.count = 0
            best_match.tensor.lerp_(inp_.data.clone().squeeze(0), eps/(best_match.age+1))
        #best_match.tensor /= best_match.tensor.norm()
        #for e in self.exemplars: e.tensor.lerp_(best_match.tensor,-eps/(e.age+1))
#         if len(self) > 4 and self.thresh < 10: self.thresh *= 1.05
#         elif len(self) < 4: self.thresh *= 0.95
        return best_match, dist 
    
    
class ConceptLearner():
    def __init__(self,enc,dec,lin,dataset,opt,loss_func,discrim=None): 
        self.enc,self.dec,self.lin,self.dataset,self.loss_func,self.ce_loss_func = enc,dec,lin,dataset,loss_func,nn.CrossEntropyLoss()
        self.opt = opt(params = [{'params':enc.parameters()},{'params':dec.parameters()}],lr=1e-3)
        self.discrim = discrim if discrim else utils.NonsenseDiscriminator().to('cuda')
        self.discrim_opt = opt(params=[{'params': self.discrim.parameters()}])
        try: rmtree('runs')
        except: pass
        self.writer = SummaryWriter()
        self.init_concepts(1.)
        
    def train_ae(self,epochs,bs):
        dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.RandomSampler(self.dataset),bs,drop_last=True),pin_memory=False)        
        self.true_labels=np.array([mnist[i][1] for i in range(len(mnist))])
        for param in self.enc.parameters(): param.requires_grad = True
        for param in self.dec.parameters(): param.requires_grad = True    
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (xb,yb,idx) in enumerate(dl): 
                try:latent = self.enc(xb)
                except: set_trace()
                rloss = self.loss_func(self.dec(latent),xb).mean(dim=[1,2,3])
                loss = rloss.mean()
                loss.backward(); self.opt.step(); self.opt.zero_grad()
                epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
            print(f'Epoch: {epoch}\tLoss: {epoch_loss.item()}')
            if not ARGS.test and epoch > 5: self.check_ae_images()
            if epoch>5 or ARGS.test:
                determin_dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.SequentialSampler(self.dataset),461,drop_last=False),pin_memory=False)        
                for i, (xb,yb,idx) in enumerate(determin_dl):
                    latent = self.enc(xb)
                    total_latents = latent.detach().cpu() if i==0 else torch.cat([total_latents,latent.detach().cpu()],dim=0)
                assert len(total_latents) == len(self.dataset)
                if ARGS.test: self.labels = np.zeros(len(total_latents),dtype=np.int32)
                else:
                    umapped_latents = umap.UMAP(random_state=42).fit_transform(total_latents.squeeze())
                    self.labels = hdbscan.HDBSCAN(min_samples=1000, min_cluster_size=1000).fit_predict(umapped_latents)
                    p = utils.scatter_clusters(umapped_latents,self.labels); p.show()
                    p = utils.scatter_clusters(umapped_latents,self.true_labels); p.show()
                self.centroids = torch.stack([total_latents[self.labels==i,:,0,0].mean(axis=0) for i in range(max(self.labels)+1)]).cuda()
                self.num_classes = max(self.labels+1)
                d = utils.Dataholder(self.dataset.x,torch.tensor(self.labels,device='cuda').long())
                self.labeled_ds = utils.TransformDataset(d,[utils.to_float_tensor,utils.add_colour_dimension],x_only=False,device='cuda')
                self.closs_func=nn.MSELoss(reduction='none')
                self.train_labels(5,bs)
                    
    def train_labels(self,epochs,bs):
        dl = data.DataLoader(self.labeled_ds,batch_sampler=data.BatchSampler(data.RandomSampler(self.labeled_ds),bs,drop_last=True),pin_memory=False)        
        for epoch in range(epochs):
            total_rloss = torch.tensor(0.,device='cuda')
            total_closs = torch.tensor(0.,device='cuda')
            total_crloss = torch.tensor(0.,device='cuda')
            for i, (xb,yb,idx) in enumerate(dl):
                print(i)
                latent = self.enc(xb)
                hmask = yb>=0
                yb_ = yb*hmask
                centroid_targets = self.centroids[yb_]
                crpred = self.dec(centroid_targets[:,:,None,None])
                rpred = self.dec(latent)
                rloss_ = self.loss_func(rpred,xb).mean(dim=[1,2,3])
                closs_ = self.closs_func(latent[:,:,0,0],centroid_targets).mean(dim=1)
                crloss_ = self.loss_func(crpred,xb).mean(dim=[1,2,3])
                crmask = crloss_ < 0.1
                mask = hmask*crmask
                closs = (crloss_[mask]).mean() if mask.any() else torch.tensor(0.,device='cuda')
                rloss = torch.tensor(0.,device='cuda') if mask.all() else (rloss_[~mask]).mean() 
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean()*(1/(i+2))
                loss = rloss + closs
                loss.backward(); self.opt.step(); self.opt.zero_grad()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test','-t',action='store_true')
    ARGS = parser.parse_args()
    enc,dec,lin = utils.get_enc_dec('cuda',latent_size=NZ)
    mnist_ds = utils.get_mnist_dset(x_only=True)
    mnist_train = utils.get_mnist_dset(x_only=False)

    ones = mnist_train.x[(mnist_train.y == 1)].to('cuda')
    twos = mnist_train.x[(mnist_train.y == 2)].to('cuda')
    threes = mnist_train.x[(mnist_train.y == 3)].to('cuda')
    fours = mnist_train.x[(mnist_train.y == 4)].to('cuda')
    fives = mnist_train.x[(mnist_train.y == 5)].to('cuda')
    sixes = mnist_train.x[(mnist_train.y == 6)].to('cuda')
    sevens = mnist_train.x[(mnist_train.y == 7)].to('cuda')
    eights = mnist_train.x[(mnist_train.y == 8)].to('cuda')
    nines = mnist_train.x[(mnist_train.y == 9)].to('cuda')
    simple_ds = utils.TransformDataset(torch.cat([ones,twos,fives,eights],dim=0), [utils.to_float_tensor,utils.add_colour_dimension],x_only=True,device='cuda')
    mnist = utils.get_mnist_dset()
    #simple_clearner = ConceptLearner(enc,dec,lin,simple_ds,opt=torch.optim.Adam,loss_func=nn.L1Loss(reduction='none'))
    simple_clearner = ConceptLearner(enc,dec,lin,mnist,opt=torch.optim.Adam,loss_func=nn.L1Loss(reduction='none'))
    simple_clearner.train_ae(epochs=20,bs=64)
    simple_clearner.check_ae_images()

    print(simple_clearner.lin1.weight.shape)
    for t in range(simple_clearner.lin1.weight.shape[0]):
        simple_clearner.vis_latent(simple_clearner.lin2((torch.arange(simple_clearner.lin1.weight.shape[0])==t).float().cuda()))
