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

def entropy(t,dim):
#     x=t+torch.logsumexp(t,dim=0)
    x=t/t.std(dim=dim,keepdim=True)
    x = F.softmax(x,dim=dim)
    y = (x+1e-5).log()
    assert t.shape == x.shape and t.shape== y.shape
    ent = (-x*y).sum(dim=dim)
    return ent

class Exemplar():
    def __init__(self,tensor): self.tensor,self.count,self.age=tensor,0,0
        
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
        
    def init_concepts(self,thresh): self.concepts = Concepts(NZ,opt=self.opt,thresh=thresh,num_starting_concepts=0)
    def test_shapes(self): test_shapes(self.ae)
    def show_one_concept(self,idx=0): 
        plt.imshow(self.dec((self.concepts.exemplars[idx].tensor)[None,:,None,None]).squeeze())
        plt.show()
        
    def train_ae(self,epochs,bs):
        dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.RandomSampler(self.dataset),bs,drop_last=True),pin_memory=False)        
        self.true_labels=np.array([mnist[i][1] for i in range(len(mnist))])
        for param in self.enc.parameters(): param.requires_grad = True
        for param in self.dec.parameters(): param.requires_grad = True    
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (xb,yb,idx) in enumerate(dl): 
                latent = self.enc(xb)
                rloss = self.loss_func(self.dec(latent),xb).mean(dim=[1,2,3])
                loss = rloss.mean()
                loss.backward(); self.opt.step(); self.opt.zero_grad()
                epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
            print(f'Epoch: {epoch}\tLoss: {epoch_loss.item()}')
            if not ARGS.test and epoch > 5: self.check_ae_images()
            if epoch>0 or ARGS.test:
                determin_dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.SequentialSampler(self.dataset),461,drop_last=False),pin_memory=False)        
                for i, (xb,yb,idx) in enumerate(determin_dl):
                    latent = self.enc(xb)
                    total_latents = latent.detach().cpu() if i==0 else torch.cat([total_latents,latent.detach().cpu()],dim=0)
                assert len(total_latents) == len(self.dataset)
                if ARGS.test: self.labels = np.zeros(len(total_latents),dtype=np.int32)
                else:
                    self.umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(total_latents.squeeze())
                    self.labels = hdbscan.HDBSCAN(min_samples=1000, min_cluster_size=1000).fit_predict(self.umapped_latents)
                    p = utils.scatter_clusters(self.umapped_latents,self.labels)
                self.centroids = torch.stack([total_latents[self.labels==i,:,0,0].mean(axis=0) for i in range(max(self.labels)+1)]).cuda()
                self.num_classes = max(self.labels+1)
                d = utils.Dataholder(self.dataset.x,torch.tensor(self.labels,device='cuda').long())
                self.labeled_ds = utils.TransformDataset(d,[utils.to_float_tensor,utils.add_colour_dimension],x_only=False,device='cuda')
                self.closs_func=nn.MSELoss(reduction='none')
                print(f'Num clusters: {self.num_classes}\t Num classified: {sum(self.labels>=0)}')
                if max(self.labels) > 0 or ARGS.test: self.train_labels(5,bs)
                    
    def train_labels(self,epochs,bs):
        dl = data.DataLoader(self.labeled_ds,batch_sampler=data.BatchSampler(data.RandomSampler(self.labeled_ds),bs,drop_last=False),pin_memory=False)        
        for epoch in range(epochs):
            total_rloss = torch.tensor(0.,device='cuda')
            total_closs = torch.tensor(0.,device='cuda')
            total_crloss = torch.tensor(0.,device='cuda')
            crlosses = torch.tensor(0.,device='cuda')
            num_easys = 0
            for i, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                hmask = yb>=0
                yb_ = yb*hmask
                centroid_targets = self.centroids[yb_]
                crpred = self.dec(centroid_targets[:,:,None,None])
                rpred = self.dec(latent)
                rloss_ = self.loss_func(rpred,xb).mean(dim=[1,2,3])
                closs_ = self.closs_func(latent,centroid_targets[:,:,None,None]).mean(dim=1)
                crloss_ = self.loss_func(crpred,xb).mean(dim=[1,2,3])
                crmask = crloss_ < 0.1
                crlosses = crloss_.detach().cpu() if i==0 else torch.cat([crlosses,crloss_.detach().cpu()],dim=0)
                mask = hmask*crmask
                num_easys += sum(mask)
                closs = (crloss_[mask]).mean() + closs_[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                rloss = torch.tensor(0.,device='cuda') if mask.all() else (rloss_[~mask]).mean() 
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean()*(1/(i+2))
                total_closs = total_closs*((i+1)/(i+2)) + closs_.mean()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean()*(1/(i+2))
                loss = rloss + closs
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            print(f'R Loss {total_rloss.item()}, C Loss: {total_closs.item()}, CR Loss: {total_crloss.item()}, num easys: {num_easys}')
            set_trace()
            bins = np.linspace(0.01,0.2,3)
            binned = np.digitize(crlosses,bins)
            utils.scatter_clusters(self.umapped_latents,binned)
            #umapped_latents = umap.UMAP(random_state=42).fit_transform(forgotten_latents.squeeze())
            #forgotten_labels = hdbscan.HDBSCAN(min_samples=500, min_cluster_size=500).fit_predict(umapped_latents)
            #num_new_classes = max(forgotten_labels)+1
            #forgotten_labels += self.num_classes
            #self.labels[forgotten_idxs] = forgotten_labels
            #self.num_classes = max(self.labels) + 1
            #print(f'Num classes {self.num_classes}')
            #set_trace()
            #p = utils.scatter_clusters(umapped_latents,self.labels); p.show()
            #p = utils.scatter_clusters(umapped_latents,self.true_labels); p.show()
    
    def check_ae_images(self,num_rows=5):
        idxs = np.random.randint(0,len(self.dataset),size=num_rows*4)
        inimgs = self.dataset[idxs][0]
        x = self.enc(inimgs)
#         x = self.lin(x[:,:,0,0])[:,:,None,None]
        outimgs = self.dec(x)
        _, axes = plt.subplots(num_rows,4,figsize=(7,7))
        for i in range(num_rows):
            axes[i,0].imshow(inimgs[i,0])
            axes[i,1].imshow(outimgs[i,0])
            axes[i,2].imshow(inimgs[i+num_rows,0])
            axes[i,3].imshow(outimgs[i+num_rows,0])
        plt.show()

    def check_concept_images(self,num_imgs=5):
        outimgs = self.dec(self.concepts.exemplars_as_tensor[:25,:,None,None]).squeeze(1)
        num_imgs = min(len(outimgs),25)
        _, axes = plt.subplots(5,5,figsize=(7,7))
        for i in range(num_imgs):
            axes.flatten()[i].imshow(outimgs[i])
    def vis_latent(self,l): plt.imshow(self.dec(l[None,:,None,None].cuda())[0,0]); plt.show()
    def vis_ae(self,x): vis_latent(self.enc(x.cuda()).squeeze())

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
