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
import torch.nn.functional as F
import utils
import hdbscan
import umap.umap_ as umap
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
import random


class ConceptLearner():
    def __init__(self,enc,dec,lin,dataset,opt,loss_func,discrim=None): 
        self.enc,self.dec,self.lin,self.dataset = enc,dec,lin,dataset
        self.ploss_func,self.loss_func,self.ce_loss_func = nn.L1Loss(),loss_func,nn.CrossEntropyLoss()
        self.opt = opt(params = enc.parameters(), lr=ARGS.enc_lr)
        self.opt.add_param_group({'params':dec.parameters(),'lr':ARGS.dec_lr})
        try: rmtree('runs')
        except: pass
        self.writer = SummaryWriter()
        
    def train_ae(self,ARGS):
        dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.RandomSampler(self.dataset),ARGS.batch_size,drop_last=True),pin_memory=False)        
        self.true_labels=np.array([mnist[i][1] for i in range(len(mnist))])
        for epoch in range(ARGS.pretrain_epochs):
            epoch_loss = 0
            for i, (xb,yb,idx) in enumerate(dl): 
                latent = self.enc(xb)
                latent = utils.noiseify(latent,ARGS.noise*(epoch/ARGS.pretrain_epochs))
                loss = self.ploss_func(self.dec(latent),xb)
                loss.backward(); self.opt.step(); self.opt.zero_grad()
                epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
                if ARGS.test: break
            if ARGS.test: break
            print(f'Epoch: {epoch}\tLoss: {epoch_loss.item()}')
        for epoch in range(ARGS.epochs):
            determin_dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.SequentialSampler(self.dataset),461,drop_last=False),pin_memory=False)        
            for i, (xb,yb,idx) in enumerate(determin_dl):
                latent = self.enc(xb)
                total_latents = latent.detach().cpu() if i==0 else torch.cat([total_latents,latent.detach().cpu()],dim=0)
            assert len(total_latents) == len(self.dataset)
            if ARGS.test: self.labels = np.zeros(len(total_latents),dtype=np.int32)
            else:
                self.umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(total_latents.squeeze())
                scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                self.full_labels = scanner.fit_predict(self.umapped_latents)
                p = utils.scatter_clusters(self.umapped_latents,self.true_labels,show=ARGS.display)
                p.savefig(f'../experiments/{ARGS.exp_name}/scatter_plot-{utils.get_datetime_stamp()}.png')
                p.savefig(f'../experiments/{ARGS.exp_name}/scatter_plot-current.{utils.get_datetime_stamp()}.png')
                self.labels = np.copy(self.full_labels)
                #self.labels[scanner.probabilities_<0.1]=-1
            self.centroids = nn.Parameter(torch.stack([total_latents[self.labels==i,:,0,0].mean(axis=0) for i in range(max(self.labels)+1)]).cuda())
            if ARGS.centroids_lr > 0: self.opt.add_param_group({'params':self.centroids, 'lr':ARGS.centroids_lr})
            else: self.centroids.requires_grad=False
            self.check_latents(self.centroids)
            self.num_classes = max(self.labels+1)
            d = utils.Dataholder(self.dataset.x,torch.tensor(self.labels,device='cuda').long())
            self.labeled_ds = utils.TransformDataset(d,[utils.to_float_tensor,utils.add_colour_dimension],x_only=False,device='cuda')
            self.closs_func=nn.MSELoss(reduction='none')
            print(f'Num clusters: {self.num_classes}\t Num classified: {sum(self.labels>=0)}')
            print(adjusted_rand_score(self.labels,self.true_labels),adjusted_mutual_info_score(self.labels,self.true_labels))
            self.train_labels(1 if ARGS.test else 6,ARGS.batch_size)
            if ARGS.test: break
        
        with open(f'../experiments/{ARGS.exp_name}/{ARGS.exp_name}_summary.txt','w') as f:
            f.write(f'Rand score: {adjusted_rand_score(self.labels,self.true_labels)}\tNMI: {adjusted_mutual_info_score(self.labels,self.true_labels)}')
            for arg in ['enc_lr','dec_lr','centroids_lr','clmbda','anneal_dec','gan_dec_centroids','clamp_closs']:
                f.write(f'{arg}: {vars(ARGS)[arg]}\n')
                    
    def train_labels(self,epochs,bs):
        dl = data.DataLoader(self.labeled_ds,batch_sampler=data.BatchSampler(data.RandomSampler(self.labeled_ds),bs,drop_last=False),pin_memory=False)        
        ce_loss_func = nn.CrossEntropyLoss()
        rlosses, closses, crlosses = [],[],[]
        for epoch in range(epochs):
            total_rloss = torch.tensor(0.,device='cuda')
            total_closs = torch.tensor(0.,device='cuda')
            total_crloss = torch.tensor(0.,device='cuda')
            total_rrloss = torch.tensor(0.,device='cuda')
            num_easys = 0
            for i, (xb,yb,idx) in enumerate(dl):
                latent = self.enc(xb)
                hmask = yb>=0
                yb_ = yb*hmask
                centroid_targets = self.centroids[yb_]
                crpred = self.dec(centroid_targets[:,:,None,None])
                rpred = self.dec(utils.noiseify(latent,ARGS.noise))
                rloss_ = self.loss_func(rpred,xb).mean(dim=[1,2,3])
                closs_ = self.closs_func(latent,centroid_targets[:,:,None,None]).mean(dim=1)
                crloss_ = self.loss_func(crpred,xb).mean(dim=[1,2,3])
                #rrloss_ = self.closs_func(rrpred,centroid_targets[:,:,None,None]).mean(dim=1)
                set_trace()
                crmask = crloss_ < 0.1
                mask = hmask*crmask
                num_easys += sum(mask)
                #inverse_dists = self.closs_func(latent[:,None,:,0,0],self.centroids).mean(dim=-1)**-1
                #inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
                #gauss_loss = ce_loss_func(inverse_dists,yb_)
                #if i%5 == 0: print(gauss_loss.item())
                #closs = 0.1*(crloss_[mask]).mean() + rrloss_[mask].mean() + torch.clamp(closs_,min=ARGS.clamp_closs)[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                closs = (crloss_[mask]).mean() + torch.clamp(closs_,min=ARGS.clamp_closs)[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                #closs = (crloss_[mask]).mean() + gauss_loss if mask.any() else torch.tensor(0.,device='cuda')
                rloss = torch.tensor(0.,device='cuda') if mask.all() else (rloss_[~mask]).mean() 
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean()*(1/(i+2))
                total_closs = total_closs*((i+1)/(i+2)) + closs_.mean()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean()*(1/(i+2))
                #total_rrloss = total_rrloss*((i+1)/(i+2)) + rrloss_.mean()*(1/(i+2))
                loss = rloss + ARGS.clmbda*closs
                rlosses.append(rloss); closses.append(closs_.mean()); crlosses.append(crloss_.mean())
                if random.random()<ARGS.mirror_prob: loss += self.closs_func(self.enc(self.dec(self.centroids[:,:,None,None]))[:,:,0,0],self.centroids).mean()
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            print(f'R Loss {total_rloss.item()}, C Loss: {total_closs.item()}, CR Loss: {total_crloss.item()}, num easys: {num_easys}')
        plt.clf(); plt.plot(rlosses,closses,crlosses)
        plt.show()
    
    def check_latents(self,latents,show=False):
        _, axes = plt.subplots(6,2,figsize=(7,7))
        for i,latent in enumerate(latents):
            try:
                outimg = self.dec(latent[None,:,None,None])
                axes.flatten()[i].imshow(outimg[0,0])
            except: set_trace()
        if show: plt.show()
        plt.savefig(f'../experiments/{ARGS.exp_name}/current.png')
        plt.savefig(f'../experiments/{ARGS.exp_name}/{utils.get_datetime_stamp()}.png')
        plt.clf()

    def check_ae_images(self,num_rows=5):
        idxs = np.random.randint(0,len(self.dataset),size=num_rows*4)
        inimgs = self.dataset[idxs][0]
        x = self.enc(inimgs)
        outimgs = self.dec(x)
        _, axes = plt.subplots(num_rows,4,figsize=(7,7))
        for i in range(num_rows):
            axes[i,0].imshow(inimgs[i,0])
            axes[i,1].imshow(outimgs[i,0])
            axes[i,2].imshow(inimgs[i+num_rows,0])
            axes[i,3].imshow(outimgs[i+num_rows,0])
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--overwrite','-o',action='store_true')
    parser.add_argument('--display','-d',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--centroids_lr',type=float,default=0.)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--mirror_lmbda',type=float,default=1.)
    parser.add_argument('--mirror_prob',type=float,default=0.1)
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--anneal_dec',action='store_true')
    parser.add_argument('--clamp_closs',type=float,default=0.)
    parser.add_argument('--exp_name',type=str,default="try")
    ARGS = parser.parse_args()
    print(ARGS)

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if ARGS.test: 
        ARGS.overwrite = True
        ARGS.epochs = 1
        ARGS.exp_name = 'try'
    utils.set_experiment_dir(ARGS.exp_name, ARGS.overwrite)

    global LOAD_START_TIME; LOAD_START_TIME = time()

    enc,dec,lin = utils.get_enc_dec('cuda',latent_size=ARGS.NZ)
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
    simple_clearner = ConceptLearner(enc,dec,lin,mnist,opt=torch.optim.Adam,loss_func=nn.L1Loss(reduction='none'))
    simple_clearner.train_ae(ARGS)
