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
from scipy.optimize import linear_sum_assignment
import random


class ConceptLearner():
    def __init__(self,enc,dec,lin,dataset,opt,loss_func,discrim=None): 
        self.enc,self.dec,self.lin,self.dataset = enc,dec,lin,dataset
        self.ploss_func,self.loss_func,self.ce_loss_func = nn.L1Loss(),loss_func,nn.CrossEntropyLoss()
        if ARGS.pretrained:
            saved_model = torch.load('pretrained.pt')
            self.enc,self.dec = saved_model['encoder'],saved_model['decoder']
        self.opt = opt(params = self.enc.parameters(), lr=ARGS.enc_lr)
        self.opt.add_param_group({'params':self.dec.parameters(),'lr':ARGS.dec_lr})
        
    def train_ae(self,ARGS):
        dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.RandomSampler(self.dataset),ARGS.batch_size,drop_last=True),pin_memory=False)        
        self.true_labels=np.array([mnist[i][1] for i in range(len(mnist))])
        if ARGS.pretrained: pass
        else:
            for epoch in range(ARGS.pretrain_epochs):
                epoch_loss = 0
                for i, (xb,yb,idx) in enumerate(dl): 
                    latent = self.enc(xb)
                    latent = utils.noiseify(latent,ARGS.noise*(epoch/ARGS.pretrain_epochs))
                    loss = self.ploss_func(self.dec(latent),xb)
                    loss.backward(); self.opt.step(); self.opt.zero_grad()
                    epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
        for epoch in range(ARGS.epochs):
            print(f'Epoch: {epoch}')
            determin_dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.SequentialSampler(self.dataset),461,drop_last=False),pin_memory=False)        
            for i, (xb,yb,idx) in enumerate(determin_dl):
                latent = self.enc(xb)
                total_latents = latent.detach().cpu() if i==0 else torch.cat([total_latents,latent.detach().cpu()],dim=0)
            if ARGS.test: self.labels = np.zeros(len(total_latents),dtype=np.int32)
            elif epoch==0 and ARGS.pretrained: self.labels = np.load('saved_labels.npy')
            else:
                self.umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(total_latents.squeeze())
                scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                self.labels = scanner.fit_predict(self.umapped_latents)
                if epoch==0: np.save('saved_labels.npy',self.labels)
            self.centroids = torch.stack([total_latents[self.labels==i,:,0,0].mean(axis=0) for i in range(max(self.labels)+1)]).cuda()
            self.num_classes = max(self.labels+1)
            d = utils.Dataholder(self.dataset.x,torch.tensor(self.labels,device='cuda').long())
            self.labeled_ds = utils.TransformDataset(d,[utils.to_float_tensor,utils.add_colour_dimension],x_only=False,device='cuda')
            self.closs_func=nn.MSELoss(reduction='none')
            self.train_labels(1,bs=ARGS.batch_size)
        
    def train_labels(self,epochs,bs):
        dl = data.DataLoader(self.labeled_ds,batch_sampler=data.BatchSampler(data.RandomSampler(self.labeled_ds),bs,drop_last=False),pin_memory=False)        
        ce_loss_func = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            num_easys = 0
            total_rloss = torch.tensor(0.,device='cuda')
            #total_rloss = 0.
            total_closs = torch.tensor(0.,device='cuda')
            #total_closs = 0.
            total_crloss = torch.tensor(0.,device='cuda')
            #total_crloss = 0.
            total_rrloss = torch.tensor(0.,device='cuda')
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
                crmask = crloss_ < 0.1
                mask = hmask*crmask
                closs = 0.1*(crloss_[mask]).mean() + closs_[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                #total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean().item()*(1/(i+2))
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean()*(1/(i+2))
                #total_closs = total_closs*((i+1)/(i+2)) + closs_.mean().item()*(1/(i+2))
                total_closs = total_closs*((i+1)/(i+2)) + closs_.mean()*(1/(i+2))
                #total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean().item()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean()*(1/(i+2))
                rloss = torch.tensor(0.,device='cuda') if mask.all() else (rloss_[~mask]).mean() 
                total_closs = total_closs*((i+1)/(i+2)) + closs_.mean()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean()*(1/(i+2))
                #loss = rloss + ARGS.clmbda*closs
                loss = rloss + closs
                #loss = closs_.mean()
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            print(f'R Loss {total_rloss}, C Loss: {total_closs}, CR Loss: {total_crloss}, num easys: {num_easys}')
    
    def check_latents(self,latents,show=False):
        _, axes = plt.subplots(6,2,figsize=(7,7))
        for i,latent in enumerate(latents):
            try:
                outimg = self.dec(latent[None,:,None,None])
                axes.flatten()[i].imshow(outimg[0,0])
            except: set_trace()
        if show: plt.show()
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


def label_assignment_cost(labels1,labels2,label1,label2):
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def translate_labellings(labels1,labels2):
    cost_matrix = np.array([[label_assignment_cost(labels1,labels2,l1,l2) for l2 in set(labels2)] for l1 in set(labels1)])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind

def get_confusion_mat(labels1,labels2):
    if max(labels1) != max(labels2): 
        print('Different numbers of clusters, no point trying'); return
    trans_labels = translate_labellings(labels1,labels2)
    num_labels = max(labels1)+1
    confusion_matrix = np.array([[len([idx for idx in range(len(labels2)) if labels1[idx]==l1 and labels2[idx]==l2]) for l2 in range(num_labels)] for l1 in range(num_labels)])
    confusion_matrix = confusion_matrix[:,trans_labels]
    idx = np.arange(num_labels)
    confusion_matrix[idx,idx]=0
    return confusion_matrix

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--overwrite','-o',action='store_true')
    parser.add_argument('--display','-d',action='store_true')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--pretrained','-pt',action='store_true')
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--noise',type=float,default=1.5)
    ARGS = parser.parse_args()
    print(ARGS)

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    global LOAD_START_TIME; LOAD_START_TIME = time()

    enc,dec,lin = utils.get_enc_dec('cuda',latent_size=ARGS.NZ)
    mnist_ds = utils.get_mnist_dset(x_only=True)
    mnist_train = utils.get_mnist_dset(x_only=False)
    mnist = utils.get_mnist_dset()
    simple_clearner = ConceptLearner(enc,dec,lin,mnist,opt=torch.optim.Adam,loss_func=nn.L1Loss(reduction='none'))
    simple_clearner.train_ae(ARGS)
