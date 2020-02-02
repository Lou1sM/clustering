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
import load_labels
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
import random



class ConceptLearner():
    def __init__(self,enc,dec,dataset,opt,loss_func,discrim=None): 
        self.enc,self.dec,self.dataset = enc,dec,dataset
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
                total_mid_loss = 0.
                total_pred_loss = 0.
                for i,(xb,yb,idx) in enumerate(dl):
                    enc_mid, latent = enc(xb)
                    dec_mid, pred = dec(utils.noiseify(latent,ARGS.noise))
                    mid_loss = self.loss_func(enc_mid,dec_mid).mean()
                    pred_loss = self.loss_func(pred,xb).mean()
                    loss = ARGS.mid_lmbda*mid_loss + pred_loss
                    loss.backward(); self.opt.step(); self.opt.zero_grad()
                    total_mid_loss = total_mid_loss*(i+1)/(i+2) + mid_loss.item()/(i+2)
                    total_pred_loss = total_pred_loss*(i+1)/(i+2) + pred_loss.item()/(i+2)
                    if ARGS.test: break
                if ARGS.test: break
                print(f'Epoch: {epoch}\tMid Loss: {total_mid_loss}, Pred Loss {total_pred_loss}')
            torch.save({'encoder':self.enc,'decoder':self.dec},f'pretrained_seed{ARGS.seed}.pt')
        for epoch in range(ARGS.epochs):
            print(f'Epoch: {epoch}')
            determin_dl = data.DataLoader(self.dataset,batch_sampler=data.BatchSampler(data.SequentialSampler(self.dataset),461,drop_last=False),pin_memory=False)        
            for i, (xb,yb,idx) in enumerate(determin_dl):
                mid, latent = self.enc(xb)
                total_mids = mid.view(mid.shape[0],-1).detach().cpu() if i==0 else torch.cat([total_mids,mid.view(mid.shape[0],-1).detach().cpu()],dim=0)
                total_latents = latent.view(latent.shape[0],-1).detach().cpu() if i==0 else torch.cat([total_latents,latent.view(latent.shape[0],-1).detach().cpu()],dim=0)
            if ARGS.test: self.labels = np.zeros(len(total_latents),dtype=np.int32)
            elif epoch==0 and ARGS.pretrained: 
                self.pixel_labels, self.mid_labels, self.latent_labels, self.pruned_pixel_labels, self.pruned_mid_labels, self.pruned_latent_labels  = np.load('saved_pixel_labels.npy'), np.load('saved_mid_labels.npy'), np.load('saved_latent_labels.npy'), np.load('saved_pruned_pixel_labels.npy'), np.load('saved_pruned_mid_labels.npy'), np.load('saved_pruned_latent_labels.npy')
                self.total_labels = np.stack([self.pixel_labels,self.mid_labels,self.latent_labels])
            else:
                if epoch==0:
                    print('Umapping pixels...')
                    self.umapped_pixels = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(self.dataset.x.view(60000,-1).detach().cpu().numpy())
                    pixel_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                    self.pixel_labels = pixel_scanner.fit_predict(self.umapped_pixels)
                print('Umapping mids...')
                self.umapped_mids = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(total_mids.squeeze())
                print('Umapping latents...')
                self.umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(total_latents.squeeze())
                mid_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                latent_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                self.mid_labels = mid_scanner.fit_predict(self.umapped_mids)
                self.latent_labels = latent_scanner.fit_predict(self.umapped_latents)
                ordered = sorted(['pixel_labels', 'mid_labels', 'latent_labels', 'true_labels'],key=lambda x: max(getattr(self,x)), reverse=True)
                lar = getattr(self,ordered[0])
                for not_lar_name in ordered[1:]:
                    not_lar = getattr(self,not_lar_name)
                    not_lar_trans = translate_labellings(not_lar,lar)
                    setattr(self,not_lar_name,not_lar_trans)
                #self.pixel_labels = np.array([pixel_trans[l] for l in self.pixel_labels])
                #self.mid_labels = np.array([mid_trans[l] for l in self.mid_labels])
                #self.latent_labels = np.array([latent_trans[l] for l in self.latent_labels])
                self.pruned_pixel_labels = np.copy(self.pixel_labels)
                self.pruned_mid_labels = np.copy(self.mid_labels)
                self.pruned_latent_labels = np.copy(self.latent_labels)
                self.pruned_pixel_labels[pixel_scanner.probabilities_<1.0] = -1
                self.pruned_mid_labels[mid_scanner.probabilities_<1.0] = -1
                self.pruned_latent_labels[latent_scanner.probabilities_<1.0] = -1
                if epoch==0 and ARGS.save_labels: 
                    #np.save(f'saved_pixel_labels_seed{ARGS.seed}.npy',self.pixel_labels)
                    np.save(f'saved_mid_labels_seed{ARGS.seed}.npy',self.mid_labels)
                    np.save(f'saved_latent_labels_seed{ARGS.seed}.npy',self.latent_labels)
                    np.save(f'saved_pruned_pixel_labels_seed{ARGS.seed}.npy',self.pruned_pixel_labels)
                    np.save(f'saved_pruned_mid_labels_seed{ARGS.seed}.npy',self.pruned_mid_labels)
                    np.save(f'saved_pruned_latent_labels_seed{ARGS.seed}.npy',self.pruned_latent_labels)
            
                if ARGS.pretrain_only: return
            self.total_labels = np.stack([self.pixel_labels,self.mid_labels,self.latent_labels])
            all_agree = np.array([np.unique(self.total_labels[:,i]).shape[0] <= 1 for i in range(60000)])
            self.pruned_pixel_labels[~all_agree] = -1
            self.pruned_mid_labels[~all_agree] = -1
            self.pruned_latent_labels[~all_agree] = -1
            self.mid_centroids = torch.stack([total_mids[(self.pruned_mid_labels==i)*(all_agree),:].mean(axis=0) for i in set(self.pruned_mid_labels) if i != -1]).cuda()
            self.latent_centroids = torch.stack([total_latents[(self.pruned_latent_labels==i)*(all_agree),:].mean(axis=0) for i in set(self.pruned_latent_labels) if i != -1]).cuda()
            self.check_latents(self.latent_centroids,stacked=True,show=False)
            self.num_pixel_classes = max(self.pixel_labels+1)
            self.num_mid_classes = max(self.mid_labels+1)
            self.num_latent_classes = max(self.latent_labels+1)
            d = utils.Dataholder(self.dataset.x,torch.tensor(self.pruned_latent_labels,device='cuda').long())
            #self.labeled_ds = utils.KwargTransformDataset(transforms=[utils.to_float_tensor,utils.add_colour_dimension],device='cuda',x=self.dataset.x,y=self.dataset.y,mid_centroids=self.mid_centroids,latent_centroids=self.latent_centroids)
            self.labeled_ds = utils.TransformDataset(d,[utils.to_float_tensor,utils.add_colour_dimension],x_only=False,device='cuda')
            self.closs_func=nn.MSELoss(reduction='none')
            print(f'Num pixel clusters: {self.num_pixel_classes}\t Num classified: {sum(self.pixel_labels>=0)}')
            print(f'Num mid clusters: {self.num_mid_classes}\t Num classified: {sum(self.mid_labels>=0)}')
            print(f'Num latent clusters: {self.num_latent_classes}\t Num classified: {sum(self.latent_labels>=0)}')
            print(adjusted_rand_score(self.pixel_labels,self.true_labels),adjusted_mutual_info_score(self.pixel_labels,self.true_labels))
            print(adjusted_rand_score(self.mid_labels,self.true_labels),adjusted_mutual_info_score(self.mid_labels,self.true_labels))
            print(adjusted_rand_score(self.latent_labels,self.true_labels),adjusted_mutual_info_score(self.latent_labels,self.true_labels))
            #ax = sns.heatmap(get_confusion_mat(self.true_labels,self.labels)); plt.show(); plt.clf()
            self.train_labels(1 if ARGS.test else 1+epoch,ARGS.batch_size)
            if ARGS.test: break
            for epoch in range(10):
                epoch_loss = 0
                for i, (xb,yb,idx) in enumerate(dl): 
                    enc_mid, latent = self.enc(xb)
                    latent = utils.noiseify(latent,ARGS.noise)
                    dec_mid, pred = self.dec(latent)
                    loss = self.ploss_func(pred,xb)
                    loss.backward(); self.opt.step(); self.opt.zero_grad()
                    epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
                print(f'\tInter Epoch: {epoch}\tLoss: {epoch_loss.item()}')
                if epoch_loss < 0.02: break
        
    def train_labels(self,epochs,bs):
        dl = data.DataLoader(self.labeled_ds,batch_sampler=data.BatchSampler(data.RandomSampler(self.labeled_ds),bs,drop_last=False),pin_memory=False)        
        ce_loss_func = nn.CrossEntropyLoss(reduction='none')
        set_trace()
        for epoch in range(epochs):
            num_easys = 0
            total_rloss = 0.
            total_closs = 0.
            total_gauss_loss = 0.
            total_crloss = 0.
            total_rrloss = 0.
            rloss_list = []
            for i, (xb,yb,idx) in enumerate(dl):
                enc_mid,latent  = self.enc(xb)
                hmask = yb>=0
                yb_ = yb*hmask
                mid_centroid_targets = self.mid_centroids[self.pruned_mid_labels[idx]]
                latent_centroid_targets = self.latent_centroids[yb_]
                cdec_mid,crpred = self.dec(latent_centroid_targets[:,:,None,None])
                dec_mid,rpred = self.dec(utils.noiseify(latent,ARGS.noise))
                rloss_ = self.loss_func(rpred,xb).mean(dim=[1,2,3])
                closs_ = self.closs_func(latent,latent_centroid_targets[:,:,None,None]).mean(dim=1)
                crloss_ = self.loss_func(crpred,xb).mean(dim=[1,2,3])
                def get_gauss_loss(preds,centroids,targets):
                    inverse_dists = self.closs_func(preds[:,None,:,0,0],centroids).mean(dim=-1)**-1
                    inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
                    return ce_loss_func(inverse_dists,targets)
                inverse_dists = self.closs_func(latent[:,None,:,0,0],self.latent_centroids).mean(dim=-1)**-1
                inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
                gauss_loss = ce_loss_func(inverse_dists,yb_)
                assert (gauss_loss == get_gauss_loss(latent,self.latent_centroids,yb_)).all()
                mid_gauss_loss = get_gauss_loss(latent,self.latent_centroids,yb_)
                #crmask = crloss_ < ARGS.crthresh
                #mask = hmask*crmask
                mask = hmask
                num_easys += sum(mask)
                #closs = 0.1*(crloss_[mask]).mean() + closs_[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                #closs = 0.1*(crloss_[mask]).mean() + torch.clamp(closs_,min=ARGS.clamp_closs)[mask].mean() if mask.any() else torch.tensor(0.,device='cuda')
                #closs = (crloss_[mask]).mean() + gauss_loss[mask].mean()  if mask.any() else torch.tensor(0.,device='cuda')
                closs = 0.1*(crloss_[mask]).mean() + torch.clamp(gauss_loss, min=ARGS.clamp_gauss_loss)[mask].mean()  if mask.any() else torch.tensor(0.,device='cuda')
                rloss = torch.tensor(0.,device='cuda') if mask.all() else (rloss_[~mask]).mean() 
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss_.mean().item()*(1/(i+2))
                total_closs = total_closs*((i+1)/(i+2)) + closs_.mean().item()*(1/(i+2))
                total_gauss_loss = total_gauss_loss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
                total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean().item()*(1/(i+2))
                loss = rloss + ARGS.clmbda*closs
                #loss = gauss_loss.mean()
                loss.backward(); self.opt.step(); self.opt.zero_grad()
            print(f'R Loss {total_rloss}, C Loss: {total_closs}, CR Loss: {total_crloss}, Gauss Loss {total_gauss_loss} num easys: {num_easys}')
    
    def check_latents(self,latents,show,stacked):
        _, axes = plt.subplots(6,2,figsize=(7,7))
        for i,latent in enumerate(latents):
            try:
                outimg = self.dec(latent[None,:,None,None])
                if stacked: outimg = outimg[-1]
                axes.flatten()[i].imshow(outimg[0,0])
            except: set_trace()
        if show: plt.show()
        plt.clf()
    
   
def train_ae(x): x.train_ae(ARGS)


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
    parser.add_argument('--num_aes',type=int,default=10)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--stacked',action='store_true')
    parser.add_argument('--pretrain_only',action='store_true')
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--mid_lmbda',type=float,default=1.)
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--crthresh',type=float,default=0.1)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--save_labels',action='store_false',default=True)
    ARGS = parser.parse_args()
    print(ARGS)

    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    global LOAD_START_TIME; LOAD_START_TIME = time()

    #enc,dec,lin = utils.get_enc_dec('cuda',latent_size=ARGS.NZ)
    mnist_ds = utils.get_mnist_dset(x_only=True)
    mnist_train = utils.get_mnist_dset(x_only=False)
    mnist = utils.get_mnist_dset()
    clearners = []
    for i in range(ARGS.num_aes):
        enc_b1, enc_b2 = utils.get_enc_blocks('cuda',ARGS.NZ)
        enc = utils.EncoderStacked(enc_b1,enc_b2)
        dec = utils.GeneratorStacked(nz=ARGS.NZ,ngf=32,nc=1,dropout_p=0.)
        dec.to('cuda')
        clearner = ConceptLearner(enc,dec,mnist,opt=torch.optim.Adam,loss_func=nn.L1Loss(reduction='none'))
        clearners.append(clearner)
    from multiprocessing import Pool
    with Pool(processes=10) as pool:
        pool.map(train_ae, clearners)
        #pool.map(print, range(5))
