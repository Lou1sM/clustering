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
from collections import Counter
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
from functools import partial
import torch.nn.functional as F
import utils
import hdbscan
import umap.umap_ as umap
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
import random


class AETrainer():
    def __init__(self,args): 
        self.args = args
        self.aeids = []
        self.latents_by_id, self.mids_by_id = {}, {}
        self.latent_labels_by_id, self.mid_labels_by_id = {}, {}
        
    def pretrain_ae(self,ae):
        self.aeids.append(ae.identifier)
        print(self.aeids)
        DATASET = utils.get_mnist_dset()
        dl = data.DataLoader(DATASET,batch_sampler=data.BatchSampler(data.RandomSampler(DATASET),self.args.batch_size,drop_last=True),pin_memory=False)        
        loss_func=nn.L1Loss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=self.args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':self.args.dec_lr})
        ae.true_labels=np.array([DATASET[i][1] for i in range(len(DATASET))])
        if self.args.pretrained: pass
        else:
            for epoch in range(self.args.pretrain_epochs):
                print(epoch)
                total_mid_loss = 0.
                total_pred_loss = 0.
                for i,(xb,yb,idx) in enumerate(dl):
                    print(i)
                    enc_mid, latent = ae.enc(xb)
                    dec_mid, pred = ae.dec(utils.noiseify(latent,self.args.noise))
                    mid_loss = loss_func(enc_mid,dec_mid).mean()
                    pred_loss = loss_func(pred,xb).mean()
                    loss = self.args.mid_lmbda*mid_loss + pred_loss
                    loss.backward(); opt.step(); opt.zero_grad()
                    total_mid_loss = total_mid_loss*(i+1)/(i+2) + mid_loss.item()/(i+2)
                    total_pred_loss = total_pred_loss*(i+1)/(i+2) + pred_loss.item()/(i+2)
                    if self.args.test: break
                if self.args.test: break
                print(f'Epoch: {epoch}\tMid Loss: {total_mid_loss}, Pred Loss {total_pred_loss}')
            torch.save({'encoder':ae.enc,'decoder':ae.dec},f'../checkpoints/pt{ae.identifier}.pt')
        generate_labels_and_vecs(DATASET,ae)
  
    def load_saved_labels(self,aeids):
        self.mid_labels_by_id = {aeid: np.load(f'../labels/mid_labels{aeid}.npy') for aeid in aeids}
        self.latent_labels_by_id = {aeid: np.load(f'../labels/latent_labels{aeid}.npy') for aeid in aeids}
        return self.mid_labels_by_id, self.latent_labels_by_id

    def build_ensemble_gt(self,ensemble_results):
        # Exclude labellings that disagree with majority num clusters
        nums_labels = [get_num_labels(ae_results[ctype]) for ae_results in ensemble_results.values() for ctype in ['mid_labels','latent_labels']]
        counts = {x:nums_labels.count(x) for x in nums_labels}
        ensemble_num_labels = sorted([x for x,c in counts.items() if c==max(counts.values())])[-1]
        self.usable_mid_labels = {aeid:result['mid_labels'] for aeid,result in ensemble_results.items() if get_num_labels(result['mid_labels']) == ensemble_num_labels}
        self.usable_latent_labels = {aeid:result['latent_labels'] for aeid,result in ensemble_results.items() if get_num_labels(result['latent_labels']) == ensemble_num_labels}
        same_lang_labels = utils.debable(list(self.usable_mid_labels.values())+list(self.usable_latent_labels.values()))
        multihots = utils.ask_ensemble(np.stack(same_lang_labels))
        self.all_agree = (multihots.max(axis=1)==(len(self.usable_mid_labels) + len(self.usable_latent_labels)))
        self.ensemble_labels = multihots.argmax(axis=1)
        self.mid_centroids_by_id = {}
        self.latent_centroids_by_id = {}
        for aeid in self.usable_mid_labels.keys():
            mid_centroids = np.stack([ensemble_results[aeid]['mids'][(self.ensemble_labels==i)*self.all_agree].mean(axis=0) for i in sorted(set(self.ensemble_labels))])
            self.mid_centroids_by_id[aeid] = mid_centroids
            np.save(f'mid_centroids{aeid}.npy', mid_centroids)
        for aeid,latent in self.usable_latent_labels.items():
            latent_centroids = np.stack([ensemble_results[aeid]['latents'][(self.ensemble_labels==i)*self.all_agree].mean(axis=0) for i in sorted(set(self.ensemble_labels))])
            self.latent_centroids_by_id[aeid] = latent_centroids
            np.save(f'latent_centroids{aeid}.npy', latent_centroids)

        np.save('ensemble_labels.npy', self.ensemble_labels)
        np.save('all_agree.npy', self.all_agree)
        return self.mid_centroids_by_id, self.latent_centroids_by_id, self.ensemble_labels, self.all_agree

    def load_ensemble(self,aeids):
        mid_centroids_by_id, latent_centroids_by_id = {}, {}
        for aeid in aeids:
            try: mid_centroids_by_id[aeid] = np.load(f'mid_centroids{aeid}.npy')
            except FileNotFoundError: pass
            try: latent_centroids_by_id[aeid] = np.load(f'latent_centroids{aeid}.npy')
            except FileNotFoundError: pass
        ensemble_labels = np.load('ensemble_labels.npy')
        all_agree = np.load('all_agree.npy')
        return mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree

def train_ae(ae,args,mid_centroids_by_id,latent_centroids_by_id,ensemble_labels,all_agree):
    try:
        mid_centroids = torch.tensor(mid_centroids_by_id[ae.identifier],device='cuda')
        latent_centroids = torch.tensor(latent_centroids_by_id[ae.identifier],device='cuda')
    except KeyError: print(f"Can't train ae {ae.identifier}"); return
    
    DATASET = utils.get_mnist_dset()
    dl = data.DataLoader(DATASET,batch_sampler=data.BatchSampler(data.RandomSampler(DATASET),args.batch_size,drop_last=True),pin_memory=False)        
    loss_func=nn.L1Loss(reduction='none')
    ce_loss_func = nn.CrossEntropyLoss(reduction='none')
    mse_loss_func = nn.MSELoss(reduction='none')
    opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
    opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
    gt_labels = torch.tensor(ensemble_labels,device='cuda')
    for epoch in range(1):
        epoch_loss = 0
        for i, (xb,yb,idx) in enumerate(dl): 
            enc_mid, latent = ae.enc(xb)
            latent = utils.noiseify(latent,args.noise)
            dec_mid, pred = ae.dec(latent)
            loss = loss_func(pred,xb).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
        print(f'\tInter Epoch: {epoch}\tLoss: {epoch_loss.item()}')
        if epoch_loss < 0.02: break
    
    for epoch in range(args.epochs):
        total_rloss = 0.
        total_closs = 0.
        total_gauss_loss = 0.
        total_crloss = 0.
        total_rrloss = 0.
        rloss_list = []
        for i, (xb,yb,idx) in enumerate(dl):
            enc_mid,latent  = ae.enc(xb)
            targets = gt_labels[idx]
            mask = all_agree[idx]
            #mid_centroid_targets = mid_centroids[targets]
            latent_centroid_targets = latent_centroids[targets]
            if ~mask.any():
                dec_mid,rpred = ae.dec(utils.noiseify(latent,args.noise)[~mask])
                rloss_ = loss_func(rpred,xb[~mask]).mean()
            else:
                rloss = torch.tensor(0.,device='cuda')
            cdec_mid,crpred = ae.dec(latent_centroid_targets[mask,:,None,None])
            closs_ = mse_loss_func(latent[mask],latent_centroid_targets[mask,:,None,None]).mean(dim=1)
            crloss_ = loss_func(crpred,xb[mask]).mean(dim=[1,2,3])
            #def get_gauss_loss(preds,centroids,targets):
            #    inverse_dists = mse_loss_func(preds,centroids).mean(dim=-1)**-1
            #    inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
            #    return ce_loss_func(inverse_dists,targets)
            #inverse_dists = closs_func(latent[:,None,:,0,0],latent_centroids).mean(dim=-1)**-1
            #inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
            #gauss_loss = ce_loss_func(inverse_dists,yb_)
            #assert (gauss_loss == get_gauss_loss(latent,latent_centroids,yb_)).all()
            #mid_gauss_loss = get_gauss_loss(enc_mid.view(args.batch_size,1,-1),mid_centroids,targets)
            #latent_gauss_loss = get_gauss_loss(latent.view(args.batch_size,1,-1),latent_centroids,targets)
            #closs = 0.1*(crloss_[mask]).mean() + torch.clamp(latent_gauss_loss, min=args.clamp_gauss_loss)[mask].mean()  if mask.any() else torch.tensor(0.,device='cuda')
            closs = 0.1*crloss_.mean() + torch.clamp(closs_, min=args.clamp_gauss_loss).mean()  if mask.any() else torch.tensor(0.,device='cuda')
            total_rloss = total_rloss*((i+1)/(i+2)) + rloss.item()*(1/(i+2))
            total_closs = total_closs*((i+1)/(i+2)) + closs_.mean().item()*(1/(i+2))
            #total_gauss_loss = total_gauss_loss*((i+1)/(i+2)) + latent_gauss_loss.mean().item()*(1/(i+2))
            total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean().item()*(1/(i+2))
            loss = rloss + args.clmbda*closs
            loss.backward(); opt.step(); opt.zero_grad()
        print(f'R Loss {total_rloss}, C Loss: {total_closs}, CR Loss: {total_crloss}, C Loss {total_closs}')
    return generate_labels_and_vecs(DATASET,ae)
    
def generate_labels_and_vecs(dataset,ae):
    print('Generating mids and latents...')
    mids, latents = generate_mids_and_latents(dataset,ae.enc)
    print('Umapping mids...')
    umapped_mids = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(mids.squeeze())
    print('Umapping latents...')
    umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(latents.squeeze())
    mid_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
    latent_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
    mid_labels = mid_scanner.fit_predict(umapped_mids)
    latent_labels = latent_scanner.fit_predict(umapped_latents)
    pruned_mid_labels = np.copy(mid_labels)
    pruned_latent_labels = np.copy(latent_labels)
    pruned_mid_labels[mid_scanner.probabilities_<1.0] = -1
    pruned_latent_labels[latent_scanner.probabilities_<1.0] = -1
    np.save(f'../labels/mid_labels{ae.identifier}.npy',mid_labels)
    np.save(f'../labels/latent_labels{ae.identifier}.npy',latent_labels)
    np.save(f'../vecs/midls{AE.identifier}.npy',mids)
    np.save(f'../vecs/latentls{AE.identifier}.npy',latentls)
    return {'aeid':AE.identifier,'mids':mids,'latents':latents,'mid_labels':mid_labels,'latent_labels':latent_labels}

def get_num_labels(labels): 
    assert labels.ndim == 1
    return  len(set([l for l in labels if l != -1]))

def generate_mids_and_latents(dset,enc):
    """Return all mids and latents as pytorch tensors"""
    determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),461,drop_last=False),pin_memory=False)        
    for i, (xb,yb,idx) in enumerate(determin_dl):
        mid, latent = enc(xb)
        total_mids = mid.view(mid.shape[0],-1).detach().cpu() if i==0 else torch.cat([total_mids,mid.view(mid.shape[0],-1).detach().cpu()],dim=0)
        total_latents = latent.view(latent.shape[0],-1).detach().cpu() if i==0 else torch.cat([total_latents,latent.view(latent.shape[0],-1).detach().cpu()],dim=0)
    return total_mids, total_latents
    
class AE(nn.Module):
    def __init__(self,enc,dec,identifier): 
        super(AE, self).__init__()
        self.enc,self.dec = enc,dec
        self.identifier = identifier
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=str,default='0')
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
    parser.add_argument('--mid_lmbda',type=float,default=0.)
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--crthresh',type=float,default=0.1)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--save_labels',action='store_false',default=True)
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--regen',action='store_true')
    ARGS = parser.parse_args()
    print(ARGS)

    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    global LOAD_START_TIME; LOAD_START_TIME = time()

    mnist_ds = utils.get_mnist_dset(x_only=True)
    mnist_train = utils.get_mnist_dset(x_only=False)
    mnist = utils.get_mnist_dset()
    AEs = []
    for i in range(ARGS.num_aes):
        enc_b1, enc_b2 = utils.get_enc_blocks('cuda',ARGS.NZ)
        enc = utils.EncoderStacked(enc_b1,enc_b2)
        dec = utils.GeneratorStacked(nz=ARGS.NZ,ngf=32,nc=1,dropout_p=0.)
        dec.to('cuda')
        AEs.append(AE(enc,dec,i))
    ARGSl = [ARGS,ARGS]
    DATASET = utils.get_mnist_dset()
    aetrainer = AETrainer(ARGS)
    if ARGS.single:
        if ARGS.build:
            mids_by_id,latents_by_id = {},{}
            for i,ae in enumerate(AEs):
                if ARGS.regen:
                    mids,latents = generate_mids_and_latents(DATASET,ae.enc)
                    mids = mids.detach().cpu().numpy()
                    latents = latents.detach().cpu().numpy()
                    mids_by_id[i] = mids
                    latents_by_id[i] = latents
                    np.save(f'../vecs/mids{i}.npy',mids)
                    np.save(f'../vecs/latents{i}.npy',latents)
                else:
                    mids_by_id[i] = np.load(f'../vecs/mids{i}.npy')
                    latents_by_id[i] = np.load(f'../vecs/latents{i}.npy')
            aetrainer.mids_by_id = mids_by_id
            aetrainer.latents_by_id = latents_by_id
            mid_labels_by_id, latent_labels_by_id = aetrainer.load_saved_labels(range(ARGS.num_aes))
            loaded_results = {aeid:{'mids':mids_by_id[aeid],'latents':latents_by_id[aeid],'mid_labels':mid_labels_by_id[aeid],'latent_labels':latent_labels_by_id[aeid]} for aeid in range(ARGS.num_aes)}
            mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree = aetrainer.build_ensemble_gt(loaded_results)
        else:
            mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree = aetrainer.load_ensemble(range(ARGS.num_aes))
        filled_train = partial(train_ae,args=ARGS,mid_centroids_by_id=mid_centroids_by_id,latent_centroids_by_id=latent_centroids_by_id,ensemble_labels=ensemble_labels,all_agree=all_agree)
        for ae in AEs:
            filled_train(ae)
        map(filled_train,AEs)
    else: 
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=10) as pool:
            if ARGS.build:
                pretrain_results = pool.map(aetrainer.pretrain_ae, AEs)
                pretrain_results = {r['aeid']: r for r in pretrain_results}
                mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree = aetrainer.build_ensemble_gt(pretrain_results)
            else:
                mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree = aetrainer.load_ensemble(range(ARGS.num_aes))
            filled_train = partial(train_ae,args=ARGS,mid_centroids_by_id=mid_centroids_by_id,latent_centroids_by_id=latent_centroids_by_id,ensemble_labels=ensemble_labels,all_agree=all_agree)
            print(ensemble_labels)
            results = pool.map(filled_train, AEs)
            train_results = {r['aeid']: r for r in train_results}
