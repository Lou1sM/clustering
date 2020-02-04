import copy
import sys
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


def pretrain_ae(ae_dict,args):
    aeid,ae = ae_dict['aeid'],ae_dict['ae']
    dset = utils.get_mnist_dset()
    dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)        
    loss_func=nn.L1Loss(reduction='none')
    opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
    opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
    for epoch in range(args.pretrain_epochs):
        print(epoch)
        total_mid_loss = 0.
        total_pred_loss = 0.
        for i,(xb,yb,idx) in enumerate(dl):
            enc_mid, latent = ae.enc(xb)
            dec_mid, pred = ae.dec(utils.noiseify(latent,args.noise))
            mid_loss = loss_func(enc_mid,dec_mid).mean()
            pred_loss = loss_func(pred,xb).mean()
            loss = args.mid_lmbda*mid_loss + pred_loss
            loss.backward(); opt.step(); opt.zero_grad()
            total_mid_loss = total_mid_loss*(i+1)/(i+2) + mid_loss.item()/(i+2)
            total_pred_loss = total_pred_loss*(i+1)/(i+2) + pred_loss.item()/(i+2)
            if args.test: break
        if args.test: break
        print(f'Epoch: {epoch}\tMid Loss: {total_mid_loss}, Pred Loss {total_pred_loss}')
    torch.save({'enc':ae.enc,'dec':ae.dec},f'../checkpoints/pt{aeid}.pt')

def build_ensemble(vecs_and_labels):
    # Exclude labellings that disagree with majority num clusters
    nums_labels = [utils.get_num_labels(ae_results[ctype]) for ae_results in vecs_and_labels.values() for ctype in ['mid_labels','latent_labels']]
    counts = {x:nums_labels.count(x) for x in nums_labels}
    ensemble_num_labels = sorted([x for x,c in counts.items() if c==max(counts.values())])[-1]
    usable_mid_labels = {aeid:result['mid_labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['mid_labels']) == ensemble_num_labels}
    usable_latent_labels = {aeid:result['latent_labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['latent_labels']) == ensemble_num_labels}
    same_lang_labels = utils.debable(list(usable_mid_labels.values())+list(usable_latent_labels.values()))
    multihots = utils.ask_ensemble(np.stack(same_lang_labels))
    all_agree = (multihots.max(axis=1)==(len(usable_mid_labels) + len(usable_latent_labels)))
    ensemble_labels = multihots.argmax(axis=1)
    mid_centroids_by_id = {}
    latent_centroids_by_id = {}
    centroids_by_id = {}
    for aeid in vecs_and_labels.keys():
        new_centroid_info={'aeid':aeid}
        if aeid in usable_mid_labels.keys():
            mid_centroids = np.stack([vecs_and_labels[aeid]['mids'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels))])
            new_centroid_info['mid_centroids'] = mid_centroids
            np.save(f'mid_centroids{aeid}.npy', mid_centroids)
        if aeid in usable_latent_labels.keys():
            latent_centroids = np.stack([vecs_and_labels[aeid]['latents'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels))])
            new_centroid_info['latent_centroids'] = latent_centroids
            np.save(f'latent_centroids{aeid}.npy', latent_centroids)
        centroids_by_id[aeid] = new_centroid_info

    np.save('ensemble_labels.npy', ensemble_labels)
    np.save('all_agree.npy', all_agree)
    return centroids_by_id, ensemble_labels, all_agree

def train_ae(ae_dict,args,centroids_by_id,ensemble_labels,all_agree):
    aeid, ae = ae_dict['aeid'],ae_dict['ae']
    try:
        mid_centroids = torch.tensor(centroids_by_id[aeid]['mid_centroids'],device='cuda')
        latent_centroids = torch.tensor(centroids_by_id[aeid]['latent_centroids'],device='cuda')
    except KeyError: print(f"Can't train ae {ae.identifier}"); return
    
    DATASET = utils.get_mnist_dset()
    dl = data.DataLoader(DATASET,batch_sampler=data.BatchSampler(data.RandomSampler(DATASET),args.batch_size,drop_last=True),pin_memory=False)        
    loss_func=nn.L1Loss(reduction='none')
    ce_loss_func = nn.CrossEntropyLoss(reduction='none')
    mse_loss_func = nn.MSELoss(reduction='none')
    opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
    opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
    gt_labels = torch.tensor(ensemble_labels,device='cuda')
    for epoch in range(10):
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
            if args.test: break
        print(f'R Loss {total_rloss}, C Loss: {total_closs}, CR Loss: {total_crloss}, C Loss {total_closs}')
        if args.test: break
    vecs = generate_vecs_single(aedict)
    labels = label_single(test=False,ae_ouput=vecs)
    return {**vecs, **labels}
    
def label_single(ae_output,test):
    aeid,mids,latents = ae_output['aeid'],ae_output['mids'],ae_output['latents']
    if test:
        print('Skipping umap and hdbscan')
        mid_labels = np.random.randint(10,size=60000)
        latent_labels = np.random.randint(10,size=60000)
    else:
        print('Umapping mids...')
        umapped_mids = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(mids.squeeze())
        print('Umapping latents...')
        umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(latents.squeeze())
        print('Scanning mids...')
        mid_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        mid_labels = mid_scanner.fit_predict(umapped_mids)
        print('Scanning latents...')
        latent_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        latent_labels = latent_scanner.fit_predict(umapped_latents)
        np.save(f'../labels/mid_labels{aeid}.npy',mid_labels)
        np.save(f'../labels/latent_labels{aeid}.npy',latent_labels)
    return {'aeid':aeid,'mid_labels':mid_labels,'latent_labels':latent_labels}

def generate_vecs_single(aedict,test):
    aeid,ae = aedict['aeid'],aedict['ae']
    if test:
        mids = np.random.random((60000,4096)).astype(np.float32)
        latents = np.random.random((60000,10)).astype(np.float32)
    else:
        dset = utils.get_mnist_dset()
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),461,drop_last=False),pin_memory=False)        
        for i, (xb,yb,idx) in enumerate(determin_dl):
            mid, latent = ae.enc(xb)
            mid = mid.view(mid.shape[0],-1).detach().cpu().numpy()
            mids = mid if i==0 else np.concatenate([mids,mid],axis=0)
            latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
            latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
        np.save(f'../vecs/mids{ae.identifier}.npy',mids)
        np.save(f'../vecs/latents{ae.identifier}.npy',latents)
    return {'aeid':aeid,'mids':mids,'latents':latents}
    
def load_ae(aeid): 
    chkpt = torch.load(f'../checkpoints/pt{aeid}.pt')
    revived_ae = utils.AE(chkpt['enc'],chkpt['dec'],aeid)
    return {'aeid': aeid, 'ae':revived_ae}
  
def load_vecs(aeid):
    mids = np.load(f'../vecs/mids{aeid}.npy')
    latents = np.load(f'../vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'mids': mids, 'latents':latents}

def load_vecs(aeid):
    mids = np.load(f'../vecs/mids{aeid}.npy')
    latents = np.load(f'../vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'mids': mids, 'latents':latents}

def load_labels(aeid):
    mid_labels = np.load(f'../labels/mid_labels{aeid}.npy')
    latent_labels = np.load(f'../labels/latent_labels{aeid}.npy')
    return {'aeid': aeid, 'mid_labels': mid_labels, 'latent_labels': latent_labels}

def load_ensemble(aeids):
    mid_centroids_by_id, latent_centroids_by_id = {}, {}
    for aeid in aeids:
        try: mid_centroids_by_id[aeid] = np.load(f'mid_centroids{aeid}.npy')
        except FileNotFoundError: pass
        try: latent_centroids_by_id[aeid] = np.load(f'latent_centroids{aeid}.npy')
        except FileNotFoundError: pass
    ensemble_labels = np.load('ensemble_labels.npy')
    all_agree = np.load('all_agree.npy')
    return mid_centroids_by_id, latent_centroids_by_id, ensemble_labels, all_agree

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_aes',type=int,default=10)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--mid_lmbda',type=float,default=0.)
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--crthresh',type=float,default=0.1)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--sections',type=int,nargs='+')
    ARGS = parser.parse_args()
    print(ARGS)

    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.gpu
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    global LOAD_START_TIME; LOAD_START_TIME = time()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=20) as pool:
        aeids = range(ARGS.num_aes)
         
        if 1 in ARGS.sections: #Pretrain aes rather than load
            pretrain_start_time = time()
            aes = []
            for aeid in aeids:
                enc_b1, enc_b2 = utils.get_enc_blocks('cuda',ARGS.NZ)
                enc = utils.EncoderStacked(enc_b1,enc_b2)
                dec = utils.GeneratorStacked(nz=ARGS.NZ,ngf=32,nc=1,dropout_p=0.)
                dec.to('cuda')
                aes.append({'aeid':aeid, 'ae':utils.AE(enc,dec,aeid)})
            filled_pretrain = partial(pretrain_ae,args=ARGS)
            [filled_pretrain(ae) for ae in aes] if ARGS.single else pool.map(filled_pretrain, aes)
            print(f'Pretraining time: {utils.asMinutes(time()-pretrain_start_time)}')
        elif 2 in ARGS.sections or 3 in ARGS.sections or 5 in ARGS.sections:
            aes = [load_ae(aeid) for aeid in aeids] if ARGS.single else pool.map(load_ae, aeids)

        if 2 in ARGS.sections:
            generate_start_time = time()
            filled_generate = partial(generate_vecs_single,test=ARGS.test)
            vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
            print(f'Generation time: {utils.asMinutes(time()-generate_start_time)}')
        elif 3 in ARGS.sections or 4 in ARGS.sections:
            vecs = [load_vecs(aeid) for aeid in aeids] if ARGS.single else pool.map(load_vecs, aeids)

        if 3 in ARGS.sections:
            label_start_time = time()
            filled_label = partial(label_single,test=ARGS.test)
            labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
            print(f'Labelling time: {utils.asMinutes(time()-label_start_time)}')
        elif 4 in ARGS.sections:
            labels = [load_labels(aeid) for aeid in aeids] if ARGS.single else pool.map(load_labels,aeids)

        if 4 in ARGS.sections:
            ensemble_build_start_time = time()
            vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in aeids}
            centroids_by_id, ensemble_labels, all_agree  = build_ensemble(vecs_and_labels)
            print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
        elif 5 in ARGS.sections:
            centroids_by_id, ensemble_labels, all_agree  = load_ensemble(aeids)
            
        if 5 in ARGS.sections:
            train_start_time = time()
            filled_train = partial(train_ae,args=ARGS,centroids_by_id=centroids_by_id,ensemble_labels=ensemble_labels,all_agree=all_agree)
            [filled_train(ae) for ae in aes] if ARGS.single else pool.map(filled_train, [copy.deepcopy(ae) for ae in aes])
            print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
