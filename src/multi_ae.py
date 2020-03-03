import signal
import sys
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
import warnings
warnings.filterwarnings('ignore')
from functools import partial
from pdb import set_trace
from scipy.stats import entropy
from time import time
from torch.utils import data
import copy
import hdbscan
import numpy as np
import random
import torch
import torch.nn as nn
import umap
import utils


def rtrain_ae(ae_dict,args,should_change):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid,ae = ae_dict['aeid'],ae_dict['ae']
        print(f'Rtraining {aeid}')
        befores = [p.detach().cpu() for p in ae.parameters()]
        dset = utils.get_mnist_dset() if args.dset == 'MNIST' else utils.get_fashionmnist_dset()
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
        loss_func=nn.L1Loss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
        for epoch in range(args.pretrain_epochs):
            total_loss = 0.
            for i,(xb,yb,idx) in enumerate(dl):
                enc_mid, latent = ae.enc(xb)
                dec_mid, pred = ae.dec(utils.noiseify(latent,args.noise))
                loss = loss_func(pred,xb).mean()
                loss.backward(); opt.step(); opt.zero_grad()
                total_loss = total_loss*(i+1)/(i+2) + loss.item()/(i+2)
                if args.test: break
            print(f'Rtraining AE {aeid}, Epoch: {epoch}, Loss {round(total_loss,4)}')
            if args.test: break
        if should_change:
            afters = [p.detach().cpu() for p in ae.parameters()]
            for b,a in zip(befores,afters): assert not (b==a).all()
    if args.save: torch.save({'enc':ae.enc,'dec':ae.dec},f'../{args.dset}/checkpoints/pt{aeid}.pt')

def generate_vecs_single(ae_dict,args):
    aeid, ae = ae_dict['aeid'],ae_dict['ae']
    if args.test:
        latents = np.random.random((60000,50)).astype(np.float32)
        return {'aeid':aeid,'latents':latents}
    else:
        device = torch.device(f'cuda:{args.gpu}')
        with torch.cuda.device(device):
            dset = utils.get_mnist_dset() if args.dset == 'MNIST' else utils.get_fashionmnist_dset()
            ae.to(device)
            determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
            for i, (xb,yb,idx) in enumerate(determin_dl):
                mid, latent = ae.enc(xb)
                latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
                latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
            if args.save:
                np.save(f'../{args.dset}/vecs/latents{ae.identifier}.npy',latents)
            return {'aeid':aeid,'latents':latents}

def label_single(ae_output,args):
    aeid,latents = ae_output['aeid'],ae_output['latents']
    if args.test:
        print('Skipping umap and hdbscan')
        latent_labels = np.tile(np.arange(10),6000)
        umapped_latents = None
    else:
        umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(latents.squeeze())
        latent_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        latent_labels = latent_scanner.fit_predict(umapped_latents)
        if args.save:
            np.save(f'../{args.dset}/labels/latent_labels{aeid}.npy',latent_labels)
            np.save(f'../{args.dset}/umaps/latent_umaps{aeid}.npy',umapped_latents)
    return {'aeid':aeid,'umapped_latents':umapped_latents,'latent_labels':latent_labels}

def build_ensemble(vecs_and_labels,args,pivot,given_gt):
    nums_labels = [utils.get_num_labels(ae_results['latent_labels']) for ae_results in vecs_and_labels.values()]
    counts = {x:nums_labels.count(x) for x in set(nums_labels)}
    ensemble_num_labels = sorted([x for x,c in counts.items() if c==max(counts.values())])[0]
    print(f'All nums labels: {nums_labels}, ensemble num labels: {ensemble_num_labels}')
    assert given_gt is None or ensemble_num_labels == utils.get_num_labels(given_gt)
    usable_latent_labels = {aeid:result['latent_labels'] for aeid,result in vecs_and_labels.items()}
    if pivot is not None and ensemble_num_labels == utils.num_labs(pivot):
        same_lang_labels = utils.debable(list(usable_latent_labels.values()),pivot=pivot)
    else:
        same_lang_labels = utils.debable(list(usable_latent_labels.values()),pivot=None)
    multihots = utils.ask_ensemble(np.stack(same_lang_labels))
    all_agree = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)==len(usable_latent_labels))
    ensemble_labels_ = multihots.argmax(axis=1)
    ensemble_labels = given_gt if given_gt is not None else ensemble_labels_
    solid_labels = [i for i in sorted(set(ensemble_labels)) if ((ensemble_labels==i)*all_agree).any()]
    ensemble_labels = utils.compress_labels([l if l in solid_labels else -1 for l in ensemble_labels])
    multihots = multihots[:,sorted(solid_labels)] # Drop columns of non-solid labels
    centroids_by_id = {}
    for aeid in set(usable_latent_labels.keys()):
        new_centroid_info={'aeid':aeid}
        if not all([((ensemble_labels==i)*all_agree).any() for i in sorted(set(ensemble_labels)) if i != -1]):
            print('No all agree vecs for centroids')
            print({i:((ensemble_labels==i)*all_agree).any() for i in sorted(set(ensemble_labels))})
            continue
        else:
            latent_centroids = np.stack([vecs_and_labels[aeid]['latents'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels)) if i!= -1])
        try:
            assert (latent_centroids == latent_centroids).all()
        except: set_trace()
        new_centroid_info['latent_centroids'] = latent_centroids
        if args.save:
            print(f'saving to ../{args.dset}/centroids/latent_centroids{aeid}.npy')
            np.save(f'../{args.dset}/centroids/latent_centroids{aeid}.npy', latent_centroids)
        centroids_by_id[aeid] = new_centroid_info

    if args.save:
        np.save(f'../{args.dset}/ensemble_labels.npy', ensemble_labels)
        np.save(f'../{args.dset}/multihots.npy', multihots)
        np.save(f'../{args.dset}/all_agree.npy', all_agree)
    try: assert (ensemble_labels[all_agree] == multihots[all_agree].argmax(-1)).all()
    except: set_trace()
    return centroids_by_id, multihots, all_agree, ensemble_labels

def train_ae(ae_dict,args,centroids_by_id,multihots,all_agree,worst3):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid, ae = ae_dict['aeid'],ae_dict['ae']
        befores = [p.detach().cpu() for p in ae.parameters()]
        if aeid not in centroids_by_id.keys():
            print(f'{aeid} is not in {list(centroids_by_id.keys())}')
            rtrain_ae(ae_dict,args=args,should_change=False)
            return aeid
        centroids_dict = centroids_by_id[aeid]
        latent_centroids = torch.tensor(centroids_by_id[aeid]['latent_centroids'],device='cuda')

        DATASET = utils.get_mnist_dset() if args.dset == 'MNIST' else utils.get_fashionmnist_dset()
        dl = data.DataLoader(DATASET,batch_sampler=data.BatchSampler(data.RandomSampler(DATASET),args.batch_size,drop_last=True),pin_memory=False)
        loss_func=nn.L1Loss(reduction='none')
        def ce_loss_func(pred,gt):
            pred_probs = pred - torch.logsumexp(pred,-1,keepdim=True)
            return -pred_probs*gt
        mse_loss_func = nn.MSELoss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
        opt.add_param_group({'params':ae.pred.parameters(),'lr':1e-3})
        opt.add_param_group({'params':ae.pred2.parameters(),'lr':1e-3})
        opt.add_param_group({'params':ae.pred3.parameters(),'lr':1e-3})
        confidence_mask = (multihots.max(axis=-1)>0)
        m = multihots+~confidence_mask[:,None]
        multihots_entropy = entropy(m,axis=-1)
        confidences = 1/(1+multihots_entropy)
        assert (confidences==confidences).all()
        multihots = torch.tensor(multihots,device='cuda')
        all_agree = torch.tensor(all_agree,device=device)
        for epoch in range(args.epochs):
            total_rloss = 0.
            total_closs = 0.
            total_gloss = 0.
            total_crloss = 0.
            total_w2loss = 0.
            total_w3loss = 0.
            rloss_list = []
            for i, (xb,yb,idx) in enumerate(dl):
                try:enc_mid,latent  = ae.enc(xb)
                except Exception as e:print(e); set_trace()
                targets = multihots[idx,:]
                wor1 = (targets.argmax(1) == worst3[0])
                wor2 = (targets.argmax(1) == worst3[1])
                wor3 = (targets.argmax(1) == worst3[2])
                mask = all_agree[idx]
                worstmask2 = (wor1 + wor2)
                worstmask3 = (wor1 + wor2 + wor3)
                try:
                    worsttargets2 = targets[worstmask2][:,torch.tensor(worst3[:-1],device=device)]
                    worsttargets3 = targets[worstmask3][:,torch.tensor(worst3,device=device)]
                except Exception as e:
                    print(e)
                    set_trace()
                dec_mid,rpred = ae.dec(utils.noiseify(latent,args.noise))
                rloss = loss_func(rpred,xb)
                if not mask.any():
                    loss = rloss.mean()
                else:
                    if worstmask2.any():
                        w2loss = ce_loss_func(ae.pred2(latent[worstmask2,:,0,0]),worsttargets2)
                    else: w2loss = torch.tensor(0.,device=device)
                    if worstmask3.any():
                        w3loss = ce_loss_func(ae.pred3(latent[worstmask3,:,0,0]),worsttargets3)
                    else: w3loss = torch.tensor(0.,device=device)
                    lin_pred = ae.pred(utils.noiseify(latent,args.noise)[:,:,0,0])
                    gauss_loss = ce_loss_func(lin_pred,targets).mean()
                    closs = torch.clamp(gauss_loss, min=args.clamp_gauss_loss).mean() + rloss[~mask].mean() + 0.1*rloss.mean() + w2loss.mean() + w3loss.mean()
                    latent_centroid_targets = latent_centroids[targets.argmax(1)]
                    mask2 = ~(latent_centroid_targets==0).max(dim=-1)[0]
                    mask *= mask2
                    try:cdec_mid,crpred = ae.dec(latent_centroid_targets[mask,:,None,None])
                    except: print(f'Not working at {aeid}')
                    crloss_ = loss_func(crpred,xb[mask]).mean(dim=[1,2,3])
                    closs += 0.1*(crloss_).mean()
                    total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean().item()*(1/(i+2))
                    total_rloss = total_rloss*((i+1)/(i+2)) + rloss.mean().item()*(1/(i+2))
                    total_w2loss = total_w2loss*((i+1)/(i+2)) + w2loss.mean().item()*(1/(i+2))
                    total_w3loss = total_w3loss*((i+1)/(i+3)) + w3loss.mean().item()*(1/(i+3))
                    total_gloss = total_gloss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
                    loss = rloss.mean() + args.clmbda*closs
                assert loss != 0
                loss.backward(); opt.step(); opt.zero_grad()
                if args.test: break
            print(f'AE: {aeid}, Epoch: {epoch} RLoss: {round(total_rloss,3)}, GaussLoss: {round(total_gloss,3)}, CRLoss: {round(total_crloss,3)}, W2: {round(total_w2loss,2)}, W3: {round(total_w3loss,3)}')
            if args.test: break
        for epoch in range(args.inter_epochs):
            epoch_loss = 0
            for i, (xb,yb,idx) in enumerate(dl):
                enc_mid, latent = ae.enc(xb)
                latent = utils.noiseify(latent,args.noise)
                dec_mid, pred = ae.dec(latent)
                loss = loss_func(pred,xb).mean()
                loss.backward(); opt.step(); opt.zero_grad()
                epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
                if args.test: break
            if args.test: break
            print(f'\tInter Epoch: {epoch}\tLoss: {epoch_loss.item()}')
            if epoch_loss < 0.02: break
        afters = [p.detach().cpu() for p in ae.parameters()]
        for b,a in zip(befores,afters): assert not (b==a).all()
        if args.save: torch.save({'enc':ae.enc,'dec':ae.dec},f'../{args.dset}/checkpoints/{aeid}.pt')
    print('Finished epochs for ae', aeid)

def load_ae(aeid,args):
    checkpoints_dir = "checkpoints_solid" if args.solid else "checkpoints"
    chkpt = torch.load(f'../{args.dset}/{checkpoints_dir}/pt{aeid}.pt', map_location=f'cuda:{args.gpu}')
    revived_ae = utils.AE(chkpt['enc'],chkpt['dec'],aeid)
    return {'aeid': aeid, 'ae':revived_ae}

def load_vecs(aeid,args):
    latents = np.load(f'../{args.dset}/vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'latents':latents}

def load_labels(aeid,args):
    latent_labels = np.load(f'../{args.dset}/labels/latent_labels{aeid}.npy')
    umapped_latents = None
    return {'aeid': aeid, 'umapped_latents':umapped_latents, 'latent_labels': latent_labels}

def load_ensemble(aeids,args):
    centroids_by_id = {}
    for aeid in aeids:
        new_centroids = {'aeid':aeid}
        try:
            new_centroids['latent_centroids'] = np.load(f'../{args.dset}/centroids/latent_centroids{aeid}.npy')
            centroids_by_id[aeid] = new_centroids
        except FileNotFoundError: pass
    multihots = np.load(f'../{args.dset}/multihots.npy')
    all_agree = np.load(f'../{args.dset}/all_agree.npy')
    return centroids_by_id, multihots, all_agree

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--dset',type=str,default='MNIST',choices=['MNIST','FashionMNIST'])
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--exp_name',type=str,default='try')
    parser.add_argument('--gen_batch_size',type=int,default=100)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--inter_epochs',type=int,default=8)
    parser.add_argument('--max_meta_epochs',type=int,default=30)
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--sections',type=int,nargs='+')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--solid',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ae_range',type=int,nargs='+')
    group.add_argument('--aeids',type=int,nargs='+')
    group.add_argument('--num_aes',type=int)
    ARGS = parser.parse_args()
    print(ARGS)

    if ARGS.short_epochs:
        ARGS.pretrain_epochs = 1
        ARGS.epochs = 1
        ARGS.inter_epochs = 1
        ARGS.pretrain_epochs = 1
        ARGS.meta_epochs = 2

    if ARGS.test and ARGS.save:
        print("shouldn't be saving for a test run")
        sys.exit()
    if ARGS.test and ARGS.max_meta_epochs == 30: ARGS.max_meta_epochs = 2
    global LOAD_START_TIME; LOAD_START_TIME = time()
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import multiprocessing as mp
    if ARGS.aeids is not None:
        aeids = ARGS.aeids
    elif ARGS.num_aes is not None:
        aeids = range(ARGS.num_aes)
    else:
        assert len(ARGS.ae_range) == 2
        aeids = range(ARGS.ae_range[0],ARGS.ae_range[1])
    ctx = mp.get_context("spawn")

    filled_pretrain = partial(rtrain_ae,args=ARGS,should_change=True)
    filled_generate = partial(generate_vecs_single,args=ARGS)
    filled_label = partial(label_single,args=ARGS)
    filled_load_ae = partial(load_ae,args=ARGS)
    filled_load_vecs = partial(load_vecs,args=ARGS)
    filled_load_labels = partial(load_labels,args=ARGS)
    filled_load_ensemble = partial(load_ensemble,args=ARGS)
    device = torch.device(f'cuda:{ARGS.gpu}')
    if 1 in ARGS.sections: #Rtrain aes rather than load
        pretrain_start_time = time()
        aes = []
        print(f"Instantiating {len(aeids)} aes...")
        with torch.cuda.device(device):
            for aeid in aeids:
                aes.append({'aeid':aeid, 'ae':utils.make_ae(aeid,device=device,NZ=ARGS.NZ)})
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Pretraining {len(aes)} aes...")
            r=[filled_pretrain(ae) for ae in aes] if ARGS.single else pool.map(filled_pretrain, aes)
        print(f'Pretraining time: {utils.asMinutes(time()-pretrain_start_time)}')
    else:
        with ctx.Pool(processes=len(aeids)) as pool:
            print(f"Loading {len(aeids)} aes...")
            aes = [filled_load_ae(aeid) for aeid in aeids] if ARGS.single else pool.map(filled_load_ae, aeids)
        try: assert all([a['ae'].enc.block2[-2][0].out_channels==ARGS.NZ for a in aes])
        except:
            print(f'Mismatch between latent size of loaded aes, and NZ argument')
            print([a['ae'].enc.block2[-2][0].out_channels for a in aes],'\n')
            sys.exit()

    if 2 in ARGS.sections:
        generate_start_time = time()
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Generating vecs for {len(aes)} aes...")
            vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
        print(f'Generation time: {utils.asMinutes(time()-generate_start_time)}')
    elif 3 in ARGS.sections or 4 in ARGS.sections:
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Loading vecs for {len(aes)} aes...")
            vecs = [filled_load_vecs(aeid) for aeid in aeids] if ARGS.single else pool.map(filled_load_vecs, aeids)
        assert all([v['latents'].shape[1] == ARGS.NZ for v in vecs])

    from sklearn.metrics import normalized_mutual_info_score as mi
    try: gt_labels = np.load(f'../{ARGS.dset}/labels/gt_labels.npy')
    except FileNotFoundError:
        d = utils.get_mnist_dset() if ARGS.dset == 'MNIST' else utils.get_fashionmnist_dset()
        set_trace()
        gt_labels = d.y
        np.save(f'../{ARGS.dset}/labels/gt_labels.npy',gt_labels)
    if 3 in ARGS.sections:
        label_start_time = time()
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Labelling {len(aes)} aes...")
            labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
        print(f'Labelling time: {utils.asMinutes(time()-label_start_time)}')
    elif 4 in ARGS.sections:
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Loading labels for {len(aes)} aes...")
            labels = [filled_load_labels(aeid) for aeid in aeids] if ARGS.single else pool.map(filled_load_labels, aeids)

    if 4 in ARGS.sections:
        scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
        concat_umap_start_time = time()
        if ARGS.test: concat_labels = labels[0]['latent_labels']
        elif len(labels) == 1: concatted_labels = labels[0]['latent_labels']
        else:
            print('Umapping concatted vecs...')
            umapped_concats = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(concatted_vecs)

            print(f'Umap concat time: {utils.asMinutes(time()-concat_umap_start_time)}')
            concat_scan_start_time = time()
            print('Scanning umapped concatted vecs...')
            concatted_labels = scanner.fit_predict(umapped_concats)
            if ARGS.save: np.save(f'../{ARGS.dset}/labels/concatted_labels.npy',concatted_labels)
            print(f'Concat scan time: {utils.asMinutes(time()-concat_scan_start_time)}')
        ensemble_build_start_time = time()
        vecs = utils.dictify_list(vecs,key='aeid')
        labels = utils.dictify_list(labels,key='aeid')
        vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
        print(f"Building ensemble from {len(aes)} aes...")
        centroids_by_id, multihots, all_agree, ensemble_labels = build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt=None)
        ensemble_labels = multihots.argmax(-1)
        assert (ensemble_labels == multihots.argmax(-1)).all()
        print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
    elif 5 in ARGS.sections:
        print(f"Loading ensemble from {len(aes)} aes...")
        centroids_by_id, multihots, all_agree = filled_load_ensemble(aeids)
        ensemble_labels = multihots.argmax(-1)
        concatted_labels = np.load(f'../{ARGS.dset}/labels/concatted_labels.npy')

    if 5 in ARGS.sections:
        train_start_time = time()
        best_acc = 0.
        count = 0
        for meta_epoch_num in range(ARGS.max_meta_epochs):
            meta_epoch_start_time = time()
            print('\nMeta epoch:', meta_epoch_num)
            copied_aes = [copy.deepcopy(ae) for ae in aes]
            if len(aes) == 0:
                print('No legit aes left')
                sys.exit()
            print('Aes remaining:', *list(aeids))
            difficult_list = [1,2,3] if ARGS.test else sorted([l for l in set(ensemble_labels) if l != -1], key=lambda x: (concatted_labels[~all_agree]==x).sum()/(concatted_labels==x).sum(),reverse=True)
            print('difficults:',difficult_list)
            worst3 = difficult_list[:3]
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Training aes {list(aeids)}...")

                try: assert set([ae['aeid'] for ae in copied_aes]) == set(aeids)
                except: set_trace()
                num_labels = utils.get_num_labels(ensemble_labels)
                for aedict in copied_aes:
                    try: assert(aedict['ae'].pred.in_features == ARGS.NZ)
                    except:aedict['ae'].pred = utils.mlp(ARGS.NZ,25,num_labels,device=device)
                    aedict['ae'].pred2 = utils.mlp(ARGS.NZ,25,2,device=device)
                    aedict['ae'].pred3 = utils.mlp(ARGS.NZ,25,3,device=device)
                quot = multihots.sum(axis=-1)
                quot += (quot==0) # Replace 0 entries with 1s
                normalized_multihots = multihots/quot[:,None]
                filled_train = partial(train_ae,args=ARGS,centroids_by_id=centroids_by_id,multihots=normalized_multihots,all_agree=all_agree,worst3=worst3)
                [filled_train(ae) for ae in copied_aes] if ARGS.single else pool.map(filled_train, copied_aes)
            aes = copied_aes
            assert set([ae['aeid'] for ae in aes]) == set(aeids)
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Generating vecs for {len(aes)} aes...")
                vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Labelling {len(aes)} aes...")
                labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
            if len(labels) == 1 or ARGS.test: concatted_labels = labels[0]['latent_labels']
            else:
                scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
                umapped_concats = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(concatted_vecs)
                concatted_labels = scanner.fit_predict(umapped_concats)
            vecs = utils.dictify_list(vecs,key='aeid')
            labels = utils.dictify_list(labels,key='aeid')
            vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
            print(f"Building ensemble from {len(aes)} aes...")
            centroids_by_id, multihots, all_agree, ensemble_labels  = build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt=None)
            copied_aes = [copy.deepcopy(ae) if ae['aeid'] in centroids_by_id.keys() else {'aeid': ae['aeid'], 'ae':utils.make_ae(ae['aeid'],device=device,NZ=ARGS.NZ)} for ae in aes]

            acc = utils.accuracy(ensemble_labels,gt_labels)
            concat_acc = utils.accuracy(concatted_labels,gt_labels)
            new_best_acc = max(acc,concat_acc)
            print('AE Scores:')
            print('L acc', [utils.accuracy(labels[x]['latent_labels'][labels[x]['latent_labels']>=0],gt_labels[labels[x]['latent_labels']>=0]) for x in aeids if x in centroids_by_id.keys()])
            print('L MI', [mi(labels[x]['latent_labels'][labels[x]['latent_labels']>=0],gt_labels[labels[x]['latent_labels']>=0]) for x in aeids if x in centroids_by_id.keys()])
            print('Ensemble Scores:')
            print('Acc:',acc)
            print('Concat Acc:', concat_acc)
            print('Acc agree:', utils.accuracy(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('NMI:',mi(ensemble_labels,gt_labels))
            print('NMI agree:',mi(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('Num agrees:', all_agree.sum())
            print(f'Meta Epoch time: {utils.asMinutes(time()-meta_epoch_start_time)}')
            if new_best_acc <= best_acc:
                count += 1
            else:
                best_acc = new_best_acc
                count = 0
                best_ensemble_labels = ensemble_labels
                best_all_agree = all_agree
            print('Best acc:', best_acc)
            print('Count:',count)
            if count == ARGS.patience: break

        ensemble_fname = f'../{ARGS.dset}/experiments/{ARGS.exp_name}_ensemble.json'
        np.savez(ensemble_fname,ensemble_labels=best_ensemble_labels,all_agree=best_all_agree)
        by_aeid = {aeid:{**vecs_and_labels[aeid],**centroids_by_id[aeid]} for aeid in set(vecs_and_labels.keys()).intersection(set(centroids_by_id.keys()))}
        for aeid,v in by_aeid.items():
            fname = f'../{ARGS.dset}/experiments/{ARGS.exp_name}_aeid{aeid}_centroids.json'
            arrays_to_save = {k:v for k,v in by_aeid[aeid].items() if k!='aeid'}
            np.savez(fname,**arrays_to_save)
        exp_outputs = {'ensemble_labels': ensemble_labels,'all_agree':all_agree,'by_aeid':by_aeid}
        print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
