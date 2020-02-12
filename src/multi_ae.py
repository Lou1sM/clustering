from functools import partial
from pdb import set_trace
from time import time
from torch.utils import data
import copy
import hdbscan
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import umap.umap_ as umap
import utils
import warnings
warnings.filterwarnings('ignore')


def pretrain_ae(ae_dict,args):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid,ae = ae_dict['aeid'],ae_dict['ae']
        dset = utils.get_mnist_dset()
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
        loss_func=nn.L1Loss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
        for epoch in range(args.pretrain_epochs):
            total_mid_loss = 0.
            total_pred_loss = 0.
            print(f'Pretraining ae {aeid}, epoch {epoch}')
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
            print(f'AE: {aeid}, Epoch: {epoch} Mid Loss: {total_mid_loss}, Pred Loss {total_pred_loss}')
    torch.save({'enc':ae.enc,'dec':ae.dec},f'../checkpoints/pt{aeid}.pt')
    return np.array([dset[i][1] for i in range(len(dset))])

def generate_vecs_single(ae_dict,args):
    aeid, ae = ae_dict['aeid'],ae_dict['ae']
    if args.test:
        mids = np.random.random((60000,4096)).astype(np.float32)
        latents = np.random.random((60000,50)).astype(np.float32)
    else:
        device = torch.device(f'cuda:{args.gpu}')
        with torch.cuda.device(device):
            dset = utils.get_mnist_dset()
            ae.to(device)
            determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
            for i, (xb,yb,idx) in enumerate(determin_dl):
                mid, latent = ae.enc(xb)
                mid = mid.view(mid.shape[0],-1).detach().cpu().numpy()
                mids = mid if i==0 else np.concatenate([mids,mid],axis=0)
                latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
                latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
            np.save(f'../vecs/mids{ae.identifier}.npy',mids)
            np.save(f'../vecs/latents{ae.identifier}.npy',latents)
    return {'aeid':aeid,'mids':mids,'latents':latents}

def label_single(ae_output,args):
    aeid,mids,latents = ae_output['aeid'],ae_output['mids'],ae_output['latents']
    if args.test:
        print('Skipping umap and hdbscan')
        mid_labels = np.load(f'../labels/mid_labels{aeid}.npy')
        latent_labels = np.load(f'../labels/latent_labels{aeid}.npy')
        umapped_latents = np.load(f'../umaps/latent_umaps{aeid}.npy')
    else:
        #print('Umapping mids...')
        #umapped_mids = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(mids.squeeze())
        print('Umapping latents...')
        umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(latents.squeeze())
        #print('Scanning mids...')
        #mid_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        #mid_labels = mid_scanner.fit_predict(umapped_mids)
        print('Scanning latents...')
        latent_scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        latent_labels = latent_scanner.fit_predict(umapped_latents)
        if args.save:
            #np.save(f'../labels/mid_labels{aeid}.npy',mid_labels)
            np.save(f'../labels/latent_labels{aeid}.npy',latent_labels)
            np.save(f'../umaps/latent_umaps{aeid}.npy',umapped_latents)
    #return {'aeid':aeid,'umapped_mids':umapped_mids,'umapped_latents':umapped_latents,'mid_labels':mid_labels,'latent_labels':latent_labels}
    return {'aeid':aeid,'umapped_latents':umapped_latents,'latent_labels':latent_labels}

def build_ensemble(vecs_and_labels,args,pivot,given_gt):
    #nums_labels = [utils.get_num_labels(ae_results[ctype]) for ae_results in vecs_and_labels.values() for ctype in ['mid_labels','latent_labels']]
    nums_labels = [utils.get_num_labels(ae_results['latent_labels']) for ae_results in vecs_and_labels.values()]
    print(nums_labels)
    counts = {x:nums_labels.count(x) for x in set(nums_labels)}
    ensemble_num_labels = sorted([x for x,c in counts.items() if c==max(counts.values())])[-1]
    assert ensemble_num_labels == utils.get_num_labels(given_gt)
    #usable_mid_labels = {aeid:result['mid_labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['mid_labels']) == ensemble_num_labels}
    usable_latent_labels = {aeid:result['latent_labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['latent_labels']) == ensemble_num_labels}
    #same_lang_labels = utils.debable(list(usable_mid_labels.values())+list(usable_latent_labels.values()),pivot=pivot)
    same_lang_labels = utils.debable(list(usable_latent_labels.values()),pivot=pivot)
    multihots = utils.ask_ensemble(np.stack(same_lang_labels))
    #all_agree = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)==(len(usable_mid_labels) + len(usable_latent_labels)))
    all_agree = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)==len(usable_latent_labels))
    ensemble_labels_ = multihots.argmax(axis=1)
    ensemble_labels = given_gt if given_gt is not None else ensemble_labels_
    centroids_by_id = {}
    for aeid in vecs_and_labels.keys():
        #print(aeid,'num_mids:',utils.get_num_labels(vecs_and_labels[aeid]['mid_labels']), 'num_latents:',utils.get_num_labels(vecs_and_labels[aeid]['latent_labels']))
        print(aeid, 'num_latents:',utils.get_num_labels(vecs_and_labels[aeid]['latent_labels']))
    #for aeid in set(usable_mid_labels.keys()).intersection(set(usable_latent_labels.keys())):
    for aeid in set(usable_latent_labels.keys()):
        new_centroid_info={'aeid':aeid}
        if  not any([((ensemble_labels==i)*all_agree).any() for i in sorted(set(ensemble_labels))]):
            print('No all agree vecs for centroids')
            continue
        #try:mid_centroids = np.stack([vecs_and_labels[aeid]['mids'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels))])
        #except:set_trace()
        #assert (mid_centroids == mid_centroids).all()
        #new_centroid_info['mid_centroids'] = mid_centroids
        #if args.save: np.save(f'../centroids/mid_centroids{aeid}.npy', mid_centroids)
        latent_centroids = np.stack([vecs_and_labels[aeid]['latents'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels)) if i!= -1])
        try:
            assert (latent_centroids == latent_centroids).all()
        except: set_trace()
        new_centroid_info['latent_centroids'] = latent_centroids
        if args.save: np.save(f'../centroids/latent_centroids{aeid}.npy', latent_centroids)
        centroids_by_id[aeid] = new_centroid_info

    #lost_aeids = [aeid for aeid in vecs_and_labels.keys() if aeid not in centroids_by_id.keys()]
    if args.save:
        np.save('ensemble_labels.npy', ensemble_labels)
        np.save('all_agree.npy', all_agree)
    return centroids_by_id, ensemble_labels_, all_agree

def train_ae(ae_dict,args,centroids_by_id,ensemble_labels,all_agree):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid, ae = ae_dict['aeid'],ae_dict['ae']
        if aeid not in centroids_by_id.keys():
            print(aeid, ', you must be new here')
            pretrain_ae(ae_dict,args=args)
            return
        centroids_dict = centroids_by_id[aeid]
        #mid_centroids = torch.tensor(centroids_dict['mid_centroids'],device='cuda')
        latent_centroids = torch.tensor(centroids_by_id[aeid]['latent_centroids'],device='cuda')

        DATASET = utils.get_mnist_dset()
        dl = data.DataLoader(DATASET,batch_sampler=data.BatchSampler(data.RandomSampler(DATASET),args.batch_size,drop_last=True),pin_memory=False)
        loss_func=nn.L1Loss(reduction='none')
        ce_loss_func = nn.CrossEntropyLoss(reduction='none')
        mse_loss_func = nn.MSELoss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
        opt.add_param_group({'params':ae.lin.parameters(),'lr':1e-3})
        ensemble_labels = torch.tensor(ensemble_labels,device='cuda')
        try:
            #assert len(ensemble_labels.unique()) == mid_centroids.shape[0]
            assert len(ensemble_labels.unique()) == latent_centroids.shape[0]
        except: set_trace()
        for epoch in range(args.epochs):
            total_rloss = 0.
            total_closs = 0.
            total_gloss = 0.
            total_crloss = 0.
            total_rrloss = 0.
            rloss_list = []
            for i, (xb,yb,idx) in enumerate(dl):
                try:enc_mid,latent  = ae.enc(xb)
                except Exception as e:print(e); set_trace()
                targets = ensemble_labels[idx]
                mask = all_agree[idx]
                #mid_centroid_targets = mid_centroids[targets]
                latent_centroid_targets = latent_centroids[targets]
                dec_mid,rpred = ae.dec(utils.noiseify(latent,args.noise))
                rloss = loss_func(rpred,xb)
                if not mask.any():
                    loss = rloss.mean()
                else:
                    cdec_mid,crpred = ae.dec(latent_centroid_targets[mask,:,None,None])
                    crloss_ = loss_func(crpred,xb[mask]).mean(dim=[1,2,3])
                    #def get_gauss_loss(preds,centroids,targets):
                    #    inverse_dists = mse_loss_func(preds,centroids).mean(dim=-1)**-1
                    #    #inverse_dists = mse_loss_func(preds.squeeze(-1).transpose(1,2),centroids).mean(dim=-1)**(-1)
                    #    inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
                    #    return ce_loss_func(inverse_dists,targets)
                    #inverse_dists = mse_loss_func(latent[:,None,:,0,0],latent_centroids).mean(dim=-1)**-1
                    #inverse_dists /= inverse_dists.min(dim=-1,keepdim=True)[0]
                    #gauss_loss = ce_loss_func(inverse_dists,targets)
                    lin_pred = ae.lin(utils.noiseify(latent,args.noise)[:,:,0,0])
                    gauss_loss = ce_loss_func(lin_pred,targets).mean()
                    #mid_gauss_loss = get_gauss_loss(enc_mid.view(args.batch_size,1,-1),mid_centroids,targets)
                    #latent_gauss_loss = get_gauss_loss(latent.view(args.batch_size,1,-1),latent_centroids,targets)
                    closs = 0.1*(crloss_).mean() + torch.clamp(gauss_loss, min=args.clamp_gauss_loss).mean() + rloss[~mask].mean() + 0.1*rloss.mean()
                    #closs = 0.1*crloss_.mean() + torch.clamp(closs_, min=args.clamp_gauss_loss).mean()
                    total_rloss = total_rloss*((i+1)/(i+2)) + rloss.mean().item()*(1/(i+2))
                    #total_closs = total_closs*((i+1)/(i+2)) + closs_.mean().item()*(1/(i+2))
                    total_gloss = total_gloss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
                    total_crloss = total_crloss*((i+1)/(i+2)) + crloss_.mean().item()*(1/(i+2))
                    loss = rloss.mean() + args.clmbda*closs
                loss.backward(); opt.step(); opt.zero_grad()
                if args.test: break
            print(f'AE: {aeid}, Epoch: {epoch} RLoss: {total_rloss}, GaussLoss: {total_gloss}, CRLoss: {total_crloss}')
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
        if args.save: torch.save({'enc':ae.enc,'dec':ae.dec},f'../checkpoints/{aeid}.pt')

    print('Finished epochs for ae', aeid)

def load_ae(aeid,args):
    checkpoints_dir = "checkpoints_solid" if args.solid else "checkpoints"
    #chkpt = torch.load(f'../{checkpoints_dir}/pt{aeid}.pt', map_location=f'cuda:{torch.cuda.current_device()}')
    chkpt = torch.load(f'../{checkpoints_dir}/pt{aeid}.pt', map_location=f'cuda:{args.gpu}')
    revived_ae = utils.AE(chkpt['enc'],chkpt['dec'],aeid)
    return {'aeid': aeid, 'ae':revived_ae}

def load_vecs(aeid):
    mids = np.load(f'../vecs/mids{aeid}.npy')
    latents = np.load(f'../vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'mids': mids, 'latents':latents}

def load_labels(aeid):
    mid_labels = np.load(f'../labels/mid_labels{aeid}.npy')
    latent_labels = np.load(f'../labels/latent_labels{aeid}.npy')
    umapped_latents = np.load(f'../umaps/latent_umaps{aeid}.npy')
    return {'aeid': aeid, 'umapped_latents':umapped_latents,'mid_labels': mid_labels, 'latent_labels': latent_labels}

def load_ensemble(aeids):
    centroids_by_id = {}
    for aeid in aeids:
        new_centroids = {'aeid':aeid}
        try: new_centroids['mid_centroids'] = np.load(f'../centroids/mid_centroids{aeid}.npy')
        except FileNotFoundError: pass
        try: new_centroids['latent_centroids'] = np.load(f'../centroids/latent_centroids{aeid}.npy')
        except FileNotFoundError: pass
        centroids_by_id[aeid] = new_centroids
    ensemble_labels = np.load('ensemble_labels.npy')
    all_agree = np.load('all_agree.npy')
    return centroids_by_id, ensemble_labels, all_agree

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--exp_name',type=str,default='try')
    parser.add_argument('--gen_batch_size',type=int,default=500)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--inter_epochs',type=int,default=8)
    parser.add_argument('--meta_epochs',type=int,default=8)
    parser.add_argument('--mid_lmbda',type=float,default=0.)
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

    filled_pretrain = partial(pretrain_ae,args=ARGS)
    filled_generate = partial(generate_vecs_single,args=ARGS)
    filled_label = partial(label_single,args=ARGS)
    filled_load = partial(load_ae,args=ARGS)
    device = torch.device(f'cuda:{ARGS.gpu}')
    if 1 in ARGS.sections: #Pretrain aes rather than load
        pretrain_start_time = time()
        aes = []
        print(f"Instantiating {len(aes)} aes...")
        with torch.cuda.device(device):
            for aeid in aeids:
                aes.append({'aeid':aeid, 'ae':utils.make_ae(aeid,device=device,NZ=ARGS.NZ)})
        with ctx.Pool(processes=20) as pool:
            print(f"Pretraining {len(aes)} aes...")
            gt_labelss = [filled_pretrain(ae) for ae in aes] if ARGS.single else pool.map(filled_pretrain, aes)
        print(f'Pretraining time: {utils.asMinutes(time()-pretrain_start_time)}')
    else:
        with ctx.Pool(processes=20) as pool:
            print(f"Loading {len(aeids)} aes...")
            aes = [filled_load(aeid) for aeid in aeids] if ARGS.single else pool.map(filled_load, aeids)
        try: assert all([a['ae'].enc.block2[-2][0].out_channels==ARGS.NZ for a in aes])
        except:
            print(f'Mismatch between latent size of loaded aes, and NZ argument')
            print([a['ae'].enc.block2[-2][0].out_channels for a in aes],'\n')
            sys.exit()

    if 2 in ARGS.sections:
        generate_start_time = time()
        with ctx.Pool(processes=20) as pool:
            print(f"Generating vecs for {len(aes)} aes...")
            vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
        print(f'Generation time: {utils.asMinutes(time()-generate_start_time)}')
    elif 3 in ARGS.sections or 4 in ARGS.sections:
        with ctx.Pool(processes=20) as pool:
            print(f"Loading vecs for {len(aes)} aes...")
            vecs = [load_vecs(aeid) for aeid in aeids] if ARGS.single else pool.map(load_vecs, aeids)
        assert all([v['latents'].shape[1] == ARGS.NZ for v in vecs])

    from sklearn.metrics import normalized_mutual_info_score as mi
    gt_labels = np.load('../labels/gt_labels.npy')
    if 3 in ARGS.sections:
        label_start_time = time()
        with ctx.Pool(processes=20) as pool:
            print(f"Labelling vecs for {len(aes)} aes...")
            labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
        print(f'Labelling time: {utils.asMinutes(time()-label_start_time)}')
    elif 4 in ARGS.sections:
        with ctx.Pool(processes=20) as pool:
            print(f"Loading labels for {len(aes)} aes...")
            labels = [load_labels(aeid) for aeid in aeids] if ARGS.single else pool.map(load_labels,aeids)

    if 4 in ARGS.sections:
        scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
        concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
        concat_umap_start_time = time()
        print('Umapping concatted vecs...')
        umapped_concats = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(concatted_vecs)
        print(f'Umap concat time: {utils.asMinutes(time()-concat_umap_start_time)}')
        concat_scan_start_time = time()
        print('Scanning umapped concatted vecs...')
        concatted_labels = scanner.fit_predict(umapped_concats)
        print(f'Concat scan time: {utils.asMinutes(time()-concat_scan_start_time)}')
        ensemble_build_start_time = time()
        vecs = utils.dictify_list(vecs,key='aeid')
        labels = utils.dictify_list(labels,key='aeid')
        vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in aeids}
        print(f"Building ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = build_ensemble(vecs_and_labels,ARGS,pivot=concatted_labels,given_gt=concatted_labels)
        print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
    elif 5 in ARGS.sections:
        print(f"Loading ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = load_ensemble(aeids)

    if 5 in ARGS.sections:
        train_start_time = time()
        for meta_epoch_num in range(ARGS.meta_epochs):
            print('\nMeta epoch:', meta_epoch_num)
            copied_aes = [copy.deepcopy(ae) if ae['aeid'] in centroids_by_id.keys() else {'aeid': ae['aeid'], 'ae':utils.make_ae(ae['aeid'],device=device,NZ=ARGS.NZ)} for ae in aes]
            if len(aes) == 0:
                print('No legit aes left')
                sys.exit()
            print('Aes remaining:', *list(aeids))
            with ctx.Pool(processes=20) as pool:
                print(f"Training aes {list(aeids)}...")

                try: assert set([ae['aeid'] for ae in copied_aes]) == set(aeids)
                except: set_trace()
                for aedict in copied_aes:
                    try: assert(aedict['ae'].lin.in_features == ARGS.NZ)
                    except:aedict['ae'].lin = nn.Linear(ARGS.NZ,len(set(ensemble_labels))).to(device)
                filled_train = partial(train_ae,args=ARGS,centroids_by_id=centroids_by_id,ensemble_labels=ensemble_labels,all_agree=all_agree)
                [filled_train(ae) for ae in copied_aes] if ARGS.single else pool.map(filled_train, copied_aes)
            aes = copied_aes
            assert set([ae['aeid'] for ae in aes]) == set(aeids)
            print(aes[0]['ae'].first_row)
            with ctx.Pool(processes=20) as pool:
                print(f"Generating vecs for {len(aes)} aes...")
                vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
            with ctx.Pool(processes=20) as pool:
                print(f"Generating labels for {len(aes)} aes...")
                labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
            scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
            concatted_umaps = np.concatenate([l['umapped_latents'] for l in labels])
            concatted_labels = scanner.fit_predict(concatted_umaps)

            concatted_vecs = np.concatenate([v['latents'] for v in vecs])
            umapped_concats = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(concatted_vecs)
            concatted_labels2 = scanner.fit_predict(umapped_concats)
            vecs = utils.dictify_list(vecs,key='aeid')
            labels = utils.dictify_list(labels,key='aeid')
            vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in aeids}
            print(f"Building ensemble from {len(aes)} aes...")
            centroids_by_id, ensemble_labels, all_agree  = build_ensemble(vecs_and_labels,ARGS,pivot=concatted_labels,given_gt=concatted_labels)
            copied_aes = [copy.deepcopy(ae) if ae['aeid'] in centroids_by_id.keys() else {'aeid': ae['aeid'], 'ae':utils.make_ae(ae['aeid'],device=device,NZ=ARGS.NZ)} for ae in aes]

            #difficult_list = sorted([l for l in set(best_labels)], key=lambda x: (best_labels[~all_agree]==x).sum()/(best_labels==x).sum())
            print('AE Scores:')
            #print('Mid Acc:', [utils.accuracy(labels[x]['mid_labels'],gt_labels) for x in aeids if x in centroids_by_id.keys()])
            #print('Mid MIs:',[mi(labels[x]['mid_labels'],gt_labels) for x in aeids if x in centroids_by_id.keys()])
            #print('L acc',[utils.accuracy(labels[x]['latent_labels'],gt_labels) for x in aeids if x in centroids_by_id.keys()])
            print('L acc', [utils.accuracy(labels[x]['latent_labels'][labels[x]['latent_labels']>=0],gt_labels[labels[x]['latent_labels']>=0]) for x in aeids if x in centroids_by_id.keys()])
            print('L acc', [mi(labels[x]['latent_labels'][labels[x]['latent_labels']>=0],gt_labels[labels[x]['latent_labels']>=0]) for x in aeids if x in centroids_by_id.keys()])
            #print('Latent MIs:',[mi(labels[x]['latent_labels'],gt_labels) for x in aeids if x in centroids_by_id.keys()])
            print('Ensemble Scores:')
            print('Acc:', utils.accuracy(ensemble_labels,gt_labels))
            print('Concat Acc:', utils.accuracy(concatted_labels,gt_labels))
            print('Concat2 Acc:', utils.accuracy(concatted_labels2,gt_labels))
            print('Acc agree:', utils.accuracy(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('NMI:',mi(ensemble_labels,gt_labels))
            print('NMI agree:',mi(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('Num agrees:', all_agree.sum())

        ensemble_fname = f'../experiments/{ARGS.exp_name}_ensemble.json'
        np.savez(ensemble_fname,ensemble_labels=ensemble_labels,all_agree=all_agree)
        by_aeid = {aeid:{**vecs_and_labels[aeid],**centroids_by_id[aeid]} for aeid in aeids}
        for aeid,v in by_aeid.items():
            fname = f'../experiments/{ARGS.exp_name}_aeid{aeid}_centroids.json'
            arrays_to_save = {k:v for k,v in by_aeid[aeid].items() if k!='aeid'}
            np.savez(fname,**arrays_to_save)
        exp_outputs = {'ensemble_labels': ensemble_labels,'all_agree':all_agree,'by_aeid':by_aeid}
        print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
