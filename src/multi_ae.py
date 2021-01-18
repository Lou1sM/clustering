import signal
import sys
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

from longformer.sliding_chunks import pad_to_window_size
from pdb import set_trace
from torch.utils import data
import hdbscan
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import umap
import utils
import warnings
warnings.filterwarnings('ignore')


def rtrain_ae(ae_dict,args,dset,should_change):
    dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
    aeid,ae = ae_dict['aeid'],ae_dict['ae']
    print(f'Rtraining {aeid}')
    befores = [p.detach().cpu().clone() for p in ae.parameters()]
    loss_func=nn.L1Loss(reduction='none')
    opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
    opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
    for epoch in range(args.pretrain_epochs):
        total_loss = 0.
        for i,(xb,yb,idx) in enumerate(dl):
            latent = ae.enc(xb)
            pred = ae.dec(utils.noiseify(latent,args.noise))
            loss = loss_func(pred,xb).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            total_loss = total_loss*(i+1)/(i+2) + loss.item()/(i+2)
            if args.test: break
        print(f'Rtraining AE {aeid}, Epoch: {epoch}, Loss {round(total_loss,4)}')
        if args.test: break
    if should_change and args.pretrain_epochs > 0:
        afters = [p.detach().cpu() for p in ae.parameters()]
        for b,a in zip(befores,afters):
            assert not (b==a).all() or args.pretrain_epochs == 0
    if args.save:
        utils.torch_save({'enc':ae.enc,'dec':ae.dec}, f'../{args.dset}/checkpoints',f'pretrained{aeid}.pt')
    if args.vis_pretrain:
        utils.check_ae_images(ae.enc, ae.dec, dset)
    # Now generate latent vecs for dataset
    if args.test:
        latents = np.random.random((args.dset_size,50)).astype(np.float32)
    else:
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
        for i, (xb,yb,idx) in enumerate(determin_dl):
            latent = ae.enc(xb)
            latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
            latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
        if args.save:
            utils.np_save(array=latents,directory=f'../{args.dset}/vecs/',fname=f'latents{ae.identifier}.npy',verbose=True)
    return {'aeid':aeid,'latents':latents}

def label_single(ae_output,args,abl=False):
    aeid,latents = ae_output['aeid'],ae_output['latents']
    if args.test:
        as_tens = np.tile(np.arange(args.num_clusters),args.dset_size//args.num_clusters)
        remainder = np.zeros(args.dset_size%args.num_clusters)
        labels = np.concatenate([as_tens,remainder]).astype(np.int)
        umapped_latents = None
    else:
        umap_neighbours = 30*args.dset_size//(70000)
        if abl:
            print("not umapping")
            umapped_latents = latents
        else:
            umapped_latents = umap.UMAP(min_dist=0,n_neighbors=umap_neighbours,n_components=2,random_state=42).fit_transform(latents.squeeze())
        if args.clusterer == 'GMM':
            c = GaussianMixture(n_components=args.num_clusters)
            labels = c.fit_predict(umapped_latents)
        elif args.clusterer == 'HDBSCAN':
            min_c = args.dset_size//(args.num_clusters*2)
            scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=min_c).fit(umapped_latents)
            n = -1
            if utils.get_num_labels(scanner.labels_) == args.num_clusters:
                labels = scanner.labels_
            elif utils.get_num_labels(scanner.labels_) > args.num_clusters:
                print(f"AE {aeid} has too MANY labels: {utils.get_num_labels(scanner.labels_)}")
                eps = 0.01
                for i in range(1000):
                    labels = hdbscan.hdbscan_._tree_to_labels(umapped_latents, scanner.single_linkage_tree_.to_numpy(), cluster_selection_epsilon=eps, min_cluster_size=min_c)[0]
                    n = utils.get_num_labels(labels)
                    if n == args.num_clusters:
                        break
                    elif n > args.num_clusters:
                        eps += 0.01
                    else:
                        eps -= 0.001
            else:
                eps = 1.
                print(f"AE {aeid} has too FEW labels: {utils.get_num_labels(scanner.labels_)}")
                best_eps = eps
                most_n = 0
                for i in range(1000):
                    labels = scanner.single_linkage_tree_.get_clusters(eps,min_cluster_size=min_c)
                    n = utils.get_num_labels(labels)
                    if n > most_n:
                        most_n = n
                        best_eps = eps
                    if n == args.num_clusters:
                        print(f'ae {aeid} using {eps}')
                        break
                    elif set(labels) == set([-1]):
                        while True:
                            labels = scanner.single_linkage_tree_.get_clusters(best_eps,min_cluster_size=min_c)
                            n = utils.get_num_labels(labels)
                            if n == args.num_clusters:
                                print('having to use min_c', min_c)
                                break
                            elif n > args.num_clusters:
                                print(f"{aeid} overshot inner to {n} with {min_c}, {eps}")
                                best_eps += 0.01
                                if best_eps >= 100: break
                            else:
                                min_c -= 25
                                if min_c <= 300: break
                        break

                    elif n < args.num_clusters:
                        if i > 100:
                            print(n,i)
                        eps -= 0.01
                    else:
                        print(f'ae {aeid} overshot to {n} with {eps} at {i}')
                        eps *= 1.1
            if utils.get_num_labels(labels) != args.num_clusters:
                print(f'WARNING: {aeid} has only {n} clusters, {min_c}, {eps}')

        if args.save:
            utils.np_save(labels,f'../{args.dset}/labels',f'labels{aeid}.npy',verbose=False)
            utils.np_save(umapped_latents,f'../{args.dset}/umaps',f'latent_umaps{aeid}.npy',verbose=False)
    return {'aeid':aeid,'umapped_latents':umapped_latents,'labels':labels}

def build_ensemble(vecs_and_labels,args,pivot,given_gt):
    nums_labels = [utils.get_num_labels(ae_results['labels']) for ae_results in vecs_and_labels.values()]
    counts = {x:nums_labels.count(x) for x in set(nums_labels)}
    print([(k,utils.get_num_labels(v['labels'])) for k,v in vecs_and_labels.items()])
    usable_labels = {aeid:result['labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['labels']) == args.num_clusters}
    if len(usable_labels) == 0:
        most_common_num_labels = max(nums_labels, key=lambda x: counts[x])
        usable_labels = {aeid:result['labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['labels']) == most_common_num_labels}
    if pivot != 'none' and args.num_clusters == utils.num_labs(pivot):
        same_lang_labels = utils.debable(list(usable_labels.values()),pivot=pivot)
    else:
        same_lang_labels = utils.debable(list(usable_labels.values()),pivot='none')
    multihots = utils.compute_multihots(np.stack(same_lang_labels),probs='none')
    all_agree = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)>=len(usable_labels)/1.2)
    all_agree_test = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)>=len(usable_labels)/5.2)
    print(all_agree.sum(), all_agree_test.sum())
    ensemble_labels_ = multihots.argmax(axis=1)
    ensemble_labels = given_gt if given_gt != 'none' else ensemble_labels_
    centroids_by_id = {}
    for l in same_lang_labels+[ensemble_labels]: print({i:(l==i).sum() for i in set(l)}) # Size of clusters
    for aeid in set(usable_labels.keys()):
        new_centroid_info={'aeid':aeid}
        if aeid == 'concat': continue
        if not all([((ensemble_labels==i)*all_agree).any() for i in sorted(set(ensemble_labels))]):
            continue
        latent_centroids = np.stack([vecs_and_labels[aeid]['latents'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels)) if i!= -1])
        try:
            assert (latent_centroids == latent_centroids).all()
        except: set_trace()
        new_centroid_info['latent_centroids'] = latent_centroids
        centroids_by_id[aeid] = new_centroid_info
        if args.save: utils.np_save(latent_centroids,f'../{args.dset}/centroids', f'centroids{aeid}.npy',verbose=False)

    if args.save:
        utils.np_save(ensemble_labels,f'../{args.dset}', 'ensemble_labels.npy',verbose=False)
        utils.np_save(all_agree,f'../{args.dset}', 'all_agree.npy',verbose=False)
    return centroids_by_id, ensemble_labels_, all_agree

def train_ae(ae_dict,args,targets,all_agree,dset,sharing_ablation):
    dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
    aeid, ae = ae_dict['aeid'],ae_dict['ae']
    befores = [p.detach().cpu() for p in ae.parameters()]
    loss_func=nn.L1Loss(reduction='none')
    ce_loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
    opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
    opt.add_param_group({'params':ae.pred.parameters(),'lr':1e-3})
    if sharing_ablation:
        targets_arr = np.array(targets[aeid]['labels'])
        ok_labels = targets_arr>=0
        filler = np.random.randint(size=targets_arr.shape,high=targets_arr.max(),low=0)
        targets_arr = np.where(targets_arr,ok_labels,filler)
        targets = torch.tensor(targets_arr,device=args.device)
        full_mask = targets >= 0
        assert full_mask.all()
    else:
        targets = torch.tensor(targets,device=args.device)
        full_mask = torch.tensor(all_agree,device=args.device)
    for epoch in range(args.epochs):
        total_rloss = 0.
        total_gloss = 0.
        for i, (xb,yb,idx) in enumerate(dl):
            latent  = ae.enc(xb)
            batch_targets = targets[idx]
            rpred = ae.dec(utils.noiseify(latent,args.noise))
            rloss = loss_func(rpred,xb)
            mask = full_mask[idx]
            if not mask.any():
                loss = rloss.mean()
            else:
                lin_pred = ae.pred(utils.noiseify(latent[mask],args.noise)[:,:,0,0])
                gauss_loss = ce_loss_func(lin_pred,batch_targets[mask]).mean()
                loss = gauss_loss + args.rlmbda*rloss.mean()
                if not mask.all():
                    loss += rloss[~mask].mean()
                total_rloss = total_rloss*((i+1)/(i+2)) + rloss.mean().item()*(1/(i+2))
                total_gloss = total_gloss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
            assert loss != 0
            loss.backward(); opt.step(); opt.zero_grad()
            if args.test: break
        print(f'AE: {aeid}, Epoch: {epoch} RLoss: {round(total_rloss,3)}, GaussLoss: {round(total_gloss,3)}')
        if args.test: break

    for epoch in range(args.inter_epochs):
        epoch_loss = 0
        for i, (xb,yb,idx) in enumerate(dl):
            latent = ae.enc(xb)
            latent = utils.noiseify(latent,args.noise)
            pred = ae.dec(latent)
            loss = loss_func(pred,xb).mean()
            loss.backward(); opt.step(); opt.zero_grad()
            epoch_loss = epoch_loss*((i+1)/(i+2)) + loss*(1/(i+2))
            if args.test: break
        if args.test: break
        if epoch_loss < 0.02: break
    afters = [p.detach().cpu() for p in list(ae.enc.parameters()) + list(ae.dec.parameters())]
    for b,a in zip(befores,afters): assert not (b==a).all()
    if args.save:
        utils.torch_save({'enc':ae.enc,'dec':ae.dec}, f'../{args.dset}/checkpoints/{args.exp_name}',f'{aeid}.pt')
    if args.vis_train: utils.check_ae_images(ae.enc,ae.dec,dset,stacked=True)
    if args.test:
        latents = np.random.random((args.dset_size,50)).astype(np.float32)
        as_tens = np.tile(np.arange(args.num_clusters),args.dset_size//args.num_clusters)
        remainder = np.zeros(args.dset_size%args.num_clusters)
        direct_labels = np.concatenate([as_tens,remainder]).astype(np.int)
    else:
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
        for i, (xb,yb,idx) in enumerate(determin_dl):
            latent = ae.enc(xb)
            new_direct_labels = ae.pred(latent[:,:,0,0]).argmax(dim=-1).detach().cpu().numpy()
            latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
            latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
            direct_labels = new_direct_labels if i==0 else np.concatenate([direct_labels,new_direct_labels],axis=0)
    return {'aeid':aeid,'latents':latents,'direct_labels':direct_labels}

def train_transformer(transformer_dict,args,targets,all_agree,dset):
    dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
    tran_id, transformer, pred = transformer_dict['tran_id'],transformer_dict['transformer'],transformer_dict['pred']
    befores = [p.detach().cpu() for p in list(transformer.parameters())]
    ce_loss_func = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(params = transformer.parameters(), lr=args.enc_lr)
    targets = torch.tensor(targets,device=args.device)
    full_mask = torch.tensor(all_agree,device=args.device)
    for epoch in range(args.epochs):
        total_loss = 0.
        for i, (xb,yb,idx) in enumerate(dl):
            mask = full_mask[idx]
            batch_targets = targets[idx]
            max_in_batch = max([len(item) for item in xb])
            attention_mask = torch.stack([torch.arange(max_in_batch) < len(item) for item in xb]).long()
            padded = torch.tensor([item + [1]*(max_in_batch - len(item)) for item in xb]).long()
            padded, attention_mask = pad_to_window_size(padded, attention_mask, 512, 1)
            transformed_word_vecs = transformer(padded, attention_mask=attention_mask)[0]
            # Zero out indices of padding
            s = transformed_word_vecs*(attention_mask.unsqueeze(-1))
            # Take mean of non-zero vectors
            doc_feature_vecs = s.sum(axis=1)/(attention_mask.sum(axis=1).unsqueeze(1))
            lin_pred = pred(utils.noiseify(doc_feature_vecs[mask],args.noise)[:,:,0,0])
            loss = ce_loss_func(lin_pred,batch_targets[mask]).mean()
            total_loss = total_loss*((i+1)/(i+2)) + loss.mean().item()*(1/(i+2))
            assert loss != 0
            loss.backward(); opt.step(); opt.zero_grad()
            if args.test: break
        print(f'Trans: {tran_id}, Epoch: {epoch} Loss: {round(total_loss,3)}')
        if args.test: break

    afters = [p.detach().cpu() for p in list(transformer.parameters())]
    for b,a in zip(befores,afters): assert not (b==a).all()
    if args.save:
        utils.torch_save({'transformer':transformer}, f'../{args.dset}/checkpoints/{args.exp_name}',f'{aeid}.pt')
    if args.test:
        latents = np.random.random((args.dset_size,50)).astype(np.float32)
        as_tens = np.tile(np.arange(args.num_clusters),args.dset_size//args.num_clusters)
        remainder = np.zeros(args.dset_size%args.num_clusters)
        direct_labels = np.concatenate([as_tens,remainder]).astype(np.int)
    else:
        determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
        for i, (xb,yb,idx) in enumerate(determin_dl):
            latent = ae.enc(xb)
            new_direct_labels = ae.pred(latent[:,:,0,0]).argmax(dim=-1).detach().cpu().numpy()
            latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
            latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
            direct_labels = new_direct_labels if i==0 else np.concatenate([direct_labels,new_direct_labels],axis=0)
    return {'aeid':aeid,'latents':latents,'direct_labels':direct_labels}

def load_ae(aeid,args):
    if args.reload_chkpt == 'none':
        path = f'../{args.dset}/checkpoints/pretrained{aeid}.pt'
    else:
        path = f'../{args.dset}/checkpoints/{args.reload_chkpt}/{aeid}.pt'
    print('Loading from',path)
    chkpt = torch.load(path, map_location=args.device)
    revived_ae = utils.AE(chkpt['enc'],chkpt['dec'],aeid)
    return {'aeid': aeid, 'ae':revived_ae}

def load_vecs(aeid,args):
    latents = np.load(f'../{args.dset}/vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'latents':latents}

def load_labels(aeid,args):
    labels = np.load(f'../{args.dset}/labels/labels{aeid}.npy')
    umapped_latents = None
    return {'aeid': aeid, 'umapped_latents':umapped_latents, 'labels': labels}

def load_ensemble(aeids,args):
    centroids_by_id = {}
    for aeid in aeids:
        new_centroids = {'aeid':aeid}
        try:
            new_centroids['latent_centroids'] = np.load(f'../{args.dset}/centroids/latent_centroids{aeid}.npy')
            centroids_by_id[aeid] = new_centroids
        except FileNotFoundError: pass
    ensemble_labels = np.load(f'../{args.dset}/ensemble_labels.npy')
    all_agree = np.load(f'../{args.dset}/all_agree.npy')
    return centroids_by_id, ensemble_labels, all_agree
