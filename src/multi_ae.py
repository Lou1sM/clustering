import signal
import sys
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
from pdb import set_trace
from torch.utils import data
import nns
import hdbscan
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import utils
import warnings
warnings.filterwarnings('ignore')
import umap


def rtrain_ae(ae_dset_dict,args,should_change):
    """Pretrain an autoencoder on reconstruction loss.

    ARGS:
        ae_dset_dict<dict>: contains the AE and the dataset to train on
        args: parameters for training
        should_change<bool>: whether the network weights should change, False only if
    """

    with torch.cuda.device(ae_dset_dict['dset'].device):
        aeid,ae,dset = ae_dset_dict['aeid'],ae_dset_dict['ae'],ae_dset_dict['dset']
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
        print(f'Rtraining {aeid} on {dset.device}')
        befores = [p.detach().cpu().clone() for p in ae.parameters()]
        loss_func=nn.L1Loss(reduction='none')
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=1e-3)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':1e-3})
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
            utils.torch_save({'enc':ae.enc,'dec':ae.dec}, f'../{args.dset}/checkpoints/{args.exp_name}',f'pretrained{aeid}.pt')
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
        print('finished', aeid)
        return {'aeid':aeid,'latents':latents}

def label_single(output,args,umap_abl=False):
    """Predict labels for a set of latent vectors.

    ARGS:
        output<dict>: contains the latent vectors to be labeled
        args: training hyperparameters
        umap_abl<bool>: whether to omit umap, True only in ablation study
    """

    aeid,all_pooled_latents = output['aeid'],output['latents']
    if args.test:
        as_tens = np.tile(np.arange(args.num_clusters),args.dset_size//args.num_clusters)
        remainder = np.zeros(args.dset_size%args.num_clusters)
        labels = np.concatenate([as_tens,remainder]).astype(np.int)
        umapped_latents = None
    else:
        umap_neighbours = 30*args.dset_size//(70000)
        if umap_abl:
            print("not umapping")
            umapped_latents = all_pooled_latents
        else:
            umapped_latents = umap.UMAP(min_dist=0,n_neighbors=umap_neighbours,n_components=2,random_state=42).fit_transform(all_pooled_latents.squeeze())
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
                print(f"ae {aeid} has too MANY labels: {utils.get_num_labels(scanner.labels_)}")
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
                print(f"ae {aeid} has too FEW labels: {utils.get_num_labels(scanner.labels_)}")
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

def build_ensemble(vecs_and_labels,args,pivot):
    """Aggregate together the labellings across and ensemble to form a single
    set of predicted labels, and to compute the indices where all labellings
    agree.

    ARGS:
        vecs_and_labels<dict>: keys are ensemble ids, values are results for
            that ensemble member
        args: training hyperparameters
        pivot: a certain labelling to use to translate all other into, if 'none'
            then the first labellings in vecs_and_labels is used
    """

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
    ensemble_labels = multihots.argmax(axis=1)
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
    return centroids_by_id, ensemble_labels, all_agree

def train_ae(ae_dset_dict,args,targets,all_agree,sharing_ablation):
    """Train autoencoder on pseudo labels, then use the autoencdoer to predict
    new pseudo labels for the dataset and return them.

    ARGS:
        ae_dset_dict<dict>: contains the ae id, the ae and the dataset to train on
        targets<np.array>: pseudo labels to train on
        args: training hyperparameters
        sharing_ablation<bool>: whether to omit label sharing, ie run ablation
            setting A2 from the paper
    """

    with torch.cuda.device(ae_dset_dict['dset'].device):
        aeid,ae,dset = ae_dset_dict['aeid'],ae_dset_dict['ae'],ae_dset_dict['dset']
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
        befores = [p.detach().cpu() for p in ae.parameters()]
        loss_func=nn.L1Loss(reduction='none')
        ce_loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=1e-3)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':1e-3})
        opt.add_param_group({'params':ae.pred.parameters(),'lr':1e-3})
        if sharing_ablation:
            targets_arr = np.array(targets[aeid]['labels'])
            ok_labels = targets_arr>=0
            filler = np.random.randint(size=targets_arr.shape,high=targets_arr.max(),low=0)
            targets_arr = np.where(targets_arr,ok_labels,filler)
            targets = torch.tensor(targets_arr,device=ae.device)
            full_mask = targets >= 0
            assert full_mask.all()
        else:
            targets = torch.tensor(targets,device=ae.device)
            full_mask = torch.tensor(all_agree,device=ae.device)
        for epoch in range(args.epochs):
            total_rloss = 0.
            total_gloss = 0.
            for i, (xb,yb,idx) in enumerate(dl):
                latent = ae.enc(xb)
                batch_targets = targets[idx]
                rpred = ae.dec(utils.noiseify(latent,args.noise))
                rloss = loss_func(rpred,xb)
                mask = full_mask[idx]
                if not mask.any():
                    loss = rloss.mean()
                else:
                    lin_pred = ae.pred(utils.noiseify(latent[mask],args.noise)[:,:,0,0])
                    gauss_loss = ce_loss_func(lin_pred,batch_targets[mask]).mean()
                    loss = gauss_loss + rloss.mean()
                    if not mask.all():
                        loss += rloss[~mask].mean()
                    total_rloss = total_rloss*((i+1)/(i+2)) + rloss.mean().item()*(1/(i+2))
                    total_gloss = total_gloss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
                assert loss != 0
                loss.backward(); opt.step(); opt.zero_grad()
                if args.test: break
            print(f'AE: {aeid}, Epoch: {epoch} RLoss: {round(total_rloss,3)}, GaussLoss: {round(total_gloss,3)}')
            if args.test: break
        afters = [p.detach().cpu() for p in list(ae.enc.parameters()) + list(ae.dec.parameters())]
        for b,a in zip(befores,afters): assert not (b==a).all()
        if args.save:
            utils.torch_save({'enc':ae.enc,'dec':ae.dec}, f'../{args.dset}/checkpoints/{args.exp_name}',f'{aeid}.pt')
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

def load_ae(ae_dict_without_ae,args):
    """Load saved autoencoder, insert into a dict and return the result.

    ARGS:
        ae_dict_without_ae<dict>: where the autoencoder will be inserted under
            the key 'ae'
        args: training hyperparameters
    """

    aeid, dset = ae_dict_without_ae['aeid'], ae_dict_without_ae['dset']
    if args.reload_chkpt == 'none':
        path = f'../{args.dset}/checkpoints/pretrained{aeid}.pt'
    else:
        path = f'../{args.dset}/checkpoints/{args.reload_chkpt}/{aeid}.pt'
    print('Loading from',path)
    chkpt = torch.load(path, map_location=dset.device)
    revived_ae = nns.AE(chkpt['enc'],chkpt['dec'],aeid)
    return dict(ae=revived_ae)

def load_vecs(aeid,args):
    """Load saved latent vectors for a given ae id."""
    latents = np.load(f'../{args.dset}/vecs/latents{aeid}.npy')
    return {'aeid':aeid, 'latents':latents}

def load_labels(aeid,args):
    """Load saved labels for a given ae id."""
    labels = np.load(f'../{args.dset}/labels/labels{aeid}.npy')
    umapped_latents = None
    return {'aeid': aeid, 'umapped_latents':umapped_latents, 'labels': labels}

def load_ensemble(aeids,args):
    """Load saved centroids, ensemble_labels and all indices for given ae ids."""
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
