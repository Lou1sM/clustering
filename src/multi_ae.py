import gc
import signal
import sys
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

from functools import partial
from pdb import set_trace
from scipy.stats import entropy
from time import time
from torch.utils import data
import copy
import gc
import hdbscan
import numpy as np
import os
import random
import torch
import torch.nn as nn
import umap
import utils
import warnings
warnings.filterwarnings('ignore')


def rtrain_ae(ae_dict,args,should_change):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid,ae = ae_dict['aeid'],ae_dict['ae']
        print(f'Rtraining {aeid}')
        befores = [p.detach().cpu() for p in ae.parameters()]
        dset = utils.get_vision_dset(args.dset)
        if args.one_image:
            dset.x = dset.x[:1]
            dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),1,drop_last=True),pin_memory=False)
            args.pretrain_epochs = 1000
        else:
            dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
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
        if should_change:
            afters = [p.detach().cpu() for p in ae.parameters()]
            for b,a in zip(befores,afters): assert not (b==a).all()
        if args.save:
            utils.torch_save({'enc':ae.enc,'dec':ae.dec}, f'../{args.dset}/checkpoints',f'pt{aeid}.pt')
        if args.vis_pretrain:
            utils.check_ae_images(ae.enc, ae.dec, dset)

def generate_vecs_single(ae_dict,args):
    aeid, ae = ae_dict['aeid'],ae_dict['ae']
    if args.test:
        latents = np.random.random((args.dset_size,50)).astype(np.float32)
    else:
        device = torch.device(f'cuda:{args.gpu}')
        with torch.cuda.device(device):
            dset = utils.get_vision_dset(args.dset)
            ae.to(device)
            determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),args.gen_batch_size,drop_last=False),pin_memory=False)
            for i, (xb,yb,idx) in enumerate(determin_dl):
                latent = ae.enc(xb)
                latent = latent.view(latent.shape[0],-1).detach().cpu().numpy()
                latents = latent if i==0 else np.concatenate([latents,latent],axis=0)
            if args.save:
                utils.np_save(array=latents,directory=f'../{args.dset}/vecs/',fname=f'latents{ae.identifier}.npy',)
    return {'aeid':aeid,'latents':latents}

def label_single(ae_output,args):
    aeid,latents = ae_output['aeid'],ae_output['latents']
    if args.test:
        as_tens = np.tile(np.arange(args.num_clusters),args.dset_size//args.num_clusters)
        remainder = np.zeros(args.dset_size%args.num_clusters)
        labels = np.concatenate([as_tens,remainder]).astype(np.int)
        umapped_latents = None
    else:
        min_c = 500
        umapped_latents = umap.UMAP(min_dist=0,n_neighbors=30,random_state=42).fit_transform(latents.squeeze())
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
                    for _ in range(50000):
                        labels = scanner.single_linkage_tree_.get_clusters(best_eps,min_cluster_size=min_c)
                        n = utils.get_num_labels(labels)
                        if n == args.num_clusters:
                            print('having to use min_c', min_c)
                            break
                        elif n > args.num_clusters:
                            eps += 0.01
                        else:
                            min_c -= 1
                    print(888)
                    break

                elif n < args.num_clusters:
                    if i > 100:
                        print(n,i)
                    eps -= 0.01
                else:
                    print(f'ae {aeid} overshot to {n} with {eps} at {i}')
                    eps *= 1.1
        if args.save:
            utils.np_save(labels,f'../{args.dset}/labels',f'labels{aeid}.npy')
            utils.np_save(umapped_latents,f'../{args.dset}/umaps',f'latent_umaps{aeid}.npy')
    return {'aeid':aeid,'umapped_latents':umapped_latents,'labels':labels}

def build_ensemble(vecs_and_labels,args,pivot,given_gt):
    nums_labels = [utils.get_num_labels(ae_results['labels']) for ae_results in vecs_and_labels.values()]
    counts = {x:nums_labels.count(x) for x in set(nums_labels)}
    ensemble_num_labels = sorted([x for x,c in counts.items() if c==max(counts.values())])[-1]
    assert given_gt is 'none' or ensemble_num_labels == utils.get_num_labels(given_gt)
    usable_labels = {aeid:result['labels'] for aeid,result in vecs_and_labels.items() if utils.get_num_labels(result['labels']) == ensemble_num_labels}
    if pivot is not 'none' and ensemble_num_labels == utils.num_labs(pivot):
        same_lang_labels = utils.debable(list(usable_labels.values()),pivot=pivot)
    else:
        same_lang_labels = utils.debable(list(usable_labels.values()),pivot='none')
    multihots = utils.compute_multihots(np.stack(same_lang_labels),probs='none')
    all_agree = np.ones(multihots.shape[0]).astype(np.bool) if args.test else (multihots.max(axis=1)==len(usable_labels))
    ensemble_labels_ = multihots.argmax(axis=1)
    ensemble_labels = given_gt if given_gt is not 'none' else ensemble_labels_
    centroids_by_id = {}
    for l in same_lang_labels+[ensemble_labels]: print({i:(l==i).sum() for i in set(l)}) # Size of clusters
    for aeid in set(usable_labels.keys()):
        new_centroid_info={'aeid':aeid}
        if aeid == 'concat': continue
        if  not all([((ensemble_labels==i)*all_agree).any() for i in sorted(set(ensemble_labels))]):
            continue
        latent_centroids = np.stack([vecs_and_labels[aeid]['latents'][(ensemble_labels==i)*all_agree].mean(axis=0) for i in sorted(set(ensemble_labels)) if i!= -1])
        try:
            assert (latent_centroids == latent_centroids).all()
        except: set_trace()
        new_centroid_info['latent_centroids'] = latent_centroids
        centroids_by_id[aeid] = new_centroid_info
        if args.save: utils.np_save(latent_centroids,f'../{args.dset}/centroids', f'centroids{aeid}.npy')

    if args.save:
        utils.np_save(ensemble_labels,f'../{args.dset}', 'ensemble_labels.npy')
        utils.np_save(all_agree,f'../{args.dset}', 'all_agree.npy')
    return centroids_by_id, ensemble_labels_, all_agree

def train_ae(ae_dict,args,centroids_by_id,worst3,targets,all_agree):
    device = torch.device(f'cuda:{args.gpu}')
    with torch.cuda.device(device):
        aeid, ae = ae_dict['aeid'],ae_dict['ae']
        befores = [p.detach().cpu() for p in ae.parameters()]
        dset = utils.get_vision_dset(args.dset)
        dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),args.batch_size,drop_last=True),pin_memory=False)
        loss_func=nn.L1Loss(reduction='none')
        ce_loss_func = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(params = ae.enc.parameters(), lr=args.enc_lr)
        opt.add_param_group({'params':ae.dec.parameters(),'lr':args.dec_lr})
        opt.add_param_group({'params':ae.pred.parameters(),'lr':1e-3})
        if args.worst3:
            opt.add_param_group({'params':ae.pred2.parameters(),'lr':1e-3})
            opt.add_param_group({'params':ae.pred3.parameters(),'lr':1e-3})
        targets = torch.tensor(targets,device='cuda')
        all_agree = torch.tensor(all_agree,device=device)
        for epoch in range(args.epochs):
            total_rloss = 0.
            total_loss = 0.
            total_gloss_ = 0.
            total_gloss = 0.
            total_crloss = 0.
            total_w2loss = 0.
            total_w3loss = 0.
            rloss_list = []
            for i, (xb,yb,idx) in enumerate(dl):
                latent  = ae.enc(xb)
                batch_targets = targets[idx]
                rpred = ae.dec(utils.noiseify(latent,args.noise))
                rloss = loss_func(rpred,xb)
                mask = all_agree[idx]
                if not mask.any():
                    loss = rloss.mean()
                else:
                    lin_pred = ae.pred(utils.noiseify(latent[mask],args.noise)[:,:,0,0])
                    gauss_loss = ce_loss_func(lin_pred,batch_targets[mask]).mean()
                    loss = torch.clamp(gauss_loss, min=args.clamp_gauss_loss).mean() + args.rlmbda*rloss.mean()
                    if not mask.all():
                        loss += rloss[~mask].mean()
                    if args.worst3:
                        wor1 = (batch_targets == worst3[0])
                        wor2 = (batch_targets == worst3[1])
                        wor3 = (batch_targets == worst3[2])
                        worstmask2 = (wor1 + wor2)*mask
                        worstmask3 = (wor1 + wor2 + wor3)*mask
                        worsttargets2 = torch.tensor(utils.compress_labels(batch_targets[worstmask2]),device=device).long()
                        worsttargets3 = torch.tensor(utils.compress_labels(batch_targets[worstmask3]),device=device).long()
                        if worstmask2.any():
                            w2loss = ce_loss_func(ae.pred2(latent[worstmask2,:,0,0]),worsttargets2).mean()
                            loss += w2loss
                            total_w2loss = total_w2loss*((i+1)/(i+2)) + w2loss.mean().item()*(1/(i+2))
                        if worstmask3.any():
                            w3loss = ce_loss_func(ae.pred3(latent[worstmask3,:,0,0]),worsttargets3).mean()
                            loss += w3loss
                            total_w3loss = total_w3loss*((i+1)/(i+3)) + w3loss.mean().item()*(1/(i+3))
                    total_rloss = total_rloss*((i+1)/(i+2)) + rloss.mean().item()*(1/(i+2))
                    total_gloss = total_gloss*((i+1)/(i+2)) + gauss_loss.mean().item()*(1/(i+2))
                assert loss != 0
                loss.backward(); opt.step(); opt.zero_grad()
                if args.test: break
            if args.test: break
        print(f'AE: {aeid}, Epoch: {epoch} RLoss: {round(total_rloss,3)}, GaussLoss: {round(total_gloss,3)}, CRLoss: {round(total_crloss,3)}, W2: {round(total_w2loss,2)}, W3: {round(total_w3loss,3)}')
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
        if args.worst3:
            for b,a in zip(befores,afters): assert not (b==a).all()
        if args.save: torch.save({'enc':ae.enc,'dec':ae.dec},f'../{args.dset}/checkpoints/{aeid}.pt')
    if args.vis_train: utils.check_ae_images(ae.enc,ae.dec,dset,stacked=True)

def load_ae(aeid,args):
    checkpoints_dir = "checkpoints_solid" if args.solid else "checkpoints"
    chkpt = torch.load(f'../{args.dset}/{checkpoints_dir}/pt{aeid}.pt', map_location=f'cuda:{args.gpu}')
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--ae_range',type=int,nargs='+')
    group.add_argument('--aeids',type=int,nargs='+')
    group.add_argument('--num_aes',type=int)
    parser.add_argument('--NZ',type=int,default=50)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--clamp_gauss_loss',type=float,default=0.1)
    parser.add_argument('--clmbda',type=float,default=1.)
    parser.add_argument('--conc',action='store_true')
    parser.add_argument('--dec_lr',type=float,default=1e-3)
    parser.add_argument('--dset',type=str,default='MNIST',choices=['MNIST','FashionMNIST','USPS','CIFAR10','coil-100', 'letterAJ'])
    parser.add_argument('--enc_lr',type=float,default=1e-3)
    parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--exp_name',type=str,default='try')
    parser.add_argument('--gen_batch_size',type=int,default=100)
    parser.add_argument('--gpu',type=str,default='0')
    parser.add_argument('--inter_epochs',type=int,default=8)
    parser.add_argument('--max_meta_epochs',type=int,default=30)
    parser.add_argument('--mask',action='store_true')
    parser.add_argument('--noise',type=float,default=1.5)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--pretrain_epochs',type=int,default=10)
    parser.add_argument('--rlmbda',type=float,default=1.)
    parser.add_argument('--save','-s',action='store_true')
    parser.add_argument('--scatter_clusters',action='store_true')
    parser.add_argument('--sections',type=int,nargs='+', default=[5])
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--short_epochs',action='store_true')
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--one_image',action='store_true')
    parser.add_argument('--show_gpu_memory',action='store_true')
    parser.add_argument('--solid',action='store_true')
    parser.add_argument('--test','-t',action='store_true')
    parser.add_argument('--vis_pretrain',action='store_true')
    parser.add_argument('--vis_train',action='store_true')
    parser.add_argument('--worst3',action='store_true')
    ARGS = parser.parse_args()
    print(ARGS)

    if ARGS.short_epochs:
        ARGS.pretrain_epochs = 1
        ARGS.epochs = 1
        ARGS.inter_epochs = 1
        ARGS.pretrain_epochs = 1
        ARGS.max_meta_epochs = 2

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
    elif ARGS.ae_range is not None:
        assert len(ARGS.ae_range) == 2
        aeids = range(ARGS.ae_range[0],ARGS.ae_range[1])
    else:
        aeids = [0,1]
    ctx = mp.get_context("spawn")

    if ARGS.short_epochs:
        ARGS.pretrain_epochs = 1
        ARGS.epochs = 1
        ARGS.inter_epochs = 1
        ARGS.pretrain_epochs = 1
        ARGS.meta_epochs = 2

    if ARGS.dset in ['MNIST','FashionMNIST']:
        ARGS.image_size = 28
        ARGS.dset_size = 60000
        ARGS.num_channels = 1
        ARGS.num_clusters = 10
    elif ARGS.dset == 'USPS':
        ARGS.image_size = 16
        ARGS.dset_size = 7291
        ARGS.num_channels = 1
        ARGS.num_clusters = 10
    elif ARGS.dset == 'CIFAR10':
        ARGS.image_size = 32
        ARGS.dset_size = 50000
        ARGS.num_channels = 3
        ARGS.num_clusters = 10
    elif ARGS.dset == 'coil-100':
        ARGS.image_size = 128
        ARGS.dset_size = 7200
        ARGS.num_channels = 3
        ARGS.num_clusters = 100
    elif ARGS.dset == 'letterAJ':
        ARGS.image_size = 28
        ARGS.dset_size = 18456
        ARGS.num_channels = 1
        ARGS.num_clusters = 10


    device = torch.device(f'cuda:{ARGS.gpu}')
    new_ae = partial(utils.make_ae,device=device,NZ=ARGS.NZ,image_size=ARGS.image_size,num_channels=ARGS.num_channels)
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
                aes.append({'aeid':aeid, 'ae':new_ae(aeid)})
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
            with torch.cuda.device(device):
                vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
        if ARGS.conc:
            concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
            vecs.append({'aeid': 'concat', 'latents': concatted_vecs})
        print(f'Generation time: {utils.asMinutes(time()-generate_start_time)}')
    elif 3 in ARGS.sections or 4 in ARGS.sections:
        with ctx.Pool(processes=len(aes)) as pool:
            print(f"Loading vecs for {len(aes)} aes...")
            with torch.cuda.device(device):
                vecs = [filled_load_vecs(ae) for ae in aeids] if ARGS.single else pool.map(filled_load_vecs, aeids)
        assert all([v['latents'].shape[1] == ARGS.NZ for v in vecs])

    from sklearn.metrics import normalized_mutual_info_score as mi_func
    def rmi_func(pred,gt): return round(mi_func(pred,gt),4)
    def racc(pred,gt): return round(utils.accuracy(pred,gt),4)
    labels_dir = f'../{ARGS.dset}/labels/'
    try: gt_labels = np.load(os.path.join(labels_dir,'gt_labels.npy'))
    except FileNotFoundError:
        d = utils.get_vision_dset(ARGS.dset)
        set_trace()
        gt_labels = d.y.cpu().detach().numpy()
        utils.np_save(gt_labels,labels_dir,'gt_labels.npy')
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
        ensemble_build_start_time = time()
        vecs = utils.dictify_list(vecs,key='aeid')
        labels = utils.dictify_list(labels,key='aeid')
        vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
        print(f"Building ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt='none')
        print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
    elif 5 in ARGS.sections:
        print(f"Loading ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = filled_load_ensemble(aeids)

    if 5 in ARGS.sections:
        train_start_time = time()
        best_acc = 0.
        best_mi = 0.
        count = 0
        num_agrees_list = []
        acc_histories, mi_histories, num_agree_histories = [],[],[]
        for meta_epoch_num in range(ARGS.max_meta_epochs):
            if ARGS.show_gpu_memory:
                mem_used = 0
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) or hasattr(obj, 'data') and torch.is_tensor(obj.data):
                            mem_used += obj.element_size() * obj.nelement()
                    except: pass
                print(f"GPU memory usage: {mem_used}")
            meta_epoch_start_time = time()
            print('\nMeta epoch:', meta_epoch_num)
            copied_aes = [copy.deepcopy(ae) for ae in aes]
            if len(aes) == 0:
                print('No legit aes left')
                sys.exit()
            print('Aes remaining:', *list(aeids))
            difficult_list = [1,2,3] if ARGS.test else sorted([l for l in set(ensemble_labels) if l != -1], key=lambda x: (ensemble_labels[~all_agree]==x).sum()/(ensemble_labels==x).sum(),reverse=True)
            print('difficults:',difficult_list)
            worst3 = difficult_list[:3]
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Training aes {list(aeids)}...")

                try: assert set([ae['aeid'] for ae in copied_aes]) == set(aeids)
                except: set_trace()
                for aedict in copied_aes:
                    try: assert(aedict['ae'].pred.in_features == ARGS.NZ)
                    except:aedict['ae'].pred = utils.mlp(ARGS.NZ,25,ARGS.num_clusters,device=device)
                    aedict['ae'].pred2 = utils.mlp(ARGS.NZ,25,2,device=device)
                    aedict['ae'].pred3 = utils.mlp(ARGS.NZ,25,3,device=device)
            filled_train = partial(train_ae,args=ARGS,centroids_by_id=centroids_by_id,worst3=worst3,targets=ensemble_labels,all_agree=all_agree)
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Training aes {list(aeids)}...")
                [filled_train(ae) for ae in copied_aes] if ARGS.single else pool.map(filled_train, copied_aes)
            aes = copied_aes
            assert set([ae['aeid'] for ae in aes]) == set(aeids)
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Generating vecs for {len(aes)} aes...")
                vecs = [filled_generate(ae) for ae in aes] if ARGS.single else pool.map(filled_generate, [copy.deepcopy(ae) for ae in aes])
            if ARGS.conc:
                scanner = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500)
                concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
                vecs.append({'aeid': 'concat', 'latents': concatted_vecs})
            with ctx.Pool(processes=len(aes)) as pool:
                print(f"Generating labels for {len(aes)} aes...")
                with torch.cuda.device(device):
                    print(f"Labelling vecs for {len(aes)} aes...")
                    labels = [filled_label(v) for v in vecs] if ARGS.single else pool.map(filled_label, vecs)
            if ARGS.scatter_clusters:
                selected = random.choice(labels)['umapped_latents']
                utils.scatter_clusters(selected, gt_labels,show=True)
            vecs = utils.dictify_list(vecs,key='aeid')
            labels = utils.dictify_list(labels,key='aeid')
            vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
            print(f"Building ensemble from {len(aes)} aes...")
            centroids_by_id, ensemble_labels, all_agree  = build_ensemble(vecs_and_labels,ARGS,pivot='none',given_gt='none')
            print(ensemble_labels)

            acc = racc(ensemble_labels,gt_labels)
            mi = rmi_func(ensemble_labels,gt_labels)
            acc_histories.append(acc)
            mi_histories.append(mi)
            num_agree_histories.append(all_agree.sum())
            new_best_acc = acc
            print('AE Scores:')
            print('L acc', [racc(label_dict['labels'][label_dict['labels']>=0],gt_labels[label_dict['labels']>=0]) for label_dict in labels.values()])
            print('L MI', [rmi_func(labels[x]['labels'][labels[x]['labels']>=0],gt_labels[labels[x]['labels']>=0]) for x in aeids])
            print('Ensemble Scores:')
            print('Acc:',acc)
            print('Acc agree:', racc(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('NMI:',mi)
            print('NMI agree:',rmi_func(ensemble_labels[all_agree],gt_labels[all_agree]))
            print('Num agrees:', all_agree.sum(), round(all_agree.sum()/ARGS.dset_size,4))
            print(f'Meta Epoch time: {utils.asMinutes(time()-meta_epoch_start_time)}')
            if new_best_acc <= best_acc:
                count += 1
            else:
                best_acc = new_best_acc
                best_mi = mi
                count = 0
                best_ensemble_labels = ensemble_labels
                best_all_agree = all_agree
            print('Best acc:', best_acc)
            print('Best MI:', best_mi)
            print('Count:',count)
            num_agrees_list.append(all_agree.sum())
            print(f'Num agrees list: {num_agrees_list}')
            if count == ARGS.patience: break

        exp_dir = f'../{ARGS.dset}/experiments/{ARGS.exp_name}'
        # Save the most accurate ensemble labels and the corresponding indices of agreement
        ensemble_dict = {'ensemble_labels': best_ensemble_labels, 'all_agree': best_all_agree}
        utils.np_savez(ensemble_dict,exp_dir,'best_ensemble')
        # Save histories of acc, nmi and num_agrees
        summary_dict = {'acc_histories':acc_histories, 'mi_histories':mi_histories, 'num_agree_histories': num_agree_histories}
        utils.np_savez(summary_dict,exp_dir,'histories.npz')

        # Save info for each ae, latent vecs and labels
        by_aeid = {aeid:{**vecs_and_labels[aeid],**centroids_by_id[aeid]} for aeid in set(vecs_and_labels.keys()).intersection(set(centroids_by_id.keys()))}
        for aeid,v in by_aeid.items():
            arrays_to_save = {k:v for k,v in by_aeid[aeid].items() if k!='aeid'}
            utils.np_savez(arrays_to_save,exp_dir,f'ae{aeid}_items.npz')
        exp_outputs = {'ensemble_labels': ensemble_labels,'all_agree':all_agree,'by_aeid':by_aeid}
        print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
