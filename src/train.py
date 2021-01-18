import copy
from pdb import set_trace
import functools
import numpy as np
import os
from time import time
import torch
from torch.utils import data
import random
import sys

import multi_ae
import options
import utils


if __name__ == "__main__": # Seems necessary to make multiproc work
    ARGS = options.get_argparse()
    global LOAD_START_TIME; LOAD_START_TIME = time()
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ARGS.aeids is not None:
        aeids = ARGS.aeids
    elif ARGS.num_aes is not None:
        aeids = range(ARGS.num_aes)
    elif ARGS.ae_range is not None:
        assert len(ARGS.ae_range) == 2
        aeids = range(ARGS.ae_range[0],ARGS.ae_range[1])
    else:
        aeids = [0,1]

    if ARGS.short_epochs:
        ARGS.pretrain_epochs = 1
        ARGS.epochs = 1
        ARGS.inter_epochs = 1
        ARGS.pretrain_epochs = 1
        ARGS.max_meta_epochs = 2

    big = False
    if ARGS.dset in ['MNISTfull','FashionMNIST']:
        ARGS.image_size = 28
        ARGS.dset_size = 70000
        ARGS.num_channels = 1
        ARGS.num_clusters = 10
    elif ARGS.dset == 'USPS':
        ARGS.image_size = 16
        ARGS.dset_size = 9298
        ARGS.num_channels = 1
        ARGS.num_clusters = 10
    elif ARGS.dset == 'MNISTtest':
        ARGS.image_size = 28
        ARGS.dset_size = 10000
        ARGS.num_channels = 1
        ARGS.num_clusters = 10
    elif ARGS.dset == 'CIFAR10':
        ARGS.image_size = 32
        ARGS.dset_size = 50000
        ARGS.num_channels = 3
        ARGS.num_clusters = 10
        big = True
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

    if ARGS.test and ARGS.save:
        print("shouldn't be saving for a test run")
        sys.exit()
    if ARGS.test and ARGS.max_meta_epochs == 30: ARGS.max_meta_epochs = 2
    if not ARGS.disable_cuda and torch.cuda.is_available():
        ARGS.device = torch.device(f'cuda:{ARGS.gpu}')
    else:
        ARGS.device = torch.device('cpu')
    if ARGS.split == -1:
        ARGS.split = len(aeids)
    print(ARGS)

    exp_dir = utils.set_experiment_dir(ARGS.exp_name,overwrite=ARGS.overwrite)
    new_ae = functools.partial(utils.make_ae,device=ARGS.device,NZ=ARGS.NZ,image_size=ARGS.image_size,num_channels=ARGS.num_channels,big=big)
    dset = utils.get_vision_dset(ARGS.dset,ARGS.device)
    dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),ARGS.batch_size,drop_last=True),pin_memory=False)
    determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),ARGS.gen_batch_size,drop_last=False),pin_memory=False)
    filled_pretrain = functools.partial(multi_ae.rtrain_ae,args=ARGS,should_change=True,dset=dset)
    filled_label = functools.partial(multi_ae.label_single,args=ARGS,abl=False)
    filled_load_ae = functools.partial(multi_ae.load_ae,args=ARGS)
    filled_load_vecs = functools.partial(multi_ae.load_vecs,args=ARGS)
    filled_load_labels = functools.partial(multi_ae.load_labels,args=ARGS)
    filled_load_ensemble = functools.partial(multi_ae.load_ensemble,args=ARGS)

    if 1 in ARGS.sections: #Rtrain aes rather than load
        pretrain_start_time = time()
        aes = []
        print(f"Instantiating {len(aeids)} aes...")
        for aeid in aeids:
            aes.append({'aeid':aeid, 'ae':new_ae(aeid)})
        print(f"Pretraining {len(aes)} aes...")
        vecs = utils.apply_maybe_multiproc(filled_pretrain,aes,split=ARGS.split,single=ARGS.single)
        if ARGS.conc:
            concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
            vecs.append({'aeid': 'concat', 'latents': concatted_vecs})
        print(f'Pretraining time: {utils.asMinutes(time()-pretrain_start_time)}')
    else:
        print(f"Loading {len(aeids)} aes...")
        aes = utils.apply_maybe_multiproc(filled_load_ae,aeids,split=ARGS.split,single=ARGS.single)
        vecs = utils.apply_maybe_multiproc(filled_load_vecs,aeids,split=ARGS.split,single=ARGS.single)


    labels_dir = f'../{ARGS.dset}/labels/'
    try: gt_labels = np.load(os.path.join(labels_dir,'gt_labels.npy'))
    except FileNotFoundError:
        print('Loading gt labels directly from the dset')
        d = utils.get_vision_dset(ARGS.dset,device=ARGS.device)
        gt_labels = d.y.cpu().detach().numpy()
        utils.np_save(gt_labels,labels_dir,'gt_labels.npy',verbose=False)
    if 2 in ARGS.sections:
        label_start_time = time()
        print(f"Labelling {len(aes)} aes...")
        labels = utils.apply_maybe_multiproc(filled_label,vecs,split=len(vecs),single=ARGS.single)
        print(f'Labelling time: {utils.asMinutes(time()-label_start_time)}')
    elif 3 in ARGS.sections:
        print(f"Loading labels for {len(aes)} aes...")
        labels = utils.apply_maybe_multiproc(filled_load_labels,aeids,split=len(aeids),single=ARGS.single)

    if 3 in ARGS.sections:
        ensemble_build_start_time = time()
        vecs = utils.dictify_list(vecs,key='aeid')
        labels = utils.dictify_list(labels,key='aeid')
        vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
        print(f"Building ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = multi_ae.build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt='none')
        print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
    elif 4 in ARGS.sections:
        print(f"Loading ensemble from {len(aes)} aes...")
        centroids_by_id, ensemble_labels, all_agree = filled_load_ensemble(aeids)
        labels = utils.apply_maybe_multiproc(filled_load_labels,aeids,split=len(aeids),single=ARGS.single)

    if 4 in ARGS.sections:
        train_start_time = time()
        best_acc = 0.
        best_mi = 0.
        count = 0
        acc_histories, mi_histories, num_agree_histories = [],[],[]
        for meta_epoch_num in range(ARGS.max_meta_epochs):
            if ARGS.show_gpu_memory: utils.show_gpu_memory()
            meta_epoch_start_time = time()
            print('\nMeta epoch:', meta_epoch_num)
            copied_aes = [copy.deepcopy(ae) for ae in aes]
            print(f"Training aes {list(aeids)}...")
            for aedict in copied_aes:
                if not hasattr(aedict['ae'],'pred'):
                    print(f"making a first pred subnet for {aedict['aeid']}")
                    nout = utils.get_num_labels(labels[aedict['aeid']]['labels']) if ARGS.ablation == 'sharing' else ARGS.num_clusters
                    aedict['ae'].pred = utils.mlp(ARGS.NZ,25,nout,device=ARGS.device)

                if aedict['ae'].pred[-1].weight.shape[0] != utils.get_num_labels(labels[aedict['aeid']]['labels']):
                    print(f"pred is the wrong shape for {aedict['aeid']}, making a new one")
                    aedict['ae'].pred = utils.mlp(ARGS.NZ,25,nout,device=ARGS.device)
            if ARGS.ablation == 'none':
                filled_train = functools.partial(multi_ae.train_ae,args=ARGS,targets=ensemble_labels,all_agree=all_agree,dset=dset,sharing_ablation=False)
            elif ARGS.ablation == 'sharing':
                filled_train = functools.partial(multi_ae.train_ae,args=ARGS,targets=labels,all_agree=np.ones(ARGS.dset_size).astype(np.bool),dset=dset,sharing_ablation=True)
            elif ARGS.ablation == 'filtering':
                filled_train = functools.partial(multi_ae.train_ae,args=ARGS,targets=ensemble_labels,all_agree=np.ones(ARGS.dset_size).astype(np.bool),dset=dset,sharing_ablation=False)
            print(f"Training aes {list(aeids)}...")
            vecs = utils.apply_maybe_multiproc(filled_train,copied_aes,split=ARGS.split,single=ARGS.single)
            aes = copied_aes
            assert set([ae['aeid'] for ae in aes]) == set(aeids)
            if ARGS.conc:
                concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
                vecs.append({'aeid': 'concat', 'latents': concatted_vecs})
            print(f"Labelling vecs for {len(aes)} aes...")
            labels = utils.apply_maybe_multiproc(filled_label,vecs,split=len(vecs),single=ARGS.single)
            if ARGS.scatter_clusters:
                selected = random.choice(labels)['umapped_latents']
                utils.scatter_clusters(selected, gt_labels,show=True)
            vecs = utils.dictify_list(vecs,key='aeid')
            labels = utils.dictify_list(labels,key='aeid')
            vecs_and_labels = {aeid:{**vecs[aeid],**labels[aeid]} for aeid in set(vecs.keys()).intersection(set(labels.keys()))}
            print(f"Building ensemble from {len(aes)} aes...")
            centroids_by_id, ensemble_labels, all_agree  = multi_ae.build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt='none')

            acc = utils.racc(ensemble_labels,gt_labels)
            mi = utils.rmi_func(ensemble_labels,gt_labels)
            acc_histories.append(acc)
            mi_histories.append(mi)
            num_agree_histories.append(all_agree.sum())
            new_best_acc = acc
            print('AE Scores:')
            print('Accs', [utils.racc(labels[x]['labels'],gt_labels) for x in aeids])
            print('MIs', [utils.rmi_func(labels[x]['labels'],gt_labels) for x in aeids])
            print('DAccs', [utils.racc(vecs[x]['direct_labels'],gt_labels) for x in aeids])
            print('DMIs', [utils.rmi_func(vecs[x]['direct_labels'],gt_labels) for x in aeids])
            print('Ensemble Scores:')
            print('Acc:',acc)
            print('NMI:',mi)
            print('Num agrees:', all_agree.sum(), round(all_agree.sum()/ARGS.dset_size,4))
            if all_agree.any():
                print('Acc agree:', utils.racc(ensemble_labels[all_agree],gt_labels[all_agree]))
                print('NMI agree:',utils.rmi_func(ensemble_labels[all_agree],gt_labels[all_agree]))
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
            print(f'Num agree histories: {num_agree_histories}')
            if count == ARGS.patience: break

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
        summary_txt_file_path = os.path.join(exp_dir,'summary.txt')
        with open(summary_txt_file_path, 'w') as f:
            f.write(str(ARGS))
            f.write(f"\nBest Acc: {best_acc}")
            f.write(f"\nBest NMI: {best_mi}")
        print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
