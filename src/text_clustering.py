import numpy as np
import math
import torch
from longformer.sliding_chunks import pad_to_window_size
from transformers import RobertaTokenizer
from sklearn.datasets import fetch_20newsgroups
from pdb import set_trace
import torch.nn as nn
import options
from time import time
import random
import sys
from ng_dset import NGDataset
from torch.utils import data
import utils
import functools
import multi_ae
import os
import copy

from transformers import LongformerForSequenceClassification
from longformer.longformer import Longformer, LongformerConfig


#config = LongformerConfig.from_pretrained('allenai/longformer-base-4096',attention_mode='sliding_chunks')
#config.num_labels = 20
#config.attention_mode = 'sliding_chunks'
#model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True,num_labels=20).cuda()
#model = LongformerForSequenceClassification.from_pretrained(config=config).cuda()
#model = Longformer.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True,output_hidden_states=True, num_labels=20).cuda()
#model.longformer.encoder.layer = model.longformer.encoder.layer[11:]
#torch.save(model,'later_layers_model.pt')
#torch.save(model,'later_layers_model.pt')
if __name__ == "__main__":
    model = torch.load('later_layers_model.pt')
    model.num_labels = 20
    model.config.num_labels = 20
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    model = nn.DataParallel(model,device_ids)

    ARGS = options.get_argparse()
    global LOAD_START_TIME; LOAD_START_TIME = time()
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    random.seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if ARGS.ids is not None:
        model_ids = ARGS.model_ids
    elif ARGS.num_aes is not None:
        model_ids = range(ARGS.num_aes)
    elif ARGS.ae_range is not None:
        assert len(ARGS.ae_range) == 2
        model_ids = range(ARGS.ae_range[0],ARGS.ae_range[1])
    else:
        model_ids = [0,1]

    ARGS.num_clusters = 20
    if ARGS.short_epochs:
        ARGS.pretrain_epochs = 1
        ARGS.epochs = 1
        ARGS.inter_epochs = 1
        ARGS.pretrain_epochs = 1
        ARGS.max_meta_epochs = 2

    ARGS.dset = 'NG'
    if ARGS.test and ARGS.save:
        print("shouldn't be saving for a test run")
        sys.exit()
    if ARGS.test and ARGS.max_meta_epochs == 30: ARGS.max_meta_epochs = 2
    if not ARGS.disable_cuda and torch.cuda.is_available():
        ARGS.device = torch.device(f'cuda:{ARGS.gpu}')
    else:
        ARGS.device = torch.device('cpu')
    if ARGS.split == -1:
        ARGS.split = len(model_ids)
    dset = NGDataset('../NG/preprocessed',ARGS.device)
    ARGS.dset_size = len(dset)
    print(ARGS)
    #model.to(ARGS.device)

    exp_dir = utils.set_experiment_dir(ARGS.exp_name,overwrite=ARGS.overwrite)

    #dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.RandomSampler(dset),ARGS.batch_size,drop_last=True),pin_memory=False, collate_fn=collate_fn)
    #determin_dl = data.DataLoader(dset,batch_sampler=data.BatchSampler(data.SequentialSampler(dset),ARGS.gen_batch_size,drop_last=False),pin_memory=False,collate_fn=collate_fn)
    filled_pretrain = functools.partial(multi_ae.rtrain_trans,args=ARGS,should_change=True,dset=dset)
    filled_label = functools.partial(multi_ae.label_single,args=ARGS,abl=False)
    filled_load_model = functools.partial(multi_ae.load_model,args=ARGS)
    filled_load_vecs = functools.partial(multi_ae.load_vecs,args=ARGS)
    filled_load_labels = functools.partial(multi_ae.load_labels,args=ARGS)
    filled_load_ensemble = functools.partial(multi_ae.load_ensemble,args=ARGS)

    if 1 in ARGS.sections: #Rtrain rather than load
        pretrain_start_time = time()
        transformers = []
        print(f"Instantiating {len(model_ids)} transformers...")
        for model_id in model_ids:
            transformers.append({'model_id':model_id, 'model':copy.deepcopy(model)})
        print(f"Pretraining {len(transformers)} transformers...")
        vecs = utils.apply_maybe_multiproc(filled_pretrain,transformers,split=ARGS.split,single=ARGS.single)
        if ARGS.conc:
            concatted_vecs = np.concatenate([v['latents'] for v in vecs],axis=-1)
            vecs.append({'model_id': 'concat', 'latents': concatted_vecs})
        print(f'Pretraining time: {utils.asMinutes(time()-pretrain_start_time)}')
    else:
        print(f"Loading {len(model_ids)} transformers...")
        transformers = utils.apply_maybe_multiproc(filled_load_model,model_ids,split=ARGS.split,single=ARGS.single)
        vecs = utils.apply_maybe_multiproc(filled_load_vecs,model_ids,split=ARGS.split,single=ARGS.single)
    #transformers = []
    #print(f"Instantiating {len(model_ids)} transformers...")
    #for model_id in model_ids:
    #    transformers.append({'model_id':model_id, 'transformer':copy.deepcopy(model)})
    #vecs = utils.apply_maybe_multiproc(filled_pretrain,transformers,split=ARGS.split,single=ARGS.single)

    gt_labels = np.load('../NG/labels/preprocessed_20ng_labels.npy')
    if 2 in ARGS.sections:
        label_start_time = time()
        print(f"Labelling {len(transformers)} transformers...")
        labels = utils. apply_maybe_multiproc(filled_label,vecs,split=len(vecs),single=ARGS.single)
        print(f'Labelling time: {utils.asMinutes(time()-label_start_time)}')
    elif 3 in ARGS.sections:
        print(f"Loading labels for {len(transformers)} transformers...")
        labels = utils.apply_maybe_multiproc(filled_load_labels,model_ids,split=len(model_ids),single=ARGS.single)

    if 3 in ARGS.sections:
        ensemble_build_start_time = time()
        vecs = utils.dictify_list(vecs,key='model_id')
        labels = utils.dictify_list(labels,key='model_id')
        vecs_and_labels = {model_id:{**vecs[model_id],**labels[model_id]} for model_id in set(vecs.keys()).intersection(set(labels.keys()))}
        print(f"Building ensemble from {len(transformers)} transformers...")
        centroids_by_id, ensemble_labels, all_agree = multi_ae.build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt='none')
        print(f'Ensemble building time: {utils.asMinutes(time()-ensemble_build_start_time)}')
    elif 4 in ARGS.sections:
        print(f"Loading ensemble from {len(transformers)} transformers...")
        centroids_by_id, ensemble_labels, all_agree = filled_load_ensemble(model_ids)
        labels = utils.apply_maybe_multiproc(filled_load_labels,model_ids,split=len(model_ids),single=ARGS.single)

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
            copied_transformers = [copy.deepcopy(trans) for trans in transformers]
            print(f"Training transformers {list(model_ids)}...")
            if ARGS.ablation == 'none':
                filled_train = functools.partial(multi_ae.train_transformer,args=ARGS,targets=ensemble_labels,all_agree=all_agree,dset=dset)
            elif ARGS.ablation == 'sharing':
                filled_train = functools.partial(multi_ae.train_transformer,args=ARGS,targets=labels,all_agree=np.ones(ARGS.dset_size).astype(np.bool),dset=dset,sharing_ablation=True)
            elif ARGS.ablation == 'filtering':
                filled_train = functools.partial(multi_ae.train_transformer,args=ARGS,targets=ensemble_labels,all_agree=np.ones(ARGS.dset_size).astype(np.bool),dset=dset,sharing_ablation=False)
            print(f"Training transformers {list(model_ids)}...")
            vecs = utils.apply_maybe_multiproc(filled_train,copied_transformers,split=ARGS.split,single=ARGS.single)
            transformers = copied_transformers
            assert set([trans['model_id'] for trans in transformers]) == set(model_ids)
            if ARGS.conc:
                concatted_vecs = np.concatenate([v['all_pooled_latents'] for v in vecs],axis=-1)
                vecs.append({'model_id': 'concat', 'all_pooled_latents': concatted_vecs})
            print(f"Labelling vecs for {len(transformers)} transformers...")
            labels = utils.apply_maybe_multiproc(filled_label,vecs,split=len(vecs),single=ARGS.single)
            if ARGS.scatter_clusters:
                selected = random.choice(labels)['umapped_latents']
                utils.scatter_clusters(selected, gt_labels,show=True)
            vecs = utils.dictify_list(vecs,key='model_id')
            labels = utils.dictify_list(labels,key='model_id')
            vecs_and_labels = {model_id:{**vecs[model_id],**labels[model_id]} for model_id in set(vecs.keys()).intersection(set(labels.keys()))}
            print(f"Building ensemble from {len(transformers)} transformers...")
            centroids_by_id, ensemble_labels, all_agree  = multi_ae.build_ensemble(vecs_and_labels,ARGS,pivot=gt_labels,given_gt='none')

            acc = utils.racc(ensemble_labels,gt_labels)
            mi = utils.rmi_func(ensemble_labels,gt_labels)
            acc_histories.append(acc)
            mi_histories.append(mi)
            num_agree_histories.append(all_agree.sum())
            new_best_acc = acc
            print('AE Scores:')
            print('Accs', [utils.racc(labels[x]['labels'],gt_labels) for x in model_ids])
            print('MIs', [utils.rmi_func(labels[x]['labels'],gt_labels) for x in model_ids])
            print('DAccs', [utils.racc(vecs[x]['direct_labels'],gt_labels) for x in model_ids])
            print('DMIs', [utils.rmi_func(vecs[x]['direct_labels'],gt_labels) for x in model_ids])
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
        by_id = {model_id:{**vecs_and_labels[model_id],**centroids_by_id[model_id]} for model_id in set(vecs_and_labels.keys()).intersection(set(centroids_by_id.keys()))}
        for model_id,v in by_id.items():
            arrays_to_save = {k:v for k,v in by_id[model_id].items() if k!='id'}
            utils.np_savez(arrays_to_save,exp_dir,f'trans{model_id}_items.npz')
        exp_outputs = {'ensemble_labels': ensemble_labels,'all_agree':all_agree,'by_id':by_id}
        summary_txt_file_path = os.path.join(exp_dir,'summary.txt')
        with open(summary_txt_file_path, 'w') as f:
            f.write(str(ARGS))
            f.write(f"\nBest Acc: {best_acc}")
            f.write(f"\nBest NMI: {best_mi}")
        print(f'Train time: {utils.asMinutes(time()-train_start_time)}')
