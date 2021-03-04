import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score as mi_func
from pdb import set_trace
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import gc
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.multiprocessing as mp


def reload():
    import importlib, utils
    importlib.reload(utils)

def save_and_check(enc,dec,fname):
    torch.save({'enc': enc, 'dec': dec},fname)
    l = torch.load(fname)
    e,d = l['enc'], l['dec']
    test_mods_eq(e,enc); test_mods_eq(d,dec)

def show_xb(xb): plt.imshow(xb[0,0]); plt.show()
def to_float_tensor(item): return item.float().div_(255.)
def add_colour_dimension(item):
    if item.dim() == 4:
        if item.shape[1] in [1,3]: # batch, channels, size, size
            return item
        elif item.shape[3] in [1,3]:  # batch, size, size, channels
            return item.permute(0,3,1,2)
    if item.dim() == 3:
        if item.shape[0] in [1,3]: # channels, size, size
            return item
        elif item.shape[2] in [1,3]: # size, size, channels
            return item.permute(2,0,1)
        else: return item.unsqueeze(1) # batch, size, size
    else: return item.unsqueeze(0) # size, size

def get_datetime_stamp(): return str(datetime.now()).split()[0][5:] + '_'+str(datetime.now().time()).split()[0][:-7]

def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y': return True
    elif answer == 'n': return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))

def set_experiment_dir(exp_name, overwrite):
    exp_name = get_datetime_stamp() if exp_name == "" else exp_name
    exp_dir = f'../experiments/{exp_name}'
    if not os.path.isdir(exp_dir): os.makedirs(exp_dir)
    elif exp_name.startswith('try') or overwrite: pass
    elif not get_user_yesno_answer(f'An experiment with name {exp_name} has already been run, do you want to overwrite?'):
        print('Please rerun command with a different experiment name')
        sys.exit()
    return exp_dir

def noiseify(pytensor,constant):
    noise = torch.randn_like(pytensor)
    noise /= noise.max()
    return pytensor + noise*constant

def oheify(x):
    target_category = torch.argmax(x, dim=1)
    ohe_target = (torch.arange(x.shape[1]).cuda() == target_category[:,None])[:,:,None,None].float()
    return target_category, ohe_target

def test_mods_eq(m1,m2):
    for a,b in zip(m1.parameters(),m2.parameters()):
        assert (a==b).all()

def compose(funcs):
    def _comp(x):
        for f in funcs: x=f(x)
        return x
    return _comp

def numpyify(x):
    if isinstance(x,np.ndarray): return x
    elif isinstance(x,list): return np.array(x)
    elif torch.is_tensor(x): return x.detach().cpu().numpy()

def scatter_clusters(embeddings,labels,show):
    palette = ['r','k','y','g','b','m','purple','brown','c','orange','thistle','lightseagreen','sienna']
    labels = numpyify([0]*len(embeddings)) if labels is None else numpyify(labels)
    palette = cm.rainbow(np.linspace(0,1,len(set(labels))))
    for i,label in enumerate(list(set(labels))):
        plt.scatter(embeddings[labels==label,0], embeddings[labels==label,1], s=0.2, c=[palette[i]], label=i)
    plt.legend()
    if show: plt.show()
    return plt

def print_tensors(*tensors):
    print(*[t.item() for t in tensors])

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def label_assignment_cost(labels1,labels2,label1,label2):
    assert len(labels1) == len(labels2), f"len labels1 {len(labels1)} must equal len labels2 {len(labels2)}"
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def translate_labellings(trans_from_labels,trans_to_labels,subsample_size):
    # What you're translating into has to be compressed, otherwise gives wrong results
    assert trans_from_labels.shape == trans_to_labels.shape
    num_from_labs = get_num_labels(trans_from_labels)
    num_to_labs = get_num_labels(trans_to_labels)
    assert subsample_size is 'none' or subsample_size > min(num_from_labs,num_to_labs)
    if num_from_labs <= num_to_labs:
        return translate_labellings_fanout(trans_from_labels,trans_to_labels,subsample_size)
    else:
        return translate_labellings_fanin(trans_from_labels,trans_to_labels,subsample_size)

def translate_labellings_fanout(trans_from_labels,trans_to_labels,subsample_size):
    unique_trans_from_labels = unique_labels(trans_from_labels)
    unique_trans_to_labels = unique_labels(trans_to_labels)
    if subsample_size is 'none':
        cost_matrix = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in unique_trans_to_labels if l2 != -1] for l1 in unique_trans_from_labels if l1 != -1])
    else:
        num_trys = 0
        while True:
            num_trys += 1
            if num_trys == 5: set_trace()
            sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
            trans_from_labels_subsample = trans_from_labels[sample_indices]
            trans_to_labels_subsample = trans_to_labels[sample_indices]
            if unique_labels(trans_from_labels_subsample) == unique_trans_from_labels and unique_labels(trans_from_labels_subsample) == unique_trans_to_labels: break
        cost_matrix = np.array([[label_assignment_cost(trans_from_labels_subsample,trans_to_labels_subsample,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_from_labels if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert len(col_ind) == len(set(trans_from_labels[trans_from_labels != -1]))
    return np.array([-1 if l == -1 else col_ind[l] for l in trans_from_labels])

def translate_labellings_fanin(trans_from_labels,trans_to_labels,subsample_size):
    unique_trans_from_labels = unique_labels(trans_from_labels)
    unique_trans_to_labels = unique_labels(trans_to_labels)
    if subsample_size is 'none':
        cost_matrix = np.array([[label_assignment_cost(trans_to_labels,trans_from_labels,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_to_labels if l1 != -1])
    else:
        while True: # Keep trying random indices unitl you reach one that contains all labels
            sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
            trans_from_labels_subsample = trans_from_labels[sample_indices]
            trans_to_labels_subsample = trans_to_labels[sample_indices]
            if unique_labels(trans_from_labels_subsample) == unique_trans_from_labels and unique_labels(trans_from_labels_subsample) == unique_trans_to_labels: break
        sample_indices = np.random.choice(range(trans_from_labels.shape[0]),subsample_size,replace=False)
        trans_from_labels_subsample = trans_from_labels[sample_indices]
        trans_to_labels_subsample = trans_to_labels[sample_indices]
        cost_matrix = np.array([[label_assignment_cost(trans_to_labels_subsample,trans_from_labels_subsample,l1,l2) for l2 in unique_trans_from_labels if l2 != -1] for l1 in unique_trans_to_labels if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert len(col_ind) == get_num_labels(trans_to_labels)
    untranslated = [i for i in range(cost_matrix.shape[1]) if i not in col_ind]
    unique_untranslated = unique_labels(untranslated)
    # Now assign the additional, unassigned items
    cost_matrix2 = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in unique_trans_to_labels if l2 != -1] for l1 in unique_untranslated if l1 != -1])
    row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
    cl = col_ind.tolist()
    trans_dict = {f:cl.index(f) for f in cl}
    for u,t in zip(untranslated,col_ind2): trans_dict[u]=t
    trans_dict[-1] = -1
    return np.array([trans_dict[i] for i in trans_from_labels])

def get_confusion_mat(labels1,labels2):
    if max(labels1) != max(labels2):
        print('Different numbers of clusters, no point trying'); return
    trans_labels = translate_labellings(labels1,labels2)
    num_labels = max(labels1)+1
    confusion_matrix = np.array([[len([idx for idx in range(len(labels2)) if labels1[idx]==l1 and labels2[idx]==l2]) for l2 in range(num_labels)] for l1 in range(num_labels)])
    confusion_matrix = confusion_matrix[:,trans_labels]
    idx = np.arange(num_labels)
    confusion_matrix[idx,idx]=0
    return confusion_matrix

def debable(labellings_list,pivot):
    labellings_list.sort(key=lambda x: x.max())
    if pivot == 'none':
        pivot = labellings_list.pop(0)
        translated_list = [pivot]
    else:
        translated_list = []
    for not_lar in labellings_list:
        not_lar_translated = translate_labellings(not_lar,pivot)
        translated_list.append(not_lar_translated)
    return translated_list

def compute_multihots(l,probs):
    assert len(l) > 0
    mold = np.expand_dims(np.arange(l.max()+1),0) # (num_aes, num_labels)
    hits = (mold==np.expand_dims(l,2)) # (num_aes, dset_size, num_labels)
    if probs != 'none': hits = np.expand_dims(probs,2)*hits
    multihots = hits.sum(axis=0) # (dset_size, num_labels)
    return multihots

def check_latents(dec,latents,show,stacked):
    _, axes = plt.subplots(6,2,figsize=(7,7))
    for i,latent in enumerate(latents):
        try:
            outimg = dec(latent[None,:,None,None])
            if stacked: outimg = outimg[-1]
            axes.flatten()[i].imshow(outimg[0,0])
        except: set_trace()
    if show: plt.show()
    plt.clf()

def unique_labels(labels):
    if isinstance(labels,np.ndarray) or isinstance(labels,list):
        return set(labels)
    elif isinstance(labels,torch.Tensor):
        unique_tensor = labels.unique()
        return set(unique_tensor.tolist())
    else:
        print("Unrecognized type for labels:", type(labels))
        raise TypeError

def get_num_labels(labels):
    assert labels.ndim == 1
    return  len([l for l in unique_labels(labels) if l != -1])

def dictify_list(x,key):
    assert isinstance(x,list)
    assert len(x) > 0
    assert isinstance(x[0],dict)
    return {item[key]: item for item in x}

def accuracy(labels1,labels2,subsample_size='none'):
    trans_labels = translate_labellings(compress_labels(labels1),compress_labels(labels2),subsample_size)
    return sum(trans_labels==numpyify(labels2))/len(labels1)

def compress_labels(labels):
    if isinstance(labels,torch.Tensor): labels = labels.detach().cpu().numpy()
    x = sorted([i for i in set(labels) if i != -1])
    new_labels = np.array([l if l == -1 else x.index(l) for l in labels])
    return new_labels

def num_labs(labels): return len(set([l for l in labels if l != -1]))

def cont_factorial(x): return (x/np.e)**x*(2*np.pi*x)**(1/2)*(1+1/(12*x))
def cont_choose(ks): return cont_factorial(np.sum(ks))/np.prod([cont_factorial(k) for k in ks if k > 0])
def prob_results_given_c(results,cluster,prior_correct):
    """For single dpoint, prob of these results given right answer for cluster.
    ARGS:
        results (np.array): votes for this dpoint
        cluster (int): right answer to condition on
        prior_correct (\in (0,1)): guess for acc of each element of ensemble
        """

    assert len(results.shape) <= 1
    prob = 1
    results_normed = np.array(results)
    results_normed = results_normed / np.sum(results_normed)
    for c,r in enumerate(results_normed):
        if c==cluster: prob_of_result = prior_correct**r
        else: prob_of_result = ((1-prior_correct)/results.shape[0])**r
        prob *= prob_of_result
    partitions = cont_choose(results_normed)
    prob *= partitions
    try:assert prob <= 1
    except:set_trace()
    return prob

def prior_for_results(results,prior_correct):
    probs = [prob_results_given_c(results,c,prior_correct) for c in range(results.shape[0])]
    set_trace()
    return sum(probs)

def all_conditionals(results,prior_correct):
    """For each class, prob of results given that class."""
    cond_probs = [prob_results_given_c(results,c,prior_correct) for c in range(len(results))]
    assert np.sum(cond_probs) < 1.01
    return np.array(cond_probs)

def posteriors(results,prior_correct):
    """Bayes to get prob of each class given these results."""
    conditionals = all_conditionals(results,prior_correct)
    posterior_array = conditionals/np.sum(conditionals)
    return posterior_array

def posterior_corrects(results):
    probs = []
    for p in np.linspace(0.6,1.0,10):
        conditional_prob = np.prod([np.sum(all_conditionals(r,p)) for r in results])
        probs.append(conditional_prob)
    probs = np.array(probs)
    posterior_for_accs = 0.1*probs/np.sum(probs) # Prior was uniform over all accs in range
    assert posterior_for_accs.max() < 1.01
    return posterior_for_accs

def votes_to_probs(multihots,prior_correct):
    """For each dpoint, compute probs for each class, given these ensemble votes.
    ARGS:
        multihots (np.array): votes for each dpoint, size N x num_classes
        prior_correct (\in (0,1)): guess for acc of each element of ensemble
        """

    probs_list = [np.ones(multihots.shape[-1])/multihots.shape[-1] if r.max() == 0 else posteriors(r,prior_correct) for r in multihots]
    probs_array = np.array(probs_list)
    return probs_array

def check_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def torch_save(checkpoint,directory,fname):
    check_dir(directory)
    torch.save(checkpoint,os.path.join(directory,fname))

def np_save(array,directory,fname,verbose):
    check_dir(directory)
    save_path = os.path.join(directory,fname)
    if verbose: print('Saving to', save_path)
    np.save(save_path,array)

def np_savez(data_dict,directory,fname):
    check_dir(directory)
    np.savez(os.path.join(directory,fname),**data_dict)

def apply_maybe_multiproc(func,input_list,split,single):
    if single:
        output_list = [func(item) for item in input_list]
    else:
        list_of_lists = []
        ctx = mp.get_context("spawn")
        num_splits = math.ceil(len(input_list)/split)
        for i in range(num_splits):
            with ctx.Pool(processes=split) as pool:
                new_list = pool.map(func, input_list[split*i:split*(i+1)])
            if num_splits != 1: print(f'finished {i}th split section')
            list_of_lists.append(new_list)
        output_list = [item for sublist in list_of_lists for item in sublist]
    return output_list

def show_gpu_memory():
    mem_used = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or hasattr(obj, 'data') and torch.is_tensor(obj.data):
                mem_used += obj.element_size() * obj.nelement()
        except: pass
    print(f"GPU memory usage: {mem_used}")

def rmi_func(pred,gt): return round(mi_func(pred,gt),4)
def racc(pred,gt): return round(accuracy(pred,gt),4)
if __name__ == "__main__":
    small_labels = np.array([1,2,3,4,0,4,4])
    big_labels = np.array([0,1,2,3,4,5,5])
    print(translate_labellings(big_labels,small_labels))
    print(translate_labellings(small_labels,big_labels))
    print(translate_labellings(big_labels,big_labels))
