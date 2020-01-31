import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score
from scipy.special import softmax

from utils import translate_labellings


def compress_labels(labels):
    x = [i for i in set(labels) if i != -1]
    new_labels = np.array([l if l == -1 else x.index(l) for l in labels])
    return new_labels

def debable(labellings_list):
    labellings_list.sort(key=lambda x: x.max(),reverse=True)
    translated_list = []
    pivot = labellings_list[0]
    for not_lar in labellings_list[1:]:
        not_lar_trans = translate_labellings(not_lar,pivot)
        not_lar = np.array([not_lar_trans[l] for l in not_lar])
        translated_list.append(not_lar)
    return translated_list

def load_labels():

    pls = [np.load(f'../labels/saved_pixel_labels_seed{i}.npy') for i in range(1,20)]
    mls = [np.load(f'../labels/saved_mid_labels_seed{i}.npy') for i in range(1,20)]
    lls = [np.load(f'../labels/saved_latent_labels_seed{i}.npy') for i in range(1,20)]

    npls = [compress_labels(l) for l in pls]
    nmls = [compress_labels(l) for l in mls]
    nlls = [compress_labels(l) for l in lls]

    npls = debable(npls)
    nmls = debable(nmls)
    nlls = debable(nlls)
    return np.stack(npls), np.stack(nmls), np.stack(nlls)


if __name__ == "__main__":
    p,m,l = load_labels()
    probl = (np.expand_dims(np.arange(l.max()+1),0)==np.expand_dims(l,2)).sum(axis=0)
    probl = softmax(probl,axes=-1)
# Next make centroids with these labels, then train
# Might want to concat pixels, mids and latents, and also consider dec mids
