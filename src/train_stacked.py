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
from functools import partial
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
import torch.nn.functional as F
import utils
import hdbscan
import umap.umap_ as umap
from sklearn.metrics import adjusted_rand_score,adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
import random


NZ=50

class EncoderStacked(nn.Module):
    def __init__(self,block1,block2): 
        super(EncoderStacked, self).__init__()
        self.block1,self.block2 = block1,block2

    def forward(self,inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        return out1, out2


enc_b1, enc_b2 = utils.get_enc_blocks('cuda',NZ)
enc = EncoderStacked(enc_b1,enc_b2)
dec = utils.GeneratorStacked(nz=NZ,ngf=32,nc=1,dropout_p=0.)
dec.to('cuda')
dl = utils.get_mnist_dloader(x_only=True)

lf = nn.L1Loss()
opt = torch.optim.Adam([{'params': enc.parameters()},{'params':dec.parameters()}])
for epoch in range(10):
    total_mid_loss = 0.
    total_pred_loss = 0.
    for i,(xb,idx) in enumerate(dl):
        enc_mid, latent = enc(xb)
        dec_mid, pred = dec(latent)
        mid_loss = lf(enc_mid,dec_mid)
        pred_loss = lf(pred,xb)
        loss = mid_loss + pred_loss
        loss.backward(); opt.step(); opt.zero_grad()
        total_mid_loss = total_mid_loss*(i+1)/(i+2) + mid_loss.item()/(i+2)
        total_pred_loss = total_pred_loss*(i+1)/(i+2) + pred_loss.item()/(i+2)
    print(f'Mid Loss: {total_mid_loss}, Pred Loss {total_pred_loss}')

utils.check_ae_images(enc,dec,dl.dataset,stacked=True)
