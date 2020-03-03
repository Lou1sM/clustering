import os
import sys
import math
from pdb import set_trace
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
import torchvision.datasets as tdatasets
from fastai import datasets,layers
import umap.umap_ as umap
from torch.utils import data
from sklearn.metrics import adjusted_rand_score


def reload():
    import importlib, utils
    importlib.reload(utils)

class AE(nn.Module):
    def __init__(self,enc,dec,identifier):
        super(AE, self).__init__()
        self.enc,self.dec = enc,dec
        self.identifier = identifier

    @property
    def device(self):
        return self.enc.block1[0][0].weight.device

    @property
    def first_row(self):
        return self.enc.block1[0][0].weight[0]

class NonsenseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.db1 = get_reslike_block([1,8,16,32],sz=7)
        self.db2 = get_reslike_block([32,64,128,256],sz=1)
        self.fc = nn.Sequential(*layers.bn_drop_lin(256,1 ,bn=False))
        self.act = nn.Sigmoid()

    def forward(self,inp_):
        x = self.db1(inp_)
        x = self.db2(x)
        out = self.act(self.fc(x[:,:,0,0]))
        assert out.min() >= 0
        return out

    def predict(self,inp_):
        return self.act(self(inp_))

class Generator(nn.Module):
    def __init__(self, nz,ngf,nc,dropout_p=0.):
        super(Generator, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inp):
        return self.main(inp)

class GeneratorStacked(nn.Module):
    def __init__(self, nz,ngf,nc,dropout_p=0.):
        super(GeneratorStacked, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.block1 = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            self.dropout)
            # state size. (ngf*8) x 4 x 4
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            self.dropout)
            # state size. (ngf*4) x 8 x 8
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            self.dropout)
            # state size. (ngf*2) x 16 x 16
        self.block4 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.dropout)
            # state size. (ngf) x 32 x 32
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self,inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        return out3, out5

class EncoderStacked(nn.Module):
    def __init__(self,block1,block2):
        super(EncoderStacked, self).__init__()
        self.block1,self.block2 = block1,block2

    def forward(self,inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        return out1, out2

    @property
    def device(self):
        return self.enc.block1[0][0].weight.device

class Dataholder():
    def __init__(self,train_data,train_labels): self.train_data,self.train_labels=train_data,train_labels

class TransformDataset(data.Dataset):
    def __init__(self,data,transforms,x_only,device):
        self.transform,self.x_only,self.device=compose(transforms),x_only,device
        if x_only:
            self.data = data.to(self.device)
        else:
            self.x,self.y = data.train_data.to(self.device), data.train_labels.to(self.device)
            self.x.to(self.device); self.y.to(self.device)
    def __len__(self): return len(self.data) if self.x_only else len(self.x)
    def __getitem__(self,idx):
        if self.x_only: return self.transform(self.data[idx]), idx
        else: return self.transform(self.x[idx]), self.y[idx], idx

class KwargTransformDataset(data.Dataset):
    def __init__(self,transforms,device,**kwdata):
        self.transform,self.device=compose(transforms),device
        self.data_names = []
        self.num_datas = len(kwdata)
        for data_name, data in kwdata.items():
            setattr(self,data_name,data.to(device))
            self.data_names.append(data_name)
        assert len(self.data_names) == self.num_datas
    def __len__(self): return len(getattr(self,self.data_names[0]))
    def __getitem__(self,idx):
        return [self.transform(getattr(self,data_name)[idx]) for data_name in self.data_names] + [idx]
        if self.x_only: return self.transform(self.data[idx]), idx
        else: return self.transform(self.x[idx]), self.y[idx], idx

def save_and_check(enc,dec,fname):
    torch.save({'enc': enc, 'dec': dec},fname)
    l = torch.load(fname)
    e,d = l['enc'], l['dec']
    test_mods_eq(e,enc); test_mods_eq(d,dec)

def show_xb(xb): plt.imshow(xb[0,0]); plt.show()
def normalize_leaf(t): return torch.tensor(t.data)/t.data.norm()
def normalize(t): return t/t.norm()
def to_float_tensor(item): return item.float().div_(255.)
def add_colour_dimension(item): return item.unsqueeze(0) if item.dim() == 2 else item.unsqueeze(1)
def umap_embed(vectors,**config): return umap.UMAP(random_state=42).fit_transform(vectors,**config)
def stats(x): return x.mean(),x.std()
def safemean(t): return 0 if t.numel() == 0 else t.mean()
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
    if not os.path.isdir('../experiments/{}'.format(exp_name)): os.mkdir('../experiments/{}'.format(exp_name    ))
    elif exp_name.startswith('try') or overwrite: pass
    elif not get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name)):
        print('Please rerun command with a different experiment name')
        sys.exit()

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
    palette = ['r','k','y','g','b','m','purple','brown','c','orange']
    palette = cm.rainbow(np.linspace(0,1,len(set(labels))))
    labels =  numpyify([0]*len(embeddings)) if labels is None else numpyify(labels)
    for i,label in enumerate(list(set(labels))):
        plt.scatter(embeddings[labels==label,0], embeddings[labels==label,1], s=0.2, c=[palette[i]], label=i)
    plt.legend()
    if show: plt.show()
    return plt

def print_tensors(*tensors):
    print(*[t.item() for t in tensors])

def get_reslike_block(nfs,sz):
    return nn.Sequential(
        *[layers.conv_layer(nfs[i],nfs[i+1],stride=2 if i==0 else 1,leaky=0.3,padding=1)
         for i in range(len(nfs)-1)], nn.AdaptiveMaxPool2d(sz))

class SuperAE(nn.Module):
    def __init__(self,num_aes,latent_size,device):
        super().__init__()
        self.num_aes = num_aes
        self.latent_size = latent_size
        self.device = device
        self.encs,self.decs = [],[]
        for ae_num in range(num_aes):
            enc,dec = get_enc_dec(self.device,self.latent_size)
            setattr(self,f'enc{ae_num}',enc)
            setattr(self,f'dec{ae_num}',dec)
            self.encs.append(enc)
            self.decs.append(dec)

    def forward(self,x):
        latents = torch.stack([enc(x) for enc in self.encs])
        return torch.stack([dec(latents[i]) for i,dec in enumerate(self.decs)])

    def encode(self,x):
        #return torch.stack([enc(x) for enc in self.encs])
        return [enc(x) for enc in self.encs]

    def decode_list(self,latent_list):
        #return torch.stack([dec(latent_list[i]) for i,dec in enumerate(self.decs)])
        return [dec(latent_list[i]) for i,dec in enumerate(self.decs)]

    def decode_all(self,x):
        return [dec(x) for dec in self.decs]

def make_ae(aeid,device,NZ):
    with torch.cuda.device(device):
        enc_b1, enc_b2 = get_enc_blocks('cuda',NZ)
        enc = EncoderStacked(enc_b1,enc_b2)
        dec = GeneratorStacked(nz=NZ,ngf=32,nc=1,dropout_p=0.)
        dec.to(device)
    return AE(enc,dec,aeid)

def get_enc_blocks(device, latent_size):
    block1 = get_reslike_block([1,4,8,16,32,64],sz=8)
    block2 = get_reslike_block([64,128,256,latent_size],sz=1)
    return block1.to(device), block2.to(device)

def get_enc_dec(device, latent_size):
    block1 = get_reslike_block([1,4,8,16,32],sz=7)
    block2 = get_reslike_block([32,64,128,256,latent_size],sz=1)
    enc = nn.Sequential(block1,block2)
    dec = Generator(nz=latent_size,ngf=32,nc=1,dropout_p=0.)
    return enc.to(device),dec.to(device)

def get_far_tensor(exemplars_tensor):
    while True:
        attempt = normalize(torch.randn_like(exemplars_tensor[0]))
        dist = torch.min((exemplars_tensor-attempt).norm(dim=1))
        if dist>1: return attempt

def get_mnist_dloader(x_only=False,device='cuda',bs=64):
    mnist_data=tdatasets.MNIST(root='~/unsupervised_object_learning/data/',train=True,download=True)
    data = mnist_data.train_data if x_only else mnist_data
    mnist_dl = get_dloader(raw_data=data,x_only=x_only,batch_size=bs,device=device)
    return mnist_dl

def get_mnist_dset(device='cuda',x_only=False):
    mnist_data=tdatasets.MNIST(root='~/unsupervised_object_learning/MNIST/data',train=True,download=True)
    data = mnist_data.train_data if x_only else mnist_data
    mnist_dataset = TransformDataset(data,[to_float_tensor,add_colour_dimension],x_only,device=device)
    return mnist_dataset

def get_fashionmnist_dset(device='cuda',x_only=False):
    fashionmnist_data=tdatasets.FashionMNIST(root='~/unsupervised_object_learning/FashionMNIST/data',train=True,download=True)
    data = fashionmnist_data.train_data if x_only else fashionmnist_data
    fashionmnist_dataset = TransformDataset(data,[to_float_tensor,add_colour_dimension],x_only,device=device)
    return fashionmnist_dataset

def get_dloader(raw_data,x_only,batch_size,device,random=True):
    ds = get_dset(raw_data,x_only,device=device)
    if random: s=data.BatchSampler(data.RandomSampler(ds),batch_size,drop_last=True)
    else: s=data.BatchSampler(data.SequentialSampler(ds),batch_size,drop_last=True)
    return data.DataLoader(ds,batch_sampler=s,pin_memory=False)

def get_dset(raw_data,x_only,device,tfms=None):
    transforms = tfms if tfms else [to_float_tensor,add_colour_dimension]
    return TransformDataset(raw_data,transforms,x_only=x_only,device=device)

def prune(enc,dec,lin,idx):
    orig_enc_size = enc[1][3][0].weight.shape[0]
    orig_dec_size = dec.main[0].weight.shape[0]
    orig_lin_size0 = lin.weight.shape[0]
    orig_lin_size1 = lin.weight.shape[1]
    assert orig_lin_size0==orig_lin_size1
    try: assert orig_enc_size == orig_dec_size and orig_enc_size == orig_lin_size0
    except: set_trace()
    enc[1][3][0].weight=nn.Parameter(torch.cat([enc[1][3][0].weight[:idx],enc[1][3][0].weight[idx+1:]],0))
    enc[1][3][0].out_channels -= 1
    enc[1][3][2].weight=nn.Parameter(torch.cat([enc[1][3][2].weight[:idx],enc[1][3][2].weight[idx+1:]]))
    enc[1][3][2].bias=nn.Parameter(torch.cat([enc[1][3][2].bias[:idx],enc[1][3][2].bias[idx+1:]]))
    enc[1][3][2].running_mean=torch.cat([enc[1][3][2].running_mean[:idx],enc[1][3][2].running_mean[idx+1:]])
    enc[1][3][2].running_var=torch.cat([enc[1][3][2].running_var[:idx],enc[1][3][2].running_var[idx+1:]])
    enc[1][3][2].num_features-=1
    dec.main[0].weight=nn.Parameter(torch.cat([dec.main[0].weight[:idx],dec.main[0].weight[idx+1:]],0))
    dec.main[0].in_channels -= 1
    lin.weight = nn.Parameter(torch.cat([lin.weight[:,:idx],lin.weight[:,idx+1:]],dim=1))
    lin.weight = nn.Parameter(torch.cat([lin.weight[:idx],lin.weight[idx+1:]]))
    lin.bias = nn.Parameter(torch.cat([lin.bias[:idx],lin.bias[idx+1:]]))
    final_enc_size = enc[1][3][0].weight.shape[0]
    final_dec_size = dec.main[0].weight.shape[0]
    final_lin_size0 = lin.weight.shape[0]
    final_lin_size1 = lin.weight.shape[1]
    try:
        assert final_enc_size == final_dec_size
        assert final_dec_size == orig_dec_size-1
        assert final_lin_size0 == final_lin_size1
        assert final_lin_size0 == orig_dec_size-1
    except:
        set_trace()

def duplicate_dim(enc,dec,lin,idx):
    orig_enc_size = enc[1][3][0].weight.shape[0]
    orig_dec_size = dec.main[0].weight.shape[0]
    orig_lin_size0 = lin.weight.shape[0]
    orig_lin_size1 = lin.weight.shape[1]
    assert orig_lin_size0==orig_lin_size1
    assert orig_enc_size == orig_dec_size
    enc[1][3][0].weight = nn.Parameter(torch.cat([enc[1][3][0].weight,enc[1][3][0].weight[idx:idx+1]+0.01*torch.randn_like(enc[1][3][0].weight[idx:idx+1])]))
    enc[1][3][0].out_channels += 1
    enc[1][3][2].weight = nn.Parameter(torch.cat([enc[1][3][2].weight,enc[1][3][2].weight[idx:idx+1]+0.01*torch.randn_like(enc[1][3][2].weight[idx:idx+1])]))
    enc[1][3][2].bias = nn.Parameter(torch.cat([enc[1][3][2].bias,enc[1][3][2].bias[idx:idx+1]+0.01*torch.randn_like(enc[1][3][2].bias[idx:idx+1])]))
    enc[1][3][2].running_mean=torch.cat([enc[1][3][2].running_mean,enc[1][3][2].running_mean[idx:idx+1:]])
    enc[1][3][2].running_var=torch.cat([enc[1][3][2].running_var,enc[1][3][2].running_var[idx:idx+1:]])
    enc[1][3][2].num_features+=1
    dec.main[0].weight=nn.Parameter(torch.cat([dec.main[0].weight,dec.main[0].weight[idx:idx+1:]],0))
    dec.main[0].in_channels += 1
    lin.weight = nn.Parameter(torch.cat([lin.weight,lin.weight[:,idx:idx+1]],dim=1))
    lin.weight = nn.Parameter(torch.cat([lin.weight,lin.weight[idx:idx+1]]))
    lin.bias = nn.Parameter(torch.cat([lin.bias,lin.bias[idx:idx+1]]))
    final_enc_size = enc[1][3][0].weight.shape[0]
    final_dec_size = dec.main[0].weight.shape[0]
    final_lin_size0 = lin.weight.shape[0]
    final_lin_size1 = lin.weight.shape[1]
    try:
        assert final_enc_size == final_dec_size
        assert final_dec_size == orig_dec_size+1
        assert final_lin_size0 == final_lin_size1
        assert final_lin_size0 == orig_dec_size+1
    except:
        set_trace()

def check_ae_images(enc,dec,dataset):
    idxs = np.random.randint(0,len(dataset),size=5*4)
    inimgs = dataset[idxs]
    x = enc(inimgs)
    outimgs = dec(x)
    _, axes = plt.subplots(5,4,figsize=(7,7))
    for i in range(5):
        axes[i,0].imshow(inimgs[i,0])
        axes[i,1].imshow(outimgs[i,0])
        axes[i,2].imshow(inimgs[i+5,0])
        axes[i,3].imshow(outimgs[i+5,0])
    plt.show()

def vis_latent(dec,latent): plt.imshow(dec(latent[None,:,None,None].cuda())[0,0]); plt.show()

def check_ohe_latents(dec,t):
    for t in range(dec.main[0].weight.shape[0]):
        vis_latent(dec,(torch.arange(dec.main[0].weight.shape[0])==t).float().cuda())

def check_ae_images(enc,dec,dataset,num_rows=5,stacked=False):
    idxs = np.random.randint(0,len(dataset),size=num_rows*4)
    inimgs = dataset[idxs][0]
    x = enc(inimgs)[-1] if stacked else enc(inimgs)
    outimgs = dec(x)[-1] if stacked else dec(x)
    _, axes = plt.subplots(num_rows,4,figsize=(7,7))
    for i in range(num_rows):
        axes[i,0].imshow(inimgs[i,0])
        axes[i,1].imshow(outimgs[i,0])
        axes[i,2].imshow(inimgs[i+num_rows,0])
        axes[i,3].imshow(outimgs[i+num_rows,0])
    plt.show()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def label_assignment_cost(labels1,labels2,label1,label2):
    assert len(labels1) == len(labels2)
    return len([idx for idx in range(len(labels2)) if labels1[idx]==label1 and labels2[idx] != label2])

def translate_labellings(trans_from_labels,trans_to_labels):
    # What you're translating into has to be compressed, otherwise gives wrong results
    unique_from_labs =  set(trans_from_labels) if isinstance(trans_from_labels,np.ndarray) else trans_from_labels.unique()
    unique_to_labs =  set(trans_to_labels) if isinstance(trans_to_labels,np.ndarray) else trans_to_labels.unique()
    num_from_labs =  len([i for i in unique_from_labs if i != -1])
    num_to_labs =  len([i for i in unique_to_labs if i != -1])
    if num_from_labs <= num_to_labs:
        return translate_labellings_fanout(trans_from_labels,trans_to_labels)
    else:
        return translate_labellings_fanin(trans_from_labels,trans_to_labels)

def translate_labellings_fanout(trans_from_labels,trans_to_labels):
    cost_matrix = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in set(trans_to_labels) if l2 != -1] for l1 in set(trans_from_labels) if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert len(col_ind) == len(set(trans_from_labels[trans_from_labels != -1]))
    return np.array([col_ind[l] for l in trans_from_labels])

def translate_labellings_fanin(trans_from_labels,trans_to_labels):
    cost_matrix = np.array([[label_assignment_cost(trans_to_labels,trans_from_labels,l1,l2) for l2 in set(trans_from_labels) if l2 != -1] for l1 in set(trans_to_labels) if l1 != -1])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assert len(col_ind) == get_num_labels(trans_to_labels)
    untranslated = [i for i in range(cost_matrix.shape[1]) if i not in col_ind]
    cost_matrix2 = np.array([[label_assignment_cost(trans_from_labels,trans_to_labels,l1,l2) for l2 in set(trans_to_labels) if l2 != -1] for l1 in set(untranslated) if l1 != -1])
    row_ind2, col_ind2 = linear_sum_assignment(cost_matrix2)
    cl = col_ind.tolist()
    trans_dict = {f:cl.index(f) for f in cl}
    for u,t in zip(untranslated,col_ind2): trans_dict[u]=t
    trans_dict[-1] = -1
    return [trans_dict[i] for i in trans_from_labels]

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
    if pivot is None:
        pivot = labellings_list.pop(0)
        translated_list = [pivot]
    else:
        translated_list = []
    for not_lar in labellings_list:
        not_lar_translated = translate_labellings(not_lar,pivot)
        translated_list.append(not_lar_translated)
    return translated_list

def ask_ensemble(l): return (np.expand_dims(np.arange(l.max()+1),0)==np.expand_dims(l,2)).sum(axis=0)

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

def get_num_labels(labels):
    assert labels.ndim == 1
    return  len(set([l for l in labels if l != -1]))

def dictify_list(x,key):
    assert isinstance(x,list)
    assert len(x) > 0
    assert isinstance(x[0],dict)
    return {item[key]: item for item in x}

def accuracy(labels1,labels2):
    try:
        trans_labels = translate_labellings(labels1,labels2)
        return sum(trans_labels==labels2)/len(labels1)
    except:
        print("Couldn't compute accuracy by translating, returning adjusted_rand_score instead")
        return adjusted_rand_score(labels1,labels2)

def compress_labels(labels):
    if isinstance(labels,torch.Tensor): labels = labels.detach().cpu().numpy()
    x = sorted([i for i in set(labels) if i != -1])
    new_labels = np.array([l if l == -1 else x.index(l) for l in labels])
    return new_labels

def mlp(inp_size,hidden_size,outp_size,device):
    l1 = nn.Linear(inp_size,hidden_size)
    l2 = nn.ReLU()
    l3 = nn.Linear(hidden_size,outp_size)
    return nn.Sequential(l1,l2,l3).to(device)

def num_labs(labels): return len(set([l for l in labels if l != -1]))
def same_num_labs(labels1,labels2): num_labs(labels1) == num_labs(labels2)

if __name__ == "__main__":
    small_labels = np.array([1,2,3,4,0,4,4])
    big_labels = np.array([0,1,2,3,4,5,5])
    print(translate_labellings(big_labels,small_labels))
    print(translate_labellings(small_labels,big_labels))
    print(translate_labellings(big_labels,big_labels))
