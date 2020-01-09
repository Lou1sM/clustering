from pdb import set_trace
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.datasets as tdatasets
from fastai import datasets,layers
import umap.umap_ as umap
from sklearn.datasets import fetch_mldata
import seaborn as sns
from torch.utils import data


def reload():
    import importlib, utils
    importlib.reload(utils)


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

    def forward(self, input):
        return self.main(input)

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
    elif isinstance(x,torch.tensor): return x.numpy()

def scatter_clusters(embeddings,labels):
    palette = ['r','k','y','g','b','m','purple','brown','c','orange']
    labels =  numpyify([0]*len(embeddings)) if labels is None else numpyify(labels)
    for i,label in enumerate(list(set(labels))):
        plt.scatter(embeddings[labels==label,0], embeddings[labels==label,1], s=0.2, c=palette[i], label=i)
    plt.legend()
    plt.show()
    return plt

def print_tensors(*tensors):
    print(*[t.item() for t in tensors])

def get_reslike_block(nfs,sz):
    return nn.Sequential(
        *[layers.conv_layer(nfs[i],nfs[i+1],stride=2 if i==0 else 1,leaky=0.3,padding=1)
         for i in range(len(nfs)-1)], nn.AdaptiveMaxPool2d(sz))

def get_enc_dec(device, latent_size):
    block1 = get_reslike_block([1,4,8,16,32],sz=7)
    block2 = get_reslike_block([32,64,128,256,latent_size],sz=1)
    lin = nn.Linear(latent_size,latent_size)
    enc = nn.Sequential(block1,block2)
    dec = Generator(nz=latent_size,ngf=32,nc=1,dropout_p=0.)
    return enc.to(device),dec.to(device),lin.to(device)

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
    mnist_data=tdatasets.MNIST(root='~/unsupervised_object_learning/data/',train=True,download=True)
    data = mnist_data.train_data if x_only else mnist_data
    mnist_dataset = TransformDataset(data,[to_float_tensor,add_colour_dimension],x_only,device=device)
    return mnist_dataset

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

