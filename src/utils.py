from datetime import datetime
from sklearn.metrics import normalized_mutual_info_score as mi_func
from fastai import layers
from pdb import set_trace
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from torch.utils import data
import gc
import kornia
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision.datasets as tdatasets
import torch.multiprocessing as mp
import umap.umap_ as umap


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
    def __init__(self, nz, ngf, nc, output_size, device, dropout_p=0.):
        super(GeneratorStacked, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.output_size = output_size
        self.device = device
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d( nz, 256, 4, 1, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 8).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 4).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False).to(device),
            nn.BatchNorm2d(ngf * 2).to(device),
            nn.LeakyReLU(0.3),
            self.dropout)
        if output_size == 16:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=True).to(device),
                #nn.BatchNorm2d(32).to(device),
                #nn.LeakyReLU(0.3),
                #self.dropout)
                nn.Tanh())
        elif output_size == 28:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 2, bias=False).to(device),
                nn.BatchNorm2d(32).to(device),
                nn.LeakyReLU(0.3),
                self.dropout)
            self.block5 = nn.Sequential(
                nn.ConvTranspose2d( 32, nc, 4, 2, 1, bias=True).to(device),
                nn.Tanh()
            )
        elif output_size in [32,128]:
            self.block4 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False).to(device),
                nn.BatchNorm2d(32).to(device),
                nn.LeakyReLU(0.3),
                self.dropout)
            if output_size == 32:
                self.block5 = nn.Sequential(
                    nn.ConvTranspose2d( 32, nc, 4, 2, 1, bias=True).to(device),
                    nn.Tanh()
                )
            elif output_size == 128:
                self.block5 = nn.Sequential(
                    nn.ConvTranspose2d(32, 32, 4, 4, 0, bias=False).to(device),
                    nn.BatchNorm2d(32).to(device),
                    nn.LeakyReLU(0.3),
                    self.dropout)
                self.block6 = nn.Sequential(
                    nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
                    nn.Tanh()
                )



    def forward(self,inp):
        out = self.block1(inp)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        if self.output_size in [28,32,128]:
            out = self.block5(out)
        if self.output_size == 128:
            out = self.block6(out)
        return out

class EncoderStacked(nn.Module):
    def __init__(self,block1,block2):
        super(EncoderStacked, self).__init__()
        self.block1,self.block2 = block1,block2

    def forward(self,inp):
        out1 = self.block1(inp)
        out2 = self.block2(out1)
        return out2

    @property
    def device(self):
        return self.enc.block1[0][0].weight.device

class Dataholder():
    def __init__(self,train_data,train_labels): self.train_data,self.train_labels=train_data,train_labels

class TransformDataset_old(data.Dataset):
    def __init__(self,data,transforms,x_only,device):
        self.transform,self.x_only,self.device=compose(transforms),x_only,device
        if x_only:
            self.x = data.to(self.device)
        else:
            x, y = data
            self.x,self.y = x.to(self.device), y.to(self.device)
    def __len__(self): return len(self.data) if self.x_only else len(self.x)
    def __getitem__(self,idx):
        if self.x_only: return self.transform(self.x[idx]), idx
        else: return self.transform(self.x[idx]), self.y[idx], idx

class TransformDataset(data.Dataset):
    def __init__(self,data,transforms,x_only,device,augment=False):
        self.x_only,self.device,self.augment=x_only,device,augment
        self.aug = kornia.augmentation.RandomAffine(degrees=(-10,10),translate=(0.1,0.1),return_transform=True)
        if x_only:
            self.x = data
            for transform in transforms:
                self.x = transform(self.x)
            self.x = self.x.to(self.device)
        else:
            self.x, self.y = data
            for transform in transforms:
                self.x = transform(self.x)
            self.x.to(self.device)
            self.x, self.y = self.x.to(self.device),self.y.to(self.device)
    def __len__(self): return len(self.data) if self.x_only else len(self.x)
    def __getitem__(self,idx):
        batch_x = self.x[idx]
        if self.augment:
            batch_x = self.aug(batch_x)[0]
            if isinstance(idx,int):
                batch_x = batch_x.squeeze(1)
        if self.x_only: return batch_x, idx
        else: return batch_x, self.y[idx], idx

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

def augment_batch(batch_tensor):
    orig_size = batch_tensor.ndim
    batch_size = batch_tensor.shape[0]
    if orig_size == 3:
        batch_tensor=batch_tensor.unsqueeze(0)
    angle = (torch.rand(batch_size)*20)-10
    center = torch.ones(batch_size,2)
    center[..., 0] = batch_tensor.shape[3] / 2
    center[..., 1] = batch_tensor.shape[2] / 2
    scale = torch.ones(batch_size)
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    _, _, h, w = batch_tensor.shape
    batch_tensor_warped = kornia.warp_affine(batch_tensor, M, dsize=(h, w))
    if orig_size == 3:
        batch_tensor_warped=batch_tensor_warped.squeeze(0)
    return batch_tensor_warped


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

def make_ae(aeid,device,NZ,image_size,num_channels):
    enc_b1, enc_b2 = get_enc_blocks(device,NZ,num_channels)
    enc = EncoderStacked(enc_b1,enc_b2)
    dec = GeneratorStacked(nz=NZ,ngf=32,nc=num_channels,output_size=image_size,device=device,dropout_p=0.)
    return AE(enc,dec,aeid)

def get_enc_blocks(device, latent_size, num_channels):
    block1 = get_reslike_block([num_channels,4,8,16,32,64],sz=8)
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

def load_coil100(x_only, data_dir='../coil-100'):
    images = np.stack([plt.imread(os.path.join(data_dir,x)) for x in os.listdir(data_dir) if x.startswith('obj')])
    labels = [int(x.split('_')[0][3:]) for x in os.listdir(data_dir) if x.startswith('obj')]
    if x_only:
        return torch.tensor(images)
    else:
        return torch.tensor(images), torch.tensor(labels)

def load_letterAJ(x_only, data_dir='../letterAJ'):
    images = np.load(os.path.join(data_dir,'data','ajimages.npy'))
    labels = np.load(os.path.join(data_dir,'labels','gt_labels.npy'))
    if x_only:
        return torch.tensor(images)
    else:
        return torch.tensor(images), torch.tensor(labels)

def get_vision_dset(dset_name,device,x_only=False):
    dirpath = f'~/unsupervised_object_learning/{dset_name}/data'
    if dset_name in ['MNISTfull', 'MNISTtest']:
        dtest=tdatasets.MNIST(root=dirpath,train=False,download=True)
        x, y = dtest.data, dtest.targets
        if dset_name == 'MNISTfull':
            dtrain=tdatasets.MNIST(root=dirpath,train=True,download=True)
            x = torch.cat([dtrain.data,x])
            y = torch.cat([dtrain.targets,y])
        data = x if x_only else (x,y)
    elif dset_name == 'FashionMNIST':
        dtrain=tdatasets.FashionMNIST(root=dirpath,train=True,download=True)
        dtest=tdatasets.FashionMNIST(root=dirpath,train=False,download=True)
        x = torch.cat([dtrain.data,dtest.data])
        y = torch.cat([dtrain.targets,dtest.targets])
        data = x if x_only else (x,y)
    elif dset_name == 'USPS':
        dtrain=tdatasets.USPS(root=dirpath,train=True,download=True)
        dtest=tdatasets.USPS(root=dirpath,train=False,download=True)
        train_data = torch.tensor(dtrain.data,device=device)
        test_data = torch.tensor(dtest.data,device=device)
        train_targets = torch.tensor(dtrain.targets,device=device)
        test_targets = torch.tensor(dtest.targets,device=device)
        x = torch.cat([train_data,test_data])
        y = torch.cat([train_targets,test_targets])
        data = x if x_only else (x,y)
        #data = torch.tensor(d.data,device=device) if x_only else (torch.tensor(d.data,device=device), torch.tensor(d.targets,device=device))
    elif dset_name == 'CIFAR10':
        d=tdatasets.CIFAR10(root=dirpath,train=True,download=True)
        data = torch.tensor(d.data,device=device) if x_only else (torch.tensor(d.data,device=device), torch.tensor(d.targets,device=device))
    elif dset_name == 'coil-100':
        data = load_coil100(x_only)
    elif dset_name == 'letterAJ':
        data = load_letterAJ(x_only)
        return TransformDataset(data,[add_colour_dimension],x_only,device=device)
    return TransformDataset(data,[to_float_tensor,add_colour_dimension],x_only,device=device)

def get_dloader(raw_data,x_only,batch_size,device,random=True):
    ds = get_dset(raw_data,x_only,device=device)
    if random: s=data.BatchSampler(data.RandomSampler(ds),batch_size,drop_last=True)
    else: s=data.BatchSampler(data.SequentialSampler(ds),batch_size,drop_last=True)
    return data.DataLoader(ds,batch_sampler=s,pin_memory=False)

def get_dset(raw_data,x_only,device,tfms=None):
    transforms = tfms if tfms else [to_float_tensor,add_colour_dimension]
    return TransformDataset(raw_data,transforms,x_only=x_only,device=device)

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
    inimgs, outimgs = inimgs.detach().cpu(), outimgs.detach().cpu()
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
    assert len(labels1) == len(labels2), f"len labels1 {len(labels1)} must equal len labels2 {len(labels2)}"
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
    return np.array([-1 if l == -1 else col_ind[l] for l in trans_from_labels])

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
    if pivot is 'none':
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
    if probs is not 'none': hits = np.expand_dims(probs,2)*hits
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
        trans_labels = translate_labellings(compress_labels(labels1),compress_labels(labels2))
        return sum(trans_labels==labels2)/len(labels1)
    except Exception as e:
        print(e)
        set_trace()
        trans_labels = translate_labellings(labels1,labels2)
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
        for i in range(math.ceil(len(input_list)/split)):
            with ctx.Pool(processes=split) as pool:
                new_list = pool.map(func, input_list[split*i:split*(i+1)])
            print(f'finished {i}th split section')
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
