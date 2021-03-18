import torchvision.datasets as tdatasets
import matplotlib.pyplot as plt
from torch.utils import data
import kornia
import torch
import os
import numpy as np
from .utils import compose, to_float_tensor, add_colour_dimension

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
        for data_name, d in kwdata.items():
            setattr(self,data_name,d.to(device))
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


def get_mnist_dloader(x_only=False,device='cuda',batch_size=64,shuffle=True):
    mnist_data=tdatasets.MNIST(root='~/unsupervised_object_learning/data/',train=True,download=True)
    data = mnist_data.data if x_only else mnist_data.data, mnist_data.targets
    mnist_dl = get_dloader(raw_data=data,x_only=x_only,batch_size=batch_size,device=device,shuffle=shuffle)
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
    elif dset_name == 'MNISTtrain':
        dtrain=tdatasets.MNIST(root=dirpath,train=True,download=True)
        data = dtrain.data if x_only else (dtrain.data,dtrain.targets)
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

def get_dloader(raw_data,x_only,batch_size,device,shuffle):
    ds = get_dset(raw_data,x_only,device=device)
    if shuffle: s=data.BatchSampler(data.RandomSampler(ds),batch_size,drop_last=True)
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
    inimgs, outimgs = inimgs.detach().cpu(), outimgs.detach().cpu()
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
