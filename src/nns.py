from fastai import layers
import torch
import torch.nn as nn


def mlp(inp_size,hidden_size,outp_size,device):
    l1 = nn.Linear(inp_size,hidden_size)
    l2 = nn.ReLU()
    l3 = nn.Linear(hidden_size,outp_size)
    return nn.Sequential(l1,l2,l3).to(device)

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

class Generator(nn.Module):
    def __init__(self, nz,ngf,nc,dropout_p=0.):
        super(Generator, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            self.dropout,
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
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
            nn.ConvTranspose2d(nz, 256, 4, 1, 1, bias=False).to(device),
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
                nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
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
                    nn.ConvTranspose2d(32, nc, 4, 2, 1, bias=True).to(device),
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


