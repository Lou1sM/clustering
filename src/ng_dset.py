import os
import torch
from torch.utils import data
import numpy as np


class NGDataset(data.Dataset):
    def __init__(self, hiddens_dir, device):
        self.hiddens_dir = hiddens_dir
        self.device = device
        #self.hiddens = [torch.tensor(h,device=device) for h in hiddens]

    def __getitem__(self, index):
        fpath = os.path.join(self.hiddens_dir,f"ng{index}.pt")
        tensor_dict = torch.load(fpath,map_location=self.device)
        hidden = tensor_dict['embeds']
        label = tensor_dict['label']
        return hidden, label, index

    def __len__(self):
        return len(os.listdir(self.hiddens_dir))

    def close(self):
        self.archive.close()
