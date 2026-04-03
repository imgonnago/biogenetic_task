import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
import os

class MyDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.gm = np.load(f'{data_dir}/X_GM.npy')
        self.SNP = np.load(f'{data_dir}/X_SNP 1.npy')
        self.labels = np.load(f'{data_dir}/Y_dis.npy')
        self.
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        gm = torch.tensor(self.gm[idx], dtype=torch.float32)
        SNP = torch.tensor(self.SNP[idx], dtype=torch.float32)
        label = torch.tensor(np.argmax(self.labels[idx]), dtype=torch.long)
        return gm, SNP, label


def get_dataloader(data_dir, batch_size=16, split='train'):
    dataset = MyDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader
    
data = get_dataloader(data_dir='./biogenetic', batch_size=16, split='train')
print("DataLoader created successfully")
print(f"Number of data: {len(data.dataset)}")
print(f' GM data shape: {data.dataset.gm.shape}')
print(f'SNP data shape: {data.dataset.SNP.shape}')
print(f'Labels shape: {data.dataset.labels.shape}')
print("DataLoader content printed successfully")

