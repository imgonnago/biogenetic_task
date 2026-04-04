import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
import os
    
class SNPDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.SNP = np.load(f'{data_dir}/X_SNP 1.npy')
        self.labels = np.load(f'{data_dir}/Y_dis.npy')

        num_samples = len(self.labels)
        idx = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(idx)
        train_end = int(0.7 * num_samples)
        val_end = train_end + int(0.15 * num_samples)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]

        if split == 'train':
            idx = train_idx
        elif split == 'val':
            idx = val_idx   
        else:
            idx = test_idx

        self.SNP = self.SNP[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        SNP = torch.tensor(self.SNP[idx], dtype=torch.float32)
        label = torch.tensor(np.argmax(self.labels[idx]), dtype=torch.long)
        return SNP, label
    
class GMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.gm = np.load(f'{data_dir}/X_GM.npy')
        self.labels = np.load(f'{data_dir}/Y_dis.npy')

        num_samples = len(self.labels)
        idx = np.arange(num_samples)
        np.random.seed(42)
        np.random.shuffle(idx)

        train_end = int(0.7 * num_samples)
        val_end = train_end + int(0.15 * num_samples)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]
        test_idx = idx[val_end:]

        if split == 'train':
            idx = train_idx
        elif split == 'val':
            idx = val_idx   
        else:
            idx = test_idx

        self.gm = self.gm[idx]
        self.labels = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gm = torch.tensor(self.gm[idx], dtype=torch.float32)
        label = torch.tensor(np.argmax(self.labels[idx]), dtype=torch.long)
        return gm, label
    
#데이터 로더를 반환하는 함수. 데이터셋 객체 생성하고 반환.
def get_dataloader(dataset, batch_size=16, split='train'):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=(split=='train')
        )
    return dataloader


data = get_dataloader(data_dir='./biogenetic', batch_size=16, split='train')
print("DataLoader created successfully")
print(f"Number of data: {len(data.dataset)}")
print(f'GM data shape: {data.dataset.gm.shape}')
print(f'SNP data shape: {data.dataset.SNP.shape}')
print(f'Labels shape: {data.dataset.labels.shape}')
print("DataLoader content printed successfully")

