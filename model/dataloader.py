import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
import os

class MyDataset(Dataset):
    #필요한 데이터셋 클래스 정의. 데이터셋 클래스는 무조건 pytorch dataset 클래스를 상속받고
    #__init__, __len__, __getitem__ 메소드가 있어야함.
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.gm = np.load(f'{data_dir}/X_GM.npy')
        self.SNP = np.load(f'{data_dir}/X_SNP 1.npy')
        self.labels = np.load(f'{data_dir}/Y_dis.npy')
    #데이터셋 길이 반환하는 메소드. 데이터셋의 길이는 레이블의 길이와 같음.
    def __len__(self):
        return len(self.labels)
    #데이터셋에서 특정 인덱스의 데이터를 반환하는 메소드.
    def __getitem__(self, idx):
        gm = torch.tensor(self.gm[idx], dtype=torch.float32)
        SNP = torch.tensor(self.SNP[idx], dtype=torch.float32)
        label = torch.tensor(np.argmax(self.labels[idx]), dtype=torch.long)
        return gm, SNP, label

#데이터 로더를 반환하는 함수. 데이터셋 객체 생성하고 반환.
def get_dataloader(data_dir, batch_size=16, split='train'):
    dataset = MyDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader
    
data = get_dataloader(data_dir='./biogenetic', batch_size=16, split='train')
print("DataLoader created successfully")
print(f"Number of data: {len(data.dataset)}")
print(f'GM data shape: {data.dataset.gm.shape}')
print(f'SNP data shape: {data.dataset.SNP.shape}')
print(f'Labels shape: {data.dataset.labels.shape}')
print("DataLoader content printed successfully")

