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
# 모델 다운로드 없이 config만 가져오기
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# LLM hidden_size (우리가 projection해야 하는 차원)
print(f"LLM hidden_size: {config.text_config.hidden_size}")      # 3584 ✅

# ViT hidden_size (vision encoder 차원)
print(f"ViT hidden_size: {config.vision_config.hidden_size}")    # 1280

# ViT → LLM projection 후 차원 (out_hidden_size)
print(f"Projected hidden_size: {config.vision_config.out_hidden_size}")
