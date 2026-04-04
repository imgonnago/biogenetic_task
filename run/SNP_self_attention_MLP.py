from model.SNP_Model import SNPClassifier
from model.dataloader import get_dataloader, SNPDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def SNP_main():
    # 모델 초기화
    print("Initializing model...")
    model = SNPClassifier()
    print("Model initialized successfully")
    # 모델 요약 출력
    print("Model architecture:")
    print(model)
    print("-" * 50)
    # 데이터 로더 생성
    print("Creating DataLoader...")
    train_x, train_y = SNPDataset(data_dir='./biogenetic', split='train')
    test_x, test_y = SNPDataset(data_dir='./biogenetic', split='test')
    val_x, val_y = SNPDataset(data_dir='./biogenetic', split='val')

    train_x = get_dataloader(train_x, batch_size=16, split='train')
    test_x = get_dataloader(test_x, batch_size=16, split='test')
    val_x = get_dataloader(val_x, batch_size=16, split='val')

    train_y = get_dataloader(train_y, batch_size=16, split='train')
    test_y = get_dataloader(test_y, batch_size=16, split='test')
    val_y = get_dataloader(val_y, batch_size=16, split='val')
    print("DataLoader created successfully")
    