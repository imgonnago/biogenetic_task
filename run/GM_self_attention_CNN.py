import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.GM_Model import GM_CNN
from model.dataloader import get_dataloader, GMDataset
from run.validation import validate
from run.test import test
from run.train import train
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def GM_main():
    # 모델 초기화
    print("Initializing model...")
    model = GM_CNN()
    print("Model initialized successfully")
    # 모델 요약 출력
    print("Model architecture:")
    print(model)
    print("-" * 50)
    # 데이터 로더 생성
    print("Creating DataLoader...")
    GM_train = GMDataset(data_dir='./biogenetic', split='train')
    GM_test = GMDataset(data_dir='./biogenetic', split='test')
    GM_val = GMDataset(data_dir='./biogenetic', split='val')

    train_loader = get_dataloader(GM_train, batch_size=16, split='train')
    test_loader = get_dataloader(GM_test, batch_size=16, split='test')
    val_loader = get_dataloader(GM_val, batch_size=16, split='val')

    print("DataLoader created successfully")
    
    train(model, train_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(), num_epochs=10)
    test(model, test_loader)
    validate(model, val_loader)

if __name__ == "__main__":
    GM_main()