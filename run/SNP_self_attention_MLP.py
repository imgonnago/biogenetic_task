from model.SNP_Model import SNPClassifier
from model.dataloader import get_dataloader, SNPDataset
from run.validation import validate
from run.test import test
from run.train import train
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
    SNP_train = SNPDataset(data_dir='./biogenetic', split='train')
    SNP_test = SNPDataset(data_dir='./biogenetic', split='test')
    SNP_val = SNPDataset(data_dir='./biogenetic', split='val')

    train_loader = get_dataloader(SNP_train, batch_size=16, split='train')
    test_loader = get_dataloader(SNP_test, batch_size=16, split='test')
    val_loader = get_dataloader(SNP_val, batch_size=16, split='val')

    print("DataLoader created successfully")
    
    train(model, train_loader, optimizer=torch.optim.Adam(model.parameters(), lr=0.001), criterion=nn.CrossEntropyLoss(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_epochs=10)
    test(model, test_loader)
    validate(model, val_loader)

if __name__ == "__main__":
    SNP_main()