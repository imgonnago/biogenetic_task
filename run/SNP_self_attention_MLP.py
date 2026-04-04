from model.SNP_Model import SNPClassifier
from model.dataloader import get_dataloader, SNPDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def SNP_main():
    # 모델 초기화
    model = SNPClassifier()
    # 모델 요약 출력
    print(model)
    # 데이터 로더 생성
    train_loader = get_dataloader(SNPDataset('./biogenetic', split='train'), batch_size=16, split='train')
    val_loader = get_dataloader(SNPDataset('./biogenetic', split='val'), batch_size=16, split='val')
    test_loader = get_dataloader(SNPDataset('./biogenetic', split='test'), batch_size=16, split='test')

    