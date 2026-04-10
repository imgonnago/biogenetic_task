import sys
from pathlib import Path

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.FusionModel import FusionModel
from model.dataloader import get_dataloader, FusionDataset
from run.validation import Fusion_validate
from run.test import Fusion_test
from run.train import Fusion_train
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def Fusion_main():
    print("Initializing model...")
    model = FusionModel()
    print("Model initialized successfully")
    print("Model architecture:")
    print(model)
    print("-" * 50)
    print("Creating DataLoader...")
    fusion_train = FusionDataset(data_dir='./biogenetic', split='train')
    fusion_test = FusionDataset(data_dir='./biogenetic', split='test')
    fusion_val = FusionDataset(data_dir='./biogenetic', split='val')

    train_loader = get_dataloader(fusion_train, batch_size=16, split='train')
    test_loader = get_dataloader(fusion_test, batch_size=16, split='test')
    val_loader = get_dataloader(fusion_val, batch_size=16, split='val')

    print("DataLoader created successfully")
    
    
    Fusion_train(model, train_loader, optimizer=torch.optim.Adam(model.parameters(), lr=5e-5), criterion=nn.CrossEntropyLoss(), num_epochs=100)
    Fusion_test(model, test_loader)
    Fusion_validate(model, val_loader)

if __name__ == "__main__":
    Fusion_main()