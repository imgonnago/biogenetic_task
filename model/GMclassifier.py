import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from typing import Optional

class GMClassifier(nn.Module):
    def __init__(self, input_dim = 93, output_dim = 4):
        super().__init__()

        self.GMclassifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256,128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256,output_dim),
            nn.Softmax(output_dim)   
        )

    def forward(self, x):
        x = self.GMclassifier(x)
        return x    
    
    