import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from typing import Optional

class GMProjection(nn.Module):
    def __init__(self, input_dim = 93, output_dim = 2048):
        super().__init__()

        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim // 2, output_dim),
            nn.LayerNorm(output_dim),   
        )

    def forward(self, x):
        x = self.projection(x)
        return x    
    
    