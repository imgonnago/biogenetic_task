import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class SNPclassifier(nn.Module):
    def __init__(self, input_dim = 2098, output_dim = 4):
        super().__init__()

        self.SNPclassifier = nn.Sequential(

        )