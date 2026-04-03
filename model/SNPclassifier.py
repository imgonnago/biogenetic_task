import torch
import torch.nn as nn
import numpy as np
import torch.nn.funcional as F
from typing import Optional

class SNPClassifier(nn.Module):
    def __init__(input_dim = 2098):
