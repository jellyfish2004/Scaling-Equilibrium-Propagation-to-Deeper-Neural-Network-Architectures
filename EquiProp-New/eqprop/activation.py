import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_sigmoid(x):
    return torch.clamp(x, 0., 1.)

def identity(x):
    return x

class LearnableClampedReLU(nn.Module):
    def __init__(self, init_threshold=6.0):
        super().__init__()
        self.t = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=self.t)