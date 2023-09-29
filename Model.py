import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(16, 16)
    def forward(x):
        return x
