import torch
import torchvision
import torch.functional as F
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
    def forward(self, x):
        return x