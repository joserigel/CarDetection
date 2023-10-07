import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

from dotenv import load_dotenv
import os
load_dotenv()

class Model(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(16, 4, )
    def forward(x):
        return x

    