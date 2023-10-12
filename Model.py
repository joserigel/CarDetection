import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

from DataVisualizer import BoundingBox

from dotenv import load_dotenv
import os
load_dotenv()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 2, 2, padding=0)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*128, 128*128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128*128, 64*64),
            nn.Dropout(),
            nn.Linear(64*64, 6),
            nn.ReLU()
        )
    def forward(self, x):
        x = torch.transpose(x, 0, 2)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.layer1(x)
        return x

def box_to_polygon(box):
    return ([box[0], box[0], box[1], box[1], box[0]],
                [box[2], box[3], box[3], box[2], box[2]])

async def train():
    instance = await BoundingBox.create()
    tile_size = 256
    datas = await instance.getImgAndBox("DataSandbox/train/4a396379-37154446.jpg", tile_size)
    
    model = Model().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.MSELoss()

    running_loss = 0
    for img, box in datas:
        optimizer.zero_grad()
        # Note: 256 not the same as tile_size, this is just based on the 8bit pixel color
        img = torch.tensor(img) / 256
        img = img.cuda()
        output = model(img)
        
        gt = None
        if (box == (0, 0, 0, 0)):
            result = output[0]
            gt = torch.tensor([[*result[:4], 1, 0]])
        else :
            gt = torch.tensor([[*box, 0, 1]], dtype=torch.float) / tile_size
        # print(output.shape == gt.shape)
        loss = criterion(output.cuda(), gt.cuda())
        loss.backward()

        running_loss += loss.item()

        optimizer.step()

        print(loss.item())

asyncio.run(train())