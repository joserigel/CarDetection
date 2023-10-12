import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    path = "./bdd100k/images/10k/train/"
    images = os.listdir(path)
    tile_size = 256

    model = Model().cuda()
    model.load_state_dict(torch.load('./models/cnn.pth'))
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.002)
    criterion = nn.MSELoss()

    running_loss = 0
    invalid = 0

    for i in  tqdm(range(len(images))):
        file = path + images[i]
        try:
            datas = await instance.getImgAndBox(file, tile_size=tile_size)
            
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
                    gt = torch.tensor([[*box, 0, tile_size]], dtype=torch.float) / tile_size
                loss = criterion(output.cuda(), gt.cuda())
                loss.backward()

                running_loss += loss.item()

                optimizer.step()
                
        except Exception as e:
            # print(e)
            invalid += 1
        if (i % 100 == 0):
            print(running_loss / (i + 1))
    torch.save(model.state_dict(), './models/cnn.pth')
    print("Total Running Loss:", running_loss)
    print("Total invalid", invalid)

async def eval():
    instance = await BoundingBox.create(test=True)
    path = "./bdd100k/images/10k/val/"
    images = os.listdir(path)
    tile_size = 256

    model = Model().cuda()
    model.load_state_dict(torch.load('./models/cnn.pth'))
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.002)
    # criterion = nn.MSELoss()

    total = 0
    correct = 0
    invalid = 0


    for i in  tqdm(range(len(images))):
        file = path + images[i]
        try:
            datas = await instance.getImgAndBox(file, tile_size=tile_size)
            
            for img, box in datas:
                # Note: 256 not the same as tile_size, this is just based on the 8bit pixel color
                plt.clf()
                plt.imshow(img)
                img = torch.tensor(img) / 256
                img = img.cuda()
                output = model(img)
                output = output[0]
                if (box == (0, 0, 0, 0)):
                    if output[4] > output[5]:
                        correct += 1
                else :
                    if output[4] < output[5]:
                        correct += 1
                        x, y = box_to_polygon(output.cpu().detach().numpy() * tile_size)
                        plt.plot(x, y)
                        # plt.show()
                        # plt.pause(0.5)
                total += 1
        except Exception as e:
            invalid += 1
    print("Accuracy:", f'{(correct/total)*100:.2f}%')
    print("invalid", invalid)



async def train_draft():
    instance = await BoundingBox.create()
    tile_size = 256
    datas = await instance.getImgAndBox("DataSandbox/train/4a396379-37154446.jpg", tile_size)
    
def sandbox():
    path = "./bdd100k/images/10k/train/"
    images = os.listdir(path)
    print(len(images))
    

# asyncio.run(train())
asyncio.run(eval())
# sandbox()