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

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 2, 2, padding=0)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128*128, 128*128),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(128*128, 64*64),
            nn.Dropout(),
            nn.Linear(64*64, 4)
        )
    def forward(self, x):
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
    
    # Variables
    tile_size = 256
    min_size = 0
    max_size = tile_size

    model = Segmentation().cuda()
    model.load_state_dict(torch.load('./models/segmentation.pth'))
    optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.00001)
    criterion = nn.MSELoss()

    running_loss = 0
    invalid = 0
    tested = 0

    for i in  tqdm(range(len(images))):
        file = path + images[i]
        try:
            datas = await instance.getImgAndBox(file, tile_size=tile_size)
            
            valid_imgs = []
            valid_boxes = []
            for img, box in datas:
                if box != (0, 0, 0, 0) and (box[1] - box[0]) >= min_size and (box[3] - box[2] >= min_size) and \
                    box[1] - box[0] <= max_size and box[3] - box[2] <= max_size:
                    valid_imgs.append(img)
                    valid_boxes.append(box)

            if (len(valid_imgs) > 0):
                valid_imgs = torch.tensor(np.array(valid_imgs), dtype=torch.float).cuda() / 256
                valid_imgs = torch.transpose(valid_imgs, 1, 3)

                valid_boxes = torch.tensor(np.array(valid_boxes), dtype=torch.float).cuda() * (2 / tile_size)
                valid_boxes = valid_boxes - 1
                
                optimizer.zero_grad()
                output = model(valid_imgs)
                loss = criterion(output, valid_boxes)
                loss.backward()

                running_loss += loss.item()
                
                optimizer.step()
                tested += len(valid_imgs)

        except Exception as e:
            # if str(e) != "Invalid Polygon":
            #     print(e)
            invalid += 1
        if (i % 1000 == 0):
            print(running_loss / (i + 1))
    torch.save(model.state_dict(), './models/segmentation.pth')
    print("Total Running Loss:", running_loss)
    print("Total invalid", invalid)
    print("Tested:", tested)

async def eval():
    instance = await BoundingBox.create(test=True)
    path = "./bdd100k/images/10k/val/"
    images = os.listdir(path)
    
    # Variables
    tile_size = 256
    min_size = 50
    max_size = 200

    model = Segmentation().cuda()
    model.load_state_dict(torch.load('./models/segmentation.pth'))

    total = 0
    correct = 0
    invalid = 0
    tested = 0

    for i in  tqdm(range(len(images))):
        file = path + images[i]
        try:
            datas = await instance.getImgAndBox(file, tile_size=tile_size)
            
            for img, box in datas:
                # Note: 256 not the same as tile_size, this is just based on the 8bit pixel color
                if box != (0, 0, 0, 0) and box[1] - box[0] >= min_size and box[3] - box[2] >= min_size \
                    and box[1] - box[0] <= max_size and box[3] - box[2] <= max_size:
                    plt.clf()
                    plt.imshow(img)
                    img = torch.tensor(img) / 256
                    img = img.cuda()
                    img = torch.transpose(img, 0, 2)
                    output = model(img)
                    # print(output)
                    output = (output[0] + 1) * (tile_size / 2)

                    # Calculate Accuracy
                    for i in range(4):
                        correct += 1 - abs(box[i] - output[i])
                    total += 4
                    
                    tested += 1
                    
                    x, y = box_to_polygon(output.cpu().detach().numpy())
                    gtX, gtY = box_to_polygon(box)
                    plt.plot(x, y, 'r')
                    plt.plot(gtX, gtY, 'b')
                    plt.show()
        except Exception as e:
            # if e != Exception("Invalid Polygon"):
            #     print(e)
            invalid += 1
    print("Accuracy:", f'{(correct/total)*100:.2f}%')
    print("Invalid:", invalid)
    print("Tested:", tested)


def sandbox():
    path = "./bdd100k/images/10k/train/"
    images = os.listdir(path)
    print(len(images))
    
for i in range(0):
    print(f"EPOCH:{i+1}====================")
    asyncio.run(train())
asyncio.run(eval())
# sandbox()