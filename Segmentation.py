import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

from DataVisualizer import BoundingBox

from dotenv import load_dotenv
import os
load_dotenv()

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("====", "USING CUDA GPU" if train_device == 'cuda' else "USING CPU (SLOW)", "====")

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=(4, 4), stride=(4, 4), padding=0)
        self.layer_r = nn.Sequential(
            nn.Linear(64*64, 128*128),
            nn.Linear(128*128, 64*64),
        )
        self.layer_g = nn.Sequential(
            nn.Linear(64*64, 128*128),
            nn.Linear(128*128, 64*64),
        )
        self.layer_b = nn.Sequential(
            nn.Linear(64*64, 128*128),
            nn.Linear(128*128, 64*64),
        )
        self.combine = nn.Sequential(
            nn.Linear(3*64*64, 32*32),
            nn.Linear(32*32, 4)
        )
    def forward(self, x):
        x = self.conv1(x)
        
        r = x[:,0]
        g = x[:,1]
        b = x[:,2]

        r = torch.flatten(r, 1)
        r = self.layer_r(r)

        g = torch.flatten(g, 1)
        g = self.layer_b(g)

        b = torch.flatten(b, 1)
        b = self.layer_b(b)

        x = torch.cat((r, g, b), 1)
        x = self.combine(x)
        return x

def box_to_polygon(box):
    return ([box[0], box[0], box[1], box[1], box[0]],
                [box[2], box[3], box[3], box[2], box[2]])


async def train(epoch, load_last = True):
    instance = await BoundingBox.create()
    path = "./bdd100k/images/10k/train/"
    images = os.listdir(path)
    random.shuffle(images)
    
    # Variables
    tile_size = 256

    model = Segmentation().to(device=train_device)
    checkpoint_lists = list(filter(lambda x: "segmentation" in x, os.listdir("./models")))
    last_checkpoint = 0
    if len(checkpoint_lists) > 0 :
        last_checkpoint = int(sorted(checkpoint_lists)[-1].split(".")[0].split("_")[-1])
        print("LAST_CHECKPOINT: ", f"segmentation_{last_checkpoint}")
        if load_last:
            model.load_state_dict(torch.load(f'./models/segmentation_{last_checkpoint}.pth'))
        else:
            print("Checkpoint not loaded")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0001)
    criterion = nn.L1Loss()

    learning_rate = 0.01
    momentum = 0.003
    decay = 1

    running_loss = 0
    invalid = 0
    tested = 0
    for i in range(epoch):
        print(f"EPOCH {i+1} ============", f"Learning Rate: {learning_rate}", f"Momentum: {momentum}")
        optimizer = optim.SGD(model.parameters(), lr=learning_rate * (decay ** i), momentum=momentum * (decay ** i))
        for i in  tqdm(range(len(images) - 4000)):
            file = path + images[i]
            try:
                datas = await instance.getImgAndBox(file, tile_size=tile_size)
                
                valid_imgs = []
                valid_boxes = []
                for img, box in datas:
                    if box != (0, 0, 0, 0) and (box[1] - box[0] >= 10) and (box[3] - box[2] >= 10):
                        valid_imgs.append(img)
                        valid_boxes.append(box)

                if (len(valid_imgs) > 0):
                    valid_imgs = torch.tensor(np.array(valid_imgs), dtype=torch.float).to(device=train_device) / 256
                    valid_imgs = torch.transpose(valid_imgs, 1, 3)

                    valid_boxes = torch.tensor(np.array(valid_boxes), dtype=torch.float).to(device=train_device) * (2 / tile_size)
                    valid_boxes = valid_boxes - 1
                    
                    optimizer.zero_grad()
                    output = model(valid_imgs)
                    loss = criterion(output, valid_boxes)
                    loss.backward()

                    running_loss += loss.item()
                    
                    optimizer.step()
                    tested += len(valid_imgs)
            

            except Exception as e:
                if str(e) not in ["Invalid Polygon", 
                                  "'Line String' object has no attribute 'geoms'",
                                  ]:
                    print(e)
                invalid += 1
            if (i % 1000 == 0):
                print(running_loss / (i + 1))

    torch.save(model.state_dict(), f'./models/segmentation_{last_checkpoint + 1}.pth')
    print("Saved to: ", f"segmentation_{last_checkpoint + 1}")
    print("Total Running Loss:", running_loss)
    print("Total invalid", invalid)
    print("Tested:", tested)

async def eval():
    instance = await BoundingBox.create(test=True)
    path = "./bdd100k/images/10k/val/"
    images = os.listdir(path)
    
    # Variables
    tile_size = 256
    min_size = 0
    max_size = tile_size

    model = Segmentation().cuda()
    
    last_checkpoint = sorted(os.listdir("./models"))[-1].split(".")[0].split("_")[-1]
    print("EVALUATING:", last_checkpoint)
    model.load_state_dict(torch.load(f'./models/segmentation_{last_checkpoint}.pth'))

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
                    img = torch.tensor(np.array([img])) / 256
                    img = img.cuda()
                    img = torch.transpose(img, 1, 3)
                    output = model(img)
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
                    plt.pause(0.5)
        except Exception as e:
            # if e != Exception("Invalid Polygon"):
            #     print(e)
            invalid += 1
    print("Accuracy:", f'{(correct/total)*100:.2f}%')
    print("Invalid:", invalid)
    print("Tested:", tested)

asyncio.run(train(1))
asyncio.run(eval())
# asyncio.run(sandbox())