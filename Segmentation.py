import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import asyncio

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import cv2

from time import time

from Connection import connect
from Preprocessor import getBatch


from dotenv import load_dotenv
import os
load_dotenv()

train_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("====", "USING CUDA GPU" if train_device == 'cuda:0' else "USING CPU (SLOW)", "====")
train_device = torch.device(train_device)

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.pooling = nn.Sequential(
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=(3,3), padding=1, stride=(3, 3)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(9, 64, kernel_size=(3,3), padding=1, stride=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=(3,3), padding=1, stride=(1, 1)),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x = self.pooling(x)
        x = self.convolutional(x)
        return x


async def eval(model):
    data, labels = await getBatch(10)
    img = torch.tensor(data).to(train_device)
    # images = images.swapaxes(1, 3)
    print(data.shape, labels.shape)
        
    x = model(img)
    x = x.cpu().detach().numpy()
    for i in range(data.shape[0]):
        mask = x[i].swapaxes(0, 2)
        print(mask.shape)
        mask = cv2.resize(mask, (1280, 720))
        img = data[i].swapaxes(0, 2)
        
        zeros = np.zeros_like(img)
        zeros[:, :, 0] = mask[:, :, 0]
        zeros[:, :, 1] = mask[:, :, 1]

        print(mask.shape)
        cv2.imshow("test", zeros)
        cv2.waitKey(0)
        cv2.imshow("test", img)
        cv2.waitKey(0)

async def eval2(model):
    data, labels = await getBatch(10)
    img = torch.tensor(data).to(train_device)
    # images = images.swapaxes(1, 3)
    print(data.shape, labels.shape)
        
    x = model(img)
    x = x.cpu().detach().numpy()
    for i in range(data.shape[0]):
        mask = x[i].swapaxes(0, 2)
        mask = cv2.resize(mask, (1280, 720))
        compare = (mask[:, :, 0]).astype(np.float16)
        img = data[i].swapaxes(0, 2)
        max_road = np.max(x[i][0])
        min_road = np.min(x[i][0])
        max_not = np.max(x[i][1])
        min_not = np.min(x[i][1])
        normalized_road = (mask[:, :, 0] - min_road) / (max_road - min_road)
        normalized_not = (mask[:, :, 1] - min_not) / (max_not - min_not)
        # print(max_road, min_road, "test", np.max(normalized), np.min(normalized))
        
        zeros = np.zeros_like(img)
        final_mask = np.clip(((2 *normalized_road - (normalized_not * 0.5))).astype(np.float16), 0, 1)
        zeros[:, :, 0] = final_mask
        zeros[:, :, 1] = final_mask
        zeros[:, :, 2] = final_mask

        subtracted = cv2.multiply(img, zeros)
        cv2.imshow("test", img)
        cv2.waitKey(1000)    
        cv2.imshow("test", zeros)
        cv2.waitKey(1000)
        cv2.imshow("test", subtracted)
        cv2.waitKey(1000)

async def eval3(model):
    data, labels = await getBatch(10, preprocessed=True, resolution=(60, 107))
    img = torch.tensor(data).to(train_device)
    # images = images.swapaxes(1, 3)
    # print(data.shape, labels.shape)
        
    x = model(img)
    x = x.cpu().detach().numpy()
    for i in range(data.shape[0]):
        mask = x[i].swapaxes(0, 2)
        mask = cv2.resize(mask, (1280, 720))
        gt = labels[i][0]
        # print(gt.shape)
        gt = cv2.resize(gt, (1280, 720))
        img = data[i].swapaxes(0, 2)
        max_road = np.max(x[i])
        min_road = np.min(x[i])
        normalized_road = (mask - min_road) / (max_road - min_road)
        # print(max_road, min_road, "test", np.max(normalized), np.min(normalized))
        
        zeros = np.zeros_like(img)
        # zeros[:, :, 1] = np.where(gt > normalized_road, 1, 0)
        # zeros[:, :, 2] = normalized_road
        # zeros[:, :, 1] = np.where(gt <= normalized_road, 1, 0)
        # zeros[:, :, 2] = mask

        subtracted = cv2.add(img, zeros)
        cv2.imshow("test", img)
        cv2.waitKey(1000)    
        cv2.imshow("test", zeros)
        cv2.waitKey(1000)
        cv2.imshow("test", subtracted)
        cv2.waitKey(1000)

async def train():
    epoch = 7
    batch_size = 1000
    gpu_batch = 5
    running_loss = 0
    decay = 1/2
    model = Segmentation().to(train_device)
    # model.load_state_dict(torch.load("./models/segmentation_1.pth"))
    loss_fn = nn.BCEWithLogitsLoss()
    
    images, labels = await getBatch(batch_size, preprocessed=True, resolution=(60, 107))
    learning_rate = 0.1
    for i in range(epoch):
        learning_rate *= decay
        print(f"=====EPOCH {i + 1}/{epoch}======  lr:", learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for j in range(gpu_batch):
            size = batch_size // gpu_batch
            print(f"Batch {j + 1}/{batch_size}")
            end = min((size * j) + size, batch_size)
            optimizer.zero_grad()

            data = torch.tensor(images[size * j: end]).to(train_device)
            target = torch.tensor(labels[size * j: end]).to(train_device)
            print(data.shape, target.shape)
            
            outputs = model.forward(data)
            loss = loss_fn(outputs, target)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
        print("RUNNING_LOSS:", running_loss)
    torch.save(model.state_dict(), f'./models/segmentation_1.pth')
    await eval3(model)


model = Segmentation().to(train_device)
model.load_state_dict(torch.load("./models/segmentation_0.pth"))
asyncio.run(eval3(model))
# asyncio.run(eval(model))

# asyncio.run(train())
