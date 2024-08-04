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
train_device = 'cpu'
print("====", "USING CUDA GPU" if train_device == 'cuda:0' else "USING CPU (SLOW)", "====")
train_device = torch.device(train_device)

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=(5,5), padding=0, stride=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25347, 64*128),
            nn.LeakyReLU(),
            nn.Linear(64*128, 64*64 * 2),
            nn.LogSoftmax(dim=1),
        )
        self.unflatten = nn.Unflatten(1, (2, 64, 64))
    def forward(self, x):
        # x = self.pooling(x)
        x = self.convolutional(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.unflatten(x)
        return x

async def eval(model):
    # data, labels = await getBatch(10, preprocessed=True, resolution=(64, 64))
    data, labels = np.load('./binaryDataset/inputs_0.npy'), np.load('./binaryDataset/labels_0.npy')
    
    img = torch.tensor(data).to(train_device)
    # images = images.swapaxes(1, 3)
    # print(data.shape, labels.shape)
        
    x = model(img)
    x = x.cpu().detach().numpy()
    for i in range(440):
        mask = x[i].swapaxes(0, 2)
        print(mask.shape)
        mask = cv2.resize(mask, (1280, 720))
        gt = labels[i][1]
        gt = cv2.resize(gt, (1280, 720))
        img = data[i].swapaxes(0, 2)
        max_road = np.max(x[i][:, :, 0])
        min_road = np.min(x[i][:, :, 0])
        print(np.max(mask), np.min(mask))
        normalized = (mask - min_road) / (max_road - min_road)
        print(max_road, min_road, "test", np.max(normalized), np.min(normalized))
        compare = (mask[:, :, 0] >= mask[:, :, 1]).astype(np.float32)
        
        zeros = np.zeros_like(img)
        zeros[:, :, 0] = compare
        zeros[:, :, 1] = compare
        zeros[:, :, 2] = compare
        # zeros[:, :, 2] = mask

        added = cv2.multiply(img, zeros)
        cv2.imshow("test", img)
        cv2.waitKey(500)    
        # cv2.imshow("test", zeros)
        # cv2.waitKey(500)
        # cv2.imshow("test", added)
        # cv2.waitKey(500)

async def train():
    epoch = 100
    gpu_batch = 4
    running_loss = 0
    decay = 99/100
    learning_rate = 0.001
    model = Segmentation().to(train_device)
    model.load_state_dict(torch.load("./models/segmentation_0.pth", weights_only=True))
    loss_fn = nn.MSELoss()
    
    images, labels = np.load('./binaryDataset/inputs_0.npy'), np.load('./binaryDataset/labels_0.npy')
    batch_size = images.shape[0]
    print("BATCH SIZE", batch_size)
    for i in range(epoch):
        learning_rate *= decay
        if i % 10 == 0:
            print(f"=====EPOCH {i + 1}/{epoch}======  lr:", learning_rate)
            print("RUNNING_LOSS:", running_loss)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        for j in range(gpu_batch):
            size = batch_size // gpu_batch
            end = min((size * j) + size, batch_size)
            # print(f"Batch {j + 1}/{gpu_batch}", end)
            optimizer.zero_grad()

            data = torch.tensor(images[size * j: end]).to(train_device)
            target = torch.tensor(labels[size * j: end]).to(train_device)
            
            outputs = model.forward(data)
            loss = loss_fn(outputs, target)
            running_loss += loss.item()
            loss.backward()

            optimizer.step()
    torch.save(model.state_dict(), f'./models/segmentation_0.pth')
    await eval(model)


model = Segmentation().to(train_device)
model.load_state_dict(torch.load("./models/segmentation_0.pth", weights_only=True))
asyncio.run(eval(model))


# asyncio.run(train())
