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

from Connection import connect


from dotenv import load_dotenv
import os
load_dotenv()

train_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("====", "USING CUDA GPU" if train_device == 'cuda' else "USING CPU (SLOW)", "====")

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=(3,3), padding=0, stride=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(9, 1, kernel_size=(3,3), padding=0, stride=(3, 3))
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Linear(11360, 64*64),
            nn.ReLU(),
            nn.Linear(64*64, 64*64)
        )

        self.deconvolutional_layer = nn.Sequential(
            nn.ConvTranspose2d(1, 3, kernel_size=(3,3), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 9, kernel_size=(3,3), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 2, kernel_size=(3,3), stride=(2, 2)),
        )

    def forward(self, x):
        x = self.convolutional_layer(x)
        x = torch.flatten(x)
        x = self.fully_connected_layer(x)
        x = torch.reshape(x, (1, 64, 64))
        x = self.deconvolutional_layer(x)
        return x
    
Model = Segmentation()

async def main():
    dbConnection = await connect("sem_seg_polygons")
    drivable = await dbConnection.find_one()
    name = drivable["name"]
    img = cv2.imread(f"bdd100k/images/10k/train/{name}")
    img = img.swapaxes(0, 2)
    img = torch.from_numpy(img).float() / 255
    # print(img)
    
    x = Model.forward(img)
    x = x.detach().numpy()
    a = x[1]
    b = x[0]
    x = a > b
    
    cv2.imshow("test", x.astype(np.uint8) * 255)
    cv2.waitKey(0)

asyncio.run(main())