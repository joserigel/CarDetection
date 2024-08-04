import torch
import torch.nn as nn
import numpy as np
import cv2
import asyncio

from Preprocessor import getBatch

# Set CUDA device
cuda_device = 0
train_device = f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu'
print("====", "USING CUDA GPU" if train_device == f'cuda:{cuda_device}' else "USING CPU (SLOW)", "====")
train_device = torch.device(train_device)

# Architecture
class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=(5,5), padding=0, stride=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(3, 3, kernel_size=(3,3), padding=0, stride=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18270, 64*128),
            nn.LeakyReLU(),
            nn.Linear(64*128, 64*64 * 2),
            nn.LogSoftmax(dim=1),
        )
        self.unflatten = nn.Unflatten(1, (2, 64, 64))

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        x = self.unflatten(x)
        return x
    
# Transformation to 1280x720 mask
def transformOutputToMask(output):
    mask = output.swapaxes(0, 2).swapaxes(0, 1)
    mask = cv2.resize(mask, (1280, 720))
    mask = (mask[:, :, 0] >= mask[:, :, 1]).astype(np.float32)
    return mask

# Evaulation
async def eval(model):
    # Get random sample
    data, target = await getBatch(100, preprocessed=True, resolution=(64, 64))
    # data, labels = np.load('./binaryDataset/inputs_0.npy'), np.load('./binaryDataset/labels_0.npy')
    
    # Feed to network
    img = torch.tensor(data).to(train_device)
        
    # Detach and show result
    output = model(img)
    output = output.cpu().detach().numpy()
    for i in range(data.shape[0]):
        img = data[i].swapaxes(0, 2)
        mask = transformOutputToMask(output[i])
        
        mask_in_rgb = np.zeros_like(img)
        mask_in_rgb[:, :, 1] = mask

        marked = cv2.add(img, mask_in_rgb)
        cv2.imshow("test", marked)
        cv2.waitKey(500)

async def train():
    # Settings for training
    epoch = 100
    gpu_batch = 4
    decay = 99/100
    learning_rate = 0.001
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Load Dataset and Model
    running_loss = 0
    model = Segmentation().to(train_device)
    model.load_state_dict(torch.load("./models/segmentation_0.pth", weights_only=True))
    images, labels = np.load('./binaryDataset/inputs_0.npy'), np.load('./binaryDataset/labels_0.npy')
    batch_size = images.shape[0]
    print("BATCH SIZE", batch_size)

    # Training
    for i in range(epoch):

        if i % 10 == 0:
            print(f"=====EPOCH {i + 1}/{epoch}======  lr:", learning_rate)
            print("RUNNING_LOSS:", running_loss)

        # Optimizer setup
        learning_rate *= decay
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for j in range(gpu_batch):
            optimizer.zero_grad()

            # Set current batch
            size = batch_size // gpu_batch
            end = min((size * j) + size, batch_size)
            data = torch.tensor(images[size * j: end]).to(train_device)
            target = torch.tensor(labels[size * j: end]).to(train_device)
            # print(f"Batch {j + 1}/{gpu_batch}", end)
            
            # Feed to network
            outputs = model.forward(data)
            loss = loss_fn(outputs, target)
            running_loss += loss.item()
            
            # Calculate loss and optimize
            loss.backward()
            optimizer.step()

    # Save to disk
    torch.save(model.state_dict(), f'./models/segmentation_0.pth')

    # Eval
    await eval(model)

# Run evaluation
model = Segmentation().to(train_device)
model.load_state_dict(torch.load("./models/segmentation_0.pth", weights_only=True))
asyncio.run(eval(model))

# Train
# asyncio.run(train())
