from Connection import connect
import numpy as np
import cv2
import os
import asyncio
from tqdm import tqdm
import random
import numpy as np

async def filter_dataset():
    dbConnection = await connect("det")
    detections = dbConnection.find({"attributes.timeofday": "daytime", "attributes.weather": "clear"})
    names = []
    async for row in detections:
        name = row["name"]
        if name in train_dataset:
            names.append(name)
    return names

def drawMask(img, polygons):
    for polygon in polygons:
        pts = np.array([polygon["vertices"]])
        pts.reshape((-1, 1, 2))

        cv2.fillPoly(img, pts.astype(dtype=np.int32), color=255)


async def preprocess(image):
    return image * (1 - 0.1*np.random.rand(720, 1280, 3))


def createTarget(in_res, data, resolution):
    cars = np.zeros((720, 1280), dtype=np.float32)
    for label in data["labels"]:
        if label["category"] in ["car", "truck", "bus"]:
            drawMask(cars, label["poly2d"])
                
    cars = cv2.resize(cars, resolution, interpolation=cv2.INTER_NEAREST) / 255
    unlabeled =  1 - cars

    return cars, unlabeled

async def getBatch(
        size, in_res=(480, 270), out_res=(240, 427), 
        preprocessed=True, dataset = "val"
    ):
    dbConnection = await connect("sem_seg_polygons" 
        + ("_val" if dataset == "val" else "")
        )
    
    inputs = np.zeros((size, 3, in_res[0], in_res[1]), dtype=np.float32)
    labels = np.zeros((size, 2, out_res[1], out_res[0]), dtype=np.float32)
    
    sample = random.sample(os.listdir(f"bdd100k/images/10k/{dataset}"), size)
    
    for i, image in enumerate(tqdm(sample)):
        inp_image = cv2.imread(f"bdd100k/images/10k/{dataset}/{image}") / 255
        if preprocessed:
            inp_image = await preprocess(inp_image)
        inp_image = cv2.resize(inp_image, in_res, interpolation=cv2.INTER_CUBIC)
        inp_image = inp_image.swapaxes(0, 2)
        
        data = await dbConnection.find_one({"name": image})
        cars, unlabeled = createTarget(in_res, data, out_res)
       
        labels[i] = [cars, unlabeled]
        inputs[i] = inp_image
    return inputs, labels


async def saveToStorage(count, in_res=(480, 270), out_res=(240, 427), preprocessed=True, name="dataset"):
    size = len(train_dataset) // count
    dbConnection = await connect("sem_seg_polygons")

    for batch in range(count):
        print("BATCH:", batch + 1, "/", count)
        actual_size = min(size, len(train_dataset) - (size * batch))
        inputs = np.zeros((actual_size, 3, in_res[0], in_res[1]), dtype=np.float32)
        labels = np.zeros((actual_size, 2, out_res[1], out_res[0]), dtype=np.float32)

        for i, image in enumerate(tqdm(train_dataset[size * batch : (size * batch) + size])):
            inp_image = cv2.imread(f"bdd100k/images/10k/train/{image}") / 255
            if preprocessed:
                inp_image = await preprocess(inp_image)
            inp_image = cv2.resize(inp_image, in_res, interpolation=cv2.INTER_CUBIC)
            inp_image = inp_image.swapaxes(0, 2)
            
            data = await dbConnection.find_one({"name": image})
            cars, unlabeled = createTarget(inp_image, data, out_res)

            labels[i] = [cars, unlabeled]
            inputs[i] = inp_image 

        with open(f'binaryDataset/inputs_{name}_{batch}.npy', 'wb') as f:
            np.save(f, inputs)

        with open(f'binaryDataset/labels_{name}_{batch}.npy', 'wb') as f:
            np.save(f, labels)

train_dataset = os.listdir("bdd100k/images/10k/train/")
filtered_dataset = asyncio.run(filter_dataset())
# dataset = filtered_dataset
# asyncio.run(saveToStorage(7, out_res=(156, 86), name="n", preprocessed=True))

        
