from Connection import connect
import numpy as np
import cv2
import os
import asyncio
from tqdm import tqdm
import random
import numpy as np

async def imageMask(name, res = (720, 1280)):
    dbConnection = await connect("sem_seg_polygons")
    data = await dbConnection.find_one({"name": name})

    img = np.zeros(res)
    for label in data["labels"]:
        if label["category"] == "road":
            for polygon in label["poly2d"]:
                pts = np.array([polygon["vertices"]])
                pts.reshape((-1, 1, 2))

                cv2.fillPoly(img, pts.astype(dtype=np.int32), color=(255, 255, 255))
    return img

async def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mid = 0.5
    mean = np.mean(gray)
    gamma = np.log(mid*255)/np.log(mean)
    corrected = np.power(image, gamma).clip(0,255).astype(np.uint8) #/ 255
    # edges = cv2.Canny(image, 100, 200) / 255
    # corrected[:, :, 0] *= edges
    # corrected[:, :, 1] *= edges
    # corrected[:, :, 2] *= edges
    return corrected #* 255
directory = os.listdir("bdd100k/images/10k/train/")
async def filter():
    dbConnection = await connect("det")
    detections = dbConnection.find({"attributes": {
        "weather": "clear",
        "timeofday": "daytime",
        "scene":"city street"}
        })
    names = []
    async for row in detections:
        name = row["name"]
        if name in directory:
            names.append(name)
    return names

samples = asyncio.run(filter())



async def getBatch(size, resolution=(240, 427), preprocessed=True):
    dbConnection = await connect("sem_seg_polygons")
    
    inputs = np.zeros((size, 3, 1280, 720), dtype=np.float32)
    labels = np.zeros((size, 2, resolution[1], resolution[0]), dtype=np.float32)
    
    sample = random.sample(directory, size)
    
    
    for i, image in enumerate(tqdm(sample)):
        input_image = cv2.imread(f"bdd100k/images/10k/train/{image}")
        if preprocessed:
            input_image = await preprocess(input_image)
        input_image = input_image.swapaxes(0, 2)
        inputs[i] = input_image
        data = await dbConnection.find_one({"name": image})

        mask = np.zeros((input_image.shape[2], input_image.shape[1]))
        for label in data["labels"]:
            if label["category"] in ["car", "truck", "bus"]:
                for polygon in label["poly2d"]:
                    pts = np.array([polygon["vertices"]])
                    pts.reshape((-1, 1, 2))

                    cv2.fillPoly(mask, pts.astype(dtype=np.int32), color=(255, 255, 255))
                    
        mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_AREA)
        inverted = 255 - mask

        
        labels[i] = [mask / 255, inverted / 255]


    return inputs / 255, labels

directory = samples
async def saveToStorage(count, resolution=(240, 427), preprocessed=True):
    size = len(directory) // count
    dbConnection = await connect("sem_seg_polygons")

    for batch in range(count):
        print("BATCH:", batch + 1, "/", count)
        actual_size = min(size, len(directory) - (size * batch))
        inputs = np.zeros((actual_size, 3, 1280, 720), dtype=np.float32)
        labels = np.zeros((actual_size, 2, resolution[1], resolution[0]), dtype=np.float32)

        for i, image in enumerate(tqdm(directory[size * batch : (size * batch) + size])):
            input_image = cv2.imread(f"bdd100k/images/10k/train/{image}")
            if preprocessed:
                input_image = await preprocess(input_image)
            input_image = input_image.swapaxes(0, 2)
            inputs[i] = input_image
            data = await dbConnection.find_one({"name": image})

            mask = np.zeros((input_image.shape[2], input_image.shape[1]))
            for label in data["labels"]:
                if label["category"] in ["car", "truck", "bus"]:
                    for polygon in label["poly2d"]:
                        pts = np.array([polygon["vertices"]])
                        pts.reshape((-1, 1, 2))

                        cv2.fillPoly(mask, pts.astype(dtype=np.int32), color=(255, 255, 255))
                        
            mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)
            inverted = 255 - mask

            
            labels[i] = [mask / 255, inverted / 255]

        with open(f'binaryDataset/inputs_{batch}.npy', 'wb') as f:
            np.save(f, inputs / 255)

        with open(f'binaryDataset/labels_{batch}.npy', 'wb') as f:
            np.save(f, labels)

# asyncio.run(saveToStorage(1, resolution=(64, 64)))

        
