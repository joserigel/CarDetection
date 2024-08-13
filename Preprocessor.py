from Connection import connect
import numpy as np
import cv2
import os
import asyncio
from tqdm import tqdm
import random
import numpy as np

IMG_DIR = "bdd100k/images/10k"
DEF_IN_RES=(480, 270)
DEF_OUT_RES=(64, 36)
TRAIN_DATASET = sorted(os.listdir(f"{IMG_DIR}/train"))

def drawMask(img, polygons):
    for polygon in polygons:
        pts = np.array([polygon["vertices"]])
        pts.reshape((-1, 1, 2))

        cv2.fillPoly(img, pts.astype(dtype=np.int32), color=255)

async def add_noise(image):
    return image * (1 - 0.1*np.random.rand(720, 1280, 3))

def createTarget(data, resolution):
    # Initialize matrix
    cars = np.zeros((720, 1280), dtype=np.float32)

    # Iterate through all labels that are car/truck/bus
    for label in data["labels"]:
        if label["category"] in ["car", "truck", "bus"]:
            drawMask(cars, label["poly2d"])
    
    # Normalize and resize to target res
    cars = cv2.resize(cars, resolution, interpolation=cv2.INTER_NEAREST)
    cars = cars.swapaxes(0, 1)

    cars = np.where(cars > 0, 1, 0)
    unlabeled =  1 - cars

    return cars, unlabeled

async def getImageFromDisk(file, in_res=DEF_IN_RES, with_noise=True, dataset="train"):
    
    # Get image from disk
    img = cv2.imread(f"{IMG_DIR}/{dataset}/{file}") / 255

    # Add noise
    if with_noise:
        img = await add_noise(img)

    # Resize input
    h, w, c = img.shape
    if w != in_res[0] or h != in_res[1]:
        img = cv2.resize(img, in_res, interpolation=cv2.INTER_CUBIC)

    # Swap axes
    img = img.swapaxes(0, 2)

    return img

async def createTargetBatch(dbConnection, files = [], out_res=DEF_OUT_RES):
    
    # Initialize array
    targets = np.zeros((len(files), 2, out_res[0], out_res[1]), dtype=np.float32)

    for i, file in enumerate(tqdm(files)):
        
        # Create mask
        data = await dbConnection.find_one({"name": file})
        cars, unlabeled = createTarget(data, out_res)

        # Broadcast to array
        targets[i] = [cars, unlabeled]

    return targets


async def getImageBatch(files, in_res=DEF_IN_RES, with_noise=True, dataset="train"):
    imgs = np.zeros((len(files), 3, in_res[0], in_res[1]), dtype=np.float32)

    # Initialize arrays
    for i, file in enumerate(files):
        imgs[i] = await getImageFromDisk(file, in_res, with_noise, dataset)
    return imgs

async def getVal(
        size, dataset = "val",
        in_res=DEF_IN_RES, out_res=DEF_OUT_RES, 
        with_noise=True, 
    ):
    dbConnection = await connect(
        "sem_seg_polygons" + ("_val" if dataset == "val" else "")
        )
    
    files = random.sample(os.listdir(f"{IMG_DIR}/{dataset}"), size)
    
    images = await getImageBatch(files, in_res, with_noise, "val")
    targets = await createTargetBatch(dbConnection, files, out_res)

    return images, targets

async def getBatch(name, batch_count, idx, in_res=DEF_IN_RES, with_noise=True):
    # Load from disk
    target = np.load(f'./binaryDataset/labels_{name}_{idx}.npy')
    
    # Calculate indices
    size = len(TRAIN_DATASET) // batch_count
    start = size * idx
    end = start + target.shape[0]

    files = TRAIN_DATASET[start:end]

    # Get images
    images = await getImageBatch(files, in_res, with_noise, "train")

    return images, target     

async def saveToStorage(size, name, out_res=DEF_OUT_RES):
    batch_count  = len(TRAIN_DATASET) // size
    dbConnection = await connect("sem_seg_polygons")
    
    for i in range(batch_count):
        print(f"BATCH: {i + 1}/{batch_count}")
        start = size * i
        end = start + min(size, len(TRAIN_DATASET) - (size * i))
        files = TRAIN_DATASET[start : end]

        targets = await createTargetBatch(dbConnection, files, out_res)

        with open(f'binaryDataset/labels_{name}_{i}.npy', 'wb') as f:
            np.save(f, targets)

# asyncio.run(saveToStorage(125, name="noise_raw"))

        
