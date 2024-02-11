import matplotlib.pyplot as plt
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()

class Polygon():
    uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
    mongodb = "bdd100k"
    async def create(test: bool = False):
        self = Polygon()
        self.collection = await Polygon.getCollection(test)
        return self
    async def getCollection(test: bool = False):
        client = AsyncIOMotorClient(Polygon.uri)
        db = client[Polygon.mongodb]
        if not test:
            return db['sem_seg_polygons']
        else:
            return db['sem_seg_polygons_val']
    async def getData(self, path: str):
        return await self.collection.find_one({
            "name": path
        })
    
async def getClassesCount():
    instance = await Polygon.create()
    dir = f"{os.environ.get('BDD100K_DIR')}/images/10k/train/"
    categories = {}
    for pic in tqdm(os.listdir(dir)):
        data = await instance.getData(pic)
        for label in data["labels"]:
            if label["category"] in categories:
                categories[label["category"]] += 1
            else:
                categories[label["category"]] = 1
    categories = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))
    for k, v in categories.items():
        print(f"{k}:", v)

async def main():
    instance = await Polygon.create()
    # datas = await instance.getData("0a0a0b1a-7c39d841.jpg")
    print(os.environ.get('BDD100K_DIR'))
    
asyncio.run(getClassesCount())
