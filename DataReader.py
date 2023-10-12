import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()

# PLEASE JUST RUN THIS ONCE FOR EACH LABEL TYPES
async def connect():
    uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
    client = AsyncIOMotorClient(uri)
    db = client['bdd100k']
    return db['sem_seg_polygons_val']

async def main():
    print("Loading!")
    with open('./bdd100k/labels/sem_seg/polygons/sem_seg_val.json') as f:
        datas = json.load(f)
        print("Loaded!")
        collection = await connect()
        for data in tqdm(datas):
            await collection.insert_one(data)

asyncio.run(main())

