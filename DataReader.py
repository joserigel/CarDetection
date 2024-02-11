import json
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from tqdm import tqdm

from dotenv import load_dotenv
import os
load_dotenv()

# PLEASE JUST RUN THIS ONCE FOR EACH LABEL TYPES
async def connect(collection):
    uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
    client = AsyncIOMotorClient(uri)
    db = client['bdd100k']
    return db[collection]

async def loadIntoMongoDB(file: str, collection):
    print(f"Copying {file.split('/')[-1]} into {collection}")
    with open(file) as f:
        datas = json.load(f)
        db_collection = await connect(collection)
        for data in tqdm(datas):
            await db_collection.insert_one(data)
        print(f"Finished copying {file.split('/')[-1]} into {collection}")

async def main():
    await loadIntoMongoDB(f'{os.environ.get("BDD100K_DIR")}/labels/sem_seg/polygons/sem_seg_train.json', 'sem_seg_polygons')
    await loadIntoMongoDB(f'{os.environ.get("BDD100K_DIR")}/labels/sem_seg/polygons/sem_seg_val.json', 'sem_seg_polygons_val')
    await loadIntoMongoDB(f'{os.environ.get("BDD100K_DIR")}/labels/lane/polygons/lane_train.json', 'lane_polygons')
    await loadIntoMongoDB(f'{os.environ.get("BDD100K_DIR")}/labels/lane/polygons/lane_val.json', 'lane_polygons_val')

asyncio.run(main())

