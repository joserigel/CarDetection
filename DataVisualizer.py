import matplotlib.pyplot as plt
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from dotenv import load_dotenv
import os
load_dotenv()

class BoundingBox():
    async def create(test: bool = False):
        self = BoundingBox()
        self.collection = await BoundingBox.getCollection(test)
        return self

    async def getCollection(test: bool = False):
        uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
        client = AsyncIOMotorClient(uri)
        db = client['bdd100k']
        if not test:
            return db['sem_seg_polygons']
        else:
            return db['sem_seg_polygons_val']

    async def getPolygons(self, file_name, categories = ["car", "truck", "bus"]):
        result = await self.collection.find_one({
            "name": file_name
        })
        labels = list(filter(lambda x: (x["category"] in categories), result["labels"]))
        return list(map(lambda x: (x["poly2d"][0]["vertices"]), labels))
    
    
    async def getImgAndBox(self, file: str, tile_size: int):
        polygons = await self.getPolygons(file.split("/")[-1])
        img = plt.imread(file)
        x = 0
        y = 0

        if False in [Polygon(x).is_valid for x in polygons]:
            raise Exception("Invalid Polygon")

        imgAndBox = []
        while y+tile_size <= img.shape[0]:
            x = 0
            while x+tile_size <= img.shape[1]:
                cropped = img[y:y+tile_size, x:x+tile_size]
                box = self.getBounds(polygons, x, y, tile_size)
                imgAndBox.append((cropped, box))
                x += tile_size
            y += tile_size
        return imgAndBox

    def getBoundWithPolygon(self, polygon, x, y, tile_size):
        box = Polygon([[x, y], 
                    [x, y + tile_size], 
                    [x + tile_size, y + tile_size], 
                    [x + tile_size, y],
                    [x, y]])
        poly = Polygon(polygon)
        if not box.intersects(poly):
            return None
        intersection = box.intersection(poly)

        if type(intersection) == Polygon:
            left = np.min(intersection.boundary.xy[0])
            right = np.max(intersection.boundary.xy[0])
            bottom = np.min(intersection.boundary.xy[1])
            top = np.max(intersection.boundary.xy[1])
            return (left, right, bottom, top)
        else: 
            left = min(list(map(lambda x: np.min(x.boundary.xy[0]), intersection.geoms)))
            right = max(list(map(lambda x: np.max(x.boundary.xy[0]), intersection.geoms)))
            bottom = min(list(map(lambda x: np.min(x.boundary.xy[1]), intersection.geoms)))
            top = max(list(map(lambda x: np.max(x.boundary.xy[1]), intersection.geoms)))
            return (left, right, bottom, top)


    def getBounds(self, polygons, x, y, tile_size):
        if len(polygons) == 0:
            return (0, 0, 0, 0)
        bounds = [self.getBoundWithPolygon(polygon, x, y, tile_size) for polygon in polygons]
        bounds = list(filter(lambda x: x != None, bounds))
        
        if (len(bounds) == 0):
            return (0, 0, 0, 0)
        bounds = sorted(bounds, key=(lambda x: (x[1] - x[0]) * (x[3] - x[2])), reverse=True)
        bound = bounds[0]
        return (bound[0] - x, bound[1] - x, bound[2] - y, bound[3] - y)


    def adjustPolygons(polygons, x, y):
        adjusted = []
        for polygon in polygons:
            vertices = []
            for vertex in polygon:
                vertices.append([
                    vertex[0] - x,
                    vertex[1] - y
                ])
            adjusted.append(vertices)
        return adjusted
            
async def main():
    instance = await BoundingBox.create()
    datas = await instance.getImgAndBox("./bdd100k/images/10k/train/0a0a0b1a-7c39d841.jpg", 256)
    for img, box in datas:
        plt.clf()
        plt.imshow(img)
        plt.plot([box[0], box[0], box[1], box[1], box[0]],
                [box[2], box[3], box[3], box[2], box[2]])
        plt.pause(0.5)

# asyncio.run(main())
