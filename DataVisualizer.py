import matplotlib.pyplot as plt
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from dotenv import load_dotenv
import os
load_dotenv()

async def connection():
    uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
    client = AsyncIOMotorClient(uri)
    db = client['bdd100k']
    return db['sem_seg_polygons']


async def getPolygons(collection, file_name, category = "car"):
    result = await collection.find_one({
        "name": file_name
    })
    labels = list(filter(lambda x: (x["category"] == category), result["labels"]))
    return list(map(lambda x: (x["poly2d"][0]["vertices"]), labels))
    


def getBoundWithPolygon(polygon, x, y, tile_size):
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


def getBounds(polygons, x, y, tile_size):
    if len(polygons) == 0:
        return (0, 0, 0, 0)
    bounds = [getBoundWithPolygon(polygon, x, y, tile_size) for polygon in polygons]
    bounds = list(filter(lambda x: x != None, bounds))
    # for i in range(len(bounds)):
    #     bounds[i] = (bounds[i][0] - x,
    #     bounds[i][1] - x,
    #     bounds[i][2] - y,
    #     bounds[i][3] - y)
    # return bounds
    
    if (len(bounds) == 0):
        return (0, 0, 0, 0)
    left = min(list(map(lambda x: x[0], bounds)))
    right = max(list(map(lambda x: x[1], bounds)))
    bottom = min(list(map(lambda x: x[2], bounds)))
    top = max(list(map(lambda x: x[3], bounds)))
    print(left, right, bottom, top)
    return (left - x, right - x, bottom - y, top - y)


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

async def visualize_polygons():
    path = "DataSandbox/train/"
    collection = await connection()
    for file in os.listdir(path):
        # Show image raw
        plt.clf()
        img = plt.imread(path + file)
        plt.imshow(img)
        
        result = await collection.find_one({
            "name": file
        })

        for label in result["labels"]:
            if label["category"] == "car":
                coords = label["poly2d"][0]["vertices"]
                xs, ys = zip(*coords)
                plt.plot(xs, ys)      
        
        plt.pause(0.5)
        plt.draw()
        

async def visualize_boxes_in_grid():
    path = "DataSandbox/train/"
    collection = await connection()
    
    for file in os.listdir(path):
        polygons = await getPolygons(collection, file)
        img = plt.imread(path + file)
        tile_size = 256
        x = 0
        y = 0

        while y+tile_size <= img.shape[0]:
            x = 0
            while x+tile_size <= img.shape[1]:
                plt.clf()

                cropped = img[y:y+tile_size, x:x+tile_size]
                plt.imshow(cropped)

                box = getBounds(polygons, x, y, tile_size)
                if (box != (0,0,0,0)):
                    plt.plot([box[0], box[0], box[1], box[1], box[0]],
                            [box[2], box[3], box[3], box[2], box[2]])
                    
                
                    # plt.xlim(128, 0)
                    # plt.ylim(128, 0)
                    plt.pause(1.5)
                x += tile_size
            y += tile_size
            

asyncio.run(visualize_boxes_in_grid())
