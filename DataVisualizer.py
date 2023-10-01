import matplotlib.pyplot as plt
import os
import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np

from dotenv import load_dotenv
import os
load_dotenv()

async def connection():
    uri = f"mongodb://{os.environ.get('MONGO_USER')}:{os.environ.get('MONGO_PASS')}@localhost:27017"
    client = AsyncIOMotorClient(uri)
    db = client['bdd100k']
    return db['sem_seg_polygons']

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

async def getPolygons(collection, file_name, category = "car"):
    result = await collection.find_one({
        "name": file_name
    })
    labels = list(filter(lambda x: (x["category"] == category), result["labels"]))
    return list(map(lambda x: (x["poly2d"][0]["vertices"]), labels))
    
    
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


def poylgon_to_box(polygons):
    boxes = []
    for polygon in polygons:
        x_min = min(*list(map(lambda x: x[0], polygon)))
        x_max = max(*list(map(lambda x: x[0], polygon)))
        y_min = min(*list(map(lambda x: x[1], polygon)))
        y_max = max(*list(map(lambda x: x[1], polygon)))
        if (x_max - x_min > 0 and y_max - y_min > 0):
            boxes.append([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
                [x_min, y_min]
            ])
    return boxes

# def sophisticated_to_box(polygons, tile_size, x, y):
#     polygons = adjustPolygons(polygons, x, y)
#     boxes = []
#     for polygon in polygons:
#         last_point = polygon[0]
#         polygon = polygon[:-1]
#         for point in polygon:
#             if (last_point[0] < 0 and point[0] < 0) or \
#                 (last_point[0] > tile_size and point[0] > tile_size) or \
#                 (last_point[1] < 0 and point[1] < 0) or \
#                 (last_point[1] > tile_size and point[1] > tile_size):
#                 last_point = point
#                 continue
#             if (last_point[0] < 0 or last_point[1] < 0 or last_point[0] > tile_size or last_point[1 > tile_size]) and
#                 (point[0] >= 0 and point[0] <= tile_size and point[1] >= 0 and point <= tile_size):



async def visualize_boxes_in_grid():
    path = "DataSandbox/train/"
    collection = await connection()
    
    for file in os.listdir(path):
        polygons = await getPolygons(collection, file)
        img = plt.imread(path + file)
        tile_size = 128
        x = 0
        y = 0

        while y+tile_size <= img.shape[0]:
            x = 0
            while x+tile_size <= img.shape[1]:
                plt.clf()

                cropped = img[y:y+tile_size, x:x+tile_size]
                plt.imshow(cropped)

                boxes = poylgon_to_box(adjustPolygons(polygons, tile_size, x, y))
                for polygon in boxes:
                    xs, ys = zip(*polygon)
                    plt.plot(xs, ys)
                    
                
                plt.xlim(128, 0)
                plt.ylim(128, 0)
                # plt.pause(10)
                if (len(boxes) > 0):
                    plt.show()
                x += tile_size
            y += tile_size
            

asyncio.run(visualize_boxes_in_grid())
