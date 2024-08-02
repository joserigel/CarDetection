import asyncio
from Connection import connect
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import cv2

async def main():
    dbConnection = await connect("sem_seg_polygons")
    drivable = await dbConnection.find_one()
    name = drivable["name"]
    
    
    img = cv2.imread(f"bdd100k/images/10k/train/{name}")
    # img = np.zeros_like(img)
    
    for label in drivable["labels"]:
        # print(label.keys())
        if label["category"] == "car":
            for polygon in label["poly2d"]:
                pts = np.array([polygon["vertices"]])
                pts.reshape((-1, 1, 2))
                

                cv2.fillPoly(img, pts.astype(dtype=np.int32), color=(255, 255, 255, 50))
    print(name)
    print(img.shape)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


