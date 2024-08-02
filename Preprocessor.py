from Connection import connect
import numpy as np
import cv2

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

