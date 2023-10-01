import matplotlib.pyplot as plt
import numpy as np
import math
import random

plt.clf()

def crop_polygon(polygon, tile_size):
    points = []
    last_point = polygon[0]
    polygon = polygon[1:]
    for point in polygon:
        last_point_inside = max(last_point) <= tile_size and min(last_point) >= 0
        point_inside = max(point) <= tile_size and min(point) >= 0

        m = (point[1] - last_point[1]) / (point[0] - last_point[0]) if point[0] != last_point[0] else 0
        c = point[1] - (m * point[0])
        
        left = min(last_point[0], point[0])
        right = max(last_point[0], point[0])
        bottom = min(last_point[1], point[1])
        top = max(last_point[1], point[1])

        # if both points are outside the box, then ignore
        if not last_point_inside and not point_inside :
            # if points go across the bounds
            if left < 0 and right > tile_size:
                left_y = y - c / m
            elif bottom < 0 and top > tile_size:
            last_point = point
            print("skipped")
            continue
        # if current is inside, but last is outside
        elif last_point_inside and not point_inside or \
            not last_point_inside and point_inside:
            if point[1] == last_point[1]:
                if not last_point_inside:
                    x = min(tile_size, max(0, last_point[0]))
                    points.append([x, point[1]])
                else:
                    x = min(tile_size, max(0, point[0]))
                    points.append([x, last_point[1]])
            else:
                if left < 0:
                    points.append([0, c])
                    
                elif right > tile_size:
                    y = (m * tile_size) + c
                    points.append([tile_size, y])
                elif bottom < 0:
                    x = - c / m
                    points.append([x, 0])
                else:
                    x = (tile_size - c) / m
                    points.append([x, 128])
                    
        

            
        if point_inside:
            points.append(point)
                
        last_point = point
    return points
            


def magnitude(vector):
    return math.sqrt(np.dot(vector, vector))

def generate_polygon(sides):
    points = []
    for i in range(sides):
        points.append([random.uniform(-128, 128), random.uniform(-128, 128)])
    points.append(points[0])
    return points

def draw_loop(sides, iter):
    # Draw Loop
    for j in range(iter):
        polygon = generate_polygon(sides)
        for i in range(2, sides):
            plt.clf()
            plt.xlim(-128, 256)
            plt.ylim(-128, 256)

            xs, ys = zip(*polygon[:i])
            plt.plot(xs, ys)
            plt.plot([0, 0, 128, 128, 0], [0, 128, 128, 0, 0])


            cropped = crop_polygon(polygon[:i], 128)
            if len(cropped) >= 2 :
                xs2, ys2 = zip(*crop_polygon(polygon[:i], 128))
                plt.plot(xs2, ys2)
            

            plt.pause(1)
            plt.draw()

def draw_one():
    polygon = [[-200, -30], [10,100], [200, 0], [200, -10], [30, 50]]
    
    plt.clf()
    plt.xlim(-128, 256)
    plt.ylim(-128, 256)

    xs, ys = zip(*polygon)
    plt.plot(xs, ys)
    plt.plot([0, 0, 128, 128, 0], [0, 128, 128, 0, 0])


    cropped = crop_polygon(polygon, 128)
    if len(cropped) >= 2 :
        xs2, ys2 = zip(*crop_polygon(polygon, 128))
        plt.plot(xs2, ys2)

    plt.show()

draw_one()