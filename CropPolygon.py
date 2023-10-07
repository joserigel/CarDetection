import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
import numpy as np

def crop_polygon(polygon, tile_size):
    box = Polygon([[0, 0], [0, tile_size], [tile_size, tile_size], [tile_size, 0], [0, 0]])
    intersection = box.intersection(Polygon(polygon))
    return intersection

            
def draw_one():
    polygon = [[-10, -30], [10, 50], [80, -5], [100, 50], [120, -5], [-10, -30]]
    
    plt.clf()
    plt.xlim(-128, 256)
    plt.ylim(-128, 256)

    xs, ys = zip(*polygon)
    plt.plot(xs, ys)
    plt.plot([0, 0, 128, 128, 0], [0, 128, 128, 0, 0])

    cropped = crop_polygon(polygon, 128)
    box = [0, 0, 0, 0]
    if type(cropped) == Polygon:
        xs2, ys2 = cropped.boundary.xy
        plt.plot(xs2, ys2)
    elif type(cropped) == MultiPolygon:
        left = min(list(map(lambda x: np.min(x.boundary.xy[0]), cropped.geoms)))
        right = max(list(map(lambda x: np.max(x.boundary.xy[0]), cropped.geoms)))
        bottom = min(list(map(lambda x: np.min(x.boundary.xy[1]), cropped.geoms)))
        top = max(list(map(lambda x: np.max(x.boundary.xy[1]), cropped.geoms)))

        plt.plot([left, left, right, right, left], [bottom, top, top, bottom , bottom])
        for poly in cropped.geoms:
            plt.plot(poly.boundary.xy[0], poly.boundary.xy[1])
    
    
    # for cropped_poly in cropped:
    #     if len(cropped_poly) >= 2:
    #         xs2, ys2 = zip(*cropped_poly)
    #         plt.plot(xs2, ys2)
    #         xs2, ys2 = zip(*filter((lambda x: x[0] == 0 or x[0] == 128 or x[1] == 0 or x[1] == 128), cropped_poly))
    #         plt.plot(xs2, ys2, 'bo')
    plt.show()

draw_one()
# draw_loop(8, 5)