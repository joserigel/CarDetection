import os
import shutil
import random

path = "./bdd100k/images/10k/train/"
label_paths = "./bdd100k/labels/lane/"
items_count = len(os.listdir(path))

# Copy 5 images and its labels to DataSandbox folder
for i in range(5):    
    while True:
        idx = random.randint(0, items_count)
        name = str(os.listdir(path)[idx])

        train_path = path + name
        colormap_path = label_paths + "colormaps/train/" + name[:-4] + ".png"
        mask_path = label_paths + "masks/train/" + name[:-4] + ".png"

        if os.path.isfile(train_path) and os.path.isfile(colormap_path) and os.path.isfile(mask_path):
            shutil.copyfile(train_path, "DataSandbox/train/" + name)
            shutil.copyfile(colormap_path, "DataSandbox/labels/colormaps/" + name[:-4] + ".png")
            shutil.copyfile(mask_path, "DataSandbox/labels/masks/" + name[:-4] + ".png")
            break

    
