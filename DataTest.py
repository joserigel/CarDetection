import os
import shutil
import random

path = "./bdd100k/images/10k/train/"
lane_paths = "./bdd100k/labels/lane/"
instance_paths = "./bdd100k/labels/ins_seg/"
semantic_paths = "./bdd100k/labels/sem_seg/"
items_count = len(os.listdir(instance_paths + "colormaps/train/"))

# Copy 5 images and its labels to DataSandbox folder
def get_sample(count, lane_color = False, lane_mask = False, 
               instance_color = False, instance_bitmasks = False,
               semantic_color = False, semantic_bitmasks = False,
                attempt_max = 100):
    if not lane_color and not lane_mask and not instance_color and not instance_bitmasks and not semantic_color and not semantic_bitmasks:
        raise Exception("at least one label type must be selected")
        
    for i in range(count):
        attempt = 0
        while True:
            idx = random.randint(0, items_count)
            name = str(os.listdir(path)[idx])

            train_path = path + name
            lane_color_path = lane_paths + "colormaps/train/" + name[:-4] + ".png"
            lane_mask_path = lane_paths + "masks/train/" + name[:-4] + ".png"
            segmentation_color_path = instance_paths + "colormaps/train/" +  name[:-4] + ".png"
            segmentation_mask_path = instance_paths + "bitmasks/train/" + name[:-4] + ".png"
            semantic_color_path = semantic_paths + "colormaps/train/" + name[:-4] + ".png"
            semantic_mask_path = semantic_paths + "masks/train/" + name[:-4] + ".png"

            available = []
            available.append(os.path.isfile(train_path))

            if lane_color:
                available.append(os.path.isfile(lane_color_path))
            if lane_mask:
                available.append(os.path.isfile(lane_mask_path))
            if instance_color:
                available.append(os.path.isfile(segmentation_color_path))
            if instance_bitmasks:
                available.append(os.path.isfile(segmentation_mask_path))
            if semantic_color:
                available.append(os.path.isfile(semantic_color_path))
            if semantic_bitmasks:
                available.append(os.path.isfile(semantic_mask_path))

            if False not in available:
                shutil.copyfile(train_path, "DataSandbox/train/" + name)
                
                if lane_color:
                    shutil.copyfile(lane_color_path, "DataSandbox/labels/lane/colormaps/" + name[:-4] + ".png")
                if lane_mask:
                    shutil.copyfile(lane_mask_path, "DataSandbox/labels/lane/masks/" + name[:-4] + ".png")
                if instance_color:
                    shutil.copyfile(segmentation_color_path, "DataSandbox/labels/ins_seg/bitmasks/" + name[:-4] + ".png")
                if instance_bitmasks:
                    shutil.copyfile(segmentation_mask_path, "DataSandbox/labels/ins_seg/colormaps/" + name[:-4] + ".png")
                if semantic_color:
                    shutil.copyfile(semantic_color_path, "DataSandbox/labels/sem_seg/colormaps/" + name[:-4] + ".png")
                if semantic_bitmasks:
                    shutil.copyfile(semantic_mask_path, "DataSandbox/labels/sem_seg/masks/" + name[:-4] + ".png")
                break
            
            attempt += 1
            if attempt >= attempt_max:
                raise Exception(f"Iteratation exceeds max! {attempt_max}<={attempt}")
            
get_sample(5, semantic_color=True)

    
