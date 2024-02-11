# Lane Detection
This project is mainly focused on Distinct grid based classification of roads using the [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) dataset.

## Setup

To setup environment, it is advisable to use a python environment. CUDA compatible device is better suited but not necessary

### List of Environment variables required
```env
MONGO_USER=root
MONGO_PASS=password
BDD100K_DIR=./bdd100k
```

### List of Dataset required inside the bdd100k folder
- [10K Images](https://doc.bdd100k.com/download.html#id1)
- [Semantic Segmentation](https://doc.bdd100k.com/download.html#semantic-segmentation)
- [Instance Segmentation](https://doc.bdd100k.com/download.html#instance-segmentation)
- [Lane Marking](https://doc.bdd100k.com/download.html#lane-marking)
<details>
<summary>Directory Structure</summary>

```
.
├── bdd100k
├── bdd100k
    ├── images
    │   └── 10k
    │       ├── test
    │       ├── train
    │       └── val
    └── labels
        ├── ins_seg
        │   ├── bitmasks
        │   │   ├── train
        │   │   └── val
        │   ├── colormaps
        │   │   ├── train
        │   │   └── val
        │   ├── polygons
        │   └── rles
        ├── lane
        │   ├── colormaps
        │   │   ├── train
        │   │   └── val
        │   ├── masks
        │   │   ├── train
        │   │   └── val
        │   └── polygons
        └── sem_seg
            ├── colormaps
            │   ├── train
            │   └── val
            ├── masks
            │   ├── train
            │   └── val
            ├── polygons
            └── rles
├── DataReader.py
├── DataSandbox
├── DataVisualizer.py
├── docker-compose.yml
├── env
├── example.json
├── README.md
├── requirements.txt
└── Segmentation.py
```

</details>

### Running on your local machine

1. install all from `requirements.txt`
```bash
pip install -r requirements.txt
```
2. run `docker-compose up`
```bash
docker-compose up
```
3. run `DataReader.py` __ONCE__ (do not run it multiple times, otherwise data would be duplicated on your MongoDB instance)
```bash
python DataReader.py
```

## Notes 

### February 11 2024
I converted this project from last year's attempt that wasn't succesful using the YOLO algorithm to also detect cars, so `Segmentation.py` is not working

<details>
    <summary>Classes Count</summary>
    
```json
{
    "car"                : 71875,
    "pole"               : 60367,
    "vegetation"         : 40929,
    "building"           : 36612,
    "static"             : 31637,
    "traffic sign"       : 21356,
    "sky"                : 18988,
    "street light"       : 16335,
    "traffic light"      : 15123,
    "road"               : 11910,
    "sidewalk"           : 11748,
    "person"             : 9963,
    "terrain"            : 6807,
    "ego vehicle"        : 6757,
    "fence"              : 5046,
    "dynamic"            : 4817,
    "guard rail"         : 4717,
    "truck"              : 3745,
    "traffic sign frame" : 3130,
    "bridge"             : 3089,
    "billboard"          : 2999,
    "banner"             : 2859,
    "ground"             : 2650,
    "wall"               : 2649,
    "lane divider"       : 1640,
    "bus"                : 1638,
    "parking"            : 1129,
    "traffic cone"       : 971,
    "bicycle"            : 830,
    "traffic device"     : 617,
    "polegroup"          : 617,
    "rider"              : 471,
    "parking sign"       : 444,
    "motorcycle"         : 396,
    "caravan"            : 359,
    "unlabeled"          : 339,
    "fire hydrant"       : 265,
    "trash can"          : 237,
    "tunnel"             : 145,
    "bus stop"           : 131,
    "mail box"           : 122,
    "rail track"         : 111,
    "trailer"            : 108,
    "garage"             : 106,
    "train"              : 65
}
```

</details>