
MaskRCNN expects dataset in this format:

dataset
├── train
│   ├── images
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── annotations
│       ├── img_0001.json
│       ├── img_0002.json
│       └── ...
└── val
    ├── images
    │   ├── img_0001.jpg
    │   ├── img_0002.jpg
    │   └── ...
    └── annotations
        ├── img_0001.json
        ├── img_0002.json
        └── ...
