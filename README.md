# Description
This code is the algorithm implementation for article `HQ-Fusion: A High-Quality Feature Fusion Framework for Multispectral Object Detection`.

# Code Information
This code is built upon the RT-DETR codebase. The RT-DETR repository is located at: https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch.

# Visualization display of dataset

### FLIR dataset
![FLIR](hq-fusion/images/flir.jpg)

### LLVIP dataset
![LLVIP](hq-fusion/images/llvip.jpg)

### M3FD dataset
![M3FD](hq-fusion/images/m3fd.jpg)

### VEDAI dataset
![VEDAI](hq-fusion/images/vedai.jpg)

# Directory Structure
```
├── hq-fusion
│   ├── configs                             # Configuration file directory
│   │   ├── dataset                         # Dataset information configuration
│   │   │   └── coco_detection*.yml     
│   │   ├── rtdetr
│   │   │   ├── include
│   │   │   │   ├── dataloader*.yml          # Data loader configuration
│   │   │   │   ├── optimizer*.yml           # Optimizer configuration
│   │   │   │   └── rtdetr_r50vd*.yml        # Network structure configuration
│   │   │   └── rtdetr_r50vd_6x_coco-*.yml  # A file for integrating configuration information, backbone-resnet-50
│   │   └── runtime.yml
│   ├── src                                 # Source code
│   │   ├── core
│   │   ├── data
│   │   ├── misc
│   │   ├── nn
│   │   ├── optim
│   │   ├── solver
│   │   ├── zoo
│   ├── tools
│   │   ├── export_onnx.py
│   │   ├── infer_flir.py
│   │   ├── infer.py
│   │   ├── misc.py
│   │   └── train.py                        # Training script
│   └── requirements.txt                    # List of project dependency packages
└── README.md
```

## Training on a Single GPU: 
```python tools/train.py -c hq-fusion/configs/rtdetr/rtdetr_r50vd_6x_coco.yml```
## Evaluation on a Single GPU:
```python tools/train.py -c hq-fusion/configs/rtdetr/rtdetr_r50vd_6x_coco.yml  -r path/to/checkpoint --test-only```

# Requirements
```
torch==2.0.1
torchvision==0.15.2
pycocotools
PyYAML
scipy
transformers
```