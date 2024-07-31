# Spatial-Wise Dynamic Distillation for MLP-Like Efficient Visual Fault Detection of Freight Trains
This is the official implementation of [Spatial-Wise Dynamic Distillation for MLP-Like Efficient Visual Fault Detection of Freight Trains](https://ieeexplore.ieee.org/abstract/document/10391271). 
# Table of Contents
* [Introduction](#jump1)
* [Installation](#jump2)
* [Usage](#jump3)
* [Citation](#jump4) 
***

# <span id="jump1">Introduction</span>
<div align="center"><img decoding="async" src="Framework.png" width="100%"/> </div>

**Fig. 2.** Overview of the proposed MLP-like spatial-wise dynamic distillation method. We adopt a novel dynamic teacher architecture comprising three modules: label encoder, appearance encoder, and feature adaptive interaction.  The dynamic teacher enables joint teacher-student training, which generates instructional representations from ground truth annotations and feature pyramids during the training stage.

# <span id="jump2">Installation</span>
1、First install Detectron2 following the official guide: [INSTALL.md](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

2、Then build SDD-FTI-FDet with:
```
git clone https://github.com/MVME-HBUT/SDD-FTI-FDet.git
cd SDD-FTI-FDet
```
3、Create a conda virtual environment and activate it:
```
conda create -n SDD python=3.7 -y
conda activate SDD
```


# <span id="jump3">Usage</span>
## Prepare your own datasets
```
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


register_coco_instances("ACH_HALF_train", {},
                        r"/home/yzhang/LGD/datasets/zy_dataset_V3/ACH_HALF/annotations/instances_train2017.json", 
                        r"/home/yzhang/LGD/datasets/zy_dataset_V3/ACH_HALF/train2017")
register_coco_instances("ACH_HALF_test", {},
                        r"/home/yzhang/LGD/datasets/zy_dataset_V3/ACH_HALF/annotations/instances_val2017.json", 
                        r"/home/yzhang/LGD/datasets/zy_dataset_V3/ACH_HALF/val2017")

MetadataCatalog.get("ACH_HALF").thing_classes = ["target", 'fault']
```

## Train
```
python3 train.py --config-file configs/Distillation/SDD.yaml --num-gpus 1
```
## Evaluation
```
python3 train.py --eval-only --config-file configs/Distillation/SDD.yaml --num-gpus 1
```
## Visualization
```
python3 SDD-FTI-FDet/utils/result_vis.py
```










# <span id="jump4">Citation</span>
If you find this repository useful in your research, please consider citing:
```
@ARTICLE{10391271,
  author={Zhang, Yang and Pan, Huilin and Li, Mingying and Wang, An and Zhou, Yang and Ren, Hongliang},
  journal={IEEE Transactions on Industrial Electronics}, 
  title={Spatial-Wise Dynamic Distillation for MLP-Like Efficient Visual Fault Detection of Freight Trains}, 
  year={2024},
  volume={71},
  number={10},
  pages={13168-13177},
  keywords={Fault detection;Detectors;Feature extraction;Training;Visualization;Task analysis;Computational modeling;Dynamic distillation;fault detection;freight train images;multilayer perceptron (MLP)},
  doi={10.1109/TIE.2023.3344837}}
```
