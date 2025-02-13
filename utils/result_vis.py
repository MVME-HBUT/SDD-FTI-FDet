import numpy as np
import cv2
import os
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
#from google.colab.patches import cv2_imshow

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import sys
sys.path.insert(0,'/home/SDD')
from utils.build import build_distillator_configs
import models.distillator
import models.adapters

# from bmaskrcnn import add_boundary_preserving_config
from detectron2.config import CfgNode as CN
def add_boundary_preserving_config(cfg):

    cfg.MODEL.BOUNDARY_MASK_HEAD = CN()
    cfg.MODEL.BOUNDARY_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.BOUNDARY_MASK_HEAD.IN_FEATURES = ("p2",)
    cfg.MODEL.BOUNDARY_MASK_HEAD.NUM_CONV = 2


im_folder= '/ACH_HALF/val2017'
save_folder = 'result_vis/predict_result_ACH'

for im_file in os.listdir(im_folder):
	im = cv2.imread(os.path.join(im_folder, im_file))

	save_result_path = os.path.join(save_folder, im_file)

	height = im.shape[0]
	width = im.shape[1]
	dpi = 500

	cfg = get_cfg()
	add_boundary_preserving_config(cfg)
	cfg = build_distillator_configs(cfg)
	# ACH
	cfg.merge_from_file('config.yaml')
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
	cfg.MODEL.WEIGHTS = 'model_0017999.pth'
	predictor = DefaultPredictor(cfg)
	outputs = predictor(im)

	pred_classes = outputs["instances"].pred_classes
	pred_boxes = outputs["instances"].pred_boxes

	#在原图上画出检测结果
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu")) #根据预测结果的实例信息

	plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
	plt.axis('off')
	plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
	plt.imshow(v.get_image())
	# plt.savefig(save_result_path)
	try: plt.savefig(save_result_path) #保存结果
	except: continue
	plt.close()		