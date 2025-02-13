# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from SwinT_detectron2 (https://github.com/xiaohu2015/SwinT_detectron2)
# Copyright (c) 2021 Hu Ye.
# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN

def add_swint_config(cfg):
    # SwinT backbone
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 96
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWINT.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWINT.WINDOW_SIZE = 7
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"


def add_asmlp_config(cfg):
    # ASMLP backbone
    cfg.MODEL.ASMLP = CN()
    cfg.MODEL.ASMLP.EMBED_DIM = 96
    cfg.MODEL.ASMLP.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.ASMLP.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.ASMLP.SHIFT_SIZE = 5
    cfg.MODEL.ASMLP.MLP_RATIO = 4
    cfg.MODEL.ASMLP.DROP_PATH_RATE = 0.1
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"
