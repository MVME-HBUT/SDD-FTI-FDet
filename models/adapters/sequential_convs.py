# ------------------------------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. All Rights Reserved.
# ------------------------------------------------------------------------------
from .build import ADAPTERS_REGISTRY
from torch import nn

@ADAPTERS_REGISTRY.register()
class SequentialConvs(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()  
        self.nr_fpn_channels = cfg.MODEL.FPN.OUT_CHANNELS
        self.adapter =  nn.Sequential(*[nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 1, 1), nn.ReLU(),
                                        nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 1, 1), nn.ReLU(),
                                        nn.Conv2d(self.nr_fpn_channels, self.nr_fpn_channels, 1, 1)]) #conv2d_3*3

    def forward(self, x):
        return self.adapter(x)
