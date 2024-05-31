import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, Conv2dNormActivation, DCNv2
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()
        
        if config["pretrained"]:
            self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features[:-1]
        else:
            self.features = mobilenet_v3_large().features[:-1]
        
        for deform_layer in config["deform_layers"]:
            InvertedResidual = self.features[deform_layer]
            in_channels = InvertedResidual.block[0][0].in_channels
            out_channels = InvertedResidual.block[-1][0].out_channels
            kernel_size = InvertedResidual.block[1][0].kernel_size
            stride = InvertedResidual.block[1][0].stride
            self.features[deform_layer] = DCNv2(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size[0],
                stride=stride[0]
            )
        
        SPPF_in_channels = 160
        SPPF_out_channels = 160
        self.SPPF = SPPF(in_channels=SPPF_in_channels,
                         out_channels=SPPF_out_channels)
        
        neck_in_channels = [40, 112, SPPF_out_channels]
        neck_out_channels = neck_in_channels
        self.FPNPAN = FPNPAN(in_channels=neck_in_channels)
        
        self.RepECPHead = RepECPHead(in_channels=neck_out_channels, 
                                     expand_ratio=config["expand_ratio"],
                                     pool_size=config["pool_size"],
                                     pos_dim=config["pos_dim"],
                                     yaw_dim=int(360 // config["stride"] + 1 + 2 * config["neighbour"]),
                                     pitch_dim=int(180 // config["stride"] + 1 + 2 * config["neighbour"]),
                                     roll_dim=int(360 // config["stride"] + 1 + 2 * config["neighbour"]))
        
    def forward(self, x: Tensor):
        p3 = self.features[:7](x)
        p4 = self.features[7:13](p3)
        p5 = self.features[13:](p4)
        
        p5 = self.SPPF(p5)
        
        p3, p4, p5 = self.FPNPAN([p3, p4, p5])
        
        pos, yaw, pitch, roll = self.RepECPHead([p3, p4, p5])
        return pos, yaw, pitch, roll

    def switch_repvggplus_to_deploy(self):
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        if hasattr(self, 'stage1_aux'):
            self.__delattr__('stage1_aux')
        if hasattr(self, 'stage2_aux'):
            self.__delattr__('stage2_aux')
        if hasattr(self, 'stage3_first_aux'):
            self.__delattr__('stage3_first_aux')
        self.deploy = True