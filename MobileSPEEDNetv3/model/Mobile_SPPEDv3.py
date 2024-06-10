import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, Conv2dNormActivation, DCNv2
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from rich import print

class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()
        
        if config["backbone"] == "mobilenet_v3_large":
            if config["pretrained"]:
                self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features[:-1]
            else:
                self.features = mobilenet_v3_large().features[:-1]
            self.stage = [7, 13]
            SPPF_in_channels = [40, 112, 160]
            SPPF_out_channels = [40, 112, 160]
            neck_in_channels = SPPF_out_channels
            neck_out_channels = neck_in_channels
        elif config["backbone"] == "EfficientNet":
            if config["pretrained"]:
                self.features = efficientnet_b0(weights = EfficientNet_B0_Weights.DEFAULT).features[:-1]
            else:
                self.features = efficientnet_b0().features[:-1]
            self.stage = [4, 6]
            SPPF_in_channels = [40, 112, 320]
            SPPF_out_channels = [40, 112, 320]
            neck_in_channels = SPPF_out_channels
            neck_out_channels = neck_in_channels
            for name, module in self.features.named_modules():
                if "stochastic_depth" in name:
                    path = name.split(".")
                    self.features[int(path[0])][int(path[1])].stochastic_depth = nn.Identity()
                    
        
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
        
        self.SPPF_p5 = SPPF(in_channels=SPPF_in_channels[-1],
                            out_channels=SPPF_out_channels[-1])
        
        self.FPNPAN = FPNPAN(in_channels=neck_in_channels)
        
        self.RepECPHead = RepECPHead(in_channels=neck_out_channels, 
                                     expand_ratio=config["expand_ratio"],
                                     pool_size=config["pool_size"],
                                     pos_dim=config["pos_dim"],
                                     yaw_dim=int(360 // config["stride"] + 1 + 2 * config["n"]),
                                     pitch_dim=int(180 // config["stride"] + 1 + 2 * config["n"]),
                                     roll_dim=int(360 // config["stride"] + 1 + 2 * config["n"]))
        
    def forward(self, x: Tensor):
        p3 = self.features[:self.stage[0]](x)
        p4 = self.features[self.stage[0]:self.stage[1]](p3)
        p5 = self.features[self.stage[1]:](p4)
        
        p5 = self.SPPF_p5(p5)
        
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