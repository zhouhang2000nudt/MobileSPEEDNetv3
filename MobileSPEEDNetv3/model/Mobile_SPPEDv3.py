import torch
import torch.nn as nn

from typing import List, Union
from torch import Tensor

from .block import SPPF, FPNPAN, RepECPHead, Conv2dNormActivation, DCNv2, TriFPN, TriFPNAtt, SCGHead
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from rich import print

class Mobile_SPEEDv3(nn.Module):
    def __init__(self, config: dict):
        super(Mobile_SPEEDv3, self).__init__()
        
        self.expand2rgb_conv = Conv2dNormActivation(1, 3, kernel_size=3, stride=1, padding=1, activation_layer=nn.Mish, bias=True)
        
        if config["backbone"] == "mobilenet_v3_large":
            if config["pretrained"]:
                self.features = mobilenet_v3_large(weights = MobileNet_V3_Large_Weights.DEFAULT).features[:-1]
            else:
                self.features = mobilenet_v3_large().features[:-1]
            self.stage = [7, 13]
            SPPF_in_channels = 160
            SPPF_out_channels = 160
            neck_in_channels = [40, 112, SPPF_out_channels]
            neck_out_channels = [40, 112, 160]
            # 修改激活函数
            for module in self.features.modules():
                if isinstance(module, Conv2dNormActivation):
                    if len(module) == 3:
                        module[2] = nn.Mish()
        
        # self.SPPF_p5 = SPPF(in_channels=SPPF_in_channels,
        #                     out_channels=SPPF_out_channels)
        
        if config["neck"] == "FPNPAN":
            self.neck = FPNPAN(in_channels=neck_in_channels)
        elif config["neck"] == "TriFPN":
            self.stage = [4, 7, 13]
            neck_in_channels = [24, 40, 112, 160]
            self.neck = TriFPN(in_channels=neck_in_channels)
        elif config["neck"] == "TriFPNAtt":
            self.stage = [4, 7, 13]
            neck_in_channels = [24, 40, 112, 160]
            self.neck = TriFPNAtt(in_channels=neck_in_channels)
        else:
            self.stage = [7, 13]
            # self.neck = nn.Identity()
        
        
        # self.head = RepECPHead(in_channels=neck_out_channels,
        #                         pool_size=config["pool_size"],
        #                         pos_dim=config["pos_dim"],
        #                         yaw_dim=int(360 // config["stride"] + 1 + 2 * config["n"]),
        #                         pitch_dim=int(180 // config["stride"] + 1 + 2 * config["n"]),
        #                         roll_dim=int(360 // config["stride"] + 1 + 2 * config["n"]))
        self.head = SCGHead(in_channels=neck_out_channels,
                            pool_size=config["pool_size"],
                            pos_dim=config["pos_dim"],
                            yaw_dim=int(360 // config["stride"] + 1 + 2 * config["n"]),
                            pitch_dim=int(180 // config["stride"] + 1 + 2 * config["n"]),
                            roll_dim=int(360 // config["stride"] + 1 + 2 * config["n"]))
        
    
    def forward(self, x: Tensor):
        x = self.expand2rgb_conv(x)
        features = [self.features[:self.stage[0]](x)]
        for i in range(len(self.stage)-1):
            features.append(self.features[self.stage[i]:self.stage[i+1]](features[-1]))
        features.append(self.features[self.stage[-1]:](features[-1]))
        
        # features[-1] = self.SPPF_p5(features[-1])
        
        # features = self.neck(features)
        
        pos, yaw, pitch, roll = self.head(features)
        return pos, yaw, pitch, roll