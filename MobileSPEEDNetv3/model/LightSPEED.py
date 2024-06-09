# type
from typing import List, Tuple, Union

import torch
from torch import nn

# baseblock
from timm.layers.conv_bn_act import ConvBnAct
from .LightSPEEDBlock import *

# n
BackboneWidths = [16, 64, 64, 64, 128, 128, 160, 160, 256, 256]
BackboneDepths = [ 1,  1,  1,  1,   1,   2,   1,   2,   1,   1]



class LightSPEED(nn.Module):
    def __init__(self, config):
        super(LightSPEED, self).__init__()
        
        if config["self_supervised"]:
            self.supervised = True
        else:
            self.supervised = False
        
        self.features = nn.Sequential(
            DownSample3x3(3, BackboneWidths[0], kernel_size=3, stride=2, act=nn.SiLU),
            C2f(BackboneWidths[0], BackboneWidths[1], n=BackboneDepths[2], shortcut=True),
            DownSample3x3(BackboneWidths[1], BackboneWidths[2], kernel_size=3, stride=2, act=nn.SiLU),
            C2f(BackboneWidths[2], BackboneWidths[3], n=BackboneDepths[2], shortcut=True),
            DownSample3x3(BackboneWidths[3], BackboneWidths[4], kernel_size=3, stride=2, act=nn.SiLU),
            C2f(BackboneWidths[4], BackboneWidths[5], n=BackboneDepths[4], shortcut=True),
            DownSample3x3(BackboneWidths[5], BackboneWidths[6], kernel_size=3, stride=2, act=nn.SiLU),
            C2f(BackboneWidths[6], BackboneWidths[7], n=BackboneDepths[6], shortcut=True),
            DownSample3x3(BackboneWidths[7], BackboneWidths[8], kernel_size=3, stride=2, act=nn.SiLU),
            C2f(BackboneWidths[8], BackboneWidths[9], n=BackboneDepths[8], shortcut=True),
        )
        
        self.SPPF = SPPF(BackboneWidths[8], 256)
        
        neck_in_channels = [BackboneWidths[4], BackboneWidths[6], BackboneWidths[8]]
        neck_out_channels = [BackboneWidths[4], BackboneWidths[6], BackboneWidths[8]]
        
        self.neck = BiFPN(neck_in_channels, neck_out_channels)
        
        self.PRC = PRC(config["pool_size"])
        
        self.Head = Head(
            in_features=sum([int(neck_out_channels[i] * config["pool_size"][i]**2) for i in range(3)]),
            pos_dim=config["pos_dim"],
            yaw_dim=int(360 // config["stride"] + 1 + 2 * config["n"]),
            pitch_dim=int(180 // config["stride"] + 1 + 2 * config["n"]),
            roll_dim=int(360 // config["stride"] + 1 + 2 * config["n"])
        )

    def forward_once(self, x):
        p3 = self.features[:6](x)
        p4 = self.features[6:8](p3)
        p5 = self.features[8:](p4)
        p5 = self.SPPF(p5)
        
        p3, p4, p5 = self.neck([p3, p4, p5])
        
        features = self.PRC([p3, p4, p5])
        
        pos, yaw, pitch, roll = self.Head(features)
        
        return pos, yaw, pitch, roll
    
    def forward(self, x1, x2=None):
        if self.supervised:
            return [self.forward_once(x1), self.forward_once(x2)]
        return self.forward_once(x1)