from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from timm.layers.conv_bn_act import ConvBnAct

from fightingcv_attention.rep.repvgg import RepBlock
# =======================block=========================

class DownSample3x3(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride, act):
        super(DownSample3x3, self).__init__()
        
        self.conv = ConvBnAct(c1, c2, kernel_size=kernel_size, stride=stride, act_layer=act)
    
    def forward(self, x):
        return self.conv(x)
    

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=[3, 3], e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, c_, kernel_size=k[0], stride=1, act_layer=nn.SiLU)
        self.cv2 = ConvBnAct(c_, c2, kernel_size=k[1], stride=1, groups=g, act_layer=nn.SiLU)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super(C2f, self).__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvBnAct(c1, 2 * self.c, kernel_size=1, stride=1, act_layer=nn.Mish)
        self.cv2 = ConvBnAct((2 + n) * self.c, c2, kernel_size=1, act_layer=nn.Mish)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=[[3, 3], [3, 3]], e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.conv1 = ConvBnAct(in_channels, c_, kernel_size=1, stride=1)
        self.conv2 = ConvBnAct(c_*4, out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y5x5 = self.pool(x)
        y9x9 = self.pool(y5x5)
        y13x13 = self.pool(y9x9) 
        return self.conv2(torch.cat([x, y5x5, y9x9, y13x13], dim=1))

# =======================neck=========================
class BiFPN(nn.Module):
    def __init__(self, in_channels: List[int], out_channels: List[int]):
        super(BiFPN, self).__init__()
        
        fused_channel_p45 = in_channels[1] + in_channels[2]
        fused_channel_p34 = in_channels[0] + in_channels[1]
        
        # 上采样通路
        self.p4_fuseconv_up = C2f(fused_channel_p45, in_channels[1])
        self.p3_fuseconv_up = C2f(fused_channel_p34, in_channels[0])
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2, act_layer=nn.SiLU)
        
        self.p4_fuseconv_down = C2f(fused_channel_p34 + in_channels[1], in_channels[1])
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, act_layer=nn.SiLU)
        
        self.p5_fuseconv_down = C2f(fused_channel_p45, in_channels[2])
    
    def forward(self, x):
        p3, p4, p5 = x

        # 上采样通路
        p4_fused_up = self.p4_fuseconv_up(torch.cat([F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=True), p4], dim=1))
        p3_fused_up = self.p3_fuseconv_up(torch.cat([F.interpolate(p4_fused_up, size=p3.shape[2:], mode="bilinear", align_corners=True), p3], dim=1))
        
        # 下采样通路
        p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up, p4], dim=1))
        p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1))
        
        return p3_fused_up, p4_fused_down, p5_fused_down

# =======================head=========================

class PRC(nn.Module):
    def __init__(self, pool_size: List[int]):
        super(PRC, self).__init__()
        
        self.Pool_p3 = nn.AdaptiveAvgPool2d((pool_size[0], pool_size[0]))
        self.Pool_p4 = nn.AdaptiveAvgPool2d((pool_size[1], pool_size[1]))
        self.Pool_p5 = nn.AdaptiveAvgPool2d((pool_size[2], pool_size[2]))
    
    def forward(self, x):
        p3, p4, p5 = x
        
        p3 = self.Pool_p3(p3)
        p4 = self.Pool_p4(p4)
        p5 = self.Pool_p5(p5)
        
        return torch.cat([p3.reshape(p3.size(0), -1), p4.reshape(p4.size(0), -1), p5.reshape(p5.size(0), -1)], dim=1)


class Head(nn.Module):
    def __init__(self, in_features: int, pos_dim: int, yaw_dim: int, pitch_dim: int, roll_dim: int):
        super(Head, self).__init__()
        feature_hide = in_features
        self.fc = nn.Sequential(
            nn.Linear(in_features, feature_hide),
            nn.SiLU(inplace=True),
        )
        self.pos_hide_features = int(feature_hide * 0.25)
        self.ori_hide_features = feature_hide - self.pos_hide_features
        self.pos_fc = nn.Sequential(
            nn.Linear(self.pos_hide_features, pos_dim),
        )
        self.yaw_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, yaw_dim),
        )
        self.pitch_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, pitch_dim),
        )
        self.roll_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, roll_dim),
        )
    
    def forward(self, x):
        x = self.fc(x)
        pos_feature, ori_feature = torch.split(x, [self.pos_hide_features, self.ori_hide_features], dim=1)
        pos = self.pos_fc(pos_feature)
        yaw = F.softmax(self.yaw_fc(ori_feature).type(torch.float32), dim=1)
        pitch = F.softmax(self.pitch_fc(ori_feature).type(torch.float32), dim=1)
        roll = F.softmax(self.roll_fc(ori_feature).type(torch.float32), dim=1)
        return pos, yaw, pitch, roll
