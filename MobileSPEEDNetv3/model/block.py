import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Union
from .RepVGG import RepVGGplusBlock
from functools import partial
from torchvision.ops import Conv2dNormActivation
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

from timm.layers.conv_bn_act import ConvBnAct
from timm.models._efficientnet_blocks import InvertedResidual

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

# ==================== block ===================
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, groups=1, dilation=1, act=True, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        padding = autopad(kernel_size, padding, dilation)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()



# ================== block end =================


# ==================== tail ====================

class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.conv1 = Conv2dNormActivation(in_channels=in_channels, out_channels=c_, kernel_size=1, stride=1)
        self.conv2 = Conv2dNormActivation(in_channels=c_*4, out_channels=out_channels, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    
    def forward(self, x):
        x = self.conv1(x)
        y5x5 = self.pool(x)
        y9x9 = self.pool(y5x5)
        y13x13 = self.pool(y9x9) 
        return self.conv2(torch.cat([x, y5x5, y9x9, y13x13], dim=1))

# ================== tail end ==================





# ==================== neck ====================

class FPNPAN(nn.Module):
    def __init__(self, in_channels: List[int], fuse_mode: str = "cat"):
        super(FPNPAN, self).__init__()
        self.UpSample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse_mode = fuse_mode
        
        if fuse_mode == "cat":
            fused_channel_p45 = in_channels[1] + in_channels[2]
            fused_channel_p34 = in_channels[0] + in_channels[1]
        elif fuse_mode == "add":
            raise NotImplementedError("Not implemented yet")
            pass
        
        # 上采样通路
        self.p4_fuseconv_up = RepVGGplusBlock(in_channels=fused_channel_p45, out_channels=in_channels[1], kernel_size=3, stride=1, padding=1)
        self.p3_fuseconv_up = RepVGGplusBlock(in_channels=fused_channel_p34, out_channels=in_channels[0], kernel_size=3, stride=1, padding=1)
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2)
        
        self.p4_fuseconv_down = RepVGGplusBlock(in_channels=fused_channel_p34, out_channels=in_channels[1], kernel_size=3, stride=1, padding=1)
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2)
        
        self.p5_fuseconv_down = RepVGGplusBlock(in_channels=fused_channel_p45, out_channels=in_channels[2], kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        p3, p4, p5 = x      # in: 40, 60, 96; p4: 112, 30, 48; p5: 160, 15, 24
        
        # 上采样通路
        if self.fuse_mode == "cat":
            p4_fused_up = self.p4_fuseconv_up(torch.cat([F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=True), p4], dim=1)) # 112, 30, 48
        
        if self.fuse_mode == "cat":
            p3_fused_up = self.p3_fuseconv_up(torch.cat([F.interpolate(p4_fused_up, size=p3.shape[2:], mode="bilinear", align_corners=True), p3], dim=1)) # 40, 60, 96    out
        
        # 下采样通路
        if self.fuse_mode == "cat":
            p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up], dim=1)) # 112, 30, 48    out
        
        if self.fuse_mode == "cat":
            p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1)) # 160, 15, 24    out
        
        return p3_fused_up, p4_fused_down, p5_fused_down


# ================== neck end ==================





# ==================== head ====================

class ECP(nn.Module):
    def __init__(self, in_channels: List[int], expand_ratio: Union[int, List[float]], pool_size: List[int]):
        super(ECP, self).__init__()
        self.ECP_p3 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[0],
                out_channels=int(in_channels[0] * expand_ratio[0]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[0], pool_size[0])),
        )
        self.ECP_p4 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[1],
                out_channels=int(in_channels[1] * expand_ratio[1]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[1], pool_size[1])),
        )
        self.ECP_p5 = nn.Sequential(
            RepVGGplusBlock(
                in_channels=in_channels[2],
                out_channels=int(in_channels[2] * expand_ratio[2]),
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.AdaptiveAvgPool2d((pool_size[2], pool_size[2])),
        )
    
    def forward(self, x):
        p3, p4, p5 = x
        p3 = self.ECP_p3(p3)
        p4 = self.ECP_p4(p4)
        p5 = self.ECP_p5(p5)
        return p3, p4, p5


class RSC(nn.Module):
    def __init__(self):
        super(RSC, self).__init__()
    
    def forward(self, x):
        p3, p4, p5 = x
        return torch.cat([p3.reshape(p3.size(0), -1), p4.reshape(p4.size(0), -1), p5.reshape(p5.size(0), -1)], dim=1)


class Head(nn.Module):
    def __init__(self, in_features: int, pos_dim: int, yaw_dim: int, pitch_dim: int, roll_dim: int):
        super(Head, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Mish(inplace=True),
        )
        self.pos_hide_features = int(in_features * 0.25)
        self.ori_hide_features = in_features - self.pos_hide_features
        self.pos_fc = nn.Sequential(
            nn.Linear(self.pos_hide_features, pos_dim),
        )
        self.yaw_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, yaw_dim),
            nn.Softmax(dim=1),
        )
        self.pitch_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, pitch_dim),
            nn.Softmax(dim=1),
        )
        self.roll_fc = nn.Sequential(
            nn.Linear(self.ori_hide_features, roll_dim),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        x = self.fc(x)
        pos_feature, ori_feature = torch.split(x, [self.pos_hide_features, self.ori_hide_features], dim=1)
        pos = self.pos_fc(pos_feature)
        yaw = self.yaw_fc(ori_feature)
        pitch = self.pitch_fc(ori_feature)
        roll = self.roll_fc(ori_feature)
        return pos, yaw, pitch, roll

class RepECPHead(nn.Sequential):
    def __init__(self, in_channels: List[int], expand_ratio: Union[int, List[float]], pool_size: List[int], pos_dim: int, yaw_dim: int, pitch_dim: int, roll_dim: int):
        if isinstance(expand_ratio, int):
            expand_ratio = [expand_ratio] * 3
        super(RepECPHead, self).__init__(
            ECP(in_channels, expand_ratio, pool_size),
            RSC(),
            Head(sum([int(in_channels[i] * expand_ratio[i] * pool_size[i]**2) for i in range(3)]), pos_dim, yaw_dim, pitch_dim, roll_dim),
        )


# ================== head end ==================



#https://oneapi.xty.app/v1/chat/completions
#sk-ReUPocItmhmGioyMC63d6e0eBf584e3a812dF4E97696Aa92