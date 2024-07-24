import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import List, Union
from .RepVGG import RepVGGplusBlock
from torchvision.ops import Conv2dNormActivation
from torchvision.models.efficientnet import FusedMBConv, FusedMBConvConfig
from torchvision.models.efficientnet import MBConv, MBConvConfig
from .LightSPEEDBlock import C2f

from timm.layers.conv_bn_act import ConvBnAct
from timm.models._efficientnet_blocks import InvertedResidual
from fightingcv_attention.attention import SEAttention

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
        self.conv1 = Conv2dNormActivation(in_channels=in_channels, out_channels=c_, kernel_size=1, stride=1, bias=False, activation_layer=nn.Mish)
        self.conv2 = Conv2dNormActivation(in_channels=c_*4, out_channels=out_channels, kernel_size=1, stride=1, bias=False, activation_layer=nn.Mish)
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
        
        fused_channel_p45 = in_channels[1] + in_channels[2]
        fused_channel_p34 = in_channels[0] + in_channels[1]
        
        # 上采样通路
        self.p4_fuseconv_up = C2f(fused_channel_p45, in_channels[1])
        self.p3_fuseconv_up = C2f(fused_channel_p34, in_channels[0])
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        self.p4_fuseconv_down = C2f(fused_channel_p34, in_channels[1])
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        self.p5_fuseconv_down = C2f(fused_channel_p45, in_channels[2])
    
    def forward(self, x):
        p3, p4, p5 = x      # in: 40, 60, 96; p4: 112, 30, 48; p5: 160, 15, 24
        
        # 上采样通路
        p4_fused_up = self.p4_fuseconv_up(torch.cat([F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=True), p4], dim=1)) # 112, 30, 48

        p3_fused_up = self.p3_fuseconv_up(torch.cat([F.interpolate(p4_fused_up, size=p3.shape[2:], mode="bilinear", align_corners=True), p3], dim=1)) # 40, 60, 96    out
        
        # 下采样通路
        p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up], dim=1)) # 112, 30, 48    out
        
        p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1)) # 160, 15, 24    out
        
        return p3_fused_up, p4_fused_down, p5_fused_down


class TriFPN(nn.Module):
    def __init__(self, in_channels: List[int], fuse_mode: str = "cat"):
        super(TriFPN, self).__init__()
        self.UpSample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse_mode = fuse_mode
        
        fused_channel_p345_up = in_channels[1] + in_channels[2] + in_channels[3]
        fused_channel_p234_up = in_channels[0] + in_channels[1] + in_channels[2]
        fused_channel_p45_down = in_channels[2] + in_channels[3]
        fused_channel_p34_down = in_channels[1] + in_channels[2]
        
        # 上采样通路
        self.p3_downconv_up = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, dilation=2, act_layer=nn.Mish)
        self.p5_upconv_up = ConvBnAct(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=1, stride=1, act_layer=nn.Mish)
        # self.p4_fuseconv_up = C2f(fused_channel_p345_up, in_channels[2])
        self.p4_fuseconv_up = InvertedResidual(fused_channel_p345_up, in_channels[2], exp_ratio=8, act_layer=nn.Mish)
        self.p2_downconv_up = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2, dilation=2, act_layer=nn.Mish)
        self.p4_upconv_up = ConvBnAct(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=1, stride=1, act_layer=nn.Mish)
        # self.p3_fuseconv_up = C2f(fused_channel_p234_up, in_channels[1])
        self.p3_fuseconv_up = InvertedResidual(fused_channel_p234_up, in_channels[1], exp_ratio=8, act_layer=nn.Mish)
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        # self.p4_fuseconv_down = C2f(fused_channel_p34_down, in_channels[2])
        self.p4_fuseconv_down = InvertedResidual(fused_channel_p34_down, in_channels[2], exp_ratio=8, act_layer=nn.Mish)
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        # self.p5_fuseconv_down = C2f(fused_channel_p45_down, in_channels[3])
        self.p5_fuseconv_down = InvertedResidual(fused_channel_p45_down, in_channels[3], exp_ratio=8, act_layer=nn.Mish)
    
    def forward(self, x):
        p2, p3, p4, p5 = x      # in: 40, 60, 96; p4: 112, 30, 48; p5: 160, 15, 24
        
        # 上采样通路
        p4_fused_up = self.p4_fuseconv_up(torch.cat([self.p3_downconv_up(p3), p4, F.interpolate(self.p5_upconv_up(p5), size=p4.shape[2:], mode="bilinear", align_corners=True)], dim=1)) # 112, 30, 48

        p3_fused_up = self.p3_fuseconv_up(torch.cat([self.p2_downconv_up(p2), p3, F.interpolate(self.p4_upconv_up(p4_fused_up), size=p3.shape[2:], mode="bilinear", align_corners=True)], dim=1)) # 40, 60, 96    out
        
        # 下采样通路
        p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up], dim=1)) # 112, 30, 48    out
        
        p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1)) # 160, 15, 24    out
        
        return p3_fused_up, p4_fused_down, p5_fused_down


class ChannelWeight(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 4):
        super(ChannelWeight, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.MaxPoll = nn.AdaptiveAvgPool2d(1)
        self.out_channels = out_channels
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, out_channels),
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.AvgPool(x).view(b, c)
        y_max = self.MaxPoll(x).view(b, c)
        weight = F.sigmoid(self.fc(0.5*y_avg+0.5*y_max)).view(b, self.out_channels, 1, 1)
        return weight


class SpatialWeight(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatialWeight, self).__init__()
        self.conv1 = ConvBnAct(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, act_layer=nn.Mish)
        self.conv2 = ConvBnAct(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2, dilation=2, act_layer=nn.Mish)
        self.conv_fuse = ConvBnAct(in_channels=2, out_channels=1, kernel_size=1, stride=1, act_layer=nn.Identity)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        weight = self.conv_fuse(torch.cat([x1, x2], dim=1))
        weight = F.sigmoid(weight)
        return weight


class TriFPNAtt(nn.Module):
    def __init__(self, in_channels: List[int], fuse_mode: str = "cat"):
        super(TriFPNAtt, self).__init__()
        self.UpSample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.fuse_mode = fuse_mode
        
        fused_channel_p345_up = in_channels[1] + in_channels[2] + in_channels[3]
        fused_channel_p234_up = in_channels[0] + in_channels[1] + in_channels[2]
        fused_channel_p45_down = in_channels[2] + in_channels[3]
        fused_channel_p34_down = in_channels[1] + in_channels[2]
        
        # 上采样通路
        self.p32P5_weight = SpatialWeight(in_channels=in_channels[1])
        self.p52p3_weight = ChannelWeight(in_channels=in_channels[3], out_channels=in_channels[1])
        self.p3_downconv_up = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, act_layer=nn.Mish)
        # self.p4_fuseconv_up = C2f(fused_channel_p345_up, in_channels[2])
        self.p4_fuseconv_up = InvertedResidual(fused_channel_p345_up, in_channels[2], exp_ratio=8, act_layer=nn.Mish)
        self.p22p4_weight = SpatialWeight(in_channels=in_channels[0])
        self.p42p2_weight = ChannelWeight(in_channels=in_channels[2], out_channels=in_channels[0])
        self.p2_downconv_up = ConvBnAct(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=2, act_layer=nn.Mish)
        # self.p3_fuseconv_up = C2f(fused_channel_p234_up, in_channels[1])
        self.p3_fuseconv_up = InvertedResidual(fused_channel_p234_up, in_channels[1], exp_ratio=8, act_layer=nn.Mish)
        
        # 下采样通路
        self.p3_downconv_down = ConvBnAct(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        # self.p4_fuseconv_down = C2f(fused_channel_p34_down, in_channels[2])
        self.p4_fuseconv_down = InvertedResidual(fused_channel_p34_down, in_channels[2], exp_ratio=8, act_layer=nn.Mish)
        self.p4_downconv_down = ConvBnAct(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, stride=2, act_layer=nn.Mish)
        
        # self.p5_fuseconv_down = C2f(fused_channel_p45_down, in_channels[3])
        self.p5_fuseconv_down = InvertedResidual(fused_channel_p45_down, in_channels[3], exp_ratio=8, act_layer=nn.Mish)
    
    def forward(self, x):
        p2, p3, p4, p5 = x      # in: 40, 60, 96; p4: 112, 30, 48; p5: 160, 15, 24
        
        # 上采样通路
        p4_fused_up = self.p4_fuseconv_up(torch.cat([self.p52p3_weight(p5)*self.p3_downconv_up(p3), p4, self.p32P5_weight(p3)*F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=True)], dim=1)) # 112, 30, 48

        p3_fused_up = self.p3_fuseconv_up(torch.cat([self.p42p2_weight(p4_fused_up)*self.p2_downconv_up(p2), p3, self.p22p4_weight(p2)*F.interpolate(p4_fused_up, size=p3.shape[2:], mode="bilinear", align_corners=True)], dim=1)) # 40, 60, 96    out
        
        # 下采样通路
        p4_fused_down = self.p4_fuseconv_down(torch.cat([self.p3_downconv_down(p3_fused_up), p4_fused_up], dim=1)) # 112, 30, 48    out
        
        p5_fused_down = self.p5_fuseconv_down(torch.cat([self.p4_downconv_down(p4_fused_down), p5], dim=1)) # 160, 15, 24    out
        
        return p3_fused_up, p4_fused_down, p5_fused_down

# ================== neck end ==================


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x



        


# ==================== head ====================

class ECP(nn.Module):
    def __init__(self, pool_size: List[int]):
        super(ECP, self).__init__()
        self.ECP_p3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size[0], pool_size[0])),
        )
        self.ECP_p4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((pool_size[1], pool_size[1])),
        )
        self.ECP_p5 = nn.Sequential(
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
        self.weight_fc = nn.Sequential(
            nn.Linear(in_features, in_features // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid(),
        )
        self.pos_hide_features = int(in_features * 0.2)
        self.ori_hide_features = in_features - self.pos_hide_features
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, self.ori_hide_features),
            nn.Tanh()
        )
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
        x = self.fc(x) * self.weight_fc(x)
        pos_feature, ori_feature = torch.split(x, [self.pos_hide_features, self.ori_hide_features], dim=1)
        # pos_feature = x[:, :self.pos_hide_features]
        # ori_feature = x
        pos = self.pos_fc(pos_feature)
        pos_embedding = self.pos_embedding(pos)
        ori_feature = ori_feature + pos_embedding
        yaw = self.yaw_fc(ori_feature)
        pitch = self.pitch_fc(ori_feature)
        roll = self.roll_fc(ori_feature)
        return pos, yaw, pitch, roll

class RepECPHead(nn.Sequential):
    def __init__(self, in_channels: List[int], pool_size: List[int], pos_dim: int, yaw_dim: int, pitch_dim: int, roll_dim: int):
        super(RepECPHead, self).__init__(
            ECP(pool_size),
            RSC(),
            Head(sum([int(in_channels[i]* pool_size[i]**2) for i in range(3)]), pos_dim, yaw_dim, pitch_dim, roll_dim),
        )



class SCGHead(nn.Module):
    def __init__(self, in_channels: List[int], pool_size: List[int], pos_dim: int, yaw_dim: int, pitch_dim: int, roll_dim: int, sg: List[int] = [4, 4]):
        super(SCGHead, self).__init__()
        
        self.sg = sg
        self.spatial_pool = nn.AdaptiveAvgPool2d(sg)
        C_ = sum(in_channels)*sg[0]*sg[1]
        g = sg[0]*sg[1]
        self.g = g
        self.group_att_conv1 = nn.Conv2d(in_channels=C_,
                                         out_channels=C_//2,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=g)
        self.act = nn.ReLU(inplace=True)
        self.group_att_conv2 = nn.Conv2d(in_channels=C_//2,
                                         out_channels=C_,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=g)
        self.softmax = nn.Softmax(dim=1)
        self.mlp = MLP(in_features=g, out_features=g, hidden_features=g//2)
        
        self.head = nn.Sequential(
            ECP(pool_size),
            RSC(),
            Head(sum([int(in_channels[i]* pool_size[i]**2) for i in range(3)]), pos_dim, yaw_dim, pitch_dim, roll_dim),
        )
    
    def forward(self, x):
        p3, p4, p5 = x
        B, C, H, W = p3.size()
        # 将p4, p5的特征图resize到p3的大小
        p4 = F.interpolate(p4, size=(H, W), mode="bilinear", align_corners=True)
        p5 = F.interpolate(p5, size=(H, W), mode="bilinear", align_corners=True)
        # cat
        x = torch.cat([p3, p4, p5], dim=1)      # B, C, H, W
        x_ = x.clone()
        B, C, H, W = x.size()
        # spatial group
        x_ = self.spatial_pool(x_)                # B, C, sg[0], sg[1]
        # 展平
        x_ = x_.reshape(B, -1, 1, 1)              # B, C*sg[0]*sg[1], 1, 1
        # group attention
        weight = self.group_att_conv1(x_)             # B, C*sg[0]*sg[1]//2, 1, 1
        weight = self.act(weight)
        weight = self.group_att_conv2(weight)             # B, C*sg[0]*sg[1], 1, 1
        weight = weight.reshape(B, -1, self.sg[0], self.sg[1])              # B, C, sg[0], sg[1]
        weight = self.softmax(weight)                     # B, C, sg[0], sg[1]
        # avg
        avg = torch.mean(weight, dim=1, keepdim=True)               # B, 1, sg[0], sg[1]
        avg = avg.squeeze(dim=1).reshape(B, -1)                # B, sg[0]*sg[1]
        avg_weight = self.mlp(avg)                            # B, sg[0]*sg[1]
        avg_weight = avg_weight.reshape(B, 1, self.sg[0], self.sg[1])  # B, 1, sg[0], sg[1]
        # weighted multi
        weight = weight * avg_weight                    # B, C, sg[0], sg[1]
        weight = weight.repeat_interleave(H//self.sg[0], dim=2).repeat_interleave(W//self.sg[1], dim=3)  # B, C, H, W
        x = x * weight
        # global pool
        p3, p4, p5 = torch.split(x, [p3.size(1), p4.size(1), p5.size(1)], dim=1)
        return self.head([p3, p4, p5])
# ================== head end ==================