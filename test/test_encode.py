import sys
sys.path.insert(0, sys.path[0]+"/../")


from scipy.spatial.transform import Rotation as R
from torch import Tensor

from MobileSPEEDNetv3.utils.vis import visualize

import json
import cv2
import torch

import numpy as np


from MobileSPEEDNetv3.utils.config import get_config
from MobileSPEEDNetv3.utils.dataset import Speed, prepare_Speed
from MobileSPEEDNetv3.utils.vis import visualize
from MobileSPEEDNetv3.utils.utils import Camera
from MobileSPEEDNetv3.utils.utils import OriEncoderDecoderGauss

class Camera:
    fwx = 0.0176  # focal length[m]
    fwy = 0.0176  # focal length[m]
    width = 1920  # number of horizontal[pixels]
    height = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fx = fwx / ppx  # horizontal focal length[pixels]
    fy = fwy / ppy  # vertical focal length[pixels]

    K = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    
class OriEncoderDecoder:
    def __init__(self, stride: int, alpha: float, neighbour: int = 0):
        assert 360 % stride == 0 and 180 % stride == 0, "stride must be a divisor of 360 and 180"
        assert neighbour >= 0, "neighbour must be greater than or equal to 0"
        assert alpha < 0.6666666666666666, "alpha must be less than 2/3"

        self.stride = stride
        self.alpha = alpha
        self.neighbour = neighbour
        self.yaw_len = int(360 // stride + 1 + 2 * neighbour)
        self.pitch_len = int(180 // stride + 1 + 2 * neighbour)
        self.roll_len = int(360 // stride + 1 + 2 * neighbour)
        self.yaw_range = torch.linspace(-neighbour * stride, 360 + neighbour * stride, self.yaw_len) - 180
        self.pitch_range = torch.linspace(-neighbour * stride, 180 + neighbour * stride, self.pitch_len) - 90
        self.roll_range = torch.linspace(-neighbour * stride, 360 + neighbour * stride, self.roll_len) - 180
        self.yaw_index_dict = {int(yaw // stride): i for i, yaw in enumerate(self.yaw_range)}
        self.pitch_index_dict = {int(pitch // stride): i for i, pitch in enumerate(self.pitch_range)}
        self.roll_index_dict = {int(roll // stride): i for i, roll in enumerate(self.roll_range)}
        
    
    def _encode_ori(self, angle: float, angle_len: int, angle_index_dict: dict):
        encode = np.zeros(angle_len)
        
        mean = angle / self.stride
        l = int(np.floor(mean))
        r = int(np.ceil(mean))
        li = angle_index_dict[l]
        ri = angle_index_dict[r]
        alpha = [self.alpha for _ in range(self.neighbour)]
        if l == r:
            encode[li] = 1
            alpha[0] /= 2
        else:
            encode[li] = r - mean
            encode[ri] = mean - l
        d = r - l
        for i in range(self.neighbour):
            pl_out = encode[li] * alpha[i]
            pr_out = encode[ri] * alpha[i]
            encode[li] -= pl_out
            encode[ri] -= pr_out
            p_out = pl_out + pr_out
            # neighbour
            li -= 1
            ri += 1
            l -= 1
            r += 1
            d = r - l
            encode[li] += p_out * (r-mean) / d
            encode[ri] += p_out * (mean-l) / d
        
        return encode
    
    def encode_ori(self, ori: np.ndarray):
        rotation = R.from_quat([ori[1], ori[2], ori[3], ori[0]])
        yaw, pitch, roll = rotation.as_euler('YXZ', degrees=True)    # 偏航角、俯仰角、翻滚角
        
        yaw_encode = self._encode_ori(yaw, self.yaw_len, self.yaw_index_dict)
        pitch_encode = self._encode_ori(pitch, self.pitch_len, self.pitch_index_dict)
        roll_encode = self._encode_ori(roll, self.roll_len, self.roll_index_dict)
        return yaw_encode, pitch_encode, roll_encode

    def decode_ori(self, yaw_encode: Tensor, pitch_encode: Tensor, roll_encode: Tensor):
        yaw_decode = torch.sum(yaw_encode * self.yaw_range)
        pitch_decode = torch.sum(pitch_encode * self.pitch_range)
        roll_decode = torch.sum(roll_encode * self.roll_range)
        
        rotation = R.from_euler('YXZ', [yaw_decode, pitch_decode, roll_decode], degrees=True)
        ori = rotation.as_quat()
        ori = [ori[3], ori[0], ori[1], ori[2]]
        
        return torch.tensor(ori)

    def decode_ori_batch(self, yaw_encode: Tensor, pitch_encode: Tensor, roll_encode: Tensor):
        # ori_decode = []
        # for b in range:
        #     ori_decode.append(self.decode_ori(yaw_encode[b], pitch_encode[b], roll_encode[b]))
        
        # return torch.tensor(ori_decode)
        
        yaw_decode = torch.sum(yaw_encode * self.yaw_range, dim=1)
        pitch_decode = torch.sum(pitch_encode * self.pitch_range, dim=1)
        roll_decode = torch.sum(roll_encode * self.roll_range, dim=1)
        
        cy = torch.cos(torch.deg2rad(yaw_decode * 0.5))
        sy = torch.sin(torch.deg2rad(yaw_decode * 0.5))
        cp = torch.cos(torch.deg2rad(pitch_decode * 0.5))
        sp = torch.sin(torch.deg2rad(pitch_decode * 0.5))
        cr = torch.cos(torch.deg2rad(roll_decode * 0.5))
        sr = torch.sin(torch.deg2rad(roll_decode * 0.5))
        
        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * sp * cr + sy * cp * sr
        q2 = sy * cp * cr - cy * sp * sr
        q3 = -sy * sp * cr + cy * cp * sr
        
        return torch.stack([q0, q1, q2, q3], dim=1)


ori_encoder_decoder = OriEncoderDecoder(5, 0.1, 0)
ori_encoder_decoder = OriEncoderDecoderGauss(10, 1)

ori = torch.tensor([-0.419541, -0.484436, -0.214179, 0.73718])
yaw_encode, pitch_encode, roll_encode = ori_encoder_decoder.encode_ori(ori)
yaw_encode = torch.tensor(yaw_encode).unsqueeze(0)
pitch_encode = torch.tensor(pitch_encode).unsqueeze(0)
roll_encode = torch.tensor(roll_encode).unsqueeze(0)

ori_decode = ori_encoder_decoder.decode_ori_batch(yaw_encode, pitch_encode, roll_encode)

print(
    2 * torch.arccos(torch.abs(torch.sum(ori * ori_decode)))
)

config = get_config()
prepare_Speed(config)
speed = Speed("train")
for i in range(len(speed)):
    image, y = speed[i]
    break
pos = y["pos"]
ori = y["ori"]
yaw_encode = y["yaw_encode"]
pitch_encode = y["pitch_encode"]
roll_encode = y["roll_encode"]
bbox = y["bbox"]
ori_decode = ori_encoder_decoder.decode_ori_batch(torch.tensor(yaw_encode).unsqueeze(0),
                                                  torch.tensor(pitch_encode).unsqueeze(0),
                                                  torch.tensor(roll_encode).unsqueeze(0))

category_ids = [1]
category_id_to_name = {1: 'satellite'}
visualize(image, [bbox], category_ids, category_id_to_name, ori_decode[0], pos, Camera.K)
