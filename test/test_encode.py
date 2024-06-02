import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch

from MobileSPEEDNetv3.utils.utils import OriEncoderDecoderGauss, OriEncoderDecoder
from MobileSPEEDNetv3.utils.vis import visualize_encode

ori_encoder_decoder = OriEncoderDecoder(5, 0.1, 5)
ori_encoder_decoder = OriEncoderDecoderGauss(5, 0.6, 5)

ori = torch.tensor([-0.419541, -0.484436, -0.214179, 0.73718])
yaw_encode, pitch_encode, roll_encode = ori_encoder_decoder.encode_ori(ori)
yaw_encode = torch.tensor(yaw_encode).unsqueeze(0)
pitch_encode = torch.tensor(pitch_encode).unsqueeze(0)
roll_encode = torch.tensor(roll_encode).unsqueeze(0)

ori_decode = ori_encoder_decoder.decode_ori_batch(yaw_encode, pitch_encode, roll_encode)

print(
    2 * torch.arccos(torch.abs(torch.sum(ori * ori_decode)))
)

visualize_encode(ori_encoder_decoder.yaw_range.numpy(), yaw_encode.squeeze(0).numpy(), stride=ori_encoder_decoder.stride)
