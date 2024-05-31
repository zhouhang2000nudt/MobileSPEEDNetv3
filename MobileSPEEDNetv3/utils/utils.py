"""
Utility functions

"""

import torch
import os
import math
import random
import numpy as np
import scipy
import skimage.color
import skimage.io
import skimage.transform
import urllib.request
import shutil
import warnings
import random
import itertools
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from torch import Tensor

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


def wrap_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        # 一个矩形框是4个点的，即4个坐标，左上、右下、左下、右上
        # tmp_box = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
        # tmp_box_1 的shape 为(n*4, 2)，即：该矩阵只有两列，代表了所有的坐标点。
        # tmp_box_1 = tmp_box.reshape(n * 4, 2)
        # 将所有的单独点的坐标，赋给xy[:, :2]前两列
        # xy中的一个元素就对应着， [x, y, 1] 齐次性，
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # 所有点，矩阵变换，求变换后的点
        xy = xy @ M.T  # transform
        # [x,y]/1 前两列坐标x,y除以第3列坐标值，代表“齐次性”; 此时，xy已经去掉了第三列了
        # 只有出现透视变换时，第三列的数值才不会为1
        # 再reshape回去，就又变成了8个点[[x1,y1,x2,y2,x1,y2,x2,y1]]
        # 上面的那句话不对，并不能说是同一个x和同一个y
        # 因为经过仿射变换后，不一定是方方正正的矩形了，只能说，对应的几个索引的代表x的值以及y的值
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        # 找到所有的变换后的x坐标、y坐标，分别将其中的min、max重新构成一个集合，就是变换后的bbox
        # 再根据变换后的图片的宽高，裁剪一下，得到合理的数值
        # 此处的x.min(1)是axis=1的意思，按照列求min和max
        # x.min(1)后的shape: (12, )
        # d1 = x.min(1)
        # d2 = y.min(1)
        # d3 = x.max(1)
        # d4 = y.max(1)
        # # tmp_box_2 ，默认按照行拼接， shape:(48, ) 。 就是将一个d一直排列下去
        # tmp_box_2 = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
        # tmp_box_3 = tmp_box_2.reshape(4, n)
        # tmp_box_3转置一下，就成了(n, 4)形式， (x1,y1,x2,y2)的结果
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width-1)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height-1)
        return xy.astype(np.float32)
    else:
        return boxes


def bbox_in_image(bbox_wrapped, bbox_area):
    # 若bbox_wrapped的面积小于原来bbox面积的0.95，就认为bbox_wrapped不在图像内
    bbox_wrapped_area = (bbox_wrapped[2] - bbox_wrapped[0]) * (bbox_wrapped[3] - bbox_wrapped[1])
    return bbox_wrapped_area >= 0.95 * bbox_area


def clamp(bbox):
    # 将标注限制到图片内
    if bbox[0][0] < 0:
        bbox[0][0] = 0
    elif bbox[0][0] > 768:
        bbox[0][0] = 768
    
    if bbox[0][1] < 0:
        bbox[0][1] = 0
    elif bbox[0][1] > 480:
        bbox[0][1] = 480
    
    if bbox[0][2] < 0:
        bbox[0][2] = 0
    elif bbox[0][2] > 768:
        bbox[0][2] = 768
    
    if bbox[0][3] < 0:
        bbox[0][3] = 0
    elif bbox[0][3] > 480:
        bbox[0][3] = 480
    return bbox


def rotate_image(image, pos, ori, camera_k, rot_max_magnitude):
    """Data augmentation: rotate image and adapt position/orientation.
    Rotation amplitude is randomly picked from [-rot_max_magnitude/2, +rot_max_magnitude/2]
    """

    image = np.array(image)

    change = (np.random.rand(1)-0.5) * rot_max_magnitude

    # r_change = rpy2r(change, 0, 0, order='xyz', unit='deg')
    rotation = R.from_euler('YXZ', [0, 0, change[0]], degrees=True)
    r_change = rotation.as_matrix()
    

    # Construct warping (perspective) matrix
    warp_matrix = camera_k @ r_change @ np.linalg.inv(camera_k)

    height, width = np.shape(image)[:2]

    image_warped = cv2.warpPerspective(image, warp_matrix, (width, height), cv2.WARP_INVERSE_MAP)

    # Update pose
    pos_new = np.array(r_change @ pos)
    ori_new = rotation * R.from_quat([ori[1], ori[2], ori[3], ori[0]])
    ori_new = ori_new.as_quat(canonical=True)
    ori_new = np.array([ori_new[3], ori_new[0], ori_new[1], ori_new[2]])

    return image_warped, pos_new, ori_new, warp_matrix


def rotate_cam(image, pos, ori, camera_k, rot_max_magnitude):
    """Data augmentation: rotate image and adapt position/orientation.
    Rotation amplitude is randomly picked from [-rot_max_magnitude/2, +rot_max_magnitude/2]
    """

    image = np.array(image)

    change = (np.random.rand(3)-0.5) * rot_max_magnitude

    # r_change = rpy2r(change, 0, 0, order='xyz', unit='deg')
    rotation = R.from_euler('YXZ', change, degrees=True)
    r_change = rotation.as_matrix()
    

    # Construct warping (perspective) matrix
    warp_matrix = camera_k @ r_change @ np.linalg.inv(camera_k)

    height, width = np.shape(image)[:2]

    image_warped = cv2.warpPerspective(image, warp_matrix, (width, height), cv2.WARP_INVERSE_MAP)

    # Update pose
    pos_new = np.array(r_change @ pos)
    ori_new = rotation * R.from_quat([ori[1], ori[2], ori[3], ori[0]])
    ori_new = ori_new.as_quat(canonical=True)
    ori_new = np.array([ori_new[3], ori_new[0], ori_new[1], ori_new[2]])

    return image_warped, pos_new, ori_new, warp_matrix


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
        self.yaw_range = np.linspace(-neighbour * stride, 360 + neighbour * stride, self.yaw_len) - 180
        self.pitch_range = np.linspace(-neighbour * stride, 180 + neighbour * stride, self.pitch_len) - 90
        self.roll_range = np.linspace(-neighbour * stride, 360 + neighbour * stride, self.roll_len) - 180
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
        yaw_decode = np.sum(yaw_encode * self.yaw_range)
        pitch_decode = np.sum(pitch_encode * self.pitch_range)
        roll_decode = np.sum(roll_encode * self.roll_range)
        
        rotation = R.from_euler('YXZ', [yaw_decode, pitch_decode, roll_decode], degrees=True)
        ori = rotation.as_quat()
        ori = [ori[3], ori[0], ori[1], ori[2]]
        return torch.tensor(ori)

    def decode_ori_batch(self, yaw_encode: Tensor, pitch_encode: Tensor, roll_encode: Tensor):
        ori_decode = []
        device = yaw_encode.device
        yaw_encode = yaw_encode.cpu().numpy()
        pitch_encode = pitch_encode.cpu().numpy()
        roll_encode = roll_encode.cpu().numpy()
        for b in range(yaw_encode.shape[0]):
            ori_decode.append(self.decode_ori(yaw_encode[b], pitch_encode[b], roll_encode[b]))
        
        return torch.stack(ori_decode, dim=0).to(device)