from torchvision.transforms import v2
from torch.utils.data import Dataset, random_split, Subset
from pathlib import Path
from threading import Thread
from tqdm import tqdm
from numpy import ndarray
from .utils import rotate_image, rotate_cam, Camera, wrap_boxes, bbox_in_image, OriEncoderDecoder
from typing import List

import albumentations as A
import cv2 as cv
import lightning as L
import numpy as np

import json
import torch



def CropAndPad(img: np.array, bbox: List[float]):
    # 对图片进行裁剪
    # 裁剪后padding回原来的大小
    x_min, y_min, x_max, y_max = bbox
    height, width = img.shape[:2]
    crop_x_min = np.random.randint(0, x_min+1)
    crop_y_min = np.random.randint(0, y_min+1)
    crop_x_max = np.random.randint(x_max, width)
    crop_y_max = np.random.randint(y_max, height)
    img = img[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    img = cv.copyMakeBorder(img, crop_y_min, 0, 0, 0, cv.BORDER_REPLICATE)
    img = cv.copyMakeBorder(img, 0, height-crop_y_max-1, 0, 0, cv.BORDER_REPLICATE)
    img = cv.copyMakeBorder(img, 0, 0, crop_x_min, 0, cv.BORDER_REPLICATE)
    img = cv.copyMakeBorder(img, 0, 0, 0, width-crop_x_max-1, cv.BORDER_REPLICATE)
    return img


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def prepare_Speed(config: dict):
    # 准备数据集

    # 为Speed类添加属性
    Speed.config = config
    Speed.data_dir = Path(config["data_dir"])
    Speed.image_dir = Speed.data_dir / "images/train"
    Speed.label_file = Speed.data_dir / "train_label.json"
    Speed.test_img_dir = Speed.data_dir / "images/train"
    Speed.real_test_img_dir = Speed.data_dir / "images/train"

    # 设置transform
    Speed.transform = {
        # 训练集的数据转化
        "train": {
            "transform": v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=config["imgsz"]),
            ]),
            "A_transform": A.Compose([
                A.OneOf([
                    A.AdvancedBlur(blur_limit=(3, 7),
                                   rotate_limit=25,
                                   p=0.2),
                    A.Blur(blur_limit=(3, 7), p=0.2),
                    A.GaussNoise(var_limit=(5, 15),
                                 p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7),
                                   p=0.2),
                    ], p=0.2),
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.3,
                              p=0.2),
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
        },
        # 验证集的数据转化
        "val": {
            "transform": v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=config["imgsz"]),
            ]),
            "A_transform": None,
        },
        "test": {
            "transform": v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=config["imgsz"]),
            ]),
            "A_transform": None,
        },
        "self_supervised_train": {
            "transform": v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=config["imgsz"]),
            ]),
            "A_transform": A.Compose([
                A.OneOf([
                    A.AdvancedBlur(blur_limit=(3, 7),
                                   rotate_limit=25,
                                   p=0.2),
                    A.Blur(blur_limit=(3, 7), p=0.2),
                    A.GaussNoise(var_limit=(5, 15),
                                 p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7),
                                   p=0.2),
                    ], p=1),
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.3,
                              p=1),
                A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                         p=1),
                A.Compose([
                    A.BBoxSafeRandomCrop(p=1.0),
                    A.PadIfNeeded(min_height=1200, min_width=1920, border_mode=cv.BORDER_REPLICATE, position="random", p=1.0),
                ], p=1),
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
        },
        "self_supervised_val": {
            "transform": v2.Compose([
                v2.ToTensor(),
                v2.Resize(size=config["imgsz"]),
            ]),
            "A_transform": [A.Compose([
                A.Flip(p=0.5),
                A.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                         p=1),
                A.SafeRotate(limit=180, p=1.0),
                A.Compose([
                    A.BBoxSafeRandomCrop(p=1.0),
                    A.PadIfNeeded(min_height=1200, min_width=1920, border_mode=cv.BORDER_REPLICATE, position="random", p=1.0),
                ], p=1)
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])),
            A.Compose([
                A.OneOf([
                    A.AdvancedBlur(blur_limit=(3, 7),
                                   rotate_limit=25,
                                   p=0.2),
                    A.Blur(blur_limit=(3, 7), p=0.2),
                    A.GaussNoise(var_limit=(5, 15),
                                 p=0.2),
                    A.GaussianBlur(blur_limit=(3, 7),
                                   p=0.2),
                    ], p=1),
                A.ColorJitter(brightness=0.3,
                              contrast=0.3,
                              saturation=0.3,
                              hue=0.3,
                              p=1)
            ],
            p=1,
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]))
            ]
        }
    }

    # 设置标签字典
    Speed.labels = json.load(open(Speed.label_file, "r"))
    
    # 采样列表
    Speed.img_name = list(Speed.labels.keys())
    num = len(Speed.img_name)
    train_num = int(num * Speed.config["split"][0])
    val_num = num - train_num
    Speed.train_index, Speed.val_index = random_split(Speed.img_name, [train_num, val_num])
    # Speed.test_index = list(Speed.test_labels.keys())

    # 缓存图片
    if Speed.config["ram"]:
        Speed.img_dict = {}
        Speed.read_img()
    
    # 设置姿态编码解码器
    Speed.ori_encoder_decoder = OriEncoderDecoder(Speed.config["stride"], Speed.config["alpha"], Speed.config["neighbour"])


class ImageReader(Thread):
    def __init__(self, img_name: list, config: dict, image_dir: Path):
        Thread.__init__(self)
        self.config: dict = config
        self.image_dir: Path = image_dir
        self.image_name: list = img_name
        self.img_dict: dict = {}
    
    def run(self):
        for img_name in tqdm(self.image_name):
            img = cv.imread(str(self.image_dir / img_name), cv.IMREAD_GRAYSCALE)
            self.img_dict[img_name] = img
    
    def get_result(self) -> dict:
        return self.img_dict


class Speed(Dataset):
    data_dir: Path          # 数据集根目录
    image_dir: Path         # 图片目录
    po_file: Path           # 位姿json文件
    bbox_file: Path         # bbox json文件
    labels: dict            # 标签字典
    test_labels: dict       # 测试集标签字典
    config: dict            # 配置字典
    img_name: list     # 样本id列表
    transform: dict   # 数据转化方法字典
    train_index: Subset     # 训练集图片名列表
    val_index: Subset       # 验证集图片名列表
    test_index: list        # 测试集图片名列表
    img_dict: dict = {} # 图片字典
    camera: Camera = Camera
    ori_encoder_decoder: OriEncoderDecoder
    

    def __init__(self, mode: str = "train"):
        self.A_transform = Speed.transform[mode]["A_transform"]
        self.transform = Speed.transform[mode]["transform"]
        self.sample_index = Speed.train_index if "train" in mode else Speed.val_index if "val" in mode else Speed.test_index
        self.mode = mode
    
    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, index) -> tuple:
        filename = self.sample_index[index].strip()                  # 图片文件名
        # filename = "img000001.jpg"
        if Speed.config["ram"]:
            image = Speed.img_dict[filename]                         # 伪图片
        else:
            image = cv.imread(str(self.image_dir / filename), cv.IMREAD_GRAYSCALE)       # 读取图片
        
        ori = np.array(self.labels[filename]["ori"])   # 姿态
        pos = np.array(self.labels[filename]["pos"])   # 位置
        
        bbox = self.labels[filename]["bbox"]
        if bbox[2] >= 1920:
            bbox[2] = 1919
        if bbox[3] >= 1200:
            bbox[3] = 1199
        
        # 先进行wrapping
        dice = np.random.rand()
        if ("train" in self.mode or "self_supervised" in self.mode) and (Speed.config["Rotate"]["Rotate_img"] or Speed.config["Rotate"]["Rotate_cam"]):
            wrapped_time = 0
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            wrapped = False
            if Speed.config["Rotate"]["Rotate_img"] and dice <= Speed.config["Rotate"]["p"]:
                while True:
                    if wrapped_time > 10:
                        wrapped = False
                        break
                    wrapped_time += 1
                    image_wrapped, pos_wrapped, ori_wrapped, M_wrapped = rotate_image(image, pos, ori, Speed.camera.K, Speed.config["Rotate"]["img_angle"])
                    bbox_wrapped = wrap_boxes(np.array([bbox]), M_wrapped, height=1200, width=1920).tolist()[0]
                    if bbox_in_image(bbox_wrapped, bbox_area):
                        wrapped = True
                        break
            elif Speed.config["Rotate"]["Rotate_cam"] and dice > Speed.config["Rotate"]["p"]:
                while True:
                    if wrapped_time > 10:
                        wrapped = False
                        break
                    wrapped_time += 1
                    image_wrapped, pos_wrapped, ori_wrapped, M_wrapped = rotate_cam(image, pos, ori, Speed.camera.K, Speed.config["Rotate"]["cam_angle"])
                    bbox_wrapped = wrap_boxes(np.array([bbox]), M_wrapped, height=1200, width=1920).tolist()[0]
                    if bbox_in_image(bbox_wrapped, bbox_area):
                        wrapped = True
                        break
            
            if wrapped:
                image = image_wrapped
                pos = pos_wrapped
                ori = ori_wrapped
                bbox = list(map(int, bbox_wrapped))
        
        # 进行Albumentation增强
        if "self_supervised" in self.mode:
            # 空间变换
            transformed = self.A_transform[0](image=image, bboxes=[bbox], category_ids=[1])
            image = transformed["image"]
            bbox = list(map(int, list(transformed["bboxes"][0])))
            # 像素变换
            transformed_1 = self.A_transform[1](image=image, bboxes=[bbox], category_ids=[1])
            image_1 = transformed_1["image"]
            bbox_1 = list(map(int, list(transformed_1["bboxes"][0])))
            image_1 = CropAndPad(image_1, bbox_1)
            transformed_2 = self.A_transform[1](image=image, bboxes=[bbox], category_ids=[1])
            image_2 = transformed_2["image"]
            bbox_2 = list(map(int, list(transformed_2["bboxes"][0])))
            image_2 = CropAndPad(image_2, bbox_2)
        else:
            if self.A_transform is not None:
                transformed = self.A_transform(image=image, bboxes=[bbox], category_ids=[1])
                image = transformed["image"]
                dice = np.random.rand()
                if dice < Speed.config["CropAndPadding"]:
                    image = CropAndPad(image, bbox)
                bbox = list(map(int, list(transformed["bboxes"][0])))
        
        if "self_supervised" in self.mode:
            image_1 = self.transform(image_1)       # (1, 480, 768)
            image_1 = image_1.repeat(3, 1, 1)       # (3, 480, 768)
            image_2 = self.transform(image_2)       # (1, 480, 768)
            image_2 = image_2.repeat(3, 1, 1)
            return image_1, image_2
        
        # 使用torchvision转换图片
        image = self.transform(image)       # (1, 480, 768)
        image = image.repeat(3, 1, 1)       # (3, 480, 768)
        
        yaw_encode, pitch_encode, roll_encode = Speed.ori_encoder_decoder.encode_ori(ori)
        
        y: dict = {
            "filename": filename,
            "pos": pos,
            "ori": ori,
            "yaw_encode": yaw_encode,
            "pitch_encode": pitch_encode,
            "roll_encode": roll_encode,
            "bbox": bbox
        }

        return image, y

    @staticmethod
    def divide_data(lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    
    @staticmethod
    def read_img(thread_num: int = 12):
        # 将采样列表中的图片读入内存
        img_divided: list = Speed.divide_data(Speed.img_name, thread_num)
        thread_list: list[ImageReader] = []
        for sub_img_name in img_divided:
            thread_list.append(ImageReader(sub_img_name, Speed.config, Speed.image_dir))
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
        for thread in thread_list:
            Speed.img_dict.update(thread.get_result())


class SpeedDataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config: dict = config
        prepare_Speed(config)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.config["self_supervised"]:
                self.speed_data_train: Speed = Speed("self_supervised_train")
                self.speed_data_val: Speed = Speed("self_supervised_val")
            else:
                self.speed_data_train: Speed = Speed("train")
                self.speed_data_val: Speed = Speed("val")
        elif stage == "validate":
            self.speed_data_val: Speed = Speed("val")
    
    def train_dataloader(self) -> MultiEpochsDataLoader:
        return MultiEpochsDataLoader(
            self.speed_data_train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
            persistent_workers=True
        )
    
    def val_dataloader(self) -> MultiEpochsDataLoader:
        return MultiEpochsDataLoader(
            self.speed_data_val,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
            persistent_workers=True
        )



if __name__ == "__main__":
    import yaml

    with open("./cfg/base.yaml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    prepare_Speed(config)
    speed_dataset = SpeedDataModule(config)
    speed_dataset.setup("fit")
    speed_dataset.speed_data_train[0]
    speed_dataset.speed_data_val[0]
    print()