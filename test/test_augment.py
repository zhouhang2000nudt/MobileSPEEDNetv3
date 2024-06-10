import sys
sys.path.insert(0, sys.path[0]+"/../")

import numpy as np

from MobileSPEEDNetv3.utils.utils import Camera
from MobileSPEEDNetv3.utils.config import get_config
from MobileSPEEDNetv3.utils.dataset import Speed, prepare_Speed
from MobileSPEEDNetv3.utils.vis import visualize_image
from MobileSPEEDNetv3.utils.utils import resize, warp_boxes
from torchvision.transforms.v2 import ToPILImage


category_ids = [1]
category_id_to_name = {1: 'satellite'}

config = get_config()
# config["ram"] = False
config["debug"] = True
config["offline"] = True
config["resize_first"] = False
prepare_Speed(config)
speed = Speed("train")
for i in range(len(speed)):
    image, y = speed[i]
    break
print(y["filename"])
print(image.shape)
image_pil = ToPILImage()(image)
image = np.array(image_pil)
pos = y["pos"]
ori = y["ori"]
bbox = y["bbox"]
print("ori", ori)
print("pos", pos)
print("bbox", bbox)

config["resize_first"] = True
camera = Camera(config)
image, pos, ori, warp_matrix = resize(image, pos, ori, camera, 0.5)
bbox = warp_boxes(np.array([bbox]), warp_matrix, height=image.shape[0], width=image.shape[1]).tolist()[0]
print("ori", ori)
print("pos", pos)
print("bbox", bbox)
visualize_image(image, [bbox], category_ids, category_id_to_name, ori, pos, camera)