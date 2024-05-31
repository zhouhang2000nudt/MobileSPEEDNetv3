import sys
sys.path.insert(0, sys.path[0]+"/../")

import numpy as np

from MobileSPEEDNetv3.utils.utils import Camera
from MobileSPEEDNetv3.utils.config import get_config
from MobileSPEEDNetv3.utils.dataset import Speed, prepare_Speed
from MobileSPEEDNetv3.utils.vis import visualize
from torchvision.transforms.v2 import ToPILImage


category_ids = [1]
category_id_to_name = {1: 'satellite'}

config = get_config()
camera = Camera()
config["ram"] = False
prepare_Speed(config)
speed = Speed("train")
for i in range(len(speed)):
    image, y = speed[i]
    break
print(y["filename"])
print(image.shape)
image_pil = ToPILImage()(image)
image_pil = image_pil.resize((1920, 1200))
image = np.array(image_pil)
pos = y["pos"]
ori = y["ori"]
bbox = y["bbox"]
print("ori", ori)
print("pos", pos)
print("bbox", bbox)

visualize(image, [bbox], category_ids, category_id_to_name, ori, pos, camera.K)