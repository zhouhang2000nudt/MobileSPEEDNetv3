from urllib.request import urlopen
from PIL import Image
import timm
import torch
from rich import print

img = Image.open("/home/zh/pythonhub/yaolu/MobileSPEEDv3/test/img000001.jpg")

model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True, features_only=True)
model = model.eval()

print(list(model.feature_info.channels()))

# print(model)
