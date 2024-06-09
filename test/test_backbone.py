import sys
sys.path.insert(0, sys.path[0]+"/../")

import torch

import timm

from rich import print
from ptflops import get_model_complexity_info
from MobileSPEEDNetv3.model import Mobile_SPEEDv3, LightSPEED
from MobileSPEEDNetv3.utils.config import get_config


def profile_model(model):
    flops, params = get_model_complexity_info(model, (3, 240, 384), as_strings=True, print_per_layer_stat=False, verbose=False, flops_units="GMac", param_units="M", output_precision=10)
    print(flops)
    print(params)


print(timm.list_models("*mobilenet*"))

# timm_model = timm.create_model("mobilenetv3_large_100", pretrained=False, features_only=True)
# timm_model = timm.create_model('mobilenetv3_small_075.lamb_in1k', pretrained=True, features_only=True)
# print(timm_model.default_cfg)
# print(timm_model)
# profile_model(timm_model)
print("=====================================")
t = torch.rand([1, 3, 600, 960])
config = get_config()
config["pretrained"] = False
full_model = LightSPEED(config)
print(full_model)
full_model.eval()
# full_model._switch_to_deploy()
print(full_model)
# print(full_model(t).shape)
profile_model(full_model)