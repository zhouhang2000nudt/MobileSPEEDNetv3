import yaml
import torch
# from .utils import build_histogram, pre_compute_ori_decode

def get_config(cfg_path: str = "MobileSPEEDv3/cfg/base.yaml"):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    # config["H_MAP"], config["REDUNDANT_FLAGS"] = build_histogram(config["N_ORI_BINS_PER_DIM"],
    #                                                              torch.tensor([-180, -90, -180]),
    #                                                              torch.tensor([180, 90, 180]))
    # config["B"] = pre_compute_ori_decode(config["H_MAP"])
    return config
