import torch.nn
import torch
import json

labels = json.load(open("/home/zh/pythonhub/yaolu/datasets/speed/train_label.json"))
ori_error_sum = torch.tensor(0.0)
ori_error_max = torch.tensor(0.0)
ori_error_min = torch.tensor(1000.0)
for key, value in labels.items():
    ori_label = torch.tensor(value["ori"])
    ori_norm = ori_label / torch.norm(ori_label)
    ori_inner_dot = torch.abs(torch.sum(ori_label * ori_norm))
    ori_inner_dot = torch.clamp(ori_inner_dot, max=1)
    ori_error = 2 * torch.arccos(ori_inner_dot)
    ori_error_sum += ori_error
    ori_error_max = torch.max(ori_error_max, ori_error)
    ori_error_min = torch.min(ori_error_min, ori_error)
print(torch.rad2deg(ori_error / len(labels)))
print(torch.rad2deg(ori_error_max))
print(torch.rad2deg(ori_error_min))