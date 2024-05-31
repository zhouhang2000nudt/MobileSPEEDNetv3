import torch
from torch import Tensor
from torchmetrics import Metric
from typing import List

class Loss(Metric):
    # 计算总Loss
    is_differentiable = True
    full_state_update = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, loss: Tensor, num: int):
        self.loss += loss * num
        self.num += num
    
    def compute(self) -> Tensor:
        return self.loss / self.num


class PosError(Metric):
    # 计算position error的类
    is_differentiable = True
    full_state_update = False
    higher_is_better = False
    
    def __init__(self):
        super().__init__()
        self.add_state("pos_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")
    
    def update(self, pos_pred: Tensor, pos_label:Tensor) -> None:
        self.pos_error += torch.sum(torch.linalg.norm(pos_label - pos_pred, dim=1))
        self.num += pos_label.shape[0]
    
    def compute(self) -> Tensor:
        return self.pos_error / self.num


class OriError(Metric):
    # 计算ori error的类
    is_differentiable = True
    full_state_update = False
    higher_is_better = False

    def __init__(self):
        super().__init__()
        self.add_state("ori_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, ori_pred: Tensor, ori_label: Tensor) -> None:
        ori_inner_dot = torch.abs(torch.sum(ori_label * ori_pred, dim=1, keepdim=True))
        if torch.any(ori_inner_dot > 1.01):
            # raise ValueError("Intermediate sum issue due to error in model prediction (orientation)")
            print("Intermediate sum issue due to error in model prediction (orientation)")
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1)
        ori_error = 2 * torch.arccos(ori_inner_dot)
        ori_error = torch.rad2deg(ori_error)
        self.ori_error += torch.sum(ori_error)
        self.num += ori_label.shape[0]
    
    def compute(self) -> Tensor:
        return self.ori_error / self.num


class Score(Metric):
    # 计算综合得分的类
    is_differentiable = True
    full_state_update = False
    higher_is_better = False
    
    def __init__(self, ALPHA: List[float]):
        super().__init__()
        self.add_state("score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.ALPHA = ALPHA
    
    def update(self, ori_error: Tensor, pos_error: Tensor):
        self.score = self.ALPHA[0] * pos_error + self.ALPHA[1] * ori_error
    
    def compute(self) -> Tensor:
        return self.score


if __name__ == "__main__":
    pass
        