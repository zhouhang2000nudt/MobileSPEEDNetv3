import torch
import lightning as L
import numpy as np
import rich
import csv

from torch.optim import SGD, AdamW

from ..model.Mobile_SPPEDv3 import Mobile_SPEEDv3
from ..utils.loss import PoseLoss, EulerLoss
from ..utils.metrics import Loss, PosError, OriError, Score
from ..utils.utils import OriEncoderDecoder


class LightningMobileSPEEDv3(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        # 配置
        self.config: dict = config
        # 模型
        self.model: Mobile_SPEEDv3 = Mobile_SPEEDv3(self.config)
        # 欧拉角编码解码器
        self.ori_encoder_decoder: OriEncoderDecoder = OriEncoderDecoder(self.config["stride"], self.config["alpha"], neighbour=self.config["neighbour"])
        # 损失函数
        self.pos_loss: PoseLoss = PoseLoss(self.config["pos_loss"])
        self.yaw_loss: EulerLoss = EulerLoss(self.config["euler_loss"])
        self.pitch_loss: EulerLoss = EulerLoss(self.config["euler_loss"])
        self.roll_loss: EulerLoss = EulerLoss(self.config["euler_loss"])
        # 损失比例
        self.BETA = self.config["BETA"]
        # 指标
        self.train_pos_loss: Loss = Loss()
        self.train_yaw_loss: Loss = Loss()
        self.train_pitch_loss: Loss = Loss()
        self.train_roll_loss: Loss = Loss()
        self.train_loss: Loss = Loss()
        self.val_pos_loss: Loss = Loss()
        self.val_yaw_loss: Loss = Loss()
        self.val_pitch_loss: Loss = Loss()
        self.val_roll_loss: Loss = Loss()
        self.val_loss: Loss = Loss()
        self.ori_error: OriError = OriError()
        self.pos_error: PosError = PosError()
        self.score: Score = Score(self.config["ALPHA"])
        self.save_hyperparameters()
        if self.config["save_csv"]:
            self.result = [["file_name",
                    "pos_label_1", "pos_label_2", "pos_label_3",
                    "ori_label_1", "ori_label_2", "ori_label_3", "ori_label_4",
                    "pos_pred_1", "pos_pred_2", "pos_pred_3",
                    "ori_pred_1", "ori_pred_2", "ori_pred_3", "ori_pred_4",
                    "pos_error(m)", "ori_error(deg)"]]


    def forward(self, x):
        return self.model(x)

    # ===========================train===========================
    def on_train_start(self):
        self.logger.experiment.log_asset_folder(folder="MobileSPEEDv3", log_file_name=True, recursive=True)


    def training_step(self, batch, batch_idx):
        if self.config["self_supervised"]:
            pass
        else:
            inputs, labels = batch
            num = inputs.shape[0]
            pos, yaw, pitch, roll = self(inputs)
            train_pos_loss = self.pos_loss(pos, labels["pos"])
            train_yaw_loss = self.yaw_loss(yaw, labels["yaw_encode"])
            train_pitch_loss = self.pitch_loss(pitch, labels["pitch_encode"])
            train_roll_loss = self.roll_loss(roll, labels["roll_encode"])

        train_loss = self.BETA[0] * train_pos_loss + self.BETA[1] * (train_yaw_loss + train_pitch_loss + train_roll_loss)

        self.train_pos_loss.update(train_pos_loss, num)
        self.train_yaw_loss.update(train_yaw_loss, num)
        self.train_pitch_loss.update(train_pitch_loss, num)
        self.train_roll_loss.update(train_roll_loss, num)
        self.train_loss.update(train_loss, num)
        return train_loss


    def on_train_epoch_end(self):
        self.log_dict({
            "train/pos_loss": self.train_pos_loss.compute(),
            "train/yaw_loss": self.train_yaw_loss.compute(),
            "train/pitch_loss": self.train_pitch_loss.compute(),
            "train/roll_loss": self.train_roll_loss.compute(),
            "train/loss": self.train_loss.compute(),
        }, on_epoch=True)
        self.train_pos_loss.reset()
        self.train_yaw_loss.reset()
        self.train_pitch_loss.reset()
        self.train_roll_loss.reset()
        self.train_loss.reset()
    

    # ===========================validation===========================
    def on_validation_start(self) -> None:
        rich.print(f"[b]{'train':<5} Epoch {self.current_epoch:>3}/{self.trainer.max_epochs:<3} pos_loss: {self.train_pos_loss.compute().item():<8.4f}  yaw_loss: {self.train_yaw_loss.compute().item():<8.4f}  pitch_loss: {self.train_pitch_loss.compute().item():<8.4f}  roll_loss: {self.train_roll_loss.compute().item():<8.4f}  loss: {self.train_loss.compute().item():<8.4f}")

    
    def validation_step(self, batch, batch_index):
        # 取出数据
        if self.config["self_supervised"]:
            pass
        else:
            inputs, labels = batch
            num = inputs.shape[0]
            # 前向传播
            pos, yaw, pitch, roll = self(inputs)
            # 计算损失
            val_pos_loss = self.pos_loss(pos, labels["pos"])
            val_yaw_loss = self.yaw_loss(yaw, labels["yaw_encode"])
            val_pitch_loss = self.pitch_loss(pitch, labels["pitch_encode"])
            val_roll_loss = self.roll_loss(roll, labels["roll_encode"])
            ori_decode = self.ori_encoder_decoder.decode_ori_batch(yaw, pitch, roll)
            self.ori_error.update(ori_decode, labels["ori"])
            self.pos_error.update(pos, labels["pos"])

        val_loss = self.BETA[0] * val_pos_loss + self.BETA[1] * (val_yaw_loss + val_pitch_loss + val_roll_loss)
        # 计算指标
        self.val_pos_loss.update(val_pos_loss, num)
        self.val_yaw_loss.update(val_yaw_loss, num)
        self.val_pitch_loss.update(val_pitch_loss, num)
        self.val_roll_loss.update(val_roll_loss, num)
        self.val_loss.update(val_loss, num)
        
        
        if self.config["save_csv"]:
            new_result = [labels["filename"][0],
                        labels["pos"][0][0].item(), labels["pos"][0][1].item(), labels["pos"][0][2].item(),
                        labels["ori"][0][0].item(), labels["ori"][0][1].item(), labels["ori"][0][2].item(), labels["ori"][0][3].item(),
                        pos[0][0].item(), pos[0][1].item(), pos[0][2].item(),
                        ori_decode[0][0].item(), ori_decode[0][1].item(), ori_decode[0][2].item(), ori_decode[0][3].item(),
                        self.pos_error.compute().item(), self.ori_error.compute().item()]
            self.result.append(new_result)
            self.pos_error.reset()
            self.ori_error.reset()


    def on_validation_epoch_end(self) -> None:
        if self.config["save_csv"]:
            with open("result.csv", "w", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(self.result)
        self.score.update(self.ori_error.compute(), self.pos_error.compute())
        self.metric_dict = {
            "val/pos_loss": self.val_pos_loss.compute(),
            "val/yaw_loss": self.val_yaw_loss.compute(),
            "val/pitch_loss": self.val_pitch_loss.compute(),
            "val/roll_loss": self.val_roll_loss.compute(),
            "val/loss": self.val_loss.compute(),
            "val/ori_error(deg)": self.ori_error.compute(),
            "val/pos_error(m)": self.pos_error.compute(),
            "val/score": self.score.compute(),
        }
        self.log_dict(self.metric_dict, on_epoch=True)
        rich.print(f"[b]{'val':<5} Epoch {self.current_epoch:>3}/{self.trainer.max_epochs:<3} pos_loss: {self.val_pos_loss.compute().item():<8.4f}  yaw_loss: {self.val_yaw_loss.compute().item():<8.4f}  pitch_loss: {self.val_pitch_loss.compute().item():<8.4f}  roll_loss: {self.val_roll_loss.compute().item():<8.4f}  loss: {self.val_loss.compute().item():<8.4f}  pos_error(m): {self.pos_error.compute().item():<8.4f}  ori_error(deg): {self.ori_error.compute().item():<8.4f}")
        self.val_pos_loss.reset()
        self.val_yaw_loss.reset()
        self.val_pitch_loss.reset()
        self.val_roll_loss.reset()
        self.val_loss.reset()
        self.ori_error.reset()
        self.pos_error.reset()
        self.score.reset()
    
    def on_fit_end(self):
        self.logger.experiment.log_asset(self.trainer.callbacks[3].best_model_path, overwrite=True)
        self.logger.experiment.log_asset(self.trainer.callbacks[3].last_model_path, overwrite=True)
    
    def configure_optimizers(self):
        # 定义优化器
        if self.config["optimizer"] == "AdamW":
            if self.config["precision"] == "half":
                eps = 1e-3
            else:
                eps = 1e-8
            optimizer = AdamW(self.parameters(), lr=self.config["lr0"],
                             weight_decay=self.config["weight_decay"],
                             eps=eps)
        elif self.config["optimizer"] == "SGD":
            optimizer = SGD(self.parameters(), lr=self.config["lr0"],
                            weight_decay=self.config["weight_decay"],
                            momentum=self.config["momentum"])
        
        # 定义学习率调度器
        lr_scheduler_config: dict = {
            "scheduler": None,
            "interval": "epoch",            # 调度间隔
            "frequency": 1,                 # 调度频率
        }
        if self.config["lr_scheduler"] == "WarmupCosineAnnealingLR":
            # 余弦退火学习率调度器
            lambda_max = self.config["lr0"] / self.config['lr0']
            lambda_min = self.config["lr_min"] / self.config["lr0"]
            warmup_epoch = self.config["warmup_epochs"]
            max_epoch = self.config["epoch"]
            lambda0 = lambda cur_iter: lambda_min + (lambda_max-lambda_min) * cur_iter / (warmup_epoch-1) if cur_iter < warmup_epoch \
                else lambda_min + (lambda_max-lambda_min)*(1 + np.cos(np.pi * (cur_iter - warmup_epoch) / (max_epoch - warmup_epoch - 1))) / 2
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        elif self.config["lr_scheduler"] == "ReduceLROnPlateau":
            # 当评价指标不再提升时，降低学习率
            factor = self.config["factor"]
            Plateaupatience = self.config["Plateaupatience"]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=Plateaupatience, verbose=True)
            lr_scheduler_config["monitor"] = "val/loss"
            lr_scheduler_config["strict"] = True
        elif self.config["lr_scheduler"] == "MultiStepLR":
            # 多步学习率调度器
            milestones = self.config["milestones"]
            gamma = self.config["gamma"]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        else:
            ValueError("Invalid lr_scheduler: {}".format(self.config["lr_scheduler"]))
        lr_scheduler_config["scheduler"] = scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }

@torch.jit.script
def get_box(box, stage):
    batch_index = torch.arange(0, box.shape[0]).unsqueeze(1).repeat(1, 4)
    bbox_index = torch.arange(4).repeat(box.shape[0], 1)
    stage_index = torch.nonzero(stage == 1)[:, 1:].repeat(1, 4)
    box = box[batch_index, bbox_index, stage_index]
    return box

@torch.jit.script
def get_box_val(box, cfd):
    batch_index = torch.arange(0, box.shape[0]).unsqueeze(1).repeat(1, 4)
    bbox_index = torch.arange(4).repeat(box.shape[0], 1)
    stage_index = torch.argmax(cfd, dim=1, keepdim=True).repeat(1, 4)
    box = box[batch_index, bbox_index, stage_index]
    return box