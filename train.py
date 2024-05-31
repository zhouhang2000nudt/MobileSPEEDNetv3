import comet_ml
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import argparse

from MobileSPEEDv3.utils.config import get_config
from MobileSPEEDv3.utils.dataset import SpeedDataModule
from MobileSPEEDv3.module.Lightning_MobileSPEEDv3 import LightningMobileSPEEDv3

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.plugins import MixedPrecision, DoublePrecision, Precision, BitsandbytesPrecision, HalfPrecision
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelSummary, RichModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import StochasticWeightAveraging



if __name__ == "__main__":
    
    # ====================配置====================
    config = get_config()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--backbone", type=str, default=config["backbone"], help="backbone")
    parser.add_argument("--stride", type=int, default=config["stride"], help="stride")
    parser.add_argument("--sigma", type=int, default=config["sigma"], help="sigma")
    parser.add_argument("--tau", type=float, default=config["tau"], help="tau")
    parser.add_argument("--img_angle", type=float, default=config["Rotate"]["img_angle"], help="img_angle")
    parser.add_argument("--Rotatep", type=float, default=config["Rotate"]["p"], help="Rotatep")
    parser.add_argument("--CropAndPadp", type=float, default=config["CropAndPad"]["p"], help="CropAndPadp")
    parser.add_argument("--DropBlockSafep", type=float, default=config["DropBlockSafe"]["p"], help="DropBlockp")
    parser.add_argument("--Augmentationp", type=float, default=config["Augmentation"]["p"], help="augmentp")
    
    args = parser.parse_args()
    
    config["backbone"] = args.backbone
    config["stride"] = args.stride
    config["sigma"] = args.sigma
    config["tau"] = args.tau
    config["Rotate"]["img_angle"] = args.img_angle
    config["Rotate"]["p"] = args.Rotatep
    config["CropAndPad"]["p"] = args.CropAndPadp
    config["DropBlockSafe"]["p"] = args.DropBlockSafep
    config["Augmentation"]["p"] = args.Augmentationp
    
    config["name"] = f"{config['backbone']}-{config['stride']}_{config['tau']}_{config['sigma']}-{config['Rotate']['img_angle']}_{config['Rotate']['p']}-{config['CropAndPad']['p']}-{config['DropBlockSafe']['p']}-{config['Augmentation']['p']}"
    
    torch.set_float32_matmul_precision("high")
    
    dirpath = f"./result/{config['name']}-{time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())}"
    # 判断是否存在路径 若不存在则创建
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # 设置随机种子
    # seed_everything(config["seed"])


    # ===================训练器===================
    # =================callbacks=================
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # 进度条
    if config["bar"] == "rich":
        bar = RichProgressBar()
    elif config["bar"] == "tqdm":
        bar = TQDMProgressBar()
    # 保存模型
    checkpoint = ModelCheckpoint(dirpath=dirpath,
                                 filename="{epoch}-best",
                                 monitor="val/score",
                                 verbose=True,
                                 save_last=True,
                                 mode="min",
                                 save_weights_only=True)
    # 监控设备状态
    device_monitor = DeviceStatsMonitor(cpu_stats=None)
    # 模型总结
    if config["summary"] == "rich":
        summary = RichModelSummary(max_depth=3)
    elif config["summary"] == "default":
        summary = ModelSummary(max_depth=3)
    SWA = StochasticWeightAveraging(swa_lrs=config["lr_min"])
    callbacks = [lr_monitor, checkpoint, summary, bar]

    # ===================plugins=================
    plugins = []
    # 精度
    if config["precision"] == "mix":
        precision = MixedPrecision(precision="16-mixed", device="cuda" if config["accelerator"] == "gpu" else config["accelerator"])
    elif config["precision"] == "full":
        precision = Precision()
    elif config["precision"] == "double":
        precision = DoublePrecision()
    elif config["precision"] == "half":
        precision = HalfPrecision()
    elif config["precision"] == "int8":
        precision = BitsandbytesPrecision("int8")
    plugins = [precision]
    
    # ===================logger==================
    comet_logger = CometLogger(
        api_key=config["comet_api"],
        save_dir=dirpath,
        project_name="MobileSPEEDv3",
        experiment_name=config["name"] + "-" + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        offline=config["offline"]
    )
    
    # =================profiler==================
    profiler = SimpleProfiler(dirpath=dirpath)


    # ===================trainer=================
    if config["debug"]:
        limit_train_batches, limit_val_batches = config["limit_train_val"]
    else:
        limit_train_batches, limit_val_batches = 1.0, 1.0
    trainer = Trainer(accelerator=config["accelerator"],        # 加速器
                      logger=comet_logger,
                      callbacks=callbacks,
                      max_epochs=config["epoch"],
                      limit_train_batches=limit_train_batches,
                      limit_val_batches=limit_val_batches,
                      accumulate_grad_batches=config["accumulate_grad_batches"],
                      deterministic=config["deterministic"],
                      benchmark=config["benchmark"],
                    #   profiler=profiler,
                      plugins=plugins,
                    #   precision=precision,
                      default_root_dir=dirpath,
                      num_sanity_val_steps=0)

    # ====================模型====================
    # TODO Efficient initialization
    # with trainer.init_module():
    module = LightningMobileSPEEDv3(config)
    
    # ====================数据====================
    dataloader = SpeedDataModule(config=config)

    # ====================训练====================
    if config["train"]:
        trainer.fit(model=module, datamodule=dataloader)

    # ====================验证====================
    if config["val"]:
        module = LightningMobileSPEEDv3.load_from_checkpoint(module.trainer.callbacks[3].best_model_path)
        trainer.validate(model=module, datamodule=dataloader)