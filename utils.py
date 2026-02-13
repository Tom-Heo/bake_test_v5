import os
import sys
import random
import logging
import torch
import numpy as np
import shutil


class ModelEMA:
    """Exponential Moving Average"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)

    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_logger(log_dir, log_filename="train.log"):
    """
    콘솔 + 파일 로거
    """
    logger_name = f"Bake_{os.path.splitext(log_filename)[0]}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # File
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(file_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    config, epoch, model, model_ema, optimizer, scheduler, is_best=False
):
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "ema_shadow": model_ema.shadow,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    last_path = config.LAST_CKPT_PATH
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, "best.pth")
        shutil.copyfile(last_path, best_path)


