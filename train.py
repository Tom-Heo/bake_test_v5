import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR

# Local Modules
from config import Config
from data.dataset import BakeDataset
from core.net import BakeNet
from core.loss import BakeLoss
from core.augment import BakeAugment
from utils import (
    seed_everything,
    get_logger,
    ModelEMA,
    save_checkpoint,
)


def train(args):
    # 1. Setup
    seed_everything(42)
    Config.create_directories()

    logger = get_logger(Config.LOG_DIR)
    device = (
        torch.device(Config.DEVICE)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("Cudnn Benchmark Enabled.")
    else:
        logger.warning("CUDA not available. Falling back to CPU.")

    logger.info(f"Initialize Bake Training on {device}")

    # 2. Dataset & Loader
    logger.info("Loading Datasets...")
    train_dataset = BakeDataset(Config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Model & Components
    logger.info(f"Initializing BakeNet (Dim: {Config.INTERNAL_DIM})...")

    # GPU Augmentation Module
    augmentor = BakeAugment().to(device)

    model = BakeNet(dim=Config.INTERNAL_DIM).to(device)
    criterion = BakeLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler Setup
    warmup_epochs = 160
    steps_per_epoch = len(train_loader) // Config.ACCUM_STEPS
    total_warmup_iters = warmup_epochs * steps_per_epoch

    scheduler_warmup = LinearLR(
        optimizer, start_factor=0.01, total_iters=total_warmup_iters
    )
    scheduler_decay = ExponentialLR(optimizer, gamma=Config.SCHEDULER_GAMMA)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[total_warmup_iters],
    )

    model_ema = ModelEMA(model, decay=Config.EMA_DECAY)

    # -------------------------------------------------------------------------
    # [Resume Logic Control]
    # -------------------------------------------------------------------------
    start_epoch = 1
    ckpt_path = Config.LAST_CKPT_PATH
    should_load = False

    if args.restart:
        logger.info("🚫 Flag [--restart] detected. Ignoring existing checkpoints.")
        should_load = False
    elif args.resume:
        logger.info("🔄 Flag [--resume] detected. Attempting to load checkpoint.")
        if os.path.exists(ckpt_path):
            should_load = True
        else:
            logger.warning(
                f"⚠️ Checkpoint not found at {ckpt_path}. Starting from scratch."
            )
    else:
        # Default: Auto-resume if exists
        if os.path.exists(ckpt_path):
            logger.info("💾 Checkpoint found. Auto-resuming...")
            should_load = True
        else:
            logger.info("✨ No checkpoint found. Starting fresh training...")

    # Load Checkpoint
    if should_load and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        if "ema_shadow" in ckpt:
            model_ema.shadow = ckpt["ema_shadow"]

        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1

        logger.info(f"✅ Successfully resumed from Epoch {start_epoch-1}.")

    # 4. Training Loop
    logger.info(">>> Start Training Loop (No Validation Mode) <<<")

    accum_steps = Config.ACCUM_STEPS

    for epoch in range(start_epoch, Config.TOTAL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        opt_step = 0
        accum_loss = 0.0
        accum_count = 0

        for step, clean_batch in enumerate(train_loader, 1):
            clean_batch = clean_batch.to(device)

            with torch.no_grad():
                input_oklabp, target_oklabp = augmentor(clean_batch)

            pred_oklabp = model(input_oklabp)
            loss = criterion(pred_oklabp, target_oklabp)

            if torch.isnan(loss):
                logger.error(
                    f"[CRITICAL] NaN Loss at Epoch {epoch} Step {step}. Skipping."
                )
                continue

            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            epoch_loss += loss.item()
            accum_loss += loss.item()
            accum_count += 1

            is_accum_boundary = step % accum_steps == 0
            is_last_step = step == len(train_loader)

            if is_accum_boundary or is_last_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                model_ema.update(model)
                scheduler.step()
                opt_step += 1

                window_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

                if opt_step % Config.LOG_INTERVAL_STEPS == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.info(
                        f"Epoch [{epoch}/{Config.TOTAL_EPOCHS}] "
                        f"OptStep [{opt_step}/{steps_per_epoch}] "
                        f"Loss: {window_loss:.6f} "
                        f"LR: {current_lr:.2e}"
                    )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"==> Epoch {epoch} Finished. Avg Loss: {avg_loss:.6f}")

        # 5. Checkpoint Update (Validation 없이 저장만 수행)
        if epoch % Config.VALID_INTERVAL_EPOCHS == 0:
            logger.info(f"💾 Saving Checkpoint at Epoch {epoch}...")

            # is_best=False로 설정하여 best.pth 생성 방지
            save_checkpoint(
                Config, epoch, model, model_ema, optimizer, scheduler, is_best=False
            )

    logger.info("Training Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bake v5 Training Script")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume", action="store_true", help="Force resume from the last checkpoint."
    )
    group.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing checkpoints and start from Epoch 1.",
    )

    args = parser.parse_args()
    train(args)
