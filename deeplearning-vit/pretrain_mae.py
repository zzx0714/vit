import argparse
import json
import logging
import math
from pathlib import Path

import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.data import build_loaders
from src.engine_mae import train_one_epoch_mae, evaluate_mae
from src.models_mae import MaskedAutoencoderViT
from src.utils import set_seed
import logging

def setup_logger(log_file):
    logger = logging.getLogger("MAE_Pretraining")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    exp_dir = Path("outputs") / config["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(exp_dir / "train.log")

    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and config["device"] == "auto" else "cpu"
    )
    logger.info(f"Using device: {device}")

    train_loader, test_loader = build_loaders(config)
    
    # Initialize MAE Model
    model_cfg = config["model"]
    model = MaskedAutoencoderViT(
        img_size=config["data"]["image_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=3,
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        decoder_embed_dim=model_cfg["decoder_embed_dim"],
        decoder_depth=model_cfg["decoder_depth"],
        decoder_num_heads=model_cfg["decoder_num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
        patch_embed_type=model_cfg.get("patch_embed_type", "linear"),
    )
    model.to(device)

    train_cfg = config["training"]
    # MAE usually benefits from an AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95)
    )

    # Warmup + Cosine Decay 调度器
    warmup_epochs = int(train_cfg.get("warmup_epochs", 20))
    total_epochs = train_cfg["epochs"]
    peak_lr = train_cfg["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性 warmup
        # cosine decay 阶段
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    mask_ratio = model_cfg.get("mask_ratio", 0.75)
    logger.info(f"Starting MAE Pretraining for {train_cfg['epochs']} epochs with mask_ratio {mask_ratio}")

    best_loss = float("inf")
    metrics_log = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_res = train_one_epoch_mae(
            model, train_loader, optimizer, device, epoch, mask_ratio=mask_ratio
        )
        test_res = evaluate_mae(model, test_loader, device, mask_ratio=mask_ratio)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_res['loss']:.4f} | "
            f"Val Loss: {test_res['loss']:.4f}"
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_res["loss"],
            "val_loss": test_res["loss"],
        }
        metrics_log.append(metrics)

        # Save last
        torch.save(model.state_dict(), exp_dir / "last.pt")
        # Save best
        if test_res["loss"] < best_loss:
            best_loss = test_res["loss"]
            torch.save(model.state_dict(), exp_dir / "best.pt")

    import csv
    with open(exp_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)

    with open(exp_dir / "summary.json", "w") as f:
        json.dump({"best_val_loss": best_loss}, f, indent=4)
        
    logger.info("MAE Pretraining completed.")

if __name__ == "__main__":
    main()
