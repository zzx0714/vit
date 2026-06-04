import argparse
import time
from pathlib import Path

import torch
from torch import nn

from src.data import build_loaders
from src.engine import evaluate, train_one_epoch
from src.models import create_model
from src.utils import append_metrics_csv, get_device, load_config, save_config, save_json, set_seed


def load_pretrained_encoder(model, checkpoint_path, device):
    """
    从 MAE checkpoint 中提取 encoder 权重，映射到 ViT 模型。

    权重映射：
      MAE: blocks.layers.{i}.*  →  ViT: encoder.layers.{i}.*
      MAE: patch_embed.*        →  跳过（分辨率不同）
      MAE: cls_token / pos_embed / norm.*  →  直接复制
      MAE: decoder_* / mask_token          →  跳过（decoder 丢弃）
      ViT: head.*               →  保持随机初始化
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # 判断 checkpoint 格式：train.py 保存为 dict，pretrain_mae.py 保存为 state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        mae_state = ckpt["model"]
    else:
        mae_state = ckpt

    vit_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []

    for mae_key, tensor in mae_state.items():
        # 跳过 decoder 权重
        if mae_key.startswith(("decoder_", "mask_token")):
            skipped_keys.append(mae_key)
            continue

        # 映射 encoder blocks → encoder
        vit_key = mae_key.replace("blocks.", "encoder.")

        # 跳过不匹配的 key（patch_embed 因分辨率不同可能 shape 不一致）
        if vit_key not in vit_state:
            skipped_keys.append(mae_key)
            continue

        if vit_state[vit_key].shape != tensor.shape:
            skipped_keys.append(f"{mae_key} (shape mismatch: {tensor.shape} vs {vit_state[vit_key].shape})")
            continue

        vit_state[vit_key] = tensor
        loaded_keys.append(vit_key)

    model.load_state_dict(vit_state)

    print(f"  ✓ 加载预训练权重: {len(loaded_keys)} 个 key 已映射")
    print(f"  ⏭ 跳过: {len(skipped_keys)} 个 key (decoder/shape 不匹配)")
    print(f"  ✓ 分类头 head 保持随机初始化")

    return model


def freeze_encoder(model):
    """冻结 encoder 和 patch_embed，只训练分类头。"""
    frozen = 0
    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False
            frozen += 1
    print(f"  ❄ 冻结了 {frozen} 个参数 (encoder)")


def unfreeze_encoder(model):
    """解冻全部参数。"""
    for param in model.parameters():
        param.requires_grad = True
    print("  🔥 解冻全部参数")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT from MAE pretrained checkpoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def run():
    args = main()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))

    experiment_name = config.get("experiment_name", Path(args.config).stem)
    output_dir = Path(args.output_dir or Path("outputs") / experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    # ── 数据 ──
    train_loader, test_loader = build_loaders(config)

    # ── 模型 ──
    model = create_model(config).to(device)

    # ── 加载预训练权重 ──
    finetune_cfg = config.get("finetune", {})
    checkpoint_path = finetune_cfg.get("checkpoint")
    freeze_epochs = int(finetune_cfg.get("freeze_epochs", 0))

    if checkpoint_path:
        print(f"从 MAE checkpoint 加载预训练 encoder: {checkpoint_path}")
        model = load_pretrained_encoder(model, checkpoint_path, device)
    else:
        print("⚠ 未指定预训练 checkpoint，将从零训练")

    # ── 损失函数 ──
    criterion = nn.CrossEntropyLoss()

    # ── 优化器 ──
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg.get("epochs", 50))
    )

    # ── 冻结阶段 ──
    if freeze_epochs > 0 and checkpoint_path:
        freeze_encoder(model)
        # 冻结阶段用更大学习率训练 head
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=float(train_cfg.get("lr", 3e-4)) * 10,
            weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        )

    best_acc = 0.0
    best_epoch = 0
    start = time.time()
    metrics_path = output_dir / "metrics.csv"

    total_epochs = int(train_cfg.get("epochs", 50))

    for epoch in range(1, total_epochs + 1):
        # 解冻时机
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and checkpoint_path:
            unfreeze_encoder(model)
            # 重建优化器，包含全部参数，用正常学习率
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(train_cfg.get("lr", 3e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 0.05)),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - freeze_epochs
            )

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        append_metrics_csv(metrics_path, row)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "test_acc": test_metrics["acc"],
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "best.pt")

        print(
            f"epoch={epoch:03d} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"test_acc={test_metrics['acc']:.4f} "
            f"best={best_acc:.4f}"
        )

    save_json(
        output_dir / "summary.json",
        {
            "experiment_name": experiment_name,
            "best_epoch": best_epoch,
            "best_test_acc": best_acc,
            "pretrained_from": checkpoint_path or "scratch",
            "freeze_epochs": freeze_epochs,
            "elapsed_seconds": round(time.time() - start, 2),
        },
    )


if __name__ == "__main__":
    run()
