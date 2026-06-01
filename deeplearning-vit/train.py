import argparse
import time
from pathlib import Path

import torch
from torch import nn

from src.data import build_loaders
from src.engine import evaluate, train_one_epoch
from src.models import create_model
from src.utils import append_metrics_csv, get_device, load_config, save_config, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT variants on CIFAR-10.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for checkpoints and metrics. Defaults to outputs/<experiment_name>.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build model and dataloaders, run one forward pass, then exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))

    experiment_name = config.get("experiment_name", Path(args.config).stem)
    output_dir = Path(args.output_dir or Path("outputs") / experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    train_loader, test_loader = build_loaders(config)
    model = create_model(config).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.dry_run:
        images, _ = next(iter(train_loader))
        with torch.no_grad():
            logits = model(images[:2].to(device))
        print(f"dry-run ok: logits shape = {tuple(logits.shape)}")
        return

    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg.get("epochs", 50))
    )

    best_acc = 0.0
    best_epoch = 0
    start = time.time()
    metrics_path = output_dir / "metrics.csv"

    for epoch in range(1, int(train_cfg.get("epochs", 50)) + 1):
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
            "elapsed_seconds": round(time.time() - start, 2),
        },
    )


if __name__ == "__main__":
    main()
