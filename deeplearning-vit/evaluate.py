import argparse
from pathlib import Path

import torch
from torch import nn

from src.data import build_loaders
from src.engine import evaluate
from src.models import create_model
from src.utils import get_device, load_config, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--output", default=None, help="Optional JSON result path.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = load_config(args.config) if args.config else checkpoint["config"]
    device = get_device(config.get("device", "auto"))

    _, test_loader = build_loaders(config)
    model = create_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"test_loss={metrics['loss']:.4f} test_acc={metrics['acc']:.4f}")

    if args.output:
        save_json(Path(args.output), metrics)


if __name__ == "__main__":
    main()
