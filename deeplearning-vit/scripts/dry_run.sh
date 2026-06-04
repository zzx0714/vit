#!/usr/bash
set -e

PYTHON="${1:-python}"

cd "$(dirname "$0")/.."

$PYTHON train.py --config configs/vanilla_vit.yaml --dry-run
$PYTHON train.py --config configs/convstem_vit.yaml --dry-run
$PYTHON train.py --config configs/resnet18.yaml --dry-run
