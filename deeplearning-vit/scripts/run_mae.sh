#!/usr/bash
set -e

PYTHON="${1:-python}"

cd "$(dirname "$0")/.."

echo "=================================="
echo "Starting MAE Pretraining (Phase 1)"
echo "=================================="
$PYTHON pretrain_mae.py --config configs/mae_pretrain_cifar10.yaml

echo "============================================="
echo "Generating MAE Reconstructions Visualizations"
echo "============================================="
$PYTHON visualize_mae.py --config configs/mae_pretrain_cifar10.yaml --checkpoint outputs/mae_pretrain_cifar10/best.pt --output results/mae_reconstruction.png

echo "Done! Please check results/mae_reconstruction.png"
