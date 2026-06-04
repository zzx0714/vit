#!/usr/bash
set -e

PYTHON="${1:-python}"

cd "$(dirname "$0")/.."

echo "========================================"
echo "Phase 1: MAE Pretraining on Tiny ImageNet"
echo "========================================"

echo ""
echo ">>> [1/2] Linear Patch Embedding MAE"
$PYTHON pretrain_mae.py --config configs/mae_pretrain_tiny_imagenet.yaml

echo ""
echo ">>> [2/2] ConvStem Patch Embedding MAE"
$PYTHON pretrain_mae.py --config configs/mae_pretrain_convstem_tiny_imagenet.yaml

echo ""
echo "========================================"
echo "Phase 2: Fine-tune on CIFAR-10"
echo "========================================"

echo ""
echo ">>> [1/2] Vanilla ViT (from Linear MAE)"
$PYTHON finetune.py --config configs/finetune_vanilla_vit_cifar10.yaml

echo ""
echo ">>> [2/2] ConvStem-ViT (from ConvStem MAE)"
$PYTHON finetune.py --config configs/finetune_convstem_vit_cifar10.yaml

echo ""
echo "========================================"
echo "Phase 3: Generate Visualizations"
echo "========================================"
$PYTHON scripts/plot_all_results.py

echo ""
echo "Done! All results saved to results/"
