#!/usr/bash
set -e

PYTHON="${1:-python}"

cd "$(dirname "$0")/.."

$PYTHON train.py --config configs/resnet18.yaml
$PYTHON train.py --config configs/vanilla_vit.yaml
$PYTHON train.py --config configs/convstem_vit.yaml
$PYTHON plot_results.py --runs outputs/resnet18_cifar10 outputs/vanilla_vit_cifar10 outputs/convstem_vit_cifar10 --output results/training_curves.png
