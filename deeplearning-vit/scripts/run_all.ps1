param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python train.py --config configs/resnet18.yaml
& $Python train.py --config configs/vanilla_vit.yaml
& $Python train.py --config configs/convstem_vit.yaml
& $Python plot_results.py --runs outputs/resnet18_cifar10 outputs/vanilla_vit_cifar10 outputs/convstem_vit_cifar10 --output results/training_curves.png
