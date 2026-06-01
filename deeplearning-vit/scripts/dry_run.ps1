param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

& $Python train.py --config configs/vanilla_vit.yaml --dry-run
& $Python train.py --config configs/convstem_vit.yaml --dry-run
& $Python train.py --config configs/resnet18.yaml --dry-run
