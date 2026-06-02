param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

Write-Host "=================================="
Write-Host "Starting MAE Pretraining (Phase 1)"
Write-Host "=================================="
& $Python pretrain_mae.py --config configs/mae_pretrain_cifar10.yaml

Write-Host "============================================="
Write-Host "Generating MAE Reconstructions Visualizations"
Write-Host "============================================="
& $Python visualize_mae.py --config configs/mae_pretrain_cifar10.yaml --checkpoint outputs/mae_pretrain_cifar10/best.pt --output results/mae_reconstruction.png

Write-Host "Done! Please check results/mae_reconstruction.png"
