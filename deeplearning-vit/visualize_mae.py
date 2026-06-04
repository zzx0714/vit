import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data import build_loaders, get_normalize_params
from src.models_mae import MaskedAutoencoderViT

def unpatchify(x, patch_size=4, img_size=32):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = w = img_size // p
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def denormalize(img, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1).to(img.device)
    std = torch.tensor(std).view(3, 1, 1).to(img.device)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file used for pretraining")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default="results/mae_reconstruction.png")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model_cfg = config["model"]
    model = MaskedAutoencoderViT(
        img_size=config["data"]["image_size"],
        patch_size=model_cfg["patch_size"],
        in_chans=3,
        embed_dim=model_cfg["embed_dim"],
        depth=model_cfg["depth"],
        num_heads=model_cfg["num_heads"],
        decoder_embed_dim=model_cfg["decoder_embed_dim"],
        decoder_depth=model_cfg["decoder_depth"],
        decoder_num_heads=model_cfg["decoder_num_heads"],
        mlp_ratio=model_cfg["mlp_ratio"],
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    # 获取归一化参数
    dataset_name = config["data"].get("dataset", "cifar10").lower()
    norm_mean, norm_std = get_normalize_params(dataset_name)

    # Get data
    config["data"]["train_limit"] = 16 # just need a few
    config["data"]["test_limit"] = 16
    _, test_loader = build_loaders(config)
    images, _ = next(iter(test_loader))
    images = images[:6].to(device) # visualize 6 images

    # Run MAE inference
    mask_ratio = model_cfg.get("mask_ratio", 0.75)
    with torch.no_grad():
        # Get predictions
        loss, pred, mask = model(images, mask_ratio=mask_ratio)
        
        # Denormalize target for visualization
        target = model.patchify(images)
        mean_tar = target.mean(dim=-1, keepdim=True)
        var_tar = target.var(dim=-1, keepdim=True)
        target_norm = (target - mean_tar) / (var_tar + 1e-6)**.5

        # We want to show original, masked, and reconstructed.
        # However, prediction is predicting normalized target pixels.
        # To visualize the reconstructed image, we un-normalize the predicted patches
        pred_unnorm = pred * (var_tar + 1e-6)**.5 + mean_tar

        # Combining visible patches and predicted patches
        # mask = 1 means masked (predict it), mask = 0 means visible (keep original)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, target.shape[2])
        im_paste = target * (1 - mask_expanded) + pred_unnorm * mask_expanded
        
        # Masked image (target multiplied by inverse mask)
        im_masked = target * (1 - mask_expanded)
        
        # Unpatchify into images
        imgs_orig = denormalize(images, norm_mean, norm_std).cpu()
        imgs_recon = unpatchify(im_paste, model_cfg["patch_size"], config["data"]["image_size"])
        imgs_recon = denormalize(imgs_recon, norm_mean, norm_std).cpu()
        
        imgs_masked = unpatchify(im_masked, model_cfg["patch_size"], config["data"]["image_size"])
        imgs_masked = denormalize(imgs_masked, norm_mean, norm_std).cpu()

    # Plotting
    import os
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    fig.suptitle('MAE Reconstruction (Top: Original, Middle: 75% Masked, Bottom: Reconstructed)', fontsize=16)

    for i in range(6):
        axes[0, i].imshow(np.transpose(imgs_orig[i].numpy(), (1, 2, 0)))
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.transpose(imgs_masked[i].numpy(), (1, 2, 0)))
        axes[1, i].axis('off')
        
        axes[2, i].imshow(np.transpose(imgs_recon[i].numpy(), (1, 2, 0)))
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Reconstruction visualization saved to {args.output}")

if __name__ == "__main__":
    main()
