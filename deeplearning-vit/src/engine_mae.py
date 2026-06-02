import torch
import math
from typing import Iterable
from .utils import AverageMeter

def train_one_epoch_mae(model: torch.nn.Module,
                        data_loader: Iterable,
                        optimizer: torch.optim.Optimizer,
                        device: torch.device,
                        epoch: int,
                        mask_ratio: float = 0.75):
    model.train()
    loss_meter = AverageMeter()

    from tqdm import tqdm
    progress = tqdm(data_loader, desc=f"Epoch {epoch} MAE train", leave=False)

    for images, _ in progress: # Labels are not needed for self-supervised learning
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # Model returns loss, predicted patches, and the mask
        loss, _, _ = model(images, mask_ratio=mask_ratio)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            import sys; sys.exit(1)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return {"loss": loss_meter.avg}

@torch.no_grad()
def evaluate_mae(model: torch.nn.Module,
                 data_loader: Iterable,
                 device: torch.device,
                 mask_ratio: float = 0.75):
    model.eval()
    loss_meter = AverageMeter()

    from tqdm import tqdm
    progress = tqdm(data_loader, desc="MAE eval", leave=False)

    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        loss, _, _ = model(images, mask_ratio=mask_ratio)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)

    return {"loss": loss_meter.avg}
