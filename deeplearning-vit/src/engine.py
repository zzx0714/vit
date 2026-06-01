import torch
from tqdm import tqdm

from .utils import AverageMeter, top1_accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    progress = tqdm(loader, desc=f"epoch {epoch} train", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits.detach(), targets), batch_size)
        progress.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}
