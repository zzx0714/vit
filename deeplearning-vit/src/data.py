from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _maybe_subset(dataset, limit):
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def build_cifar10_loaders(config):
    data_cfg = config["data"]
    train_cfg = config["training"]

    image_size = int(data_cfg.get("image_size", 32))
    data_root = data_cfg.get("root", "data")
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 2))
    download = bool(data_cfg.get("download", True))

    if image_size == 32:
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        test_transforms = []
    else:
        train_transforms = [
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size, padding=max(4, image_size // 16)),
            transforms.RandomHorizontalFlip(),
        ]
        test_transforms = [transforms.Resize(image_size)]

    train_transform = transforms.Compose(
        train_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        test_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = datasets.CIFAR10(
        root=data_root, train=True, transform=train_transform, download=download
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, transform=test_transform, download=download
    )

    train_set = _maybe_subset(train_set, data_cfg.get("train_limit"))
    test_set = _maybe_subset(test_set, data_cfg.get("test_limit"))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_loaders(config):
    dataset = config["data"].get("dataset", "cifar10").lower()
    if dataset != "cifar10":
        raise ValueError(f"Unsupported dataset: {dataset}")
    return build_cifar10_loaders(config)
