import os
import shutil
import zipfile

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

DATASET_STATS = {
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
    "tiny_imagenet": (TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
}


def get_normalize_params(dataset_name):
    """返回 (mean, std) 元组，供外部代码使用。"""
    if dataset_name not in DATASET_STATS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_STATS[dataset_name]


def _maybe_subset(dataset, limit):
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


# ──────────────────────────────────────────────
#  CIFAR-10
# ──────────────────────────────────────────────

def build_cifar10_loaders(config):
    data_cfg = config["data"]
    train_cfg = config["training"]

    image_size = int(data_cfg.get("image_size", 32))
    data_root = data_cfg.get("root", "data")
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 2))
    download = bool(data_cfg.get("download", True))

    # ── 检查 CIFAR-10 数据集是否已存在 ──
    cifar10_data_dir = os.path.join(data_root, "cifar-10-batches-py")
    expected_files = [f"data_batch_{i}" for i in range(1, 6)] + [
        "test_batch",
        "batches.meta",
    ]
    data_ready = os.path.isdir(cifar10_data_dir) and all(
        os.path.isfile(os.path.join(cifar10_data_dir, f)) for f in expected_files
    )

    if data_ready:
        print(f"✓ CIFAR-10 数据集已就绪: {os.path.abspath(cifar10_data_dir)}")
    else:
        print(
            f"⏳ CIFAR-10 数据集未找到，正在下载到 {os.path.abspath(data_root)} ..."
        )
        os.makedirs(data_root, exist_ok=True)

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

    # ── 下载后验证 ──
    if not data_ready and download:
        if all(os.path.isfile(os.path.join(cifar10_data_dir, f)) for f in expected_files):
            print(f"✓ CIFAR-10 数据集下载完成: {os.path.abspath(cifar10_data_dir)}")
        else:
            raise RuntimeError("CIFAR-10 数据集下载失败，请检查网络连接后重试")

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


# ──────────────────────────────────────────────
#  Tiny ImageNet
# ──────────────────────────────────────────────

_TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _prepare_tiny_imagenet(data_root):
    """检查、下载、解压并整理 Tiny ImageNet 数据集。"""
    dataset_dir = os.path.join(data_root, "tiny-imagenet-200")
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    val_reorganized = os.path.join(dataset_dir, "val_reorganized")

    # 检查是否已就绪
    if os.path.isdir(train_dir) and os.path.isdir(val_reorganized):
        print(f"✓ Tiny ImageNet 数据集已就绪: {os.path.abspath(dataset_dir)}")
        return dataset_dir

    os.makedirs(data_root, exist_ok=True)

    # 下载
    zip_path = os.path.join(data_root, "tiny-imagenet-200.zip")
    if not os.path.isfile(zip_path):
        print(f"⏳ Tiny ImageNet 数据集未找到，正在下载 ({_TINY_IMAGENET_URL}) ...")
        import urllib.request
        urllib.request.urlretrieve(_TINY_IMAGENET_URL, zip_path)
        print("✓ 下载完成，正在解压 ...")
    else:
        print("⏳ 发现已有 zip 文件，正在解压 ...")

    # 解压
    if not os.path.isdir(dataset_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_root)
        print("✓ 解压完成")

    # 重组 val 目录为 ImageFolder 格式
    if not os.path.isdir(val_reorganized):
        print("⏳ 正在整理验证集目录结构 ...")
        annotations_path = os.path.join(val_dir, "val_annotations.txt")
        val_images_dir = os.path.join(val_dir, "images")

        os.makedirs(val_reorganized, exist_ok=True)

        with open(annotations_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                img_name = parts[0]
                class_name = parts[1]
                class_dir = os.path.join(val_reorganized, class_name)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_images_dir, img_name)
                dst = os.path.join(class_dir, img_name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        print(f"✓ 验证集目录整理完成: {os.path.abspath(val_reorganized)}")

    return dataset_dir


def build_tiny_imagenet_loaders(config):
    data_cfg = config["data"]
    train_cfg = config["training"]

    image_size = int(data_cfg.get("image_size", 64))
    data_root = data_cfg.get("root", "data")
    batch_size = int(train_cfg.get("batch_size", 256))
    num_workers = int(train_cfg.get("num_workers", 2))

    dataset_dir = _prepare_tiny_imagenet(data_root)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val_reorganized")

    # Transforms
    if image_size == 64:
        train_transforms = [
            transforms.RandomCrop(64, padding=4),
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
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )
    test_transform = transforms.Compose(
        test_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )

    train_set = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(root=val_dir, transform=test_transform)

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


# ──────────────────────────────────────────────
#  调度器
# ──────────────────────────────────────────────

def build_loaders(config):
    dataset = config["data"].get("dataset", "cifar10").lower()
    if dataset == "cifar10":
        return build_cifar10_loaders(config)
    elif dataset == "tiny_imagenet":
        return build_tiny_imagenet_loaders(config)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
