import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


def _has_images(folder: str) -> bool:
    """Return True if the folder contains at least one class subdir with images."""
    if not os.path.isdir(folder):
        return False
    for entry in os.listdir(folder):
        sub = os.path.join(folder, entry)
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f.lower().endswith(IMG_EXTS):
                    return True
    return False


def _download_cifar_as_imagefolder(root: str) -> None:
    """Download CIFAR-10 and write it as train/ and val/ image folders."""
    from PIL import Image

    print("[data] No images found. Downloading CIFAR-10 and building image folders...")
    cache = os.path.join(root, "_cifar_cache")
    os.makedirs(cache, exist_ok=True)
    classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    for split, is_train in (("train", True), ("val", False)):
        ds = datasets.CIFAR10(root=cache, train=is_train, download=True)
        for cls in classes:
            os.makedirs(os.path.join(root, split, cls), exist_ok=True)
        for idx, (img, label) in enumerate(ds):
            cls = classes[label]
            out = os.path.join(root, split, cls, f"{split}_{idx:06d}.png")
            if isinstance(img, Image.Image):
                img.save(out)
    print("[data] CIFAR-10 image folders ready.")


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training (augmented) and evaluation image transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def build_dataloaders(config) -> Tuple[DataLoader, DataLoader, int, list]:
    """Build train/val dataloaders, downloading a public dataset if needed."""
    dcfg = config["data"]
    root = dcfg["root"]
    os.makedirs(root, exist_ok=True)

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    if not _has_images(train_dir):
        if dcfg.get("auto_download", True):
            _download_cifar_as_imagefolder(root)
        else:
            raise FileNotFoundError(
                f"No images found under {train_dir} and auto_download is disabled."
            )

    train_tf, eval_tf = _build_transforms(dcfg["image_size"])

    full_train = datasets.ImageFolder(train_dir, transform=train_tf)
    classes = full_train.classes
    num_classes = len(classes)

    if _has_images(val_dir):
        train_ds = full_train
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    else:
        val_len = max(1, int(len(full_train) * dcfg.get("val_split", 0.2)))
        train_len = len(full_train) - val_len
        generator = torch.Generator().manual_seed(config.get("seed", 42))
        train_ds, val_subset = random_split(
            full_train, [train_len, val_len], generator=generator
        )
        # Use eval transforms for the validation subset.
        val_base = datasets.ImageFolder(train_dir, transform=eval_tf)
        val_ds = torch.utils.data.Subset(val_base, val_subset.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=dcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=dcfg["batch_size"],
        shuffle=False,
        num_workers=dcfg["num_workers"],
        drop_last=False,
    )
    return train_loader, val_loader, num_classes, classes
