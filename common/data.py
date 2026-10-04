"""Dataset, preprocessing and the frozen train / val / test split.

Shared by all four models (work-plan item 1).

* Training/ (5,600 images) is split 90 / 10 into train and validation with a
  fixed seed, so every model sees exactly the same images.
* Testing/ (1,600 images) is the official hold-out set and is only used for the
  final evaluation.
* Train images get light augmentation (horizontal flip, +/-10 deg rotation);
  validation and test images are only resized and normalised.
"""
import glob
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms

from .config import (
    BATCH_SIZE, CLASSES, IMG_SIZE, NUM_WORKERS, SEED, TEST_DIR, TRAIN_DIR, VAL_FRACTION,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for idx, cls in enumerate(CLASSES):
            # sorted() makes the file order (and therefore the split) identical on every machine
            for path in sorted(glob.glob(os.path.join(root_dir, cls, "*"))):
                self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def describe_dataset():
    """Print the class folders and image counts found on disk."""
    print("Train classes found:", sorted(os.listdir(TRAIN_DIR)))
    print("Test classes found: ", sorted(os.listdir(TEST_DIR)))

    for cls in sorted(os.listdir(TRAIN_DIR)):
        n = len(os.listdir(os.path.join(TRAIN_DIR, cls)))
        print(f"  {cls}: {n} images")


def get_splits():
    """Return (train_ds, val_ds, test_ds)."""
    train_full = BrainTumorDataset(TRAIN_DIR, transform=train_transform)
    val_full = BrainTumorDataset(TRAIN_DIR, transform=val_transform)  # same files, no augmentation

    n_val = int(VAL_FRACTION * len(train_full))
    n_train = len(train_full) - n_val
    train_idx, val_idx = random_split(
        range(len(train_full)), [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_ds = Subset(train_full, list(train_idx))
    val_ds = Subset(val_full, list(val_idx))
    test_ds = BrainTumorDataset(TEST_DIR, transform=val_transform)
    return train_ds, val_ds, test_ds


def get_loaders():
    """Return (train_loader, val_loader, test_loader) and print the split sizes."""
    train_ds, val_ds, test_ds = get_splits()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
