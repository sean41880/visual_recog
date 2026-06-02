"""Dataset classes for paired image restoration (Rain + Snow)."""

import os
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class PairDataset(Dataset):
    """Loads (degraded, clean) image pairs for rain and snow restoration.

    Args:
        root: path to dataset root (expects train/degraded, train/clean)
        patch_size: training crop size (None = use full image)
        augment: apply random flip/rotate augmentation
        split: 'train' or 'val'
        val_split: fraction of data held out for validation
        seed: random seed for deterministic split
    """

    def __init__(
        self,
        root,
        patch_size=128,
        augment=True,
        split="train",
        val_split=0.05,
        seed=42,
    ):
        self.patch_size = patch_size
        self.augment = augment

        deg_dir = os.path.join(root, "degraded")
        clean_dir = os.path.join(root, "clean")

        pairs = []
        for fname in sorted(os.listdir(deg_dir)):
            if not fname.lower().endswith(".png"):
                continue
            # rain-1.png  -> rain_clean-1.png
            # snow-1.png  -> snow_clean-1.png
            if fname.startswith("rain-"):
                clean_name = fname.replace("rain-", "rain_clean-")
            elif fname.startswith("snow-"):
                clean_name = fname.replace("snow-", "snow_clean-")
            else:
                continue
            deg_path = os.path.join(deg_dir, fname)
            clean_path = os.path.join(clean_dir, clean_name)
            if os.path.exists(clean_path):
                pairs.append((deg_path, clean_path))

        rng = random.Random(seed)
        rng.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_split))

        if split == "val":
            self.pairs = pairs[:n_val]
        else:
            self.pairs = pairs[n_val:]

    def __len__(self):
        return len(self.pairs)

    def _random_crop(self, deg, clean):
        ps = self.patch_size
        _, h, w = deg.shape
        if h < ps or w < ps:
            deg = TF.resize(deg, [ps, ps])
            clean = TF.resize(clean, [ps, ps])
            return deg, clean
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        deg = deg[:, top:top + ps, left:left + ps]
        clean = clean[:, top:top + ps, left:left + ps]
        return deg, clean

    def _augment(self, deg, clean):
        # Random horizontal flip
        if random.random() < 0.5:
            deg = TF.hflip(deg)
            clean = TF.hflip(clean)
        # Random vertical flip
        if random.random() < 0.5:
            deg = TF.vflip(deg)
            clean = TF.vflip(clean)
        # Random 90/180/270 rotation
        k = random.randint(0, 3)
        if k > 0:
            deg = torch.rot90(deg, k, dims=[1, 2])
            clean = torch.rot90(clean, k, dims=[1, 2])
        return deg, clean

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg = TF.to_tensor(Image.open(deg_path).convert("RGB"))
        clean = TF.to_tensor(Image.open(clean_path).convert("RGB"))

        if self.patch_size is not None:
            deg, clean = self._random_crop(deg, clean)
        if self.augment:
            deg, clean = self._augment(deg, clean)

        return deg, clean


class TestDataset(Dataset):
    """Loads test images (no clean counterpart).

    Returns (image_tensor, filename) for each test image.
    """

    def __init__(self, root):
        deg_dir = os.path.join(root, "degraded")
        self.files = sorted(
            [f for f in os.listdir(deg_dir) if f.lower().endswith(".png")],
            key=lambda f: int(os.path.splitext(f)[0]),
        )
        self.deg_dir = deg_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = TF.to_tensor(Image.open(os.path.join(self.deg_dir, fname)).convert("RGB"))
        return img, fname
