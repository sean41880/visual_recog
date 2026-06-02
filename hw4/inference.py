"""Generate pred.npz from a trained PromptIR checkpoint.

Usage:
    python inference.py --checkpoint checkpoints/best.pth \\
                        --test_root /share/sean/visual_recog/hw4/hw4_realse_dataset/test \\
                        --output pred.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import TestDataset
from models import PromptIR


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pth")
    p.add_argument("--test_root",
                   default="/share/sean/visual_recog/hw4/hw4_realse_dataset/test")
    p.add_argument("--output", default="pred.npz")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_prompts", type=int, default=5)
    p.add_argument("--tile", type=int, default=0,
                   help="tile size for large images (0 = no tiling)")
    p.add_argument("--tile_overlap", type=int, default=32)
    return p.parse_args()


def tile_forward(model, img, tile_size, overlap, device):
    """Process one image (1, C, H, W) in overlapping tiles."""
    _, C, H, W = img.shape
    stride = tile_size - overlap
    out = torch.zeros_like(img)
    cnt = torch.zeros(1, 1, H, W, device=device)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1, x1 = min(y, H - tile_size), min(x, W - tile_size)
            y2, x2 = y1 + tile_size, x1 + tile_size
            tile = img[:, :, y1:y2, x1:x2]
            with torch.amp.autocast("cuda"):
                pred_tile = model(tile).clamp(0, 1)
            out[:, :, y1:y2, x1:x2] += pred_tile
            cnt[:, :, y1:y2, x1:x2] += 1

    return (out / cnt.clamp(min=1)).clamp(0, 1)


def main():
    args = get_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")

    model = PromptIR(dim=args.dim, num_prompts=args.num_prompts)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    # Handle DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device).eval()

    ds = TestDataset(args.test_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    results = {}
    for img, (fname,) in loader:
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            if args.tile > 0:
                pred = tile_forward(model, img, args.tile, args.tile_overlap, device)
            else:
                with torch.amp.autocast("cuda"):
                    pred = model(img).clamp(0, 1)

        # Convert to uint8 (3, H, W)
        arr = (pred[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        results[fname] = arr
        print(f"  {fname}: {arr.shape}")

    np.savez(args.output, **results)
    print(f"\nSaved {len(results)} images to {args.output}")


if __name__ == "__main__":
    main()
