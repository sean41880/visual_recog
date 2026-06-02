"""Inference with Test-Time Augmentation (TTA).

Averages predictions across 8 augmented views (4 flips × 2 rotations subset)
to squeeze extra PSNR without any retraining.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    return p.parse_args()


def forward_aug(model, img, aug):
    """Apply augmentation, run model, reverse augmentation."""
    x = img.clone()
    if aug & 1:
        x = torch.flip(x, dims=[3])   # horizontal flip
    if aug & 2:
        x = torch.flip(x, dims=[2])   # vertical flip
    if aug & 4:
        x = torch.rot90(x, 1, dims=[2, 3])

    with torch.amp.autocast("cuda"):
        pred = model(x).clamp(0, 1)

    # Reverse augmentation
    if aug & 4:
        pred = torch.rot90(pred, -1, dims=[2, 3])
    if aug & 2:
        pred = torch.flip(pred, dims=[2])
    if aug & 1:
        pred = torch.flip(pred, dims=[3])
    return pred


def main():
    args = get_args()
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")

    model = PromptIR(dim=args.dim, num_prompts=args.num_prompts)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device).eval()

    ds = TestDataset(args.test_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    results = {}
    aug_modes = [0, 1, 2, 3, 4, 5, 6, 7]   # all 8 flip+rot90 combinations

    with torch.no_grad():
        for img, (fname,) in loader:
            img = img.to(device, non_blocking=True)
            preds = []
            for aug in aug_modes:
                preds.append(forward_aug(model, img, aug))
            pred = torch.stack(preds, dim=0).mean(dim=0).clamp(0, 1)
            arr = (pred[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            results[fname] = arr
            print(f"  {fname}: {arr.shape}")

    np.savez(args.output, **results)
    print(f"\nSaved {len(results)} images (TTA x{len(aug_modes)}) to {args.output}")


if __name__ == "__main__":
    main()
