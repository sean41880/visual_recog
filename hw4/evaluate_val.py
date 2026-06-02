"""Compute val PSNR from a checkpoint, with optional TTA."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PairDataset
from models import PromptIR
from utils import AverageMeter, calc_psnr


def forward_aug(model, img, aug):
    x = img.clone()
    if aug & 1:
        x = torch.flip(x, dims=[3])
    if aug & 2:
        x = torch.flip(x, dims=[2])
    if aug & 4:
        x = torch.rot90(x, 1, dims=[2, 3])
    with torch.amp.autocast("cuda"):
        pred = model(x).clamp(0, 1)
    if aug & 4:
        pred = torch.rot90(pred, -1, dims=[2, 3])
    if aug & 2:
        pred = torch.flip(pred, dims=[2])
    if aug & 1:
        pred = torch.flip(pred, dims=[3])
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root",
                   default="/share/sean/visual_recog/hw4/hw4_realse_dataset/train")
    p.add_argument("--gpus", default="0")
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_prompts", type=int, default=5)
    p.add_argument("--tta", action="store_true")
    args = p.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")

    model = PromptIR(dim=args.dim, num_prompts=args.num_prompts)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).eval()

    ds = PairDataset(args.data_root, patch_size=None, augment=False, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    aug_modes = list(range(8)) if args.tta else [0]

    psnrs = AverageMeter()
    with torch.no_grad():
        for deg, clean in loader:
            deg = deg.to(device)
            clean = clean.to(device)
            preds = [forward_aug(model, deg, a) for a in aug_modes]
            pred = torch.stack(preds).mean(0).clamp(0, 1)
            psnrs.update(calc_psnr(pred, clean))

    tag = "TTA" if args.tta else "no-TTA"
    print(f"Val PSNR ({tag}): {psnrs.avg:.4f} dB")


if __name__ == "__main__":
    main()
