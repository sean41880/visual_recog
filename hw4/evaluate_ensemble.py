"""Evaluate val PSNR for an ensemble of checkpoints with TTA."""

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


def load_model(checkpoint, dim, num_prompts, device):
    model = PromptIR(dim=dim, num_prompts=num_prompts)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--dims", nargs="+", type=int, required=True)
    p.add_argument("--data_root", default="/share/sean/visual_recog/hw4/hw4_realse_dataset/train")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_prompts", type=int, default=5)
    args = p.parse_args()

    assert len(args.checkpoints) == len(args.dims)
    device = torch.device(f"cuda:{args.gpu}")

    models = [load_model(c, d, args.num_prompts, device)
              for c, d in zip(args.checkpoints, args.dims)]
    print(f"Loaded {len(models)} models")

    ds = PairDataset(args.data_root, patch_size=None, augment=False, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    aug_modes = list(range(8))

    psnrs = AverageMeter()
    with torch.no_grad():
        for deg, clean in loader:
            deg = deg.to(device)
            clean = clean.to(device)
            all_preds = []
            for model in models:
                for aug in aug_modes:
                    all_preds.append(forward_aug(model, deg, aug))
            pred = torch.stack(all_preds).mean(0).clamp(0, 1)
            psnrs.update(calc_psnr(pred, clean))

    print(f"Ensemble val PSNR (TTA x{len(aug_modes)} x{len(models)} models): {psnrs.avg:.4f} dB")


if __name__ == "__main__":
    main()
