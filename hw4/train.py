"""Training script for PromptIR image restoration.

Usage:
    python train.py [options]

Run with 4 GPUs (DataParallel):
    python train.py --gpus 0,1,2,3
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import PairDataset
from losses import CombinedLoss
from models import PromptIR
from utils import AverageMeter, calc_psnr


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/share/sean/visual_recog/hw4/hw4_realse_dataset/train")
    p.add_argument("--save_dir", default="/share/sean/visual_recog/hw4/checkpoints")
    p.add_argument("--gpus", default="0,1,2,3")
    # Model
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_prompts", type=int, default=5)
    # Training
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)  # per GPU
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--resume", default="")
    p.add_argument("--reset_best_psnr", action="store_true",
                   help="Reset best_psnr to 0 when resuming (useful for fine-tune runs)")
    p.add_argument("--lambda_fft", type=float, default=0.1)
    p.add_argument("--lambda_ssim", type=float, default=0.1)
    return p.parse_args()


def cosine_lr(optimizer, epoch, warmup_epochs, total_epochs, base_lr, min_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    losses = AverageMeter()
    psnrs = AverageMeter()

    for deg, clean in loader:
        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            pred = model(deg)
            pred_clamp = pred.clamp(0, 1)
            loss = criterion(pred_clamp, clean)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        scaler.step(optimizer)
        scaler.update()

        bs = deg.size(0)
        losses.update(loss.item(), bs)
        with torch.no_grad():
            psnrs.update(calc_psnr(pred_clamp, clean), bs)

    return losses.avg, psnrs.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    psnrs = AverageMeter()

    for deg, clean in loader:
        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            pred = model(deg).clamp(0, 1)
            loss = criterion(pred, clean)
        bs = deg.size(0)
        losses.update(loss.item(), bs)
        psnrs.update(calc_psnr(pred, clean), bs)

    return losses.avg, psnrs.avg


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")

    # Data
    train_ds = PairDataset(args.data_root, args.patch_size, augment=True, split="train")
    val_ds = PairDataset(args.data_root, None, augment=False, split="val")
    total_bs = args.batch_size * len(gpu_ids)
    train_loader = DataLoader(
        train_ds, batch_size=total_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=len(gpu_ids) * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = PromptIR(dim=args.dim, num_prompts=args.num_prompts)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)

    criterion = CombinedLoss(args.lambda_fft, args.lambda_ssim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best_psnr = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = 0.0 if args.reset_best_psnr else ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PSNR {best_psnr:.2f}")

    log_path = os.path.join(args.save_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,lr,train_loss,train_psnr,val_loss,val_psnr\n")

    for epoch in range(start_epoch, args.epochs):
        lr = cosine_lr(optimizer, epoch, args.warmup_epochs, args.epochs, args.lr, args.min_lr)
        t0 = time.time()

        tr_loss, tr_psnr = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_psnr = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  lr={lr:.2e}  "
            f"train: loss={tr_loss:.4f} PSNR={tr_psnr:.2f}  "
            f"val: loss={val_loss:.4f} PSNR={val_psnr:.2f}  "
            f"({elapsed:.0f}s)"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch},{lr:.6f},{tr_loss:.6f},{tr_psnr:.4f},{val_loss:.6f},{val_psnr:.4f}\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_psnr": best_psnr,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "last.pth"))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(ckpt, os.path.join(args.save_dir, "best.pth"))
            print(f"  -> New best PSNR: {best_psnr:.2f} dB")

    print(f"Training complete. Best val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
