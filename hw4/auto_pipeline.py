"""Autonomous improvement pipeline.

Run once after run-1 training completes. Automatically:
  1. Evaluates val PSNR with and without TTA from run-1 checkpoint
  2. Generates pred.npz (with TTA) from run-1
  3. Fine-tunes with larger 192x192 patches for 100 more epochs (run-2)
  4. Evaluates run-2 + TTA
  5. Picks the best model and generates final pred.npz
  6. Optionally runs a run-3 with 256x256 full-res fine-tune (20 epochs)

Usage:
    python auto_pipeline.py [--run1_ckpt checkpoints/best.pth]
"""

import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PairDataset, TestDataset
from models import PromptIR
from utils import AverageMeter, calc_psnr

PYTHON = sys.executable
HW4 = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/share/sean/visual_recog/hw4/hw4_realse_dataset/train"
TEST_ROOT = "/share/sean/visual_recog/hw4/hw4_realse_dataset/test"
CKPT_DIR = os.path.join(HW4, "checkpoints")


def log(msg):
    print(f"[auto_pipeline] {msg}", flush=True)


def run(cmd):
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=HW4)
    if result.returncode != 0:
        log(f"WARNING: command exited {result.returncode}")
    return result.returncode


def eval_val_psnr(ckpt_path, tta=False, dim=48, num_prompts=5, device="cuda:0"):
    """Return val PSNR for a checkpoint."""
    gpu_ids = [0]
    dev = torch.device(device)

    model = PromptIR(dim=dim, num_prompts=num_prompts)
    state = torch.load(ckpt_path, map_location="cpu").get("model", {})
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(dev).eval()

    ds = PairDataset(DATA_ROOT, patch_size=None, augment=False, split="val")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    aug_modes = list(range(8)) if tta else [0]

    psnrs = AverageMeter()
    with torch.no_grad():
        for deg, clean in loader:
            deg, clean = deg.to(dev), clean.to(dev)
            preds = []
            for aug in aug_modes:
                x = deg.clone()
                if aug & 1:
                    x = torch.flip(x, [3])
                if aug & 2:
                    x = torch.flip(x, [2])
                if aug & 4:
                    x = torch.rot90(x, 1, [2, 3])
                with torch.amp.autocast("cuda"):
                    p = model(x).clamp(0, 1)
                if aug & 4:
                    p = torch.rot90(p, -1, [2, 3])
                if aug & 2:
                    p = torch.flip(p, [2])
                if aug & 1:
                    p = torch.flip(p, [3])
                preds.append(p)
            pred = torch.stack(preds).mean(0).clamp(0, 1)
            psnrs.update(calc_psnr(pred, clean))
    return psnrs.avg


def gen_pred_npz(ckpt_path, output_path, dim=48, num_prompts=5, tta=True):
    """Generate pred.npz with TTA from checkpoint."""
    import numpy as np
    dev = torch.device("cuda:0")
    model = PromptIR(dim=dim, num_prompts=num_prompts)
    state = torch.load(ckpt_path, map_location="cpu").get("model", {})
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(dev).eval()

    ds = TestDataset(TEST_ROOT)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    aug_modes = list(range(8)) if tta else [0]
    results = {}

    with torch.no_grad():
        for img, (fname,) in loader:
            img = img.to(dev)
            preds = []
            for aug in aug_modes:
                x = img.clone()
                if aug & 1:
                    x = torch.flip(x, [3])
                if aug & 2:
                    x = torch.flip(x, [2])
                if aug & 4:
                    x = torch.rot90(x, 1, [2, 3])
                with torch.amp.autocast("cuda"):
                    p = model(x).clamp(0, 1)
                if aug & 4:
                    p = torch.rot90(p, -1, [2, 3])
                if aug & 2:
                    p = torch.flip(p, [2])
                if aug & 1:
                    p = torch.flip(p, [3])
                preds.append(p)
            pred = torch.stack(preds).mean(0).clamp(0, 1)
            arr = (pred[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            results[fname] = arr
            log(f"  {fname} done")

    np.savez(output_path, **results)
    log(f"Saved {len(results)} images to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run1_ckpt", default=os.path.join(CKPT_DIR, "best.pth"))
    p.add_argument("--skip_run2", action="store_true")
    p.add_argument("--skip_run3", action="store_true")
    args = p.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ------------------------------------------------------------------ Run-1 eval
    log("=" * 60)
    log("Evaluating run-1 checkpoint ...")
    r1_psnr = eval_val_psnr(args.run1_ckpt, tta=False)
    r1_psnr_tta = eval_val_psnr(args.run1_ckpt, tta=True)
    log(f"Run-1 val PSNR: {r1_psnr:.4f} dB  | TTA: {r1_psnr_tta:.4f} dB")

    log("Generating run-1 pred.npz with TTA ...")
    gen_pred_npz(args.run1_ckpt, os.path.join(HW4, "pred_run1_tta.npz"), tta=True)

    best_ckpt = args.run1_ckpt
    best_psnr = r1_psnr_tta

    # ------------------------------------------------------------------ Run-2: fine-tune at 192
    if not args.skip_run2:
        log("=" * 60)
        log("Starting run-2: fine-tune with 192x192 patches ...")
        r2_ckpt_dir = os.path.join(CKPT_DIR, "run2")
        os.makedirs(r2_ckpt_dir, exist_ok=True)
        run([
            PYTHON, "train.py",
            "--data_root", DATA_ROOT,
            "--save_dir", r2_ckpt_dir,
            "--gpus", "0,1,2,3",
            "--epochs", "100",
            "--batch_size", "4",      # smaller batch for larger patches
            "--patch_size", "192",
            "--lr", "1e-4",
            "--min_lr", "1e-6",
            "--warmup_epochs", "3",
            "--num_workers", "8",
            "--dim", "48",
            "--num_prompts", "5",
            "--lambda_fft", "0.1",
            "--lambda_ssim", "0.1",
            "--resume", args.run1_ckpt,
        ])
        r2_ckpt = os.path.join(r2_ckpt_dir, "best.pth")
        if os.path.exists(r2_ckpt):
            r2_psnr_tta = eval_val_psnr(r2_ckpt, tta=True)
            log(f"Run-2 val PSNR TTA: {r2_psnr_tta:.4f} dB")
            if r2_psnr_tta > best_psnr:
                best_psnr = r2_psnr_tta
                best_ckpt = r2_ckpt
                log(f"Run-2 is better! New best: {best_psnr:.4f} dB")
            else:
                log(f"Run-1 still best ({best_psnr:.4f} dB)")
        else:
            log("Run-2 checkpoint not found, skipping.")

    # ------------------------------------------------------------------ Run-3: fine-tune at 256
    if not args.skip_run3:
        log("=" * 60)
        log("Starting run-3: fine-tune at full 256x256 resolution ...")
        r3_ckpt_dir = os.path.join(CKPT_DIR, "run3")
        os.makedirs(r3_ckpt_dir, exist_ok=True)
        run([
            PYTHON, "train.py",
            "--data_root", DATA_ROOT,
            "--save_dir", r3_ckpt_dir,
            "--gpus", "0,1,2,3",
            "--epochs", "30",
            "--batch_size", "2",      # 256x256 is memory-heavy
            "--patch_size", "256",
            "--lr", "5e-5",
            "--min_lr", "1e-6",
            "--warmup_epochs", "2",
            "--num_workers", "8",
            "--dim", "48",
            "--num_prompts", "5",
            "--lambda_fft", "0.1",
            "--lambda_ssim", "0.1",
            "--resume", best_ckpt,
        ])
        r3_ckpt = os.path.join(r3_ckpt_dir, "best.pth")
        if os.path.exists(r3_ckpt):
            r3_psnr_tta = eval_val_psnr(r3_ckpt, tta=True)
            log(f"Run-3 val PSNR TTA: {r3_psnr_tta:.4f} dB")
            if r3_psnr_tta > best_psnr:
                best_psnr = r3_psnr_tta
                best_ckpt = r3_ckpt
                log(f"Run-3 is better! New best: {best_psnr:.4f} dB")
            else:
                log(f"Previous best still holds ({best_psnr:.4f} dB)")
        else:
            log("Run-3 checkpoint not found, skipping.")

    # ------------------------------------------------------------------ Final inference
    log("=" * 60)
    log(f"Generating FINAL pred.npz from best checkpoint ({best_ckpt}) ...")
    log(f"Best val PSNR (TTA): {best_psnr:.4f} dB")
    gen_pred_npz(best_ckpt, os.path.join(HW4, "pred.npz"), tta=True)
    log("Done! pred.npz is ready for submission.")


if __name__ == "__main__":
    main()
