"""Utility helpers."""

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


def calc_psnr(pred, target, max_val=1.0):
    """Compute mean PSNR over a batch (both tensors in [0, 1])."""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    psnr = 10 * torch.log10(max_val ** 2 / (mse + 1e-8))
    return psnr.mean().item()
