"""Loss functions for image restoration.

Modifications / additions over vanilla L1:
  - FFT loss: penalises frequency-domain differences, which helps remove
    structured periodic artifacts (rain streaks, snow patterns).
  - SSIM loss: structural similarity perceptual term.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTLoss(nn.Module):
    """L1 loss in the 2-D frequency domain.

    Hypothesis: rain streaks and snow flakes are quasi-periodic spatial
    patterns, so they manifest as localised energy in the Fourier spectrum.
    Explicitly penalising frequency differences encourages the model to
    remove these structured artifacts that are hard to capture with pixel-
    space L1 alone.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_f = torch.fft.rfft2(pred, norm="ortho")
        tgt_f = torch.fft.rfft2(target, norm="ortho")
        # L1 on both real and imaginary parts
        loss = F.l1_loss(pred_f.real, tgt_f.real) + F.l1_loss(pred_f.imag, tgt_f.imag)
        return loss


class SSIMBlock(nn.Module):
    """Single-scale SSIM (returns 1 - SSIM as a loss)."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        kernel = self._gaussian_kernel(window_size, sigma)
        # shape: (1, 1, W, W) — applied channel-wise via groups
        self.register_buffer("kernel", kernel)

    @staticmethod
    def _gaussian_kernel(size, sigma):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel = g[:, None] * g[None, :]
        return kernel[None, None]

    def forward(self, pred, target):
        C = pred.shape[1]
        k = self.kernel.expand(C, 1, -1, -1)
        pad = self.window_size // 2

        mu_x = F.conv2d(pred, k, padding=pad, groups=C)
        mu_y = F.conv2d(target, k, padding=pad, groups=C)
        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sig_x2 = F.conv2d(pred ** 2, k, padding=pad, groups=C) - mu_x2
        sig_y2 = F.conv2d(target ** 2, k, padding=pad, groups=C) - mu_y2
        sig_xy = F.conv2d(pred * target, k, padding=pad, groups=C) - mu_xy

        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = (2 * mu_xy + c1) * (2 * sig_xy + c2)
        ssim /= (mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2)
        return 1.0 - ssim.mean()


class CombinedLoss(nn.Module):
    """L1 + lambda_fft * FFT_L1 + lambda_ssim * SSIM loss.

    Args:
        lambda_fft: weight for frequency-domain loss (default 0.1)
        lambda_ssim: weight for SSIM loss (default 0.1)
    """

    def __init__(self, lambda_fft=0.1, lambda_ssim=0.1):
        super().__init__()
        self.lambda_fft = lambda_fft
        self.lambda_ssim = lambda_ssim
        self.fft = FFTLoss()
        self.ssim = SSIMBlock()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        fft = self.fft(pred, target)
        ssim = self.ssim(pred, target)
        return l1 + self.lambda_fft * fft + self.lambda_ssim * ssim
