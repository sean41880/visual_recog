"""
PromptIR: Prompting for All-in-One Blind Image Restoration
Reference: Potlapalli et al., NeurIPS 2023 (arXiv:2306.13090)

Architecture: Restormer-style encoder-decoder with prompt injection blocks.
Modifications in this impl:
  - Channel attention (SE) added inside GDFN for better feature selection
  - FFT-based frequency loss support (losses computed outside model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Channel-first layer normalisation (B, C, H, W)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DWConv(nn.Module):
    """Depth-wise 3×3 convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Transformer building blocks  (Restormer)
# ---------------------------------------------------------------------------

class MDTA(nn.Module):
    """Multi-Dconv Head Transposed Attention.

    Computes attention over channels (C×C) rather than spatial positions,
    so complexity is O(C²·HW) instead of O((HW)²·C).
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, 1, 1, groups=dim * 3, bias=bias)
        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # (B, num_heads, C_per_head, H*W)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v                                   # (B, h, C_h, HW)
        out = out.view(B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv Feed-Forward Network with SE channel attention.

    Modification vs. vanilla Restormer: adds a squeeze-excitation gate
    after the gated activation, letting the network recalibrate which
    channels carry useful degradation cues.
    """

    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.proj_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw = nn.Conv2d(hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=bias)
        self.proj_out = nn.Conv2d(hidden, dim, 1, bias=bias)
        # Squeeze-excitation on the hidden dimension
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, hidden // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden // 4, hidden, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1, x2 = self.dw(self.proj_in(x)).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = out * self.se(out)
        return self.proj_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = GDFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 2, 4, 2, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, bias=False)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Prompt blocks
# ---------------------------------------------------------------------------

class PromptGenBlock(nn.Module):
    """Generates prompt features conditioned on the degraded input image.

    A lightweight CNN pools the input to a fixed spatial size, then uses
    a learned dictionary of K prompt components; the attention over that
    dictionary is derived from global-average-pooled *feature* context.
    The generated prompt is bilinearly resized to match the feature map.

    Args:
        prompt_dim: channel dimension of generated prompt feature
        num_prompts: dictionary size K
        prompt_size: spatial resolution of the prompt dictionary entries
        context_dim: channel dimension of the feature map that generates
                     the attention weights (equals prompt_dim at each level)
    """

    def __init__(self, prompt_dim=48, num_prompts=5, prompt_size=64, context_dim=48):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.randn(1, num_prompts, prompt_dim, prompt_size, prompt_size) * 0.02
        )
        # Maps global-average-pooled feature to attention weights over dictionary
        self.attn_gen = nn.Linear(context_dim, num_prompts)
        # Refine the blended prompt spatially
        self.refine = nn.Conv2d(prompt_dim, prompt_dim, 3, 1, 1, bias=False)

    def forward(self, x):
        """x: (B, C, H, W) feature map."""
        B, C, H, W = x.shape
        gap = x.mean(dim=(-2, -1))                           # (B, C)
        w = F.softmax(self.attn_gen(gap), dim=1)             # (B, K)
        prompt = self.prompt_param.expand(B, -1, -1, -1, -1)  # (B, K, C_p, S, S)
        # Weighted combination of dictionary entries
        prompt = (w[:, :, None, None, None] * prompt).sum(dim=1)  # (B, C_p, S, S)
        # Resize to match feature map
        prompt = F.interpolate(prompt, size=(H, W), mode="bilinear", align_corners=False)
        return self.refine(prompt)


class PromptBlock(nn.Module):
    """Injects prompt into feature map via concatenation + 1×1 conv fusion."""

    def __init__(self, dim, num_prompts=5, prompt_size=64):
        super().__init__()
        self.prompt_gen = PromptGenBlock(
            prompt_dim=dim, num_prompts=num_prompts,
            prompt_size=prompt_size, context_dim=dim,
        )
        self.fusion = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x):
        prompt = self.prompt_gen(x)
        return self.fusion(torch.cat([x, prompt], dim=1))


# ---------------------------------------------------------------------------
# Full PromptIR model
# ---------------------------------------------------------------------------

class PromptIR(nn.Module):
    """
    Prompt-based image restoration network (PromptIR).

    Encoder-decoder (U-Net style) built from Restormer transformer blocks,
    with a PromptBlock injected at every encoder and decoder level to provide
    degradation-type conditioning via a learned prompt dictionary.

    Default config matches the medium-scale variant used in the paper.
    """

    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        num_prompts=5,
        prompt_sizes=(64, 32, 16),   # spatial size for each encoder level
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_blocks[0])]
        )
        self.pb_enc1 = PromptBlock(dim, num_prompts, prompt_sizes[0])
        self.down1 = Downsample(dim)

        self.enc2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
              for _ in range(num_blocks[1])]
        )
        self.pb_enc2 = PromptBlock(dim * 2, num_prompts, prompt_sizes[1])
        self.down2 = Downsample(dim * 2)

        self.enc3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
              for _ in range(num_blocks[2])]
        )
        self.pb_enc3 = PromptBlock(dim * 4, num_prompts, prompt_sizes[2])
        self.down3 = Downsample(dim * 4)

        # ---------- Bottleneck ----------
        self.bottleneck = nn.Sequential(
            *[TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias)
              for _ in range(num_blocks[3])]
        )

        # ---------- Decoder ----------
        self.up3 = Upsample(dim * 8)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dec3 = nn.Sequential(
            *[TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
              for _ in range(num_blocks[2])]
        )
        self.pb_dec3 = PromptBlock(dim * 4, num_prompts, prompt_sizes[2])

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dec2 = nn.Sequential(
            *[TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
              for _ in range(num_blocks[1])]
        )
        self.pb_dec2 = PromptBlock(dim * 2, num_prompts, prompt_sizes[1])

        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=bias)
        self.dec1 = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_blocks[0])]
        )
        self.pb_dec1 = PromptBlock(dim, num_prompts, prompt_sizes[0])

        # ---------- Refinement + output ----------
        self.refinement = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
              for _ in range(num_refinement_blocks)]
        )
        self.output = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, x):
        inp = x
        x = self.patch_embed(x)

        # Encoder
        e1 = self.pb_enc1(self.enc1(x))
        x = self.down1(e1)

        e2 = self.pb_enc2(self.enc2(x))
        x = self.down2(e2)

        e3 = self.pb_enc3(self.enc3(x))
        x = self.down3(e3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.reduce3(torch.cat([self.up3(x), e3], dim=1))
        x = self.pb_dec3(self.dec3(x))

        x = self.reduce2(torch.cat([self.up2(x), e2], dim=1))
        x = self.pb_dec2(self.dec2(x))

        x = self.reduce1(torch.cat([self.up1(x), e1], dim=1))
        x = self.pb_dec1(self.dec1(x))

        x = self.refinement(x)
        x = self.output(x) + inp   # residual: predict residual, not clean image
        return x
