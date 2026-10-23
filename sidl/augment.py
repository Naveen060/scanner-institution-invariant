"""Generic scanner-variation augmentations (the non-learned baseline) with explicit strength and per-sample probability.

All ops work on float tensors (B,C,H,W) in [0,1]. `strength` in [0,1] scales every range; `p` is the probability that a
given SAMPLE is augmented at all (the rest stay clean so BatchNorm statistics remain compatible with clean test images).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rand(b, lo, hi, device):
    return torch.empty(b, 1, 1, 1, device=device).uniform_(lo, hi)


def random_gamma(x, s=1.0):
    return x.clamp(1e-4, 1).pow(_rand(x.size(0), 1 - 0.4 * s, 1 + 0.6 * s, x.device))


def random_contrast_brightness(x, s=1.0):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * _rand(x.size(0), 1 - 0.3 * s, 1 + 0.3 * s, x.device) + mean
            + _rand(x.size(0), -0.1 * s, 0.1 * s, x.device)).clamp(0, 1)


def random_blur(x, s=1.0):
    sigma = float(torch.empty(1).uniform_(0.1, max(0.15, 1.5 * s)))
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.size(1)
    kh = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kv = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kh, groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), kv, groups=c)


def random_noise(x, s=1.0):
    std = _rand(x.size(0), 0.0, 0.05 * s, x.device)
    return (x + torch.randn_like(x) * std).clamp(0, 1)


def random_bias_field(x, s=1.0, grid=4):
    b, _, h, w = x.shape
    coarse = 1 + (torch.rand(b, 1, grid, grid, device=x.device) - 0.5) * 2 * 0.3 * s
    return (x * F.interpolate(coarse, size=(h, w), mode="bicubic", align_corners=False)).clamp(0, 1)


def random_color_jitter(x, s=1.0):
    if x.size(1) != 3:
        return x
    b = x.size(0)
    scale = 1 + (torch.rand(b, 3, 1, 1, device=x.device) - 0.5) * 2 * 0.15 * s
    shift = (torch.rand(b, 3, 1, 1, device=x.device) - 0.5) * 0.15 * s
    return (x * scale + shift).clamp(0, 1)


class Harmonizer(torch.nn.Module):
    """Randomly composes site-simulating transforms on a random subset of the batch. Identity at eval time."""

    def __init__(self, p: float = 0.5, strength: float = 1.0, op_p: float = 0.6, modality: str = "cxr"):
        super().__init__()
        self.p, self.strength, self.op_p, self.modality = p, strength, op_p, modality

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        x0 = x
        ops = [random_gamma, random_contrast_brightness, random_blur, random_noise, random_bias_field]
        if self.modality in ("histo", "fundus"):
            ops.append(random_color_jitter)
        for op in ops:
            if torch.rand(1).item() < self.op_p:
                x = op(x, self.strength)
        keep_aug = (torch.rand(x.size(0), 1, 1, 1, device=x.device) < self.p)
        return torch.where(keep_aug, x, x0)
