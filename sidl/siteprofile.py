"""Site profiles: label-free, per-site acquisition statistics estimated from the training images,
plus a parametric translator that re-renders an image as if it came from another observed site.

Why this differs from generic augmentation: strengths are *measured* from the sites actually present
in training (contrast, noise floor, sharpness, low-frequency bias, per-channel gain), not hand-tuned.
For each image from site s we sample a target site t, sample a target profile from t's empirical
distribution, and apply only the *relative* change needed to go s -> t:
  - affine contrast/brightness match,
  - extra Gaussian noise of variance max(n_t^2 - n_s^2, 0),
  - extra blur (or mild unsharp) so sharpness matches, using a calibrated sharpness-vs-sigma curve,
  - multiplicative low-frequency bias of variance max(b_t^2 - b_s^2, 0),
  - per-channel gain (RGB).

Limitations (state these in the paper): the estimators are proxies (e.g. noise MAD includes fine texture),
the translation is not a physical simulator, and sites that differ in anatomy/population rather than
acquisition are not modelled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PARAM_NAMES = ["mean", "std", "noise", "sharp", "bias"]
_MEAN, _STD, _NOISE, _SHARP, _BIAS = range(5)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.05:
        return x
    k = int(2 * round(3 * sigma) + 1)
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.size(1)
    kh = g.view(1, 1, 1, k).repeat(c, 1, 1, 1)
    kv = g.view(1, 1, k, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), kh, groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), kv, groups=c)


def _box3(x):
    k = torch.ones(x.size(1), 1, 3, 3, device=x.device, dtype=x.dtype) / 9
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=x.size(1))


def _noise_sigma(g: torch.Tensor) -> torch.Tensor:
    """Robust white-noise std proxy: MAD of the 3x3 box high-pass residual (var(resid) = 8/9 sigma^2 for white noise)."""
    r = (g - _box3(g)).flatten(1)
    med = r.median(1).values
    return 1.4826 * (r - med[:, None]).abs().median(1).values * (9.0 / 8.0) ** 0.5


def _sharpness(g: torch.Tensor, compensate: bool = False) -> torch.Tensor:
    """Gradient-magnitude sharpness normalized by contrast. With compensate=True the white-noise contribution to the
    gradient energy (4 sigma^2 for adjacent-pixel differences) is removed first, so added noise no longer looks like sharpness."""
    if not compensate:
        gx = (g[..., :, 1:] - g[..., :, :-1]).abs().mean((1, 2, 3))
        gy = (g[..., 1:, :] - g[..., :-1, :]).abs().mean((1, 2, 3))
        return (gx + gy) / (g.std((1, 2, 3)) + 1e-6)
    gx2 = ((g[..., :, 1:] - g[..., :, :-1]) ** 2).mean((1, 2, 3))
    gy2 = ((g[..., 1:, :] - g[..., :-1, :]) ** 2).mean((1, 2, 3))
    energy = (gx2 + gy2 - 4 * _noise_sigma(g) ** 2).clamp(min=1e-8)
    return energy.sqrt() / (g.std((1, 2, 3)) + 1e-6)


@torch.no_grad()
def estimate_params(x: torch.Tensor, compensate: bool = False):
    """x: (B,C,H,W) in [0,1]. Returns (params (B,5), channel_mean (B,C)). compensate: noise-compensated sharpness."""
    g = x.mean(1, keepdim=True)
    r = (g - _box3(g)).flatten(1)
    med = r.median(1).values
    noise = 1.4826 * (r - med[:, None]).abs().median(1).values
    mean = g.mean((1, 2, 3))
    std = g.std((1, 2, 3))
    bias = F.adaptive_avg_pool2d(g, 4).std((1, 2, 3)) / (mean + 1e-6)
    P = torch.stack([mean, std, noise, _sharpness(g, compensate), bias], 1)
    return P, x.mean((2, 3))


class SiteProfiles:
    def __init__(self, num_sites: int, num_classes: int, compensate: bool = False):
        self.S, self.C, self.comp = num_sites, num_classes, compensate
        self.gmean = self.gstd = None     # global profile mean/std (for standardized adversary targets)
        self.mu = self.sd = None          # (S,5)
        self.cmu = self.csd = None        # (S,Cch)
        self.prior = None                 # (S,C) smoothed P(y|s)
        self.global_prior = None          # (C,)
        self.blur_sigmas = self.blur_ratio = None
        self.counts = None

    @torch.no_grad()
    def fit(self, ds, batch_size=64, num_workers=0, max_batches=None):
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        Ps, Cs, Ss, Ys, curve_imgs = [], [], [], [], None
        for i, (x, y, s) in enumerate(dl):
            if max_batches and i >= max_batches:
                break
            P, cm = estimate_params(x, self.comp)
            Ps.append(P); Cs.append(cm); Ss.append(s); Ys.append(y)
            if curve_imgs is None:
                curve_imgs = x[: min(32, len(x))]
        P, Cm, S, Y = torch.cat(Ps), torch.cat(Cs), torch.cat(Ss), torch.cat(Ys)
        mu, sd, cmu, csd = [], [], [], []
        counts = torch.zeros(self.S, self.C)
        for s in range(self.S):
            m = S == s
            if m.sum() < 2:  # unseen/tiny site: fall back to global stats
                m = torch.ones_like(S, dtype=torch.bool)
            mu.append(P[m].mean(0)); sd.append(P[m].std(0) + 1e-6)
            cmu.append(Cm[m].mean(0)); csd.append(Cm[m].std(0) + 1e-6)
            for c in range(self.C):
                counts[s, c] = ((S == s) & (Y == c)).sum()
        self.gmean, self.gstd = P.mean(0), P.std(0) + 1e-6
        self.mu, self.sd = torch.stack(mu), torch.stack(sd)
        self.cmu, self.csd = torch.stack(cmu), torch.stack(csd)
        self.counts = counts
        sm = counts + 1.0  # Laplace smoothing
        self.prior = sm / sm.sum(1, keepdim=True)
        self.global_prior = counts.sum(0) / counts.sum().clamp(min=1)
        # calibrate sharpness-vs-blur curve on a sample of images
        sig = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
        base = _sharpness(curve_imgs.mean(1, keepdim=True), self.comp).mean()
        ratio = torch.stack([_sharpness(gaussian_blur(curve_imgs, float(s_)).mean(1, keepdim=True), self.comp).mean() / base for s_ in sig])
        self.blur_sigmas, self.blur_ratio = sig, ratio
        return self

    def prior_weights(self) -> torch.Tensor:
        """w[s,y] = P_global(y) / P(y|s): importance weights that remove site-specific label priors."""
        gp = (self.global_prior + 1e-6) / (self.global_prior + 1e-6).sum()
        return (gp[None, :] / self.prior).clamp(max=10.0)

    def sigma_for_ratio(self, r: torch.Tensor) -> torch.Tensor:
        """Invert the (monotone decreasing) sharpness ratio curve; r<1 means blurrier target."""
        xs, ys = self.blur_ratio.numpy()[::-1], self.blur_sigmas.numpy()[::-1]
        return torch.from_numpy(np.interp(r.cpu().numpy(), xs, ys).astype(np.float32))

    def to(self, device):
        for k in ("mu", "sd", "cmu", "csd", "prior", "global_prior", "counts", "gmean", "gstd"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        return self

    def summary(self) -> dict:
        return {"param_names": PARAM_NAMES, "site_mean": self.mu.tolist(), "site_std": self.sd.tolist(),
                "prior": self.prior.tolist(), "blur_ratio": self.blur_ratio.tolist()}


class SiteTranslator(nn.Module):
    """Re-render each image as if acquired at a randomly chosen observed site. Identity in eval mode."""

    def __init__(self, profiles: SiteProfiles, jitter: float = 1.0, p: float = 0.9, alpha: float = 1.0):
        """p: probability a sample is translated; alpha in [0,1]: interpolation between clean (0) and fully translated (1)."""
        super().__init__()
        self.pr, self.jitter, self.p, self.alpha = profiles, jitter, p, alpha

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        pr, B, dev = self.pr, x.size(0), x.device
        pr.to(dev)
        t = torch.randint(0, pr.S, (B,), device=dev)
        s_ = s.clamp(0, pr.S - 1)
        tgt = pr.mu[t] + self.jitter * pr.sd[t] * torch.randn(B, len(PARAM_NAMES), device=dev)
        src = pr.mu[s_]
        keep = (torch.rand(B, device=dev) > self.p).view(B, 1, 1, 1)  # sometimes leave clean
        x0 = x
        # 1) contrast / brightness
        scale = (tgt[:, _STD].abs() / src[:, _STD].clamp(min=1e-4)).clamp(0.5, 2.0).view(B, 1, 1, 1)
        shift = (tgt[:, _MEAN] - src[:, _MEAN]).view(B, 1, 1, 1)
        mean_x = x.mean((1, 2, 3), keepdim=True)
        x = ((x - mean_x) * scale + mean_x + shift).clamp(0, 1)
        # 2) blur / unsharp so sharpness matches
        ratio = (tgt[:, _SHARP].abs() / src[:, _SHARP].clamp(min=1e-4)).clamp(0.2, 3.0)
        sig = pr.sigma_for_ratio(ratio.clamp(max=1.0)).to(dev)
        sig = torch.where(ratio < 1.0, sig, torch.zeros_like(sig))
        out = x.clone()
        for v in sig.round(decimals=1).unique():
            m = (sig.round(decimals=1) == v)
            if v > 0.05:
                out[m] = gaussian_blur(x[m], float(v))
        sharpen = (ratio >= 1.0).float().mul((ratio - 1.0).clamp(0, 1)).view(B, 1, 1, 1)
        out = out + sharpen * (out - gaussian_blur(out, 1.0))
        x = out.clamp(0, 1)
        # 3) extra noise
        extra_n = (tgt[:, _NOISE].abs() ** 2 - src[:, _NOISE] ** 2).clamp(min=0).sqrt().view(B, 1, 1, 1)
        x = (x + torch.randn_like(x) * extra_n).clamp(0, 1)
        # 4) extra low-frequency bias field
        extra_b = (tgt[:, _BIAS].abs() ** 2 - src[:, _BIAS] ** 2).clamp(min=0).sqrt().view(B, 1, 1, 1)
        coarse = 1 + (torch.rand(B, 1, 4, 4, device=dev) * 2 - 1) * extra_b * 1.732
        field = F.interpolate(coarse, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0.2, 2.0)
        x = (x * field).clamp(0, 1)
        # 5) per-channel gain (RGB only)
        if x.size(1) == 3 and pr.cmu.size(1) == 3:
            gain = (pr.cmu[t] / pr.cmu[s_].clamp(min=1e-4)).clamp(0.6, 1.6).view(B, 3, 1, 1)
            gain = 1 + self.jitter * (gain - 1)
            x = (x * gain).clamp(0, 1)
        x = x0 + self.alpha * (x - x0)
        return torch.where(keep, x0, x)
