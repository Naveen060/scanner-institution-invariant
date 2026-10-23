"""Semi-synthetic multi-site benchmark with KNOWN acquisition effects and CONTROLLED label shift.

Real chest X-ray images (pooled from the open collections) are randomly assigned to K virtual sites. Each site k has
  - a class prior pi_k (prior_spread controls how different the priors are; 0 = identical priors),
  - a known acquisition transform T_k (gamma, contrast, blur sigma, noise std, low-frequency bias), ordered along k so that
    acquisition signature is correlated with prior (the worst case for site shortcuts).
Because T_k is known, we can (a) sweep label shift in isolation and (b) check that estimated site profiles recover the truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .siteprofile import gaussian_blur


def site_transform_params(K: int, severity: float = 1.0):
    """Monotone family of acquisition settings across sites (site 0 = 'clean', site K-1 = most degraded).
    severity scales how far the most degraded site is from the clean one."""
    t = np.linspace(0, 1, K) * severity
    return [dict(gamma=1.0 + 0.8 * a, contrast=1.0 - 0.35 * a, blur=0.2 + 1.6 * a, noise=0.01 + 0.07 * a, bias=0.02 + 0.30 * a)
            for a in t]


def apply_site(x: torch.Tensor, p: dict, gen: torch.Generator) -> torch.Tensor:
    x = x.clamp(1e-4, 1).pow(p["gamma"])
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * p["contrast"] + m).clamp(0, 1)
    x = gaussian_blur(x, p["blur"])
    B, _, H, W = x.shape
    coarse = 1 + (torch.rand(B, 1, 4, 4, generator=gen) * 2 - 1) * p["bias"] * 1.732
    x = x * F.interpolate(coarse, size=(H, W), mode="bicubic", align_corners=False).clamp(0.3, 2.0)
    x = x + torch.randn(x.shape, generator=gen) * p["noise"]
    return x.clamp(0, 1)


def make_virtual_sites(X: np.ndarray, y: np.ndarray, K: int = 5, prior_spread: float = 0.3, per_site: int = 140,
                       seed: int = 0, size: int = 64, severity: float = 1.0):
    """Returns (Xv uint8 [N,size,size], df[row,label,site,patient_id], true_params list)."""
    rng = np.random.RandomState(seed)
    gen = torch.Generator().manual_seed(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng.shuffle(pos); rng.shuffle(neg)
    params = site_transform_params(K, severity)
    priors = [float(np.clip(0.5 + prior_spread * (2 * k / max(K - 1, 1) - 1), 0.12, 0.88)) for k in range(K)]
    xs, rows = [], []
    pp = nn = 0
    for k in range(K):
        n_pos = int(round(per_site * priors[k])); n_neg = per_site - n_pos
        ip = pos[pp:pp + n_pos]; ineg = neg[nn:nn + n_neg]
        if len(ip) < n_pos or len(ineg) < n_neg:  # recycle if pool exhausted (different images get different noise draws)
            ip = rng.choice(pos, n_pos, replace=False); ineg = rng.choice(neg, n_neg, replace=False)
        pp += n_pos; nn += n_neg
        idx = np.concatenate([ip, ineg])
        t = torch.from_numpy(X[idx]).float().div(255)[:, None]
        t = F.interpolate(t, size=(size, size), mode="area")
        t = apply_site(t, params[k], gen)
        xs.append((t[:, 0].numpy() * 255).astype(np.uint8))
        rows += [dict(label=int(y[i]), site=f"V{k}", patient_id=f"v{k}_{i}") for i in idx]
    Xv = np.concatenate(xs)
    df = pd.DataFrame(rows)
    df["row"] = np.arange(len(df))
    return Xv, df, params, priors
