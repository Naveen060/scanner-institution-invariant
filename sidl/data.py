"""Data layer.

Everything runs from a *manifest* CSV with columns: path, label, site  (optional: patient_id).
This one format covers public datasets (see scripts/prepare_manifest.py) and private hospital data.
A synthetic generator is included for smoke tests and for sanity-checking the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    site_names: List[str]


def patient_level_split(df: pd.DataFrame, val_frac: float, seed: int):
    """Split by patient_id when available (avoids leakage), else by row."""
    rng = np.random.RandomState(seed)
    if "patient_id" in df.columns:
        key = df["patient_id"].astype(str) + "_" + df["site"].astype(str)
        uniq = np.array(key.unique(), dtype=object)
        rng.shuffle(uniq)
        val_keys = set(uniq[: int(len(uniq) * val_frac)])
        m = key.isin(val_keys)
    else:
        idx = rng.permutation(len(df))
        m = np.zeros(len(df), dtype=bool)
        m[idx[: int(len(df) * val_frac)]] = True
    return df[~m].reset_index(drop=True), df[m].reset_index(drop=True)


def leave_one_site_out(df: pd.DataFrame, held_out: str, val_frac: float = 0.15, seed: int = 0) -> Splits:
    """Train/val on all sites except `held_out`; test = entire held-out site (true OOD).

    Validation is drawn from *training sites only* so no information from the held-out site
    is used for model selection (no target-domain leakage).
    """
    sites = sorted(df["site"].unique())
    assert held_out in sites, f"{held_out} not in {sites}"
    train_sites = [s for s in sites if s != held_out]
    src = df[df["site"].isin(train_sites)].reset_index(drop=True)
    tr, va = patient_level_split(src, val_frac, seed)
    te = df[df["site"] == held_out].reset_index(drop=True)
    return Splits(tr, va, te, train_sites)


def in_distribution_split(df: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=0) -> Splits:
    """All sites seen in training; held-out *samples* — used as the ID reference."""
    sites = sorted(df["site"].unique())
    rest, te = patient_level_split(df, test_frac, seed)
    tr, va = patient_level_split(rest, val_frac, seed + 1)
    return Splits(tr, va, te, sites)


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, site_to_idx: dict, image_size: int = 224, in_chans: int = 3,
                 loader: Optional[callable] = None, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.site_to_idx = site_to_idx
        self.size, self.in_chans, self.train = image_size, in_chans, train
        self.loader = loader or self._default_loader

    def _default_loader(self, path: str) -> np.ndarray:
        """Returns HxW or HxWxC float array in [0,1]. Handles png/jpg/tif and .npy/.npz (MRI slices)."""
        if path.endswith(".npy"):
            a = np.load(path).astype(np.float32)
        else:
            from PIL import Image
            a = np.asarray(Image.open(path)).astype(np.float32)
            if a.max() > 1.0:
                a = a / (65535.0 if a.max() > 255 else 255.0)
        lo, hi = np.percentile(a, [0.5, 99.5])
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        a = self.loader(r["path"])
        x = torch.from_numpy(a).float()
        x = x.unsqueeze(0) if x.ndim == 2 else x.permute(2, 0, 1)
        if x.size(0) != self.in_chans:
            x = x[:1].repeat(self.in_chans, 1, 1) if x.size(0) == 1 or self.in_chans == 3 and x.size(0) != 3 else x[: self.in_chans]
        x = torch.nn.functional.interpolate(x[None], size=(self.size, self.size), mode="bilinear", align_corners=False)[0]
        if self.train and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])
        return x, int(r["label"]), self.site_to_idx.get(r["site"], -1)


class SyntheticSiteDataset(Dataset):
    """Synthetic multi-site data: class signal is a bright blob; each site has its own
    gain/offset/blur/noise 'scanner signature'. Lets us verify invariance methods
    behave sensibly: ERM should exploit site shortcuts when class priors differ per site."""

    def __init__(self, n_per_site=200, num_sites=4, size=64, seed=0, site_prior_shift=0.3, sites: Optional[Sequence[int]] = None):
        rng = np.random.RandomState(seed)
        self.X, self.y, self.s = [], [], []
        sites = list(range(num_sites)) if sites is None else list(sites)
        for s in sites:
            prior = 0.5 + site_prior_shift * (s / max(num_sites - 1, 1) - 0.5) * 2
            gain, off, noise = 0.6 + 0.15 * s, 0.05 * s, 0.03 + 0.02 * s
            for _ in range(n_per_site):
                yv = int(rng.rand() < prior)
                img = rng.rand(size, size).astype(np.float32) * 0.1
                if yv:
                    cx, cy = rng.randint(size // 4, 3 * size // 4, 2)
                    yy, xx = np.ogrid[:size, :size]
                    img += 0.6 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size / 10) ** 2)))
                img = np.clip(img * gain + off + rng.randn(size, size) * noise, 0, 1)
                self.X.append(img)
                self.y.append(yv)
                self.s.append(s)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i])[None].float(), self.y[i], self.s[i]
