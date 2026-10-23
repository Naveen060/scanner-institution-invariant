"""Array-backed datasets for small cached benchmarks (fast CPU training)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SMALL_SITES = {"Taiwan", "Canada", "Egypt", "Iran", "Portugal", "Korea", "Hungary", "Malta"}


def load_covid_cxr(root: str = "data/covid_cxr", size: int = 128):
    X = np.load(f"{root}/images_{size}.npz")["X"]
    df = pd.read_csv(f"{root}/meta.csv")
    df["site"] = df["site"].where(~df["site"].isin(SMALL_SITES), "Other")  # merge tiny sources
    df["row"] = np.arange(len(df))
    return X, df


class ArrayDataset(Dataset):
    """Holds a subset (df rows) of a uint8 image array; returns (x[1,S,S] in [0,1], label, site_idx)."""

    def __init__(self, X: np.ndarray, df: pd.DataFrame, site_to_idx: dict, size: int = 64, train: bool = False):
        self.rows = df["row"].values
        self.y = df["label"].values.astype(int)
        self.site_names = df["site"].values
        self.s = np.array([site_to_idx.get(v, -1) for v in self.site_names])
        t = torch.from_numpy(X[self.rows]).float().div(255)[:, None]
        self.t = F.interpolate(t, size=(size, size), mode="area") if size != t.size(-1) else t
        self.size, self.train = size, train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.t[i]
        if self.train:  # base augmentation shared by ALL methods: random translation
            p = self.size // 8
            xp = F.pad(x[None], (p, p, p, p), mode="reflect")[0]
            dx, dy = np.random.randint(0, 2 * p + 1, 2)
            x = xp[:, dy:dy + self.size, dx:dx + self.size]
        return x, int(self.y[i]), int(self.s[i])
