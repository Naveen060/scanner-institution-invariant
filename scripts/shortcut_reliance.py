"""Shortcut-reliance score: how much do a trained model's held-out predictions track acquisition statistics?

For each (held-out site, method, seed): fit a profile-only logistic regression on SOURCE sites (5 acquisition numbers -> label),
score the held-out images with it, and compute the Spearman correlation between that profile score and the model's
P(COVID), computed WITHIN each class and averaged (so the label itself cannot explain the correlation).
Higher = the model's predictions co-vary with acquisition statistics beyond what the label explains.

  python scripts/shortcut_reliance.py runs/it1 --folds Actualmed Italy China Spain --methods erm groupdro
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import load_covid_cxr  # noqa: E402
from sidl.siteprofile import estimate_params  # noqa: E402


def all_profiles(X, size=64):
    t = torch.nn.functional.interpolate(torch.from_numpy(X).float().div(255)[:, None], size=(size, size), mode="area")
    return torch.cat([estimate_params(t[i:i + 128])[0] for i in range(0, len(t), 128)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--data", default="data/covid_cxr")
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--methods", nargs="*")
    a = ap.parse_args()
    X, df = load_covid_cxr(a.data)
    P = all_profiles(X)
    rows = []
    for f in sorted(glob.glob(f"{a.path}/*/*/seed*/preds.npz")):
        held, method, seed = f.split("/")[-4], f.split("/")[-3], int(f.split("/")[-2][4:])
        if (a.folds and held not in a.folds) or (a.methods and method not in a.methods):
            continue
        z = np.load(f)
        src = (df.site.values != held)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(P[src], df.label.values[src])
        rows_t = z["test_rows"]
        ps = clf.decision_function(P[rows_t])
        yt, pt = z["yt"], z["pt"][:, 1]
        cs = [spearmanr(ps[yt == c], pt[yt == c])[0] for c in (0, 1) if (yt == c).sum() > 5]
        rows.append(dict(holdout=held, method=method, seed=seed, reliance=float(np.nanmean(cs))))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(d.groupby(["method", "holdout"]).reliance.mean().unstack("holdout").assign(MEAN=lambda t: t.mean(1)).round(3))


if __name__ == "__main__":
    main()
