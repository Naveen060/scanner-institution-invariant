"""Evaluate conformal calibration modes (incl. label-free profile-aware ones) on saved predictions.

  python scripts/analyze_conformal.py runs/it1 --methods erm sidl_full [--data data/covid_cxr] [--alpha 0.1]
Walks <path>/<holdout>/<method>/seed*/preds.npz; recomputes 5-d acquisition profiles of val/test images from the cached array.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import load_covid_cxr  # noqa: E402
from sidl.conformal import calibrate, evaluate_sets, prediction_sets  # noqa: E402
from sidl.pwconformal import profile_weighted_sets, site_nn_sets  # noqa: E402
from sidl.siteprofile import estimate_params  # noqa: E402


def all_profiles(X, size=64):
    t = torch.nn.functional.interpolate(torch.from_numpy(X).float().div(255)[:, None], size=(size, size), mode="area")
    return torch.cat([estimate_params(t[i:i + 128])[0] for i in range(0, len(t), 128)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--data", default="data/covid_cxr")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X, df = load_covid_cxr(a.data)
    P = all_profiles(X)
    rows = []
    for f in sorted(glob.glob(f"{a.path}/*/*/seed*/preds.npz")):
        held, method, seed = f.split("/")[-4], f.split("/")[-3], int(f.split("/")[-2][4:])
        if a.methods and method not in a.methods:
            continue
        if a.folds and held not in a.folds:
            continue
        z = np.load(f)
        pv, yv, sv, pt, yt = z["pv"], z["yv"], z["sv"], z["pt"], z["yt"]
        Pv, Pt = P[z["val_rows"]], P[z["test_rows"]]
        res = {}
        for mode in ("pooled", "site_balanced", "worst_site"):
            q = calibrate(pv, yv, sv, a.alpha, mode)
            res[mode] = evaluate_sets(prediction_sets(pt, q), yt)
        sets, qs, ess = profile_weighted_sets(pv, yv, Pv, pt, Pt, a.alpha, a.C)
        res["profile_weighted"] = evaluate_sets(sets, yt); res["profile_weighted"]["ess"] = ess
        sets, q = site_nn_sets(pv, yv, sv, Pv, pt, Pt, a.alpha, a.k)
        res["site_nn"] = evaluate_sets(sets, yt)
        for mode, r in res.items():
            rows.append(dict(holdout=held, method=method, seed=seed, mode=mode, coverage=r["coverage"], size=r["set_size"]))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    fold = d.groupby(["method", "mode", "holdout"])[["coverage", "size"]].mean().reset_index()
    summ = fold.groupby(["method", "mode"]).agg(cov_mean=("coverage", "mean"), cov_min=("coverage", "min"),
                                                 gap_abs=("coverage", lambda v: float(np.mean(np.abs(v - (1 - a.alpha))))),
                                                 under=("coverage", lambda v: float(np.mean(v < (1 - a.alpha) - 0.05))),
                                                 size=("size", "mean")).round(3)
    print(summ)
    if a.out:
        d.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
