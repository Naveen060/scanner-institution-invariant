"""Leave-site-out cross-conformal (LSO-CV+) vs standard calibration, evaluated on the held-out site.

Calibration scores R_i = 1 - p_{-k(i)}(y_i | x_i) come from the inner model that never saw site k(i).
Modes
  lso_split        pooled quantile of R applied to the FULL model's test probabilities (cheap: one deployed model)
  lso_split_bal    same with equal total weight per calibration site
  lso_worst        max over calibration sites of the per-site quantile (sites with >= 25 images)
  lso_cvplus       CV+ rule (Barber et al. 2021) with sites as folds: class y is included iff
                   #{ i : R_i < 1 - p_{-k(i)}(y | x) } < (1-alpha)(n+1)       (test scored by EACH inner model)
  lso_cvplus_bal   weighted variant (weights n/(K n_k))
Baselines from the full model's seen-site validation split: pooled, site_balanced, worst_site.
Also reports, across an alpha grid, the coverage-calibration error and the set size at matched 90% coverage.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.conformal import _quantile, calibrate, evaluate_sets, prediction_sets  # noqa: E402

ALPHAS = [0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]


def lso_modes(R, kid, n_k, Q, full_pt, alpha):
    """R:(n,) scores; kid:(n,) site index per score; n_k:(K,) counts; Q:(K,nH,C) inner-model probs on H; full_pt:(nH,C)."""
    n = len(R); K = len(n_k); out = {}
    q = _quantile(R, alpha)
    out["lso_split"] = prediction_sets(full_pt, q)
    w = n / (K * n_k[kid])
    out["lso_split_bal"] = prediction_sets(full_pt, _quantile(R, alpha, w))
    qs = [_quantile(R[kid == k], alpha) for k in range(K) if n_k[k] >= 25]
    out["lso_worst"] = prediction_sets(full_pt, max(qs) if qs else q)
    S = 1.0 - Q  # (K,nH,C)
    for name, ww in (("lso_cvplus", np.ones(n)), ("lso_cvplus_bal", w)):
        cnt = np.zeros(Q.shape[1:])  # (nH,C)
        for k in range(K):
            m = kid == k
            cnt += (ww[m][:, None, None] * (R[m][:, None, None] < S[k][None, :, :])).sum(0)
        sets = cnt < (1 - alpha) * (ww.sum() + 1)
        empty = ~sets.any(1)
        sets[empty, full_pt[empty].argmax(1)] = True
        out[name] = sets
    return out


def load_fold(fold_dir):
    inner = sorted(glob.glob(f"{fold_dir}/inner_*"))
    if not inner or not all(os.path.exists(f"{d}/extra_H.npz") for d in inner):
        return None
    R, kid, Q, n_k = [], [], [], []
    for k, d in enumerate(inner):
        z = np.load(f"{d}/preds.npz"); h = np.load(f"{d}/extra_H.npz")
        R.append(1 - z["pt"][np.arange(len(z["yt"])), z["yt"]]); kid.append(np.full(len(z["yt"]), k))
        n_k.append(len(z["yt"])); Q.append(h["p"])
    return np.concatenate(R), np.concatenate(kid), np.stack(Q), np.array(n_k)


def collect(a, alphas):
    rows = []
    for fold_dir in sorted(glob.glob(f"{a.lso_dir}/*/seed*")):
        held, seed = fold_dir.split("/")[-2], int(fold_dir.split("/")[-1][4:])
        if a.folds and held not in a.folds:
            continue
        lf = load_fold(fold_dir)
        if lf is None:
            continue
        R, kid, Q, n_k = lf
        fz = np.load(f"{a.full_dir}/{held}/{a.method}/seed{seed}/preds.npz")
        yt, pt = fz["yt"], fz["pt"]
        for alpha in alphas:
            res = {}
            for mode in ("pooled", "site_balanced", "worst_site"):
                res[mode] = evaluate_sets(prediction_sets(pt, calibrate(fz["pv"], fz["yv"], fz["sv"], alpha, mode)), yt)
            for mode, sets in lso_modes(R, kid, n_k, Q, pt, alpha).items():
                res[mode] = evaluate_sets(sets, yt)
            for mode, r in res.items():
                rows.append(dict(alpha=alpha, holdout=held, seed=seed, mode=mode, coverage=r["coverage"], size=r["set_size"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lso_dir")
    ap.add_argument("--full_dir", required=True, help="LOSO run dir with the deployed full models (e.g. runs/it1)")
    ap.add_argument("--method", default="erm")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--folds", nargs="*")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    D = collect(a, sorted(set(ALPHAS + [a.alpha])))
    d = D[D.alpha == a.alpha]
    fold = d.groupby(["mode", "holdout"])[["coverage", "size"]].mean().reset_index()
    summ = fold.groupby("mode").agg(cov_mean=("coverage", "mean"), cov_min=("coverage", "min"),
                                    gap_abs=("coverage", lambda v: float(np.mean(np.abs(v - (1 - a.alpha))))),
                                    n_under=("coverage", lambda v: int((v < (1 - a.alpha) - 0.05).sum())),
                                    size=("size", "mean")).round(3)
    print(f"== alpha={a.alpha} (target coverage {1 - a.alpha:.2f}); folds={sorted(fold.holdout.unique())} ==")
    print(summ.sort_values("gap_abs"))
    print("\nper-fold coverage:\n", fold.pivot(index="mode", columns="holdout", values="coverage").round(3))
    F = D.groupby(["mode", "alpha", "holdout"])[["coverage", "size"]].mean().reset_index()
    C = F.groupby(["mode", "alpha"])[["coverage", "size"]].mean().reset_index()
    out = []
    for mode, g in C.groupby("mode"):
        g = g.sort_values("coverage")
        cce = float(np.mean(np.abs(g.coverage.values - (1 - g.alpha.values))))
        size90 = float(np.interp(0.90, g.coverage.values, g["size"].values))
        Fm = F[F["mode"] == mode]
        under = float(np.mean(Fm.coverage < 1 - Fm.alpha - 0.05))
        out.append(dict(mode=mode, cov_calib_err=round(cce, 3), size_at_cov90=round(size90, 3), frac_under=round(under, 3)))
    print("\nAcross alpha grid", ALPHAS, "\n", pd.DataFrame(out).set_index("mode").sort_values("cov_calib_err"))
    if a.out:
        D.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
