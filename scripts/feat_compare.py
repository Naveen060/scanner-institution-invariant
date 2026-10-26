"""Paired comparison of frozen-feature heads and the from-scratch CNN on identical held-out images.

  python scripts/feat_compare.py --folds Actualmed Italy China Spain
Prints mean AUROC per fold for each source and a hierarchical paired bootstrap (resample folds, then images stratified by class).
"""
import argparse
import glob
import sys

import numpy as np

sys.path.insert(0, "scripts")
from final_report import _auc  # noqa: E402


def load(pattern, fold):
    out = []
    for pat in pattern.split(","):  # several comma-separated glob patterns are allowed (seeds stored in different folders)
        for f in sorted(glob.glob(pat.format(fold=fold))):
            z = np.load(f)
            out.append((z["yt"], z["pt"][:, 1]))
    return out


def seed_mean_auc(runs, idx):
    return float(np.mean([_auc(y[idx], p[idx]) for y, p in runs]))


def boot(a, b, folds, B=2000, seed=0):
    """a, b: {fold: [(y, p) per seed]}; returns (mean diff, lo, hi) of mean-over-folds AUROC(a) - AUROC(b)."""
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(B):
        d = []
        for h in rng.choice(folds, len(folds)):
            y = a[h][0][0]
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            d.append(seed_mean_auc(a[h], idx) - seed_mean_auc(b[h], idx))
        ds.append(np.mean(d))
    full = np.mean([seed_mean_auc(a[h], np.arange(len(a[h][0][0]))) - seed_mean_auc(b[h], np.arange(len(b[h][0][0]))) for h in folds])
    return float(full), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def ensemble(runs_a, runs_b, w=0.5):
    n = min(len(runs_a), len(runs_b))
    return [(runs_a[i][0], w * runs_a[i][1] + (1 - w) * runs_b[i][1]) for i in range(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--scratch", default="runs/it1/{fold}/erm/seed*/preds.npz,runs/cnn_extra/{fold}/erm/seed*/preds.npz")
    ap.add_argument("--feat", default="runs/it11/{fold}/erm/seed*/preds.npz")
    ap.add_argument("--other", nargs="*", default=[], help="name=pattern for extra feature heads")
    a = ap.parse_args()
    S = {h: load(a.scratch, h) for h in a.folds}
    P = {h: load(a.feat, h) for h in a.folds}
    E = {h: ensemble(S[h], P[h]) for h in a.folds}
    srcs = {"scratch ERM": S, "pretrained ERM": P, "ensemble (mean prob.)": E}
    for kv in a.other:
        k, v = kv.split("=")
        srcs[k] = {h: load(v, h) for h in a.folds}
    print("mean AUROC per fold")
    for k, d in srcs.items():
        v = [seed_mean_auc(d[h], np.arange(len(d[h][0][0]))) for h in a.folds]
        print(f"  {k:24s}", " ".join(f"{x:.3f}" for x in v), f"mean {np.mean(v):.3f}  (seeds: {len(d[a.folds[0]])})")
    for x, y in [("ensemble (mean prob.)", "scratch ERM"), ("ensemble (mean prob.)", "pretrained ERM"), ("pretrained ERM", "scratch ERM")] + [(k, "pretrained ERM") for k in srcs if k not in ("scratch ERM", "pretrained ERM", "ensemble (mean prob.)")]:
        m, lo, hi = boot(srcs[x], srcs[y], a.folds)
        print(f"  {x} - {y}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
