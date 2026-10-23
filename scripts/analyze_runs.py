"""Summaries and paired statistics for results.jsonl files.

  python scripts/analyze_runs.py runs/it1 [--ref erm] [--metric auroc]
Prints: per-method mean over (fold,seed); per-fold table; paired differences vs --ref with
cluster-bootstrap CI (resampling held-out sites, then seeds) and a one-sided sign count over sites; conformal summary.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def load(path):
    rows = []
    for l in open(f"{path}/results.jsonl"):
        r = json.loads(l)
        for k, v in r.pop("cov").items():
            r[f"cov_{k}"] = v
        for k, v in r.pop("size").items():
            r[f"size_{k}"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def paired(df, ref, metric, B=4000, seed=0):
    rng = np.random.RandomState(seed)
    piv = df.pivot_table(index=["holdout", "seed"], columns="method", values=metric)
    fold = piv.groupby(level=0).mean()  # average seeds within fold
    out = []
    for m in fold.columns:
        if m == ref or ref not in fold.columns:
            continue
        d = (fold[m] - fold[ref]).dropna()
        bs = [d.sample(len(d), replace=True, random_state=rng.randint(1 << 30)).mean() for _ in range(B)]
        out.append(dict(method=m, mean_diff=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        folds_better=int((d > 0).sum()), n_folds=len(d)))
    return pd.DataFrame(out).set_index("method") if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default="erm")
    ap.add_argument("--metric", default="auroc")
    ap.add_argument("--folds", nargs="*", default=None, help="restrict to these held-out sites (e.g. dev folds)")
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    df = load(a.path)
    if a.folds:
        df = df[df.holdout.isin(a.folds)]
    if a.methods:
        df = df[df.method.isin(a.methods)]
    print(f"{len(df)} runs | folds={sorted(df.holdout.unique())} | seeds={sorted(df.seed.unique())}")
    fold_mean = df.groupby(["method", "holdout"])[[a.metric, "acc", "bal_acc", "ece"]].mean()
    g = df.groupby("method")
    summ = pd.DataFrame({
        a.metric: df.groupby(["method", "holdout"])[a.metric].mean().groupby("method").mean(),
        "worst_fold": df.groupby(["method", "holdout"])[a.metric].mean().groupby("method").min(),
        "bal_acc": df.groupby(["method", "holdout"])["bal_acc"].mean().groupby("method").mean(),
        "ece": df.groupby(["method", "holdout"])["ece"].mean().groupby("method").mean(),
        "n": g.size()})
    print("\n== task metrics (mean over folds of seed-mean) ==\n", summ.round(3).sort_values(a.metric, ascending=False))
    print("\n== per-fold", a.metric, "==\n", df.pivot_table(index="method", columns="holdout", values=a.metric).round(3))
    print("\n== paired vs", a.ref, "(fold-level, bootstrap over folds) ==\n", paired(df, a.ref, a.metric).round(3))
    cc = [c for c in df.columns if c.startswith("cov_")]
    sc = [c.replace("cov_", "size_") for c in cc]
    print("\n== conformal (target 0.90) coverage mean/min over folds, size ==")
    cm = df.groupby(["method", "holdout"])[cc + sc].mean().groupby("method")
    print(pd.concat([cm.mean().add_suffix("_mean"), cm.min()[cc].add_suffix("_min")], axis=1).round(3).T)


if __name__ == "__main__":
    main()
