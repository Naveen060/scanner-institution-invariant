"""Compare configurations stored in different run directories on the same folds.
  python scripts/compare_dirs.py --ref runs/it1:erm --cfg label=runs/it2_x:method ... [--folds ...] [--metric auroc]
Prints per-fold and mean metric plus paired difference vs the reference (fold-level, bootstrap over folds)."""
import argparse, json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from analyze_runs import load

def get(spec, folds):
    path, method = spec.split(":")
    d = load(path); d = d[(d.method == method)]
    if folds: d = d[d.holdout.isin(folds)]
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--cfg", nargs="+", required=True)
    ap.add_argument("--folds", nargs="*"); ap.add_argument("--metric", default="auroc")
    a = ap.parse_args(); pd.set_option("display.width", 200)
    rows = {"REF " + a.ref: get(a.ref, a.folds)}
    for c in a.cfg:
        lab, spec = c.split("=", 1); rows[lab] = get(spec, a.folds)
    tab = pd.DataFrame({k: v.groupby("holdout")[a.metric].mean() for k, v in rows.items()}).T
    tab["MEAN"] = tab.mean(1)
    print(tab.round(3))
    ref = tab.iloc[0, :-1]
    rng = np.random.RandomState(0); out = []
    for k in tab.index[1:]:
        d = (tab.loc[k].iloc[:-1] - ref).dropna()
        bs = [rng.choice(d.values, len(d)).mean() for _ in range(4000)]
        out.append((k, round(d.mean(), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3), int((d > 0).sum()), len(d)))
    print(pd.DataFrame(out, columns=["cfg", "mean_diff", "ci_lo", "ci_hi", "folds_better", "n"]).to_string(index=False))
main()
