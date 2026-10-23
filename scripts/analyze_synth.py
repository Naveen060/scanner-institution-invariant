"""Summarize semi-synthetic label-shift sweeps: AUROC / balanced accuracy / conformal coverage by prior spread and method."""
import argparse, json, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--ref", default="erm"); ap.add_argument("--metric", default="auroc")
a = ap.parse_args(); pd.set_option("display.width", 220)
rows = []
for l in open(f"{a.path}/results.jsonl"):
    r = json.loads(l)
    for k, v in r.pop("cov").items(): r[f"cov_{k}"] = v
    for k, v in r.pop("size").items(): r[f"size_{k}"] = v
    rows.append(r)
d = pd.DataFrame(rows)
print(len(d), "runs; spreads", sorted(d.spread.unique()), "holdouts", sorted(d.holdout.unique()), "seeds", sorted(d.seed.unique()))
for m in [a.metric, "bal_acc", "cov_pooled", "size_pooled"]:
    t = d.groupby(["spread", "method"])[m].mean().unstack("method")
    print(f"\n== {m} (mean over holdouts, seeds) ==\n", t.round(3))
print("\n== per-holdout", a.metric, "==")
print(d.pivot_table(index=["spread", "holdout"], columns="method", values=a.metric).round(3))
rng = np.random.RandomState(0); out = []
for sp in sorted(d.spread.unique()):
    piv = d[d.spread == sp].pivot_table(index=["holdout", "seed"], columns="method", values=a.metric)
    for m in piv.columns:
        if m == a.ref: continue
        x = (piv[m] - piv[a.ref]).dropna().values
        bs = [rng.choice(x, len(x)).mean() for _ in range(4000)]
        out.append((sp, m, round(x.mean(), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3), int((x > 0).sum()), len(x)))
print("\n== paired diff vs", a.ref, "(over holdout x seed units) ==")
print(pd.DataFrame(out, columns=["spread", "method", "mean_diff", "ci_lo", "ci_hi", "n_better", "n"]).to_string(index=False))
