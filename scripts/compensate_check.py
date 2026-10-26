"""Iteration 23: post hoc sensitivity check of the site-profile statistics.

The reported site-profile augmentation (phys_aug) and profile-adversarial invariance (pai) runs used the default,
uncompensated sharpness statistic (compensate=False): added noise raises it, so noise can masquerade as sharpness.
runs/it23.sh repeats both on all eight held-out sites with compensate=1 (noise removed from the gradient energy first),
all other settings frozen. This script compares the two versions against ERM with the same hierarchical paired
bootstrap used for the main table. It was not used for any decision.

  python scripts/compensate_check.py   # writes runs/compensate_check.json
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import final_report as fr  # noqa: E402


def load(dirs_keys):
    """{fold: [(y, p) per seed]} from (dir, key, folds) triples."""
    out = {}
    for dd, key, folds in dirs_keys:
        for h in folds:
            fs = sorted(glob.glob(f"{dd}/{h}/{key}/seed*/preds.npz"))
            if fs:
                out[h] = [(np.load(f)["yt"], np.load(f)["pt"][:, 1]) for f in fs]
    return out


def hier(a, e, B=1500, seed=0):
    folds = [h for h in fr.ALL if h in a and h in e]
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(B):
        ds = []
        for h in rng.choice(folds, len(folds)):
            y = e[h][0][0]
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            ds.append(np.mean([fr._auc(yy[idx], pp[idx]) for yy, pp in a[h]]) - np.mean([fr._auc(yy[idx], pp[idx]) for yy, pp in e[h]]))
        diffs.append(np.mean(ds))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def site_means(p):
    return {h: float(np.mean([fr._auc(y, s) for y, s in v])) for h, v in p.items()}


def main():
    erm = fr.load_preds("ERM")
    spec = {m[0]: m for m in fr.METHODS}
    variants = {
        "site-profile aug (reported, uncompensated)": fr.load_preds("Site-profile aug."),
        "site-profile aug (compensated sharpness)": load([("runs/it23_phys", "phys_aug", fr.ALL)]),
        "PAI (reported, uncompensated)": fr.load_preds("Profile adversary (PAI)"),
        "PAI (compensated sharpness)": load([("runs/it23_pai", "pai", fr.ALL)]),
    }
    res = {"ERM": {"mean": float(np.mean(list(site_means(erm).values()))), "sites": site_means(erm)}}
    for k, p in variants.items():
        sm = site_means(p)
        if len(sm) < 8:
            print(f"{k}: only {len(sm)} sites"); continue
        lo, hi = hier(p, erm)
        m = float(np.mean(list(sm.values())))
        res[k] = {"mean": m, "diff_vs_erm": m - res["ERM"]["mean"], "ci": [lo, hi], "sites": sm,
                  "seeds": {h: len(v) for h, v in p.items()}}
        print(f"{k}: mean {m:.3f}, diff to ERM {m - res['ERM']['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # the label-leak diagnostic uses the same five statistics; repeat it with the corrected sharpness
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sidl.arraydata import load_covid_cxr
    from sidl.siteprofile import estimate_params
    X, df = load_covid_cxr()
    t = torch.nn.functional.interpolate(torch.from_numpy(X).float().div(255)[:, None], size=(64, 64), mode="area")
    sites, y = df.site.values, df.label.values
    leak = {}
    for name, comp in (("uncompensated", False), ("compensated", True)):
        P = torch.cat([estimate_params(t[i:i + 128], comp)[0] for i in range(0, len(t), 128)]).numpy()
        a = {}
        for held in fr.ALL:
            m = sites == held
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(P[~m], y[~m])
            a[held] = float(roc_auc_score(y[m], clf.predict_proba(P[m])[:, 1]))
        leak[name] = {"mean": float(np.mean(list(a.values()))), "sites": a}
        print(f"leak AUROC (logistic regression on five statistics), {name}: {leak[name]['mean']:.3f}")
    res["leak"] = leak
    json.dump(res, open("runs/compensate_check.json", "w"), indent=2)


if __name__ == "__main__":
    main()
