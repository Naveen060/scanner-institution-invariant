"""Label-free diagnostics on the cached benchmark (cheap; sklearn only).

  D1  site predictability   : how well can the SITE be predicted from the 5-d acquisition profile vs raw downsampled pixels?
  D2  profile shortcut      : how well can the DISEASE LABEL be predicted from the acquisition profile ALONE, leave-one-site-out?
                              AUROC well above 0.5 on held-out sites means acquisition statistics leak the label even across sites.
  D3  label-prior spread    : per-site prevalence (determines how much label-shift-aware invariance can matter).
Outputs JSON to --out.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import load_covid_cxr  # noqa: E402
from sidl.siteprofile import PARAM_NAMES, estimate_params  # noqa: E402


def profiles(X, size=64):
    t = torch.from_numpy(X).float().div(255)[:, None]
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="area")
    P = []
    for i in range(0, len(t), 128):
        p, _ = estimate_params(t[i:i + 128])
        P.append(p)
    return torch.cat(P).numpy(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/diagnostics.json")
    a = ap.parse_args()
    X, df = load_covid_cxr()
    P, t = profiles(X)
    sites, y = df.site.values, df.label.values
    out = {"param_names": PARAM_NAMES}
    # D3 prior spread
    prev = df.groupby("site").label.agg(["count", "mean"])
    out["prevalence"] = {s: {"n": int(r["count"]), "prev": float(r["mean"])} for s, r in prev.iterrows()}
    out["prior_spread_std"] = float(prev["mean"].std())
    # D1 site predictability (5-fold stratified-ish via random split)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(df)); cut = int(0.7 * len(df)); tr, te = idx[:cut], idx[cut:]
    pix = torch.nn.functional.adaptive_avg_pool2d(t, 16).flatten(1).numpy()
    for name, F in [("profile5", P), ("pixels16x16", pix)]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(F[tr], sites[tr])
        out[f"site_acc_{name}"] = float(clf.score(F[te], sites[te]))
    out["site_acc_chance_majority"] = float(pd_majority(sites))
    # D2 profile-only LOSO label prediction
    res = {}
    for held in ["Actualmed", "Other", "Italy", "UK", "Australia", "China", "Spain", "Unknown"]:
        m = sites == held
        for name, mk in [("logreg", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                         ("gbm", lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0))]:
            clf = mk().fit(P[~m], y[~m])
            res.setdefault(name, {})[held] = float(roc_auc_score(y[m], clf.predict_proba(P[m])[:, 1]))
    out["profile_only_label_auroc_loso"] = res
    out["profile_only_label_auroc_mean"] = {k: float(np.mean(list(v.values()))) for k, v in res.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


def pd_majority(s):
    _, c = np.unique(s, return_counts=True)
    return c.max() / c.sum()


if __name__ == "__main__":
    main()
