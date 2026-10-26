"""Round-2 analysis: frozen pretrained-encoder family vs from-scratch CNN, all 8 held-out sites.

  python scripts/round2_report.py          # prints tables; writes runs/round2.json
Dev folds: Actualmed, Italy, China, Spain; confirmation folds: UK, Australia, Other, Unknown (frozen configs, run once).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from feat_compare import boot, load, seed_mean_auc  # noqa: E402

DEV = ["Actualmed", "Italy", "China", "Spain"]
CONF = ["UK", "Australia", "Other", "Unknown"]
ALL = DEV + CONF
CNN = {**{h: "runs/it1/{fold}/erm/seed*/preds.npz,runs/cnn_extra/{fold}/erm/seed*/preds.npz" for h in DEV}, **{h: "runs/conf_erm/{fold}/erm/seed*/preds.npz,runs/cnn_extra/{fold}/erm/seed*/preds.npz" for h in CONF}}


def feat(method):
    return {**{h: f"runs/it12_a05/{{fold}}/{method}/seed*/preds.npz" for h in DEV}, **{h: f"runs/it14_feat/{{fold}}/{method}/seed*/preds.npz" for h in CONF}}


def get(pat):
    return {h: load(pat[h], h) for h in ALL}


def seedens(r):
    return [(r[0][0], np.mean([x[1] for x in r], 0))]


def combo(a, b):
    return {h: [(a[h][0][0], 0.5 * seedens(a[h])[0][1] + 0.5 * seedens(b[h])[0][1])] for h in a}


def fold_means(d, folds):
    return [seed_mean_auc(d[h], np.arange(len(d[h][0][0]))) for h in folds]


if __name__ == "__main__":
    S, P = get(CNN), get(feat("erm"))
    models = {"CNN ERM": S, "Pretrained ERM": P, "CNN 5-model ens": {h: seedens(S[h]) for h in ALL},
              "Pretrained 5-model ens": {h: seedens(P[h]) for h in ALL}, "CNN + pretrained (5+5 ens)": combo(S, P)}
    out = {"fold_auroc": {}, "boot": {}}
    print(f"{'model':28s}", " ".join(f"{h[:6]:>6s}" for h in ALL), "  dev   conf   all  worst")
    for k, d in models.items():
        v = fold_means(d, ALL)
        out["fold_auroc"][k] = dict(zip(ALL, v))
        print(f"{k:28s}", " ".join(f"{x:6.3f}" for x in v), f" {np.mean(v[:4]):.3f} {np.mean(v[4:]):.3f} {np.mean(v):.3f} {min(v):.3f}")
    print("\nprimary / secondary paired comparisons (hierarchical bootstrap, 2000 reps):")
    pairs = [("CNN + pretrained (5+5 ens)", "CNN 5-model ens", "A"), ("CNN + pretrained (5+5 ens)", "CNN ERM", "B"),
             ("CNN + pretrained (5+5 ens)", "Pretrained 5-model ens", "-"), ("Pretrained ERM", "CNN ERM", "-"),
             ("CNN 5-model ens", "CNN ERM", "-"), ("Pretrained 5-model ens", "CNN 5-model ens", "-"), ("Pretrained 5-model ens", "CNN ERM", "-")]
    for x, y, tag in pairs:
        for name, fl in (("dev", DEV), ("conf", CONF), ("all", ALL)):
            m, lo, hi = boot(models[x], models[y], fl)
            out["boot"][f"{x}|{y}|{name}"] = [m, lo, hi]
            print(f"  [{tag}] {name:4s} {x} - {y}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\nfeature-head methods vs pretrained ERM (seed-averaged single heads):")
    out["feat_methods"] = {}
    for m in ["erm", "erm_p", "groupdro", "dann", "cdann", "cdann_p", "coral", "cc_full", "sitecenter"]:
        d = get(feat(m))
        v = fold_means(d, ALL)
        lo_hi = boot(d, P, ALL) if m != "erm" else (0, 0, 0)
        out["feat_methods"][m] = dict(folds=dict(zip(ALL, v)), dev=float(np.mean(v[:4])), conf=float(np.mean(v[4:])), all=float(np.mean(v)), diff_ci=list(lo_hi))
        print(f"  {m:11s} dev {np.mean(v[:4]):.3f} conf {np.mean(v[4:]):.3f} all {np.mean(v):.3f}  diff vs erm (all) {lo_hi[0]:+.3f} [{lo_hi[1]:+.3f}, {lo_hi[2]:+.3f}]")
    json.dump(out, open("runs/round2.json", "w"), indent=1)
