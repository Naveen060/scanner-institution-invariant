"""Site-bagged ensemble (mean of the K leave-one-source-site-out ERM models) vs the single full ERM model on the held-out site.
Free by-product of the LSO conformal run. Evaluated on all 8 folds (dev: runs/it6 + runs/it1; confirmation: runs/conf_lso + runs/conf_erm).
Hierarchical paired bootstrap as in final_report.py (fold resampling, then stratified test-image resampling)."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from final_report import _auc  # noqa: E402
from sidl.metrics import ece as ece_fn  # noqa: E402

CFG = [("Actualmed", "runs/it6", "runs/it1"), ("Italy", "runs/it6", "runs/it1"), ("China", "runs/it6", "runs/it1"), ("Spain", "runs/it6", "runs/it1"),
       ("UK", "runs/conf_lso", "runs/conf_erm"), ("Australia", "runs/conf_lso", "runs/conf_erm"), ("Other", "runs/conf_lso", "runs/conf_erm"), ("Unknown", "runs/conf_lso", "runs/conf_erm")]
data = {}
for held, ldir, fdir in CFG:
    per = []
    for seed in (0, 1):
        inner = sorted(glob.glob(f"{ldir}/{held}/seed{seed}/inner_*"))
        z = np.load(f"{fdir}/{held}/erm/seed{seed}/preds.npz")
        P = np.stack([np.load(f"{d}/extra_H.npz")["p"][:, 1] for d in inner])
        per.append((z["yt"], z["pt"][:, 1], P.mean(0), len(inner)))
    data[held] = per
rows = []
for held, per in data.items():
    a_full = np.mean([_auc(y, p) for y, p, _, _ in per]); a_ens = np.mean([_auc(y, q) for y, _, q, _ in per])
    e_full = np.mean([ece_fn(y, p) for y, p, _, _ in per]); e_ens = np.mean([ece_fn(y, q) for y, _, q, _ in per])
    rows.append((held, a_full, a_ens, e_full, e_ens))
    print(f"{held:10s} AUROC full {a_full:.3f} ens {a_ens:.3f} | ECE full {e_full:.3f} ens {e_ens:.3f}")
A = np.array([[r[1], r[2], r[3], r[4]] for r in rows]); print("mean", A.mean(0).round(3))
rng = np.random.RandomState(0); folds = list(data); da, de = [], []
for _ in range(1500):
    s1, s2 = [], []
    for h in rng.choice(folds, len(folds)):
        y = data[h][0][0]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        s1.append(np.mean([_auc(yy[idx], q[idx]) - _auc(yy[idx], p[idx]) for yy, p, q, _ in data[h]]))
        s2.append(np.mean([ece_fn(yy[idx], q[idx]) - ece_fn(yy[idx], p[idx]) for yy, p, q, _ in data[h]]))
    da.append(np.mean(s1)); de.append(np.mean(s2))
out = dict(auroc_full=float(A[:, 0].mean()), auroc_ens=float(A[:, 1].mean()), ece_full=float(A[:, 2].mean()), ece_ens=float(A[:, 3].mean()),
           d_auroc=float(A[:, 1].mean() - A[:, 0].mean()), d_auroc_ci=[float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
           d_ece=float(A[:, 3].mean() - A[:, 2].mean()), d_ece_ci=[float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
           folds_ens_better_auroc=int((A[:, 1] > A[:, 0]).sum()), n_members=int(data[folds[0]][0][3]))
print(json.dumps(out, indent=1)); json.dump(out, open("runs/ensemble_check.json", "w"), indent=1)
