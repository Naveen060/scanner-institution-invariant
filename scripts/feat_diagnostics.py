"""Two descriptive checks on the frozen pretrained embeddings (no model selection uses them).

 1. Site decodability: patient-grouped 5-fold cross-validated logistic regression predicting the (merged) site from the 1024-d embedding.
 2. Seen-site validation optimism of the pretrained ERM head: validation AUROC minus held-out-site AUROC, 8 sites x 5 seeds.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
from run_feat_loso import load_meta  # noqa: E402
from final_report import _auc  # noqa: E402

DEV = ["Actualmed", "Italy", "China", "Spain"]
CONF = ["UK", "Australia", "Other", "Unknown"]
df = load_meta("data/covid_cxr224")
F = np.load("data/covid_cxr224/xrv_features.npz")["feats"].astype(np.float32)
site = df.site.values
groups = (df.patient_id.astype(str) + "_" + df.site.astype(str)).values
pred = np.empty(len(df), dtype=object)
for tr, te in GroupKFold(5).split(F, site, groups):
    sc = StandardScaler().fit(F[tr])
    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(F[tr]), site[tr])
    pred[te] = clf.predict(sc.transform(F[te]))
acc = float((pred == site).mean())
major = float(pd.Series(site).value_counts(normalize=True).iloc[0])
print(f"site decodability from pretrained embedding: accuracy {acc:.3f} (majority {major:.3f}; {len(set(site))} sites)")

rows = []
for h in DEV + CONF:
    root = "runs/it11" if h in DEV else "runs/it14_feat"
    for s in range(5):
        z = np.load(f"{root}/{h}/erm/seed{s}/preds.npz")
        rows.append(dict(site=h, seed=s, val=_auc(z["yv"], z["pv"][:, 1]), test=_auc(z["yt"], z["pt"][:, 1])))
r = pd.DataFrame(rows)
r["gap"] = r.val - r.test
print(f"pretrained ERM: seen-site validation AUROC {r.val.mean():.3f}, held-out {r.test.mean():.3f}, optimism {r.gap.mean():+.3f} "
      f"(MAE {r.gap.abs().mean():.3f}; units {len(r)})")
print(r.groupby("site")[["val", "test", "gap"]].mean().round(3))
json.dump(dict(site_acc=acc, site_majority=major, n_sites=int(len(set(site))), val_auc=float(r.val.mean()), test_auc=float(r.test.mean()),
               optimism=float(r.gap.mean()), optimism_mae=float(r.gap.abs().mean()), units=int(len(r))), open("runs/feat_diagnostics.json", "w"), indent=1)
