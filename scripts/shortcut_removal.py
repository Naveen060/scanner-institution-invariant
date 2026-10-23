"""Mechanism test for site-profile augmentation (no network training).

Question: does re-rendering images toward a random *observed* site (independent of the label) reduce how much the
5-d acquisition profile reveals the disease label on held-out sites?
For each held-out site: fit SiteProfiles on the source sites only, translate SOURCE images (and, as a control,
the held-out images are left untouched), recompute profiles, train profile-only classifier on translated source
images, test on clean held-out images. Compare to training on clean source images. Also compares the generic augmenter.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import ArrayDataset, load_covid_cxr  # noqa: E402
from sidl.augment import Harmonizer  # noqa: E402
from sidl.siteprofile import SiteProfiles, SiteTranslator, estimate_params  # noqa: E402

HELD = ["Actualmed", "Other", "Italy", "UK", "Australia", "China", "Spain", "Unknown"]


def prof(t):
    return torch.cat([estimate_params(t[i:i + 128])[0] for i in range(0, len(t), 128)]).numpy()


def main(reps=5):
    torch.set_num_threads(1)
    X, df = load_covid_cxr()
    out = {"clean": {}, "site_profile_aug": {}, "generic_aug": {}}
    for held in HELD:
        src = df[df.site != held].reset_index(drop=True)
        tgt = df[df.site == held].reset_index(drop=True)
        sites = sorted(src.site.unique()); s2i = {s: i for i, s in enumerate(sites)}
        ds = ArrayDataset(X, src, s2i, 64); dt = ArrayDataset(X, tgt, s2i, 64)
        pr = SiteProfiles(len(sites), 2).fit(ds)
        trn = SiteTranslator(pr, p=1.0).train(); gen = Harmonizer(p=0.8, style_mix_p=0.0).train()
        xs, ys, ss = ds.t, torch.from_numpy(ds.y), torch.from_numpy(ds.s)
        Pt = prof(dt.t); yt = dt.y
        def fit_eval(Ptr, ytr):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Ptr, ytr)
            return float(roc_auc_score(yt, clf.predict_proba(Pt)[:, 1]))
        out["clean"][held] = fit_eval(prof(xs), ds.y)
        a_sp, a_gen = [], []
        for r in range(reps):
            torch.manual_seed(r)
            a_sp.append(fit_eval(prof(trn(xs, ss)), ds.y))
            a_gen.append(fit_eval(prof(gen(xs)), ds.y))
        out["site_profile_aug"][held] = float(np.mean(a_sp)); out["generic_aug"][held] = float(np.mean(a_gen))
    for k in ("clean", "site_profile_aug", "generic_aug"):
        out[k]["MEAN"] = float(np.mean([v for kk, v in out[k].items() if kk != "MEAN"]))
    os.makedirs("runs", exist_ok=True)
    json.dump(out, open("runs/shortcut_removal.json", "w"), indent=2)
    print(json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
