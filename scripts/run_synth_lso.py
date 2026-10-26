"""LSO conformal on the controlled semi-synthetic benchmark (known acquisition-shift severity).

For each severity, seed and held-out virtual site H: train the FULL ERM model on all other sites (deployed model) and one inner model
per source site k (never seeing k), saving predictions in the same layout analyze_lso_conformal.py expects:
  <out>/sev{S}/{H}/seed{seed}/inner_{k}/ (preds.npz, extra_H.npz)      <out>/sev{S}_full/{H}/erm/seed{seed}/preds.npz
"""
import argparse, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import ArrayDataset, load_covid_cxr
from sidl.data import leave_one_site_out, patient_level_split
from sidl.synth import make_virtual_sites
from sidl.trainer import TrainConfig, train_and_eval

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--severities", nargs="+", type=float, default=[0.5, 1.0, 1.5])
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1]); ap.add_argument("--holdouts", nargs="+", default=["V0", "V2", "V4"])
ap.add_argument("--spread", type=float, default=0.25); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--per-site", type=int, default=160); ap.add_argument("--strict", action="store_true")
a = ap.parse_args(); torch.set_num_threads(1)
X, df0 = load_covid_cxr()
mk = lambda **kw: TrainConfig(backbone="small", pretrained=False, in_chans=1, modality="cxr", epochs=a.epochs, batch_size=64, lr=2e-3,
                              num_workers=0, device="cpu", amp=False, aug_p=0.0, **kw)
for sev in a.severities:
    for seed in a.seeds:
        Xv, df, params, priors = make_virtual_sites(X, df0.label.values, 5, a.spread, a.per_site, seed, severity=sev, strict=a.strict)
        for held in a.holdouts:
            fd = f"{a.out}/sev{sev}_full/{held}/erm/seed{seed}"
            if not os.path.exists(f"{fd}/preds.npz"):
                sp = leave_one_site_out(df, held, seed=seed); s2i = {s: i for i, s in enumerate(sp.site_names)}
                train_and_eval(mk(method="erm", seed=seed, out_dir=fd), ArrayDataset(Xv, sp.train, s2i, 64, train=True),
                               ArrayDataset(Xv, sp.val, s2i, 64), ArrayDataset(Xv, sp.test, s2i, 64), len(s2i))
            src = df[df.site != held].reset_index(drop=True)
            for k in sorted(src.site.unique()):
                d = f"{a.out}/sev{sev}/{held}/seed{seed}/inner_{k}"
                if os.path.exists(f"{d}/extra_H.npz"): continue
                inner = src[src.site != k].reset_index(drop=True)
                tr_df, va_df = patient_level_split(inner, 0.15, seed); s2i = {s: i for i, s in enumerate(sorted(inner.site.unique()))}
                t0 = time.time()
                train_and_eval(mk(method="erm", seed=seed, out_dir=d), ArrayDataset(Xv, tr_df, s2i, 64, train=True), ArrayDataset(Xv, va_df, s2i, 64),
                               ArrayDataset(Xv, src[src.site == k], s2i, 64), len(s2i), extra_eval={"H": ArrayDataset(Xv, df[df.site == held], s2i, 64)})
                print(sev, seed, held, k, round(time.time() - t0, 1), flush=True)
