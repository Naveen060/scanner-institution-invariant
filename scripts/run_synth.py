"""Semi-synthetic label-shift sweep (known acquisition transforms, controlled priors) + site-profile recovery check.

  python scripts/run_synth.py --out runs/synth1 --spreads 0 0.2 0.4 --methods erm dann sidl_cc --seeds 0 1 --holdouts V0 V2 V4
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.arraydata import ArrayDataset, load_covid_cxr  # noqa: E402
from sidl.data import leave_one_site_out  # noqa: E402
from sidl.siteprofile import SiteProfiles  # noqa: E402
from sidl.synth import make_virtual_sites  # noqa: E402
from sidl.trainer import TrainConfig, train_and_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spreads", nargs="+", type=float, default=[0.0, 0.2, 0.4])
    ap.add_argument("--methods", nargs="+", default=["erm", "dann", "sidl_cc"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--holdouts", nargs="+", default=["V0", "V2", "V4"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--per-site", type=int, default=160)
    ap.add_argument("--strict", action="store_true", help="use every source image at most once across virtual sites")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(log):
        for l in open(log):
            r = json.loads(l); done.add((r["spread"], r["holdout"], r["method"], r["seed"]))
    X, df0 = load_covid_cxr()
    y0 = df0.label.values
    extra = {}
    for kv in a.set:
        k, v = kv.split("="); extra[k] = (v.lower() in ("1", "true", "yes")) if isinstance(getattr(TrainConfig, k), bool) else type(getattr(TrainConfig, k))(v)
    for spread in a.spreads:
        for seed in a.seeds:
            Xv, df, params, priors = make_virtual_sites(X, y0, a.K, spread, a.per_site, seed, strict=a.strict)
            # profile recovery: estimated per-site profile means vs. true transform strength (Spearman over sites)
            s2i_all = {s: i for i, s in enumerate(sorted(df.site.unique()))}
            pr = SiteProfiles(a.K, 2).fit(ArrayDataset(Xv, df, s2i_all, 64))
            rec = {}
            for j, nm in enumerate(["mean", "std", "noise", "sharp", "bias"]):
                rec[nm] = float(spearmanr(range(a.K), pr.mu[:, j].numpy())[0])
            with open(os.path.join(a.out, "profile_recovery.jsonl"), "a") as f:
                f.write(json.dumps({"spread": spread, "seed": seed, "spearman_vs_degradation": rec}) + "\n")
            for held in a.holdouts:
                sp = leave_one_site_out(df, held, seed=seed)
                s2i = {s: i for i, s in enumerate(sp.site_names)}
                tr = ArrayDataset(Xv, sp.train, s2i, 64, train=True)
                va = ArrayDataset(Xv, sp.val, s2i, 64)
                te = ArrayDataset(Xv, sp.test, s2i, 64)
                for m in a.methods:
                    if (spread, held, m, seed) in done:
                        continue
                    cfg = TrainConfig(method=m, backbone="small", pretrained=False, in_chans=1, modality="cxr",
                                      epochs=a.epochs, batch_size=a.batch, lr=a.lr, num_workers=0, device="cpu", amp=False,
                                      seed=seed, out_dir=f"{a.out}/s{spread}/{held}/{m}/seed{seed}", **extra)
                    t0 = time.time()
                    r = train_and_eval(cfg, tr, va, te, len(sp.site_names))
                    row = {"spread": spread, "holdout": held, "method": m, "seed": seed, "prev_test": float(te.y.mean()),
                           "sec": round(time.time() - t0, 1), **r["test_overall"],
                           "cov": {k: v["coverage"] for k, v in r["conformal"].items()},
                           "size": {k: v["set_size"] for k, v in r["conformal"].items()}}
                    with open(log, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(json.dumps({k: row[k] for k in ("spread", "holdout", "method", "seed", "auroc", "sec")}), flush=True)


if __name__ == "__main__":
    main()
