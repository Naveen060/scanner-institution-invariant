"""Leave-site-out (LSO) inner models on frozen-encoder features, for cross-site conformal calibration.

Same layout as scripts/run_lso_conformal.py (so scripts/analyze_lso_conformal.py works unchanged): for each outer held-out
site H, seed and SOURCE site k, train an inner head on the source sites except k, then store its predictions on site k
(calibration scores) and on H (extra_H.npz, key 'p').

  python scripts/run_feat_lso_conformal.py --out runs/it15_lso --holdouts Actualmed Italy China Spain --seeds 0 1 2 --method erm
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.data import patient_level_split  # noqa: E402
from sidl.featheads import FEAT_METHODS, predict, train_head  # noqa: E402
from run_feat_loso import HP_TYPES, load_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/covid_cxr224")
    ap.add_argument("--feats", default="data/covid_cxr224/xrv_features.npz")
    ap.add_argument("--holdouts", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--method", default="erm")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(1)
    hp = {k: HP_TYPES[k](v) for k, v in (kv.split("=") for kv in a.set)}
    df = load_meta(a.root)
    F = np.load(a.feats)["feats"].astype(np.float32)
    flags = FEAT_METHODS[a.method]
    for held in a.holdouts:
        src = df[df.site != held].reset_index(drop=True)
        H = df[df.site == held]
        for seed in a.seeds:
            for k in sorted(src.site.unique()):
                d = f"{a.out}/{held}/seed{seed}/inner_{k}"
                if os.path.exists(f"{d}/preds.npz") and os.path.exists(f"{d}/extra_H.npz"):
                    continue
                inner = src[src.site != k].reset_index(drop=True)
                tr_df, va_df = patient_level_split(inner, 0.15, seed)
                s2i = {s: i for i, s in enumerate(sorted(inner.site.unique()))}
                Xtr, Xva = F[tr_df.row.values], F[va_df.row.values]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                t0 = time.time()
                model, _ = train_head((Xtr - mu) / sd, tr_df.label.values, np.array([s2i[v] for v in tr_df.site]),
                                      (Xva - mu) / sd, va_df.label.values, len(s2i), flags, seed, **hp)
                cal = src[src.site == k]
                pc = predict(model, (F[cal.row.values] - mu) / sd)
                ph = predict(model, (F[H.row.values] - mu) / sd)
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(f"{d}/preds.npz", pt=pc, yt=cal.label.values)
                np.savez_compressed(f"{d}/extra_H.npz", p=ph)
                print(held, seed, k, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
