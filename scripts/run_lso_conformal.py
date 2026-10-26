"""Leave-site-out (LSO) calibration models for cross-site conformal prediction.

For each outer held-out site H and seed, and for each SOURCE site k: train an inner model on the source sites except k
(with its own source-only validation split), then store its predictions on (a) all images of site k  -> calibration scores that
mimic how a model behaves on a site it has NEVER seen, and (b) the held-out site H (for CV+-style test scoring).

  python scripts/run_lso_conformal.py --out runs/it6 --holdouts Actualmed Italy China Spain --seeds 0 1 --epochs 15
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.arraydata import ArrayDataset, load_covid_cxr  # noqa: E402
from sidl.data import patient_level_split  # noqa: E402
from sidl.trainer import TrainConfig, train_and_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/covid_cxr")
    ap.add_argument("--holdouts", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--method", default="erm")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    torch.set_num_threads(1)
    X, df = load_covid_cxr(a.data)
    extra = {}
    for kv in a.set:
        k, v = kv.split("="); extra[k] = (v.lower() in ("1", "true", "yes")) if isinstance(getattr(TrainConfig, k), bool) else type(getattr(TrainConfig, k))(v)
    for held in a.holdouts:
        src = df[df.site != held].reset_index(drop=True)
        for seed in a.seeds:
            for k in sorted(src.site.unique()):
                d = f"{a.out}/{held}/seed{seed}/inner_{k}"
                if os.path.exists(f"{d}/preds.npz") and os.path.exists(f"{d}/extra_H.npz"):
                    continue
                inner = src[src.site != k].reset_index(drop=True)
                tr_df, va_df = patient_level_split(inner, 0.15, seed)
                s2i = {s: i for i, s in enumerate(sorted(inner.site.unique()))}
                tr = ArrayDataset(X, tr_df, s2i, 64, train=True)
                va = ArrayDataset(X, va_df, s2i, 64)
                cal = ArrayDataset(X, src[src.site == k], s2i, 64)
                H = ArrayDataset(X, df[df.site == held], s2i, 64)
                cfg = TrainConfig(method=a.method, backbone="small", pretrained=False, in_chans=1, modality="cxr",
                                  epochs=a.epochs, batch_size=a.batch, lr=a.lr, num_workers=0, device="cpu", amp=False,
                                  seed=seed, out_dir=d, aug_p=0.0, **extra)
                t0 = time.time()
                train_and_eval(cfg, tr, va, cal, len(s2i), extra_eval={"H": H})
                print(held, seed, k, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
