"""Resumable leave-one-site-out runner for the cached multi-source CXR benchmark.

  python scripts/run_array_loso.py --out runs/it1 --methods erm sidl_full --seeds 0 1 --holdouts Italy UK --epochs 15
Each (holdout, method, seed) appends one JSON line to <out>/results.jsonl; finished keys are skipped on restart.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.arraydata import ArrayDataset, load_covid_cxr  # noqa: E402
from sidl.data import leave_one_site_out  # noqa: E402
from sidl.trainer import TrainConfig, train_and_eval  # noqa: E402

ELIGIBLE = ["Actualmed", "Other", "Italy", "UK", "Australia", "China", "Spain", "Unknown"]  # both classes present (>=5 each)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/covid_cxr")
    ap.add_argument("--methods", nargs="+", default=["erm", "sidl_full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--holdouts", nargs="+", default=ELIGIBLE)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--set", nargs="*", default=[], help="extra TrainConfig overrides key=value")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(log):
        for l in open(log):
            r = json.loads(l)
            done.add((r["holdout"], r["method"], r["seed"]))
    X, df = load_covid_cxr(a.data)
    extra = {}
    for kv in a.set:
        k, v = kv.split("=")
        extra[k] = type(getattr(TrainConfig, k))(v) if not isinstance(getattr(TrainConfig, k), bool) else v.lower() in ("1", "true", "yes")
    for held in a.holdouts:
        for seed in a.seeds:
            sp = leave_one_site_out(df, held, seed=seed)
            s2i = {s: i for i, s in enumerate(sp.site_names)}
            tr = ArrayDataset(X, sp.train, s2i, a.size, train=True)
            va = ArrayDataset(X, sp.val, s2i, a.size)
            te = ArrayDataset(X, sp.test, s2i, a.size)
            for m in a.methods:
                if (held, m, seed) in done:
                    continue
                cfg = TrainConfig(method=m, backbone="small", pretrained=False, in_chans=1, modality="cxr",
                                  epochs=a.epochs, batch_size=a.batch, lr=a.lr, num_workers=0, device="cpu", amp=False,
                                  seed=seed, out_dir=f"{a.out}/{held}/{m}/seed{seed}", **extra)
                t0 = time.time()
                r = train_and_eval(cfg, tr, va, te, len(sp.site_names))
                row = {"holdout": held, "method": m, "seed": seed, "n_test": len(te), "prev_test": float(te.y.mean()),
                       "sec": round(time.time() - t0, 1), **r["test_overall"],
                       "cov": {k: v["coverage"] for k, v in r["conformal"].items()},
                       "size": {k: v["set_size"] for k, v in r["conformal"].items()}, "best_val": r["best_val"]}
                with open(log, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("holdout", "method", "seed", "auroc", "acc", "sec")}), flush=True)


if __name__ == "__main__":
    main()
