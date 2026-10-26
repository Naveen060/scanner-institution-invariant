"""Resumable leave-one-site-out runner on frozen encoder features (pretrained chest X-ray encoder).

  python scripts/run_feat_loso.py --out runs/it11 --methods erm erm_linear --seeds 0 1 2 3 4 --holdouts Actualmed Italy China Spain
Each (holdout, method, seed) appends one JSON line to <out>/results.jsonl and saves preds.npz; finished keys are skipped.
Splits, site merging and validation are identical to scripts/run_array_loso.py (patient-level 15% validation from source sites only).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.arraydata import SMALL_SITES  # noqa: E402
from sidl.conformal import conformal_report  # noqa: E402
from sidl.data import leave_one_site_out  # noqa: E402
from sidl.featheads import FEAT_METHODS, predict, train_head  # noqa: E402
from sidl.metrics import evaluate  # noqa: E402

ELIGIBLE = ["Actualmed", "Other", "Italy", "UK", "Australia", "China", "Spain", "Unknown"]
HP_TYPES = dict(epochs=int, bs=int, lr=float, wd=float, adv_weight=float, coral_weight=float, proj_dim=int, drop=float)


def load_meta(root):
    df = pd.read_csv(f"{root}/meta.csv")
    df["site"] = df["site"].where(~df["site"].isin(SMALL_SITES), "Other")
    df["row"] = np.arange(len(df))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/covid_cxr224")
    ap.add_argument("--feats", default="data/covid_cxr224/xrv_features.npz")
    ap.add_argument("--methods", nargs="+", default=["erm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--holdouts", nargs="+", default=ELIGIBLE)
    ap.add_argument("--set", nargs="*", default=[], help="hyperparameters key=value (epochs, lr, adv_weight, ...)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(log):
        for l in open(log):
            r = json.loads(l)
            done.add((r["holdout"], r["method"], r["seed"]))
    hp = {k: HP_TYPES[k](v) for k, v in (kv.split("=") for kv in a.set)}
    df = load_meta(a.root)
    F = np.load(a.feats)["feats"].astype(np.float32)
    assert len(F) == len(df)
    for held in a.holdouts:
        for seed in a.seeds:
            sp = leave_one_site_out(df, held, seed=seed)
            s2i = {s: i for i, s in enumerate(sp.site_names)}
            Xtr, Xva, Xte = F[sp.train.row.values], F[sp.val.row.values], F[sp.test.row.values]
            ytr, yva, yte = sp.train.label.values, sp.val.label.values, sp.test.label.values
            str_ = np.array([s2i[v] for v in sp.train.site]); sva = np.array([s2i[v] for v in sp.val.site])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6  # source-only standardization
            Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
            for m in a.methods:
                if (held, m, seed) in done:
                    continue
                flags = FEAT_METHODS[m]
                A, B, C = Xtr, Xva, Xte
                if flags.get("center"):  # transductive per-site centering (target mean from unlabeled test features)
                    A, B = Xtr.copy(), Xva.copy()
                    for s, i in s2i.items():
                        pool = np.concatenate([Xtr[str_ == i], Xva[sva == i]])
                        A[str_ == i] -= pool.mean(0); B[sva == i] -= pool.mean(0)
                    C = Xte - Xte.mean(0)
                t0 = time.time()
                model, best = train_head(A, ytr, str_, B, yva, len(s2i), flags, seed, **hp)
                pt, pv = predict(model, C), predict(model, B)
                met = evaluate(yte, pt[:, 1])
                rep = conformal_report(pv, yva, sp.val.site.values, pt, yte, sp.test.site.values)
                row = {"holdout": held, "method": m, "seed": seed, "n_test": len(yte), "prev_test": float(yte.mean()),
                       "sec": round(time.time() - t0, 1), **met, "cov": {k: v["coverage"] for k, v in rep.items()},
                       "size": {k: v["set_size"] for k, v in rep.items()}, "best_val": best, "hp": hp}
                d = f"{a.out}/{held}/{m}/seed{seed}"
                os.makedirs(d, exist_ok=True)
                np.savez_compressed(f"{d}/preds.npz", pv=pv, yv=yva, sv=np.asarray(list(sp.val.site.values), dtype="U32"), pt=pt, yt=yte,
                                    st=np.asarray(list(sp.test.site.values), dtype="U32"))
                with open(log, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("holdout", "method", "seed", "auroc", "sec")}), flush=True)


if __name__ == "__main__":
    main()
