"""Train/evaluate on a manifest CSV under leave-one-site-out (LOSO) or in-distribution protocol.

Examples
  python scripts/run_experiment.py --config configs/cxr.yaml --manifest data/cxr_manifest.csv --protocol loso --methods erm dann sidl --seeds 0 1 2
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.data import ManifestDataset, in_distribution_split, leave_one_site_out  # noqa: E402
from sidl.trainer import TrainConfig, train_and_eval  # noqa: E402
import pandas as pd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", choices=["loso", "id"], default="loso")
    ap.add_argument("--methods", nargs="+",
                    default=["erm", "harmonize", "dann", "groupdro", "sidl", "sidl_phys", "sidl_cc", "sidl_full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--holdouts", nargs="*", default=None, help="Subset of sites to hold out (default: all)")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()

    base = yaml.safe_load(open(a.config))
    if a.epochs:
        base["epochs"] = a.epochs
    df = pd.read_csv(a.manifest)
    sites = sorted(df["site"].unique())
    folds = (a.holdouts or sites) if a.protocol == "loso" else ["ID"]
    size = base.pop("image_size", 224)

    all_res = []
    for held in folds:
        for seed in a.seeds:
            sp = leave_one_site_out(df, held, seed=seed) if a.protocol == "loso" else in_distribution_split(df, seed=seed)
            s2i = {s: i for i, s in enumerate(sp.site_names)}
            mk = lambda d, tr=False: ManifestDataset(d, s2i, size, base.get("in_chans", 3), train=tr)
            train_ds, val_ds, test_ds = mk(sp.train, True), mk(sp.val), mk(sp.test)
            for m in a.methods:
                cfg = TrainConfig(**{**base, "method": m, "seed": seed, "out_dir": f"{a.out}/{held}/{m}/seed{seed}"})
                r = train_and_eval(cfg, train_ds, val_ds, test_ds, len(sp.site_names), sp.site_names)
                row = {"holdout": held, "method": m, "seed": seed, **r["test_overall"]}
                for mode, v in r["conformal"].items():
                    row[f"cov_{mode}"], row[f"size_{mode}"] = v["coverage"], v["set_size"]
                all_res.append(row)
                print(json.dumps(row))
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(all_res).to_csv(os.path.join(a.out, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
