"""Build prediction directories for the CNN + pretrained-head ENSEMBLE in the layout scripts/analyze_lso_conformal.py reads.

Ensemble = mean predicted probability of the from-scratch CNN and the frozen-encoder head with the same seed. Validation images,
leave-site-out calibration images and held-out images are identical across the two families (checked below), so averaging is
row-wise. Seeds 0 and 1 (the CNN leave-site-out models exist for two seeds).

  python scripts/make_ensemble_runs.py --out_lso runs/it16_ens_lso --out_full runs/it16_ens_full
  python scripts/analyze_lso_conformal.py runs/it16_ens_lso --full_dir runs/it16_ens_full --method erm
"""
import argparse
import glob
import os

import numpy as np

DEV = ["Actualmed", "Italy", "China", "Spain"]
CONF = ["UK", "Australia", "Other", "Unknown"]
CNN_FULL = {**{h: "runs/it1" for h in DEV}, **{h: "runs/conf_erm" for h in CONF}}
CNN_LSO = {**{h: "runs/it6" for h in DEV}, **{h: "runs/conf_lso" for h in CONF}}
FEAT_FULL = {**{h: "runs/it11" for h in DEV}, **{h: "runs/it14_feat" for h in CONF}}
FEAT_LSO = "runs/it15_lso"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_lso", required=True)
    ap.add_argument("--out_full", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--family", choices=["ens", "cnn", "feat"], default="ens")
    a = ap.parse_args()
    w_cnn = {"ens": 0.5, "cnn": 1.0, "feat": 0.0}[a.family]
    for h in DEV + CONF:
        for s in a.seeds:
            c = np.load(f"{CNN_FULL[h]}/{h}/erm/seed{s}/preds.npz", allow_pickle=True)
            f = np.load(f"{FEAT_FULL[h]}/{h}/erm/seed{s}/preds.npz")
            assert (c["yv"] == f["yv"]).all() and (c["yt"] == f["yt"]).all(), (h, s)
            d = f"{a.out_full}/{h}/erm/seed{s}"
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(d + "/preds.npz", pv=w_cnn * c["pv"] + (1 - w_cnn) * f["pv"], yv=f["yv"], sv=f["sv"],
                                pt=w_cnn * c["pt"] + (1 - w_cnn) * f["pt"], yt=f["yt"], st=f["st"])
            for dc in sorted(glob.glob(f"{CNN_LSO[h]}/{h}/seed{s}/inner_*")):
                k = os.path.basename(dc)
                df_ = f"{FEAT_LSO}/{h}/seed{s}/{k}"
                zc, zf = np.load(dc + "/preds.npz", allow_pickle=True), np.load(df_ + "/preds.npz")
                assert (zc["yt"] == zf["yt"]).all(), (h, s, k)
                hc, hf = np.load(dc + "/extra_H.npz")["p"], np.load(df_ + "/extra_H.npz")["p"]
                do = f"{a.out_lso}/{h}/seed{s}/{k}"
                os.makedirs(do, exist_ok=True)
                np.savez_compressed(do + "/preds.npz", pt=w_cnn * zc["pt"] + (1 - w_cnn) * zf["pt"], yt=zf["yt"])
                np.savez_compressed(do + "/extra_H.npz", p=w_cnn * hc + (1 - w_cnn) * hf)
    print("done")


if __name__ == "__main__":
    main()
