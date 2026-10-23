"""Does EM prior estimation + label-shift conformal fix under-coverage / balanced accuracy / calibration on unseen sites?
  python scripts/analyze_labelshift.py runs/it1 --folds Actualmed Italy China Spain --methods erm"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import load_covid_cxr
from sidl.conformal import calibrate, evaluate_sets, prediction_sets
from sidl.labelshift import adjust_probs, em_prior, label_shift_sets
from sidl.metrics import balanced_accuracy, ece, auroc

ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--folds", nargs="*"); ap.add_argument("--methods", nargs="*")
ap.add_argument("--alpha", type=float, default=0.1); a = ap.parse_args(); pd.set_option("display.width", 200)
X, df = load_covid_cxr(); rows = []
for f in sorted(glob.glob(f"{a.path}/*/*/seed*/preds.npz")):
    held, method, seed = f.split("/")[-4], f.split("/")[-3], int(f.split("/")[-2][4:])
    if (a.folds and held not in a.folds) or (a.methods and method not in a.methods): continue
    z = np.load(f); pv, yv, pt, yt = z["pv"], z["yv"], z["pt"], z["yt"]; sv = z["sv"]
    pi_s = np.bincount(yv, minlength=2) / len(yv)
    pi_hat = em_prior(pt, pi_s); pi_true = np.bincount(yt, minlength=2) / len(yt)
    res = {}
    for name, pit in [("none", pi_s), ("em", pi_hat), ("oracle", pi_true)]:
        pt_adj = adjust_probs(pt, pi_s, pit)
        res[f"bal_acc[{name}]"] = balanced_accuracy(yt, pt_adj); res[f"ece[{name}]"] = ece(yt, pt_adj)
    res["prior_err_em"] = abs(pi_hat[1] - pi_true[1]); res["prior_err_none"] = abs(pi_s[1] - pi_true[1])
    q = calibrate(pv, yv, sv, a.alpha, "pooled"); ev = evaluate_sets(prediction_sets(pt, q), yt); res["cov[pooled]"], res["size[pooled]"] = ev["coverage"], ev["set_size"]
    q = calibrate(pv, yv, sv, a.alpha, "worst_site"); ev = evaluate_sets(prediction_sets(pt, q), yt); res["cov[worst_site]"], res["size[worst_site]"] = ev["coverage"], ev["set_size"]
    for name, pit in [("em", pi_hat), ("oracle", pi_true)]:
        sets, _ = label_shift_sets(pv, yv, pt, pi_s, pit, a.alpha); ev = evaluate_sets(sets, yt)
        res[f"cov[ls_{name}]"], res[f"size[ls_{name}]"] = ev["coverage"], ev["set_size"]
    rows.append(dict(holdout=held, method=method, seed=seed, **res))
d = pd.DataFrame(rows)
fold = d.groupby(["method", "holdout"]).mean(numeric_only=True).drop(columns="seed").groupby("method").mean()
print(fold.round(3).T)
