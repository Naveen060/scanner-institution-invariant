"""Robustness checks a reviewer would ask for (descriptive; none of this changed any earlier decision).

 1. Coverage uncertainty: hierarchical bootstrap (sites, then images within a site) of the mean coverage at alpha = 0.1 for the
    standard and leave-site-out calibrations, for the CNN, the pretrained head and their ensemble; and coverage by class.
 2. Sensitivity of the accuracy comparisons to the two tiniest test sets (China: 5 negatives, Australia: 6 positives).
 3. Multiplicity: the smallest raw bootstrap p-value among the pre-listed comparisons and its Holm-adjusted value.

  python scripts/reviewer_checks.py   -> prints a report and writes runs/reviewer_checks.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
import final_report as fr  # noqa: E402
from analyze_lso_conformal import load_fold, lso_modes  # noqa: E402
from sidl.conformal import calibrate, prediction_sets  # noqa: E402

DEV, CONF = fr.DEV, fr.CONF
ALL = DEV + CONF
ALPHA = 0.1
MODES = [("pooled", "pooled"), ("worst_site", "worst-site"), ("lso_split", "LSO-split"), ("lso_cvplus", "LSO-CV+")]
FAM = {"CNN": ("runs/it16_cnn_lso", "runs/it16_cnn_full"), "Pretrained": ("runs/it16_feat_lso", "runs/it16_feat_full"),
       "Ensemble": ("runs/it16_ens_lso", "runs/it16_ens_full")}
OUT = {}


def per_image_cover(lso_dir, full_dir):
    """{mode: {site: (y, covered[img] averaged over seeds)}}"""
    acc = {m: {} for m, _ in MODES}
    for h in ALL:
        for seed in (0, 1):
            fd = f"{lso_dir}/{h}/seed{seed}"
            lf = load_fold(fd)
            fz = np.load(f"{full_dir}/{h}/erm/seed{seed}/preds.npz")
            R, kid, Q, n_k = lf
            yt, pt = fz["yt"], fz["pt"]
            sets = {}
            for mode in ("pooled", "worst_site"):
                sets[mode] = prediction_sets(pt, calibrate(fz["pv"], fz["yv"], fz["sv"], ALPHA, mode))
            ls = lso_modes(R, kid, n_k, Q, pt, ALPHA)
            sets["lso_split"], sets["lso_cvplus"] = ls["lso_split"], ls["lso_cvplus"]
            for m, _ in MODES:
                cov = sets[m][np.arange(len(yt)), yt].astype(float)
                acc[m].setdefault(h, []).append((yt, cov))
    return {m: {h: (v[0][0], np.mean([c for _, c in v], 0)) for h, v in d.items()} for m, d in acc.items()}


def boot_cov(d, sites, B=3000, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(B):
        vals = []
        for h in rng.choice(sites, len(sites)):
            y, c = d[h]
            vals.append(c[rng.randint(0, len(c), len(c))].mean())
        out.append(np.mean(vals))
    return float(np.mean([d[h][1].mean() for h in sites])), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float(np.mean(np.array(out) >= 1 - ALPHA))


print("== 1. coverage at alpha=0.1 (target 0.90): mean over 8 sites [hier. bootstrap 95% CI]; share of replicates >= 0.90")
OUT["coverage"] = {}
for fam, (ld, fd) in FAM.items():
    cov = per_image_cover(ld, fd)
    OUT["coverage"][fam] = {}
    for m, name in MODES:
        mean, lo, hi, pge = boot_cov(cov[m], ALL)
        # per-class coverage: mean over sites that have >= 5 images of the class
        pc = {}
        for c in (0, 1):
            v = [cov[m][h][1][cov[m][h][0] == c].mean() for h in ALL if (cov[m][h][0] == c).sum() >= 5]
            pc[c] = float(np.mean(v)); pc[f"n{c}"] = len(v)
        six = [h for h in ALL if h not in ("China", "Australia")]
        m6 = float(np.mean([cov[m][h][1].mean() for h in six]))
        OUT["coverage"][fam][m] = dict(mean=mean, lo=lo, hi=hi, p_ge_target=pge, cov_other=pc[0], cov_covid=pc[1], six_site_mean=m6)
        print(f"  {fam:10s} {name:10s} {mean:.3f} [{lo:.3f}, {hi:.3f}]  P(>=0.90)={pge:.3f} | other {pc[0]:.3f} covid {pc[1]:.3f} | 6 large sites {m6:.3f}")

print("\n== 2. accuracy comparisons without China and Australia (6 sites)")
SIX = [h for h in ALL if h not in ("China", "Australia")]
OUT["six_site"] = {}
fr.ALL = SIX
for lab in ["GroupDRO", "DANN", "Generic aug.", "Site-profile aug.", "Profile adversary (PAI)", "SIDL-cc", "SIDL-phys", "SIDL-full"]:
    lo, hi, p = fr.hier_boot_diff(lab, B=1500)
    a, e = fr.load_preds(lab), fr.load_preds("ERM")
    d = np.mean([np.mean([fr._auc(y, p_) for y, p_ in a[h]]) - np.mean([fr._auc(y, p_) for y, p_ in e[h]]) for h in SIX])
    OUT["six_site"][lab] = dict(diff=float(d), lo=lo, hi=hi)
    print(f"  CNN {lab:26s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
fr.ALL = ALL
import round2_report as r2  # noqa: E402
from feat_compare import boot, seed_mean_auc  # noqa: E402
S, P = r2.get(r2.CNN), r2.get(r2.feat("erm"))
models = {"CNN": S, "Pre": P, "Ens": r2.combo(S, P), "CnnEns": {h: r2.seedens(S[h]) for h in ALL}}
for x, y in (("Pre", "CNN"), ("Ens", "CNN"), ("Ens", "CnnEns")):
    m_, lo, hi = boot(models[x], models[y], SIX, B=2000)
    OUT["six_site"][f"{x}-{y}"] = dict(diff=m_, lo=lo, hi=hi)
    print(f"  {x} - {y}: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
for k in ["groupdro", "erm_p", "dann", "cdann", "cdann_p", "coral", "cc_full", "sitecenter"]:
    d_ = r2.get(r2.feat(k))
    m_, lo, hi = boot(d_, P, SIX, B=2000)
    OUT["six_site"][f"feat_{k}"] = dict(diff=m_, lo=lo, hi=hi)
    print(f"  feature head {k:10s} - ERM: {m_:+.3f} [{lo:+.3f}, {hi:+.3f}]")
print("  worst site (6 sites): CNN %.3f  pretrained %.3f  ensemble %.3f" % tuple(min(seed_mean_auc(models[k][h], np.arange(len(models[k][h][0][0]))) for h in SIX) for k in ("CNN", "Pre", "Ens")))

print("\n== 3. multiplicity")
R = json.load(open("runs/round2.json"))
ps = []
rng = np.random.RandomState(1)
# two-sided bootstrap p-values from the hierarchical bootstrap on all 8 sites for the pre-listed comparisons
def p_two_sided(a, b, folds, B=2000):
    r = np.random.RandomState(0); ds = []
    for _ in range(B):
        dd = []
        for h in r.choice(folds, len(folds)):
            y = a[h][0][0]; pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            idx = np.concatenate([r.choice(pos, len(pos)), r.choice(neg, len(neg))])
            dd.append(seed_mean_auc(a[h], idx) - seed_mean_auc(b[h], idx))
        ds.append(np.mean(dd))
    ds = np.array(ds)
    return float(min(1.0, 2 * min(np.mean(ds <= 0), np.mean(ds >= 0))))


names, pv = [], []
for k in ["groupdro", "erm_p", "dann", "cdann", "cdann_p", "coral", "cc_full", "sitecenter"]:
    names.append("feat_" + k); pv.append(p_two_sided(r2.get(r2.feat(k)), P, ALL))
for x, y in (("Ens", "CNN"), ("Ens", "CnnEns"), ("Pre", "CNN")):
    names.append(f"{x}-{y}"); pv.append(p_two_sided(models[x], models[y], ALL))
order = np.argsort(pv); m = len(pv)
holm = np.minimum(1.0, np.maximum.accumulate([(m - i) * pv[j] for i, j in enumerate(order)]))
OUT["multiplicity"] = {names[j]: dict(p_raw=pv[j], p_holm=float(holm[i])) for i, j in enumerate(order)}
for i, j in enumerate(order):
    print(f"  {names[j]:18s} raw p={pv[j]:.3f}  Holm p={holm[i]:.3f}")
json.dump(OUT, open("runs/reviewer_checks.json", "w"), indent=1)
