"""Build every table, figure and number used in the paper from the raw run directories.

  python scripts/final_report.py            # writes paper/tables/*.tex, paper/figures/*.{png,pdf}, runs/results_summary.json
Development folds: Actualmed, Italy, China, Spain. Confirmation folds (frozen configs, run once): UK, Australia, Other, Unknown.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from sidl.metrics import auroc  # noqa: E402
from figstyle import FS, finish, size, title  # noqa: E402  (FIG_PROFILE=ieee switches to the IEEE figure style)

DEV = ["Actualmed", "Italy", "China", "Spain"]
CONF = ["UK", "Australia", "Other", "Unknown"]
ALL = DEV + CONF
METHODS = [  # label, (dev dir, key), (confirmation dir, key)
    ("ERM", ("runs/it1", "erm"), ("runs/conf_erm", "erm")),
    ("GroupDRO", ("runs/it1", "groupdro"), ("runs/conf_groupdro", "groupdro")),
    ("DANN", ("runs/it1", "dann"), ("runs/conf_dann", "dann")),
    ("Generic aug.", ("runs/it2_gen_p03_s05", "harmonize"), ("runs/conf_gen", "harmonize")),
    ("Site-profile aug.", ("runs/it2_phys_p03_s05", "phys_aug"), ("runs/conf_phys", "phys_aug")),
    ("Profile adversary (PAI)", ("runs/it5_l01", "pai"), ("runs/conf_pai", "pai")),
    ("SIDL-cc", ("runs/it8_mild", "sidl_cc"), ("runs/conf_sidl_cc", "sidl_cc")),
    ("SIDL-phys", ("runs/it8_mild", "sidl_phys"), ("runs/conf_sidl_phys", "sidl_phys")),
    ("SIDL-full", ("runs/it8_mild", "sidl_full"), ("runs/conf_sidl_full", "sidl_full")),
]
TEX = lambda s: s.replace("_", r"\_")
os.makedirs("paper/tables", exist_ok=True); os.makedirs("paper/figures", exist_ok=True)
SUM = {}


def read(dirpath, key, folds):
    f = f"{dirpath}/results.jsonl"
    if not os.path.exists(f):
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    return d[(d.method == key) & d.holdout.isin(folds)] if len(d) else d


def task_table():
    frames = []
    for label, dev, conf in METHODS:
        for (dd, key), folds, split in ((dev, DEV, "dev"), (conf, CONF, "conf")):
            d = read(dd, key, folds)
            if len(d):
                frames.append(d.assign(label=label, split=split)[["label", "split", "holdout", "seed", "auroc", "acc", "bal_acc", "ece"]])
    D = pd.concat(frames, ignore_index=True)
    cnt = D.groupby(["label", "holdout"]).seed.nunique()
    bad = cnt[cnt != 2]
    expected = len(METHODS) * len(ALL)
    if len(cnt) != expected or len(bad):
        print(f"WARNING: incomplete runs: {len(cnt)}/{expected} (method,fold) cells present; cells with !=2 seeds: {dict(bad)}")
    fold = D.groupby(["label", "split", "holdout"])[["auroc", "bal_acc", "ece"]].mean().reset_index()
    return D, fold


def boot(x, B=5000, seed=0):
    rng = np.random.RandomState(seed); x = np.asarray(x, float)
    b = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _auc(y, sc):
    from scipy.stats import rankdata
    r = rankdata(sc); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_preds(label):
    """{fold: [(y, p_pos) per seed]} for a method label, using dev dir for dev folds and conf dir for confirmation folds."""
    spec = {m[0]: m for m in METHODS}[label]
    out = {}
    for (dd, key), folds in ((spec[1], DEV), (spec[2], CONF)):
        for h in folds:
            fs = sorted(glob.glob(f"{dd}/{h}/{key}/seed*/preds.npz"))
            if fs:
                out[h] = [(np.load(f)["yt"], np.load(f)["pt"][:, 1]) for f in fs]
    return out


def hier_boot_diff(label, B=1500, seed=0):
    """Hierarchical paired bootstrap of mean-over-folds AUROC difference vs ERM: resample folds, then resample test images
    (stratified by class) within each sampled fold; the same images are used for both methods; seeds are averaged."""
    a, e = load_preds(label), load_preds("ERM")
    folds = [h for h in ALL if h in a and h in e]
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(B):
        ds = []
        for h in rng.choice(folds, len(folds)):
            y = e[h][0][0]
            pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            da = np.mean([_auc(yy[idx], pp[idx]) for yy, pp in a[h]]); de = np.mean([_auc(yy[idx], pp[idx]) for yy, pp in e[h]])
            ds.append(da - de)
        diffs.append(np.mean(ds))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(np.array(diffs) <= 0))


def main_table(fold):
    labels = [m[0] for m in METHODS if m[0] in set(fold.label)]
    piv = fold.pivot_table(index="label", columns="holdout", values="auroc").reindex(labels)
    erm = piv.loc["ERM"]
    rows, out = [], {}
    for l in labels:
        all_f = piv.loc[l].dropna(); dev = piv.loc[l, [h for h in DEV if h in piv.columns]].dropna(); conf = piv.loc[l, [h for h in CONF if h in piv.columns]].dropna()
        diff = (piv.loc[l] - erm).dropna()
        lo, hi, p_le0 = hier_boot_diff(l) if l != "ERM" and len(diff) > 1 else (np.nan, np.nan, np.nan)
        bal = fold[fold.label == l].bal_acc.mean(); ece = fold[fold.label == l].ece.mean()
        out[l] = dict(auroc_all=float(all_f.mean()), auroc_dev=float(dev.mean()), auroc_conf=float(conf.mean()) if len(conf) else None,
                      worst_fold=float(all_f.min()), bal_acc=float(bal), ece=float(ece), diff_vs_erm=float(diff.mean()) if l != "ERM" else 0.0,
                      diff_ci=[lo, hi], p_boot_le0=float(p_le0), folds_better=int((diff > 0).sum()) if l != "ERM" else None, n_folds=int(len(all_f)))
        o = out[l]
        d_s = "--" if l == "ERM" else f"{o['diff_vs_erm']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(f"{l} & {o['auroc_dev']:.3f} & {o['auroc_conf'] if o['auroc_conf'] is None else format(o['auroc_conf'], '.3f')} & {o['auroc_all']:.3f} & {o['worst_fold']:.3f} & {d_s} & {o['bal_acc']:.3f} & {o['ece']:.3f} \\\\")
    tex = "\\begin{tabular}{lcccccc c}\n\\toprule\nMethod & AUROC dev & AUROC conf. & AUROC all & Worst fold & $\\Delta$ vs ERM [95\\% CI] & Bal.\\ acc. & ECE \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    tex = tex.replace("{lcccccc c}", "{lccccccc}")
    open("paper/tables/main.tex", "w").write(tex)
    SUM["task"] = out
    from scipy.stats import spearmanr
    ms = [l for l in labels if out[l]["auroc_conf"] is not None]
    SUM["rank_corr_dev_conf"] = float(spearmanr([out[l]["auroc_dev"] for l in ms], [out[l]["auroc_conf"] for l in ms])[0])
    cols = [h for h in ALL if h in piv.columns]
    hdr = " & ".join(f"{c}$^{{{'d' if c in DEV else 'c'}}}$" for c in cols)
    body = "\n".join(f"{l} & " + " & ".join(f"{piv.loc[l, c]:.3f}" for c in cols) + " \\\\" for l in labels)
    open("paper/tables/perfold.tex", "w").write("\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\nMethod & " + hdr + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}")
    return piv


def fig_methods(piv):
    labels = list(piv.index)
    fig, ax = plt.subplots(figsize=size((7.2, 3.6), (7.16, 2.35)))
    for i, l in enumerate(labels):
        v = piv.loc[l].dropna()
        jit = np.random.RandomState(i).uniform(-.12, .12, len(v))
        for is_dev, col, mk in ((True, "#4477aa", "o"), (False, "#cc6677", "^")):
            sel = np.array([(h in DEV) == is_dev for h in v.index])
            ax.scatter((np.full(len(v), i) + jit)[sel], v.values[sel], s=18, alpha=.8, c=col, marker=mk)
        lo, hi = boot(v.values); ax.plot([i - .25, i + .25], [v.mean()] * 2, c="k", lw=2); ax.plot([i, i], [lo, hi], c="k", lw=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FS(8))
    ax.set_ylabel("Held-out-site AUROC"); ax.axhline(piv.loc["ERM"].mean(), ls=":", c="gray", lw=1)
    ax.scatter([], [], c="#4477aa", s=18, marker="o", label="development sites"); ax.scatter([], [], c="#cc6677", s=18, marker="^", label="confirmation sites"); ax.legend(**size(dict(fontsize=7, loc="lower left"), dict(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)))
    title(ax, "Leave-one-site-out AUROC (each dot = one held-out site, mean of 2 seeds)", fontsize=FS(9))
    finish(fig, "fig_methods")


def conformal_summary():
    parts = [pd.read_csv(f) for f in ("runs/conformal_dev.csv", "runs/conformal_conf.csv") if os.path.exists(f)]
    if not parts:
        return
    D = pd.concat(parts, ignore_index=True)
    d = D[np.isclose(D.alpha, 0.1)]
    fold = d.groupby(["mode", "holdout"])[["coverage", "size"]].mean().reset_index()
    S = fold.groupby("mode").agg(cov_mean=("coverage", "mean"), cov_min=("coverage", "min"), size=("size", "mean"),
                                 n_under=("coverage", lambda v: int((v < 0.85).sum())), n=("coverage", "size"))
    F = D.groupby(["mode", "alpha", "holdout"])[["coverage", "size"]].mean().reset_index()
    C = F.groupby(["mode", "alpha"])[["coverage", "size"]].mean().reset_index()
    sweep = {}
    for mode, g in C.groupby("mode"):
        g = g.sort_values("coverage")
        Fm = F[F["mode"] == mode]
        sweep[mode] = dict(cov_calib_err=float(np.mean(np.abs(g.coverage.values - (1 - g.alpha.values)))),
                           size_at_cov90=float(np.interp(0.90, g.coverage.values, g["size"].values)),
                           frac_under=float(np.mean(Fm.coverage < 1 - Fm.alpha - 0.05)))
    S = S.join(pd.DataFrame(sweep).T)
    names = {"pooled": "Pooled (seen-site val.)", "site_balanced": "Site-balanced (seen-site val.)", "worst_site": "Worst-site (seen-site val.)",
             "lso_split": "LSO-split", "lso_split_bal": "LSO-split, balanced", "lso_worst": "LSO-worst-site", "lso_cvplus": "LSO-CV+", "lso_cvplus_bal": "LSO-CV+, balanced"}
    order = ["pooled", "site_balanced", "worst_site", "lso_split", "lso_split_bal", "lso_worst", "lso_cvplus", "lso_cvplus_bal"]
    rows = []
    for m in order:
        if m in S.index:
            r = S.loc[m]
            rows.append(f"{names[m]} & {r.cov_mean:.3f} & {r.cov_min:.3f} & {int(r.n_under)}/{int(r.n)} & {r['size']:.2f} & {r.cov_calib_err:.3f} & {r.frac_under:.3f} & {r.size_at_cov90:.2f} \\\\")
    tex = ("\\begin{tabular}{lccccccc}\n\\toprule\nCalibration & Cov. & Min cov. & Folds $<$0.85 & Size & Cov.\\ calib.\\ err. & Frac.\\ under & Size@90\\% \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    open("paper/tables/conformal.tex", "w").write(tex)
    SUM["conformal_alpha0.1"] = {m: {k: (float(v) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in S.loc[m].items()} for m in S.index}
    # figure: per-fold coverage by mode
    fig, axs = plt.subplots(1, 2, figsize=size((9, 3.4), (7.16, 2.65)), gridspec_kw={"width_ratios": [1.15, 1]} if size(0, 1) else None)
    P = fold.pivot(index="mode", columns="holdout", values="coverage")
    cols = [h for h in ALL if h in P.columns]
    for i, m in enumerate([m for m in order if m in P.index]):
        jit = np.random.RandomState(i).uniform(-.15, .15, len(cols))
        for is_dev, col, mk in ((True, "#4477aa", "o"), (False, "#cc6677", "^")):
            sel = np.array([(h in DEV) == is_dev for h in cols])
            axs[0].scatter(P.loc[m, cols].values[sel], (np.full(len(cols), i) + jit)[sel], s=22, c=col, marker=mk, alpha=.8)
        axs[0].plot([P.loc[m, cols].mean()] * 2, [i - .3, i + .3], c="k")
    axs[0].axvline(0.9, c="gray", ls=":"); axs[0].set_yticks(range(len([m for m in order if m in P.index]))); axs[0].set_yticklabels([names[m] for m in order if m in P.index], fontsize=FS(7))
    axs[0].scatter([], [], c="#4477aa", s=22, marker="o", label="development sites"); axs[0].scatter([], [], c="#cc6677", s=22, marker="^", label="confirmation sites"); axs[0].legend(**size(dict(fontsize=6, loc="lower left"), dict(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False))); axs[0].set_xlabel("Coverage on held-out site (target 0.90)"); title(axs[0], "Per-site coverage, alpha=0.1", fontsize=FS(9))
    SHORT = {"pooled": "Pooled", "site_balanced": "Site-balanced", "worst_site": "Worst-site", "lso_split": "LSO-split", "lso_split_bal": "LSO-split bal.", "lso_worst": "LSO-worst", "lso_cvplus": "LSO-CV+", "lso_cvplus_bal": "LSO-CV+ bal."}
    for j, m in enumerate([m for m in order if m in C["mode"].unique()]):
        g = C[C["mode"] == m].sort_values("coverage"); axs[1].plot(g["size"], g.coverage, marker="osv^DPX*"[j % 8], ms=3, label=size(names[m], SHORT[m]), lw=1, ls=["-", "--", "-.", ":"][j % 4])
    axs[1].set_xlabel("Mean set size"); axs[1].set_ylabel("Coverage"); axs[1].legend(**size(dict(fontsize=6), dict(fontsize=8, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6, labelspacing=0.25, borderpad=0.3))); axs[1].set_ylim(*size((None, None), (0.66, 1.08))); axs[1].set_yticks([0.7, 0.8, 0.9, 1.0]) if size(0, 1) else None; title(axs[1], "Coverage vs. size across alpha", fontsize=FS(9))
    finish(fig, "fig_conformal", axs, ["(a)", "(b)"])


def optimism():
    rows = []
    for held in ALL:
        ldir, fdir = ("runs/conf_lso", "runs/conf_erm") if held in CONF else ("runs/it6", "runs/it1")
        for seed in (0, 1):
            fd = f"{ldir}/{held}/seed{seed}"
            inner = sorted(glob.glob(f"{fd}/inner_*"))
            rj = f"{fdir}/{held}/erm/seed{seed}/results.json"
            if not inner or not os.path.exists(rj) or not all(os.path.exists(f"{d}/preds.npz") for d in inner):
                continue
            a = [auroc(np.load(f"{d}/preds.npz")["yt"], np.load(f"{d}/preds.npz")["pt"][:, 1]) for d in inner]
            a = [x for x in a if not np.isnan(x)]
            r = json.load(open(rj))
            rows.append(dict(holdout=held, seed=seed, seen_val=r["best_val"], lso_est=float(np.mean(a)), actual=r["test_overall"]["auroc"]))
    if not rows:
        return
    d = pd.DataFrame(rows); g = d.groupby("holdout").mean(numeric_only=True).reindex([h for h in ALL if h in set(d.holdout)])
    from scipy.stats import spearmanr
    rho_seen = float(spearmanr(g.seen_val, g.actual)[0]) if len(g) > 2 else float("nan")
    rho_lso = float(spearmanr(g.lso_est, g.actual)[0]) if len(g) > 2 else float("nan")
    rho_gap_cov = float("nan")
    parts = [pd.read_csv(f) for f in ("runs/conformal_dev.csv", "runs/conformal_conf.csv") if os.path.exists(f)]
    if parts and len(g) > 2:
        C_ = pd.concat(parts); pc = C_[(np.isclose(C_.alpha, 0.1)) & (C_["mode"] == "pooled")].groupby("holdout").coverage.mean()
        gap = (g.seen_val - g.actual); common = [h for h in g.index if h in pc.index]
        rho_gap_cov = float(spearmanr(gap[common], pc[common])[0])
    SUM["optimism"] = dict(rho_gap_cov=rho_gap_cov, rho_seen=rho_seen, rho_lso=rho_lso, per_fold=g.round(4).to_dict("index"), mean_signed_seen=float((d.seen_val - d.actual).mean()), mean_signed_lso=float((d.lso_est - d.actual).mean()),
                           mae_seen=float((d.seen_val - d.actual).abs().mean()), mae_lso=float((d.lso_est - d.actual).abs().mean()), n_units=int(len(d)))
    fig, ax = plt.subplots(figsize=size((4.6, 2.9), (3.5, 2.7))); x = np.arange(len(g)); w = .27
    ax.bar(x - w, g.seen_val, w, label="seen-site validation", color="#ddaa33", hatch="///", edgecolor="k", lw=.3); ax.bar(x, g.lso_est, w, label="leave-site-out estimate", color="#4477aa", hatch="...", edgecolor="k", lw=.3); ax.bar(x + w, g.actual, w, label="actual held-out", color="#228833", edgecolor="k", lw=.3)
    ax.set_xticks(x); ax.set_xticklabels(g.index, fontsize=FS(7), rotation=35, ha="right"); ax.set_ylim(0.3, size(1.05, 1.28)); ax.set_ylabel("AUROC", fontsize=FS(8)); ax.tick_params(axis="y", labelsize=FS(7)); ax.set_yticks([0.4, 0.6, 0.8, 1.0]) if size(0, 1) else None; ax.legend(fontsize=FS(6.5), ncol=size(3, 1), loc="upper center", columnspacing=0.8, handlelength=1.2)
    title(ax, "Validation optimism (ERM)", fontsize=FS(8))
    finish(fig, "fig_optimism")


def synth():
    path = "runs/it3/results.jsonl"
    if not os.path.exists(path):
        return
    d = pd.DataFrame([json.loads(l) for l in open(path)])
    t = d.groupby(["spread", "method"]).auroc.mean().unstack("method")
    sd = d.groupby(["spread", "method"]).auroc.std().unstack("method")
    SUM["synth_label_shift_lam0.5"] = t.round(4).to_dict("index")
    names = {"erm": "ERM", "erm_p": "ERM + prior reweight", "dann": "DANN", "cdann": "cDANN", "cdann_p": "cDANN + prior reweight"}
    fig, axs = plt.subplots(1, 2, figsize=size((9, 3.2), (7.16, 2.2)))
    for m in ["erm", "erm_p", "dann", "cdann", "cdann_p"]:
        axs[0].errorbar(t.index, t[m], yerr=sd[m] / np.sqrt(6), marker="o", ms=4, capsize=2, label=names[m], lw=1)
    axs[0].set_xlabel("Prior spread across virtual sites"); axs[0].set_ylabel("Held-out AUROC"); axs[0].legend(fontsize=FS(6)); title(axs[0], "Label-shift sweep (adv. weight 0.5)", fontsize=FS(9))
    rows = []
    for tag, lam in [("runs/it3", 0.5), ("runs/it4_l01", 0.1), ("runs/it4_l002", 0.02)]:
        f = f"{tag}/results.jsonl"
        if os.path.exists(f):
            x = pd.DataFrame([json.loads(l) for l in open(f)]); x = x[x.spread == 0.4]; x["lam"] = lam; rows.append(x)
    R = pd.concat(rows); erm = R[R.method == "erm"].auroc.mean() if (R.method == "erm").any() else d[(d.method == "erm") & (d.spread == .4)].auroc.mean()
    for m in ["dann", "cdann_p"]:
        g = R[R.method == m].groupby("lam").auroc.mean(); axs[1].plot(g.index, g.values, marker="o", label=names[m])
    axs[1].axhline(d[(d.method == "erm") & (d.spread == .4)].auroc.mean(), c="k", ls=":", label="ERM"); axs[1].set_xscale("log"); axs[1].set_xticks([0.02, 0.1, 0.5]); axs[1].set_xticklabels(["0.02", "0.1", "0.5"]); axs[1].minorticks_off()
    axs[1].set_xlabel("Adversary weight"); axs[1].set_ylabel("Held-out AUROC (spread 0.4)"); axs[1].legend(fontsize=FS(7), loc="lower left"); title(axs[1], "Adversary strength sweep (spread 0.4)", fontsize=FS(9))
    finish(fig, "fig_synth", axs, ["(a)", "(b)"])


def synth_tables():
    """Semi-synthetic benchmark tables: label-shift sweep, adversary-weight sweep, LSO conformal by severity."""
    def L(path):
        return pd.DataFrame([json.loads(l) for l in open(path)]) if os.path.exists(path) else pd.DataFrame()
    d3 = L("runs/it3/results.jsonl")
    if len(d3):
        names = [("erm", "ERM"), ("erm_p", "ERM+prior"), ("dann", "DANN"), ("cdann", "cDANN"), ("cdann_p", "cDANN+prior")]
        rows = []
        out = {}
        for sp, g in d3.groupby("spread"):
            m = g.groupby("method").auroc.mean(); n = g.groupby("method").auroc.count().min()
            cells = []
            for k, lab in names:
                v = m.get(k, np.nan); cells.append(f"{v:.3f}")
                out[f"{sp}_{k}"] = float(v)
            rows.append(f"{sp:.2f} & " + " & ".join(cells) + f" & {n} \\\\")
        # paired differences vs ERM at the largest spread
        mx = d3.spread.max(); g = d3[d3.spread == mx]
        key = ["holdout", "seed"]; base = g[g.method == "erm"].set_index(key).auroc
        ds = []
        for k, lab in names[1:]:
            dd = (g[g.method == k].set_index(key).auroc - base).dropna().values
            lo, hi = boot(dd); ds.append(f"{lab}: {dd.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
            out[f"diff_{mx}_{k}"] = [float(dd.mean()), lo, hi]
        SUM["synth_label_shift"] = out
        open("paper/tables/synth_labelshift.tex", "w").write("\\begin{tabular}{lccccc c}\n\\toprule\nPrior spread & " + " & ".join(l for _, l in names) + " & Units \\\\\n\\midrule\n" + "\n".join(rows).replace("c c}", "cc}") + "\n\\bottomrule\n\\end{tabular}")
        SUM["synth_label_shift"]["paired_text"] = "; ".join(ds)
    # adversary weight sweep
    parts = []
    for tag, lam in (("runs/it4_l002", 0.02), ("runs/it4_l01", 0.1), ("runs/it3", 0.5)):
        x = L(f"{tag}/results.jsonl")
        if len(x):
            x["lam"] = lam; parts.append(x)
    if parts:
        R = pd.concat(parts); rows = []; out = {}
        for sp in sorted(R.spread.unique()):
            if sp not in (0.0, 0.4):
                continue
            erm = R[(R.spread == sp) & (R.method == "erm")].auroc.mean()
            for m, lab in (("dann", "DANN"), ("cdann_p", "cDANN+prior")):
                v = [R[(R.spread == sp) & (R.method == m) & (R.lam == lam)].auroc.mean() for lam in (0.02, 0.1, 0.5)]
                rows.append(f"{lab} & {sp:.1f} & " + " & ".join(f"{x:.3f}" for x in v) + f" & {erm:.3f} \\\\")
                out[f"{m}_{sp}"] = [float(x) for x in v]; out[f"erm_{sp}"] = float(erm)
        SUM["synth_adv_weight"] = out
        open("paper/tables/synth_advweight.tex", "w").write("\\begin{tabular}{llcccc}\n\\toprule\nMethod & Spread & $\\lambda{=}0.02$ & $\\lambda{=}0.1$ & $\\lambda{=}0.5$ & ERM \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    # LSO conformal by severity
    rows = []; out = {}
    for sev in ("0.5", "1.0", "1.5"):
        f = f"runs/it7_sev{sev}_conformal.csv"
        if not os.path.exists(f):
            continue
        D = pd.read_csv(f); d = D[np.isclose(D.alpha, 0.1)]
        fold = d.groupby(["mode", "holdout"])[["coverage", "size"]].mean().reset_index()
        S = fold.groupby("mode").agg(cov=("coverage", "mean"), mn=("coverage", "min"), size=("size", "mean"))
        cells = []
        for m in ("pooled", "site_balanced", "worst_site", "lso_split", "lso_cvplus", "lso_worst"):
            r = S.loc[m]; cells.append(f"{r['cov']:.3f} ({r.mn:.3f}) / {r['size']:.2f}"); out[f"{sev}_{m}"] = [float(r["cov"]), float(r.mn), float(r["size"])]
        rows.append(f"{sev} & " + " & ".join(cells) + " \\\\")
    if rows:
        SUM["synth_conformal"] = out
        open("paper/tables/synth_conformal.tex", "w").write("\\begin{tabular}{lcccccc}\n\\toprule\nSeverity & Pooled & Site-bal. & Worst-site & LSO-split & LSO-CV+ & LSO-worst \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")


def reliance_table():
    """Shortcut-reliance: within-class Spearman between a model's held-out P(COVID) and a profile-only (5 acquisition numbers) score."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from shortcut_reliance import all_profiles
    from sidl.arraydata import load_covid_cxr
    X, df = load_covid_cxr("data/covid_cxr")
    P = all_profiles(X)
    rows = []
    for label, dev, conf in METHODS:
        for (dd, key), folds, split in ((dev, DEV, "dev"), (conf, CONF, "conf")):
            for h in folds:
                for f in sorted(glob.glob(f"{dd}/{h}/{key}/seed*/preds.npz")):
                    z = np.load(f); src = df.site.values != h
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(P[src], df.label.values[src])
                    ps = clf.decision_function(P[z["test_rows"]]); yt, pt = z["yt"], z["pt"][:, 1]
                    cs = [spearmanr(ps[yt == c], pt[yt == c])[0] for c in (0, 1) if (yt == c).sum() > 5]
                    rows.append(dict(label=label, split=split, holdout=h, reliance=float(np.nanmean(cs))))
    R = pd.DataFrame(rows)
    fold = R.groupby(["label", "split", "holdout"]).reliance.mean().reset_index()
    out, tex = {}, []
    for label, _, _ in METHODS:
        g = fold[fold.label == label]
        out[label] = dict(dev=float(g[g.split == "dev"].reliance.mean()), conf=float(g[g.split == "conf"].reliance.mean()), all=float(g.reliance.mean()))
        tex.append(f"{label} & {out[label]['dev']:.3f} & {out[label]['conf']:.3f} & {out[label]['all']:.3f} \\\\")
    SUM["reliance"] = out
    open("paper/tables/reliance.tex", "w").write("\\begin{tabular}{lccc}\n\\toprule\nMethod & Dev.\\ folds & Conf.\\ folds & All 8 \\\\\n\\midrule\n" + "\n".join(tex) + "\n\\bottomrule\n\\end{tabular}")


def dataset_table():
    f = "runs/diagnostics.json"
    if not os.path.exists(f):
        return
    pv = json.load(open(f))["prevalence"]
    role = {h: "development" for h in DEV}; role.update({h: "confirmation" for h in CONF})
    rows = []
    for s_, v in sorted(pv.items(), key=lambda kv: (kv[0] not in DEV, kv[0] not in CONF, kv[0])):
        npos = int(round(v["prev"] * v["n"]))
        rows.append(f"{s_} & {v['n']} & {npos} & {v['n'] - npos} & {v['prev']:.2f} & {role.get(s_, 'training only')} \\\\")
    tot = sum(v["n"] for v in pv.values()); totp = sum(int(round(v["prev"] * v["n"])) for v in pv.values())
    rows.append("\\midrule\nTotal & %d & %d & %d & %.2f & \\\\" % (tot, totp, tot - totp, totp / tot))
    open("paper/tables/datasets.tex", "w").write("\\begin{tabular}{lrrrrl}\n\\toprule\nSource & Images & COVID-19 & Other & Prevalence & Role\\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    SUM["dataset"] = dict(total=tot, positives=totp)


def diagnostics():
    f = "runs/diagnostics.json"
    if not os.path.exists(f):
        return
    d = json.load(open(f)); SUM["diagnostics"] = {k: d[k] for k in d if k != "param_names"}
    fig, axs = plt.subplots(1, 2, figsize=size((9, 3.2), (7.16, 2.2)))
    pv = d["prevalence"]; names = sorted(pv, key=lambda s: -pv[s]["n"])
    axs[0].bar(range(len(names)), [pv[s]["prev"] for s in names], color="#4477aa"); axs[0].set_xticks(range(len(names))); axs[0].set_xticklabels(names, rotation=50, ha="right", fontsize=FS(7))
    axs[0].set_ylabel("COVID-19 prevalence"); title(axs[0], f"Label prior per source (SD={d['prior_spread_std']:.2f})", fontsize=FS(9))
    L = d["profile_only_label_auroc_loso"]["logreg"]; ks = list(L)
    axs[1].bar(range(len(ks)), [L[k] for k in ks], color="#cc6677"); axs[1].axhline(0.5, c="k", ls=":"); axs[1].set_xticks(range(len(ks))); axs[1].set_xticklabels(ks, rotation=50, ha="right", fontsize=FS(7))
    axs[1].set_ylabel("AUROC"); title(axs[1], "Label from 5 acquisition numbers alone (LOSO)", fontsize=FS(9))
    finish(fig, "fig_diagnostics", axs, ["(a)", "(b)"])


def _name(*parts):
    """camelCase macro name from parts, letters only."""
    out = ""
    for p in parts:
        w = re.sub(r"[^A-Za-z]+", " ", str(p)).split()
        out += "".join(x.capitalize() if out or i else x.capitalize() for i, x in enumerate(w))
    return out


def emit_numbers():
    """Every number quoted in the paper text is a macro defined here (so text and tables cannot drift apart)."""
    L = ["% auto-generated by scripts/final_report.py -- do not edit"]
    def add(name, val, fmt="{:.3f}"):
        txt = fmt.format(val) if isinstance(val, (int, float, np.floating, np.integer)) else val
        txt = txt.replace("-", "\\ensuremath{-}") if txt.startswith("-") else txt
        L.append("\\newcommand{\\%s}{%s}" % (name, txt))
    for m, r in SUM.get("task", {}).items():
        k = _name("t", m)
        for key in ("auroc_all", "auroc_dev", "auroc_conf", "worst_fold", "bal_acc", "ece", "diff_vs_erm"):
            if r.get(key) is not None:
                add(k + _name(key), r[key])
        if r.get("diff_ci") and not np.isnan(r["diff_ci"][0]):
            add(k + "Lo", r["diff_ci"][0]); add(k + "Hi", r["diff_ci"][1])
        if r.get("folds_better") is not None:
            add(k + "FoldsBetter", r["folds_better"], "{}")
    for m, r in SUM.get("conformal_alpha0.1", {}).items():
        k = _name("c", m)
        for key in ("cov_mean", "cov_min", "size", "n_under", "n", "cov_calib_err", "frac_under", "size_at_cov90"):
            if key in r and r[key] == r[key]:
                add(k + _name(key), int(round(r[key])) if key in ("n_under", "n") else r[key], "{}" if key in ("n_under", "n") else "{:.3f}")
    o = SUM.get("optimism")
    if o:
        for key in ("mean_signed_seen", "mean_signed_lso", "mae_seen", "mae_lso", "n_units", "rho_seen", "rho_lso", "rho_gap_cov"):
            add(_name("o", key), o[key], "{}" if key == "n_units" else "{:.3f}")
    if "dataset" in SUM:
        add("datasetTotal", SUM["dataset"]["total"], "{}")
    dg = SUM.get("diagnostics")
    if dg:
        add("profLeakLogreg", dg["profile_only_label_auroc_mean"]["logreg"]); add("profLeakGbm", dg["profile_only_label_auroc_mean"]["gbm"])
        add("priorSpreadSd", dg["prior_spread_std"]); add("siteAccProfile", dg["site_acc_profile5"]); add("siteAccChance", dg["site_acc_chance_majority"])
    sl = SUM.get("synth_label_shift", {})
    for k in ("erm_p", "dann", "cdann", "cdann_p"):
        v = sl.get(f"diff_0.4_{k}")
        if v:
            add("synthDiff" + _name(k), v[0]); add("synthDiff" + _name(k) + "Lo", v[1]); add("synthDiff" + _name(k) + "Hi", v[2])
    if "rank_corr_dev_conf" in SUM:
        add("rankCorrDevConf", SUM["rank_corr_dev_conf"])
    for m, r in SUM.get("reliance", {}).items():
        add("rel" + _name(m), r["all"])
    if os.path.exists("runs/ensemble_check.json"):
        e = json.load(open("runs/ensemble_check.json"))
        add("ensAurocFull", e["auroc_full"]); add("ensAurocEns", e["auroc_ens"]); add("ensDAuroc", e["d_auroc"])
        add("ensDAurocLo", e["d_auroc_ci"][0]); add("ensDAurocHi", e["d_auroc_ci"][1]); add("ensDEce", e["d_ece"])
        add("ensFoldsBetter", e["folds_ens_better_auroc"], "{}"); add("ensMembers", e["n_members"], "{}")
    open("paper/numbers.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    D, fold = task_table()
    piv = main_table(fold); fig_methods(piv)
    conformal_summary(); optimism(); synth(); diagnostics(); dataset_table(); synth_tables(); reliance_table()
    emit_numbers()
    json.dump(SUM, open("runs/results_summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: (v if k != "task" else {m: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items()} for m, r in v.items()}) for k, v in SUM.items() if k in ("task", "optimism")}, indent=1, default=float)[:3500])
