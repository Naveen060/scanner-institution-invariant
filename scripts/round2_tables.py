"""Round-2 tables, figure and number macros for the manuscript (reads runs/round2.json, runs/conformal_{cnn,feat,ens}_all.csv, runs/feat_diagnostics.json).

  python scripts/round2_tables.py
Writes paper/tables/r2_models.tex, r2_heads.tex, r2_conformal.tex, paper/figures/fig_round2.{pdf,png}, paper/numbers_r2.tex.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts (IEEE PDF eXpress rejects Type 3)
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
import pandas as pd

DEV = ["Actualmed", "Italy", "China", "Spain"]
CONF = ["UK", "Australia", "Other", "Unknown"]
ALL = DEV + CONF
R = json.load(open("runs/round2.json"))
DG = json.load(open("runs/feat_diagnostics.json"))
MAC = {}


def sg(x, d=3):
    s = f"{abs(x):.{d}f}"
    return ("\\ensuremath{-}" + s) if x < 0 else s


def sp(x, d=3):  # signed with explicit plus
    return ("+" if x >= 0 else "") + sg(x, d) if x >= 0 else sg(x, d)


def ci(m, lo, hi):
    return f"{sp(m)} [{sp(lo)}, {sp(hi)}]"


# ---- Table: model families
M = ["CNN ERM", "Pretrained ERM", "CNN 5-model ens", "Pretrained 5-model ens", "CNN + pretrained (5+5 ens)"]
lab = {"CNN ERM": "CNN, single (ERM)", "Pretrained ERM": "Pretrained head, single (ERM)", "CNN 5-model ens": "CNN, 5-model ensemble",
       "Pretrained 5-model ens": "Pretrained head, 5-model ensemble", "CNN + pretrained (5+5 ens)": "CNN + pretrained head (5+5 ensemble)"}
rows = []
for m in M:
    v = [R["fold_auroc"][m][h] for h in ALL]
    key = {"CNN ERM": None}.get(m, f"{m}|CNN ERM|all")
    d = "--" if m == "CNN ERM" else ci(*R["boot"][key]) if key in R["boot"] else "--"
    rows.append(f"{lab[m]} & {np.mean(v[:4]):.3f} & {np.mean(v[4:]):.3f} & {np.mean(v):.3f} & {min(v):.3f} & {d} \\\\")
# extra comparison rows not in boot keys
open("paper/tables/r2_models.tex", "w").write("\\begin{tabular}{lccccc}\n\\toprule\nModel & Dev & Conf. & All & Worst site & $\\Delta$ vs single CNN [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- Table: feature heads
HL = {"erm": "ERM", "groupdro": "GroupDRO", "erm_p": "Prior reweighting", "dann": "DANN", "cdann": "Class-cond.\\ adversary", "cdann_p": "Class-cond.\\ adv.\\ + prior",
      "coral": "CORAL", "cc_full": "Class-cond.\\ recipe", "sitecenter": "Site centring (uses target images)"}
rows = []
for k in ["erm", "groupdro", "erm_p", "dann", "cdann", "cdann_p", "coral", "cc_full", "sitecenter"]:
    f = R["feat_methods"][k]
    d = "--" if k == "erm" else ci(*f["diff_ci"])
    rows.append(f"{HL[k]} & {f['dev']:.3f} & {f['conf']:.3f} & {f['all']:.3f} & {d} \\\\")
open("paper/tables/r2_heads.tex", "w").write("\\begin{tabular}{lcccc}\n\\toprule\nHead on frozen embedding & Dev & Conf. & All & $\\Delta$ vs ERM [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- conformal across families
fam = {"CNN": "cnn", "Pretrained": "feat", "Ensemble": "ens"}
modes = [("pooled", "Pooled (seen-site val.)"), ("worst_site", "Worst-site (seen-site val.)"), ("lso_split", "LSO-split"), ("lso_cvplus", "LSO-CV+"), ("lso_worst", "LSO-worst-site")]
summ = {}
for name, f in fam.items():
    D = pd.read_csv(f"runs/conformal_{f}_all.csv")
    d = D[np.isclose(D.alpha, 0.1)]
    fold = d.groupby(["mode", "holdout"])[["coverage", "size"]].mean().reset_index()
    F = D.groupby(["mode", "alpha", "holdout"])[["coverage", "size"]].mean().reset_index()
    C = F.groupby(["mode", "alpha"])[["coverage", "size"]].mean().reset_index()
    for m, _ in modes:
        fm = fold[fold["mode"] == m]
        g = C[C["mode"] == m].sort_values("coverage")
        summ[(name, m)] = dict(cov=fm.coverage.mean(), mn=fm.coverage.min(), under=int((fm.coverage < 0.85).sum()), size=fm["size"].mean(),
                               s90=float(np.interp(0.90, np.maximum.accumulate(g.coverage.values), g["size"].values)))
rows = []
for m, nm in modes:
    rows.append(nm + " & " + " & ".join(f"{summ[(n, m)]['cov']:.3f} ({summ[(n, m)]['under']}) & {summ[(n, m)]['size']:.2f} & {summ[(n, m)]['s90']:.2f}" for n in fam) + " \\\\")
open("paper/tables/r2_conformal.tex", "w").write(
    "\\begin{tabular}{lccccccccc}\n\\toprule\n& \\multicolumn{3}{c}{CNN} & \\multicolumn{3}{c}{Pretrained head} & \\multicolumn{3}{c}{CNN + pretrained} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nCalibration & Cov.\\ (under) & Size & @90\\% & Cov.\\ (under) & Size & @90\\% & Cov.\\ (under) & Size & @90\\% \\\\\n\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")

# ---- macros
def mac(name, val):
    MAC[name] = val


for tag, m in (("Cnn", "CNN ERM"), ("Pre", "Pretrained ERM"), ("CnnEns", "CNN 5-model ens"), ("PreEns", "Pretrained 5-model ens"), ("Ens", "CNN + pretrained (5+5 ens)")):
    v = [R["fold_auroc"][m][h] for h in ALL]
    mac(f"Rt{tag}Dev", f"{np.mean(v[:4]):.3f}"); mac(f"Rt{tag}Conf", f"{np.mean(v[4:]):.3f}"); mac(f"Rt{tag}All", f"{np.mean(v):.3f}"); mac(f"Rt{tag}Worst", f"{min(v):.3f}")
for nm, key in (("EnsVsCnnEns", "CNN + pretrained (5+5 ens)|CNN 5-model ens"), ("EnsVsCnn", "CNN + pretrained (5+5 ens)|CNN ERM"), ("PreVsCnn", "Pretrained ERM|CNN ERM"),
                ("EnsVsPreEns", "CNN + pretrained (5+5 ens)|Pretrained 5-model ens"), ("CnnEnsVsCnn", "CNN 5-model ens|CNN ERM")):
    for sp_ in ("dev", "conf", "all"):
        m_, lo, hi = R["boot"][f"{key}|{sp_}"]
        mac(f"Rt{nm}{sp_.capitalize()}", sp(m_)); mac(f"Rt{nm}{sp_.capitalize()}Lo", sp(lo)); mac(f"Rt{nm}{sp_.capitalize()}Hi", sp(hi))
f = R["feat_methods"]
for k, nm in (("groupdro", "Gdro"), ("sitecenter", "Center"), ("cdann_p", "CdannP"), ("cc_full", "CcFull")):
    mac(f"Rt{nm}Diff", sp(f[k]["diff_ci"][0])); mac(f"Rt{nm}Lo", sp(f[k]["diff_ci"][1])); mac(f"Rt{nm}Hi", sp(f[k]["diff_ci"][2]))
mac("RtGdroDev", sp(f["groupdro"]["dev"] - f["erm"]["dev"])); mac("RtGdroConf", sp(f["groupdro"]["conf"] - f["erm"]["conf"]))
mac("RtFeatErmDev", f"{f['erm']['dev']:.3f}")
for name, fm in (("Cnn", "CNN"), ("Pre", "Pretrained"), ("Ens", "Ensemble")):
    for mm, tag in (("pooled", "Pooled"), ("lso_cvplus", "Cvplus"), ("worst_site", "Worst")):
        s = summ[(fm, mm)]
        mac(f"Rc{name}{tag}Cov", f"{s['cov']:.3f}"); mac(f"Rc{name}{tag}Under", str(s["under"])); mac(f"Rc{name}{tag}Size", f"{s['size']:.2f}"); mac(f"Rc{name}{tag}Min", f"{s['mn']:.3f}")
        mac(f"Rc{name}{tag}Sninety", f"{s['s90']:.2f}")
# paired per-site set size at matched 90% coverage, ensemble minus CNN (seeds 0-1)
def size90_by_site(f, mode):
    D = pd.read_csv(f"runs/conformal_{f}_all.csv"); out = {}
    for h, g in D[D["mode"] == mode].groupby("holdout"):
        g = g.groupby("alpha")[["coverage", "size"]].mean().reset_index().sort_values("coverage")
        out[h] = float(np.interp(0.90, np.maximum.accumulate(g.coverage.values), g["size"].values))
    return pd.Series(out)


for mm, tag in (("lso_cvplus", "Cvplus"), ("pooled", "Pooled")):
    d = size90_by_site("ens", mm) - size90_by_site("cnn", mm)
    rng_ = np.random.RandomState(0)
    bs = [d.values[rng_.randint(0, len(d), len(d))].mean() for _ in range(5000)]
    mac(f"RcEnsCnn{tag}Diff", sp(d.mean())); mac(f"RcEnsCnn{tag}Lo", sp(float(np.percentile(bs, 2.5)))); mac(f"RcEnsCnn{tag}Hi", sp(float(np.percentile(bs, 97.5))))
    mac(f"RcEnsCnn{tag}N", str(int((d < 0).sum())))
mac("RdSiteAcc", f"{DG['site_acc']:.3f}"); mac("RdSiteMaj", f"{DG['site_majority']:.3f}"); mac("RdOptim", f"{DG['optimism']:.3f}"); mac("RdOptimMae", f"{DG['optimism_mae']:.3f}")
mac("RdValAuc", f"{DG['val_auc']:.3f}"); mac("RdTestAuc", f"{DG['test_auc']:.3f}")
with open("paper/numbers_r2.tex", "w") as fh:
    fh.write("% auto-generated by scripts/round2_tables.py -- do not edit\n")
    for k, v in MAC.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- figure
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
cols = {"CNN ERM": ("#4477aa", "CNN, single", "o"), "Pretrained ERM": ("#ee7733", "Pretrained head, single", "s"), "CNN + pretrained (5+5 ens)": ("#228833", "CNN + pretrained (5+5)", "D")}
for j, (m, (c, l, mk)) in enumerate(cols.items()):
    y = [R["fold_auroc"][m][h] for h in ALL]
    ax.scatter(np.arange(8) + (j - 1) * 0.2, y, c=c, s=20, marker=mk, label=l, zorder=3)
    ax.axhline(np.mean(y), c=c, lw=0.8, ls=":")
ax.axvline(3.5, c="gray", lw=0.6)
ax.set_xticks(range(8)); ax.set_xticklabels([h if h != "Actualmed" else "Actual." for h in ALL], rotation=40, ha="right", fontsize=7)
ax.set_ylabel("Held-out-site AUROC", fontsize=8); ax.tick_params(axis="y", labelsize=7)
ax.text(1.5, 0.995, "development", ha="center", fontsize=7, color="gray"); ax.text(5.5, 0.995, "confirmation", ha="center", fontsize=7, color="gray")
ax.legend(fontsize=5.5, loc="lower center", ncol=3, columnspacing=0.8, handletextpad=0.2, borderpad=0.3); ax.set_ylim(0.47, 1.03); ax.set_title("(a) Per-site AUROC (dotted: mean)", fontsize=8)
ax = axs[1]
xs, labs = [], []
k = 0
for name, fm, c in (("CNN", "CNN", "#4477aa"), ("Pretrained", "Pretrained", "#ee7733"), ("CNN+pre.", "Ensemble", "#228833")):
    f_ = {"CNN": "cnn", "Pretrained": "feat", "Ensemble": "ens"}[fm]
    D = pd.read_csv(f"runs/conformal_{f_}_all.csv"); d = D[np.isclose(D.alpha, 0.1)]
    fold = d.groupby(["mode", "holdout"]).coverage.mean().reset_index()
    for mm, hatch in (("pooled", ""), ("lso_cvplus", "//")):
        v = fold[fold["mode"] == mm].coverage.values
        ax.bar(k, v.mean(), width=0.8, color=c, alpha=0.35 if hatch == "" else 0.85, hatch=hatch, edgecolor="k", lw=0.4)
        ax.scatter(np.full(len(v), k) + np.random.RandomState(int(k * 10)).uniform(-.18, .18, len(v)), v, s=7, c="k", zorder=3)
        labs.append(("pooled" if mm == "pooled" else "LSO-CV+")); xs.append(k); k += 1
    k += 0.6
ax.axhline(0.9, c="gray", ls="--", lw=0.8); ax.set_ylim(0.65, 1.02)
ax.set_xticks(xs); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=7); ax.tick_params(axis="y", labelsize=7)
ax.set_ylabel("Coverage (target 0.90)", fontsize=8)
for xx, nm in ((0.5, "CNN"), (3.1, "Pretrained"), (5.7, "CNN + pretrained")):
    ax.text(xx, 1.005, nm, ha="center", fontsize=7)
ax.set_title("(b) Coverage per site, mean (bar) and sites (dots)", fontsize=8, pad=12)
fig.tight_layout(); fig.savefig("paper/figures/fig_round2.pdf"); fig.savefig("paper/figures/fig_round2.png", dpi=200); plt.close(fig)
print("ok", len(MAC), "macros")
