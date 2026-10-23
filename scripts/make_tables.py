"""Aggregate summary.csv -> LaTeX tables:
  main.tex       task metrics (AUROC, worst-site AUROC, bal. acc., ECE) per method
  conformal.tex  coverage / set size per conformal mode for each method (target 1-alpha)
"""
import argparse
import os

import pandas as pd

NAMES = {"erm": "ERM", "harmonize": "ERM + generic aug.", "dann": "DANN", "groupdro": "GroupDRO",
         "sidl": "SIDL (generic aug.)", "sidl_phys": "+ site-profile aug.", "sidl_cc": "+ label-shift-aware adv.",
         "sidl_full": "\\textbf{SIDL-Full (ours)}"}
ORDER = list(NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="paper/tables/main.tex")
    ap.add_argument("--alpha", type=float, default=0.1)
    a = ap.parse_args()
    df = pd.read_csv(a.summary)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    g = df.groupby(["method", "seed"])
    per_seed = g.agg(auroc=("auroc", "mean"), worst_auroc=("auroc", "min"), bal_acc=("bal_acc", "mean"), ece=("ece", "mean")).reset_index()
    agg = per_seed.groupby("method").agg(["mean", "std"])
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r"Method & AUROC $\uparrow$ & Worst-site AUROC $\uparrow$ & Bal.\ Acc.\ $\uparrow$ & ECE $\downarrow$ \\", r"\midrule"]
    for m in ORDER:
        if m in agg.index:
            f = lambda c: f"{agg.loc[m, (c, 'mean')]:.3f} $\\pm$ {agg.loc[m, (c, 'std')]:.3f}"
            L.append(f"{NAMES[m]} & {f('auroc')} & {f('worst_auroc')} & {f('bal_acc')} & {f('ece')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(a.out, "w").write("\n".join(L))
    print("\n".join(L))

    if "cov_pooled" in df.columns:
        modes = ["pooled", "site_balanced", "worst_site"]
        # coverage averaged over held-out sites and seeds; also worst held-out-site coverage
        cols = {}
        for md in modes:
            cols[f"cov_{md}"] = (f"cov_{md}", "mean"); cols[f"cmin_{md}"] = (f"cov_{md}", "min"); cols[f"size_{md}"] = (f"size_{md}", "mean")
        c = df.groupby("method").agg(**cols)
        L = [r"\begin{tabular}{l" + "ccc" * len(modes) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{md.replace('_', ' ')}}}" for md in modes) + r" \\",
             "Method & " + " & ".join("Cov. & Min cov. & Size" for _ in modes) + r" \\", r"\midrule"]
        for m in ORDER:
            if m in c.index:
                L.append(NAMES[m] + " & " + " & ".join(
                    f"{c.loc[m, f'cov_{md}']:.3f} & {c.loc[m, f'cmin_{md}']:.3f} & {c.loc[m, f'size_{md}']:.2f}" for md in modes) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(os.path.dirname(a.out) or ".", "conformal.tex"), "w").write("\n".join(L))
        print("\n".join(L))


if __name__ == "__main__":
    main()
