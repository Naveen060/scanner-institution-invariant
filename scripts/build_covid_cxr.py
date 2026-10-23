"""Build the open multi-source chest X-ray benchmark (COVID-19 vs. other) from three public GitHub collections.

Sources (all public):
  ieee8023/covid-chestxray-dataset   (Cohen et al.)  -> sites by reported institution/country
  agchung/Actualmed-COVID-chestxray-dataset           -> one clinical source ("Actualmed")
  agchung/Figure1-COVID-chestxray-dataset             -> one source ("Figure1")

Label: 1 = COVID-19, 0 = any other finding (No Finding, other pneumonias, TB, ...). Unlabeled/'todo' rows and lateral views are dropped.
Images are squashed to SxS grayscale uint8 after 0.5-99.5 percentile intensity normalization and cached to one .npz.
CAVEAT: this benchmark is small, retrospective and known to be shortcut-prone; it is used here as a stress test
for site shortcuts and label shift, NOT as evidence of clinical COVID-19 detection performance.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

LOC = [("hannover", "Hannover-DE"), ("italy", "Italy"), ("milan", "Italy"), ("brescia", "Italy"), ("rozzano", "Italy"),
       ("australia", "Australia"), ("melbourne", "Australia"), ("united kingdom", "UK"), ("edinburgh", "UK"),
       ("nottingham", "UK"), ("spain", "Spain"), ("valencia", "Spain"), ("china", "China"), ("wuhan", "China"),
       ("wenzhou", "China"), ("hong kong", "China"), ("taiwan", "Taiwan"), ("korea", "Korea"), ("canada", "Canada"),
       ("egypt", "Egypt"), ("iran", "Iran"), ("portugal", "Portugal"), ("hungary", "Hungary"), ("malta", "Malta")]
MAJOR = {"Hannover-DE", "Italy", "Australia", "UK", "Spain", "China"}


def site_of(loc):
    if not isinstance(loc, str):
        return "Unknown"
    l = loc.lower()
    for k, n in LOC:
        if k in l:
            return n
    return "Other"


def load(path, size):
    im = Image.open(path).convert("L")
    a = np.asarray(im).astype(np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/claude")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="data/covid_cxr")
    a = ap.parse_args()
    rows = []
    m = pd.read_csv(f"{a.root}/ieee8023/covid-chestxray-dataset/metadata.csv")
    m = m[(m.modality == "X-ray") & (m.view != "L") & (m.finding != "todo") & m.finding.notna()]
    for _, r in m.iterrows():
        p = f"{a.root}/ieee8023/covid-chestxray-dataset/{r.folder}/{r.filename}"
        if os.path.isfile(p):
            s = site_of(r.location)
            rows.append(dict(path=p, label=int("COVID-19" in r.finding), site=s, patient_id=f"ieee_{r.patientid}", src="ieee8023"))
    f = pd.read_csv(f"{a.root}/agchung/figure1-covid-chestxray-dataset/metadata.csv", encoding="latin1")
    for _, r in f.iterrows():
        if not isinstance(r.finding, str):
            continue
        d = f"{a.root}/agchung/figure1-covid-chestxray-dataset/images"
        cand = [x for x in os.listdir(d) if x.startswith(r.patientid)]
        if cand:
            rows.append(dict(path=f"{d}/{cand[0]}", label=int("COVID-19" in r.finding), site="Figure1", patient_id=f"fig1_{r.patientid}", src="figure1"))
    ac = pd.read_csv(f"{a.root}/agchung/actualmed-covid-chestxray-dataset/metadata.csv", encoding="latin1")
    for _, r in ac.iterrows():
        if not isinstance(r.finding, str):
            continue
        p = f"{a.root}/agchung/actualmed-covid-chestxray-dataset/images/{r.imagename}"
        if os.path.isfile(p):
            rows.append(dict(path=p, label=int("COVID-19" in r.finding), site="Actualmed", patient_id=f"act_{r.patientid}", src="actualmed"))
    df = pd.DataFrame(rows).reset_index(drop=True)
    X = np.stack([load(p, a.size) for p in df.path])
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(f"{a.out}/images_{a.size}.npz", X=X)
    df.drop(columns=["path"]).to_csv(f"{a.out}/meta.csv", index=False)
    t = df.groupby("site").label.agg(["count", "sum"]).assign(prevalence=lambda d: (d["sum"] / d["count"]).round(3))
    print(X.shape, t.sort_values("count", ascending=False).to_string())


if __name__ == "__main__":
    main()
