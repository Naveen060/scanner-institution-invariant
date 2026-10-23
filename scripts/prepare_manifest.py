"""Build manifest CSVs (path,label,site[,patient_id]) from dataset metadata.

Public datasets must be downloaded by you (most require credentialed access / DUAs).
Each function below documents what it expects; adapt the column mapping as needed.

Chest X-ray (binary: e.g. Cardiomegaly / Pneumonia / Pleural Effusion vs. normal):
    python scripts/prepare_manifest.py cxr --label-name "Pleural Effusion" \
        --chexpert /data/CheXpert --mimic /data/MIMIC-CXR-JPG --nih /data/NIH-CXR14 --padchest /data/PadChest --out data/cxr_manifest.csv
Brain MRI / Histopathology / Fundus: provide a CSV with columns path,label,site via `generic`.
Private hospital data: `generic` with your own CSV; site = institution (and optionally scanner, e.g. "HospA_GE").
"""
import argparse
import os

import pandas as pd


def _chexpert(root, label):
    d = pd.read_csv(os.path.join(root, "train.csv"))
    d = d[d[label].isin([0.0, 1.0])]  # drop uncertain (-1) & missing
    return pd.DataFrame({"path": root + "/../" + d["Path"], "label": d[label].astype(int), "site": "CheXpert",
                         "patient_id": d["Path"].str.extract(r"(patient\d+)")[0]})


def _mimic(root, label):
    lab = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-chexpert.csv.gz"))
    meta = pd.read_csv(os.path.join(root, "mimic-cxr-2.0.0-metadata.csv.gz"))
    d = lab.merge(meta[["subject_id", "study_id", "dicom_id"]], on=["subject_id", "study_id"])
    d = d[d[label].isin([0.0, 1.0])]
    p = d.apply(lambda r: f"{root}/files/p{str(r.subject_id)[:2]}/p{r.subject_id}/s{r.study_id}/{r.dicom_id}.jpg", axis=1)
    return pd.DataFrame({"path": p, "label": d[label].astype(int), "site": "MIMIC", "patient_id": d["subject_id"].astype(str)})


def _nih(root, label):
    d = pd.read_csv(os.path.join(root, "Data_Entry_2017.csv"))
    nih_label = label.replace(" ", "_")
    y = d["Finding Labels"].str.contains(label.split()[-1], case=False).astype(int)
    imgs = {os.path.basename(p): p for p in __import__("glob").glob(f"{root}/images*/images/*.png")}
    d["path"] = d["Image Index"].map(imgs)
    return pd.DataFrame({"path": d["path"], "label": y, "site": "NIH", "patient_id": d["Patient ID"].astype(str)}).dropna()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    c = sub.add_parser("cxr")
    c.add_argument("--label-name", default="Pleural Effusion")
    c.add_argument("--chexpert"); c.add_argument("--mimic"); c.add_argument("--nih"); c.add_argument("--padchest")
    c.add_argument("--out", required=True)
    g = sub.add_parser("generic")
    g.add_argument("--csv", required=True, help="CSV with path,label,site[,patient_id]")
    g.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.kind == "cxr":
        parts = []
        if a.chexpert: parts.append(_chexpert(a.chexpert, a.label_name))
        if a.mimic: parts.append(_mimic(a.mimic, a.label_name))
        if a.nih: parts.append(_nih(a.nih, a.label_name))
        df = pd.concat(parts, ignore_index=True)
    else:
        df = pd.read_csv(a.csv)
        assert {"path", "label", "site"} <= set(df.columns)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby("site")["label"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
