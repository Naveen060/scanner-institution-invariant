"""Near-duplicate check on the real benchmark (iteration 22).

Correlation between 32x32 area-downsampled, mean-centred images; pairs above a threshold are reported, split into
pairs inside one source and pairs that cross sources. Writes runs/duplicate_check.json.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sidl.arraydata import load_covid_cxr  # noqa: E402

X, df = load_covid_cxr()
t = F.interpolate(torch.from_numpy(X).float().div(255)[:, None], size=(32, 32), mode="area")[:, 0].reshape(len(X), -1)
t = t - t.mean(1, keepdim=True)
t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
C = t @ t.T
C.fill_diagonal_(-1)
site, pid = df.site.values, df.patient_id.values
out = {}
for thr in (0.99, 0.98, 0.95):
    pairs = [(i.item(), j.item()) for i, j in (C > thr).nonzero() if i < j]
    cross = [(i, j) for i, j in pairs if site[i] != site[j]]
    out[str(thr)] = {"pairs": len(pairs), "cross_source": len(cross), "same_patient_id": sum(pid[i] == pid[j] for i, j in pairs),
                     "cross_source_detail": [[site[i], site[j], int(df.label.values[i]), int(df.label.values[j]), round(C[i, j].item(), 3)] for i, j in cross]}
json.dump(out, open("runs/duplicate_check.json", "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "cross_source_detail"} for k, v in out.items()}))
print(out["0.95"]["cross_source_detail"])
