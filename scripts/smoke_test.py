"""Smoke test: synthetic 4-site data, leave-one-site-out, all methods, tiny CNN on CPU.

Verifies the full pipeline runs end to end. Not a scientific result.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from torch.utils.data import ConcatDataset, Subset  # noqa: E402

from sidl.data import SyntheticSiteDataset  # noqa: E402
from sidl.trainer import METHODS, TrainConfig, train_and_eval  # noqa: E402


def main():
    held = 3
    train_sites = [0, 1, 2]
    full = SyntheticSiteDataset(n_per_site=120, num_sites=4, size=32, seed=0, site_prior_shift=0.35, sites=train_sites)
    n = len(full)
    idx = list(range(n))
    val = Subset(full, idx[: n // 6])
    tr = Subset(full, idx[n // 6:])
    test = SyntheticSiteDataset(n_per_site=150, num_sites=4, size=32, seed=1, site_prior_shift=0.35, sites=[held])
    out = {}
    for m in METHODS:
        cfg = TrainConfig(method=m, backbone="tiny", pretrained=False, in_chans=1, epochs=4, batch_size=32,
                          lr=3e-3, num_workers=0, device="cpu", amp=False, out_dir=f"/tmp/sidl_smoke/{m}")
        r = train_and_eval(cfg, tr, val, test, num_train_sites=len(train_sites))
        out[m] = {k: round(v, 3) for k, v in r["test_overall"].items()}
        out[m].update({f"cov_{k}": round(v["coverage"], 3) for k, v in r["conformal"].items()})
        out[m].update({f"size_{k}": round(v["set_size"], 2) for k, v in r["conformal"].items()})
        print(m, out[m])
    json.dump(out, open("/tmp/sidl_smoke/summary.json", "w"), indent=2)
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
