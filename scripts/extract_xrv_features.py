"""Extract frozen features from a pretrained chest X-ray encoder (torchxrayvision DenseNet121, 'all' weights).

The 'all' weights were trained on NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI and RSNA pneumonia images, none of which
carries a COVID-19 label. We use only the penultimate 1024-d embedding (and the 18 pathology logits as a zero-shot baseline).
We did NOT check whether individual images in our benchmark also appear in those training sets.

  python scripts/extract_xrv_features.py --cache data/covid_cxr224/images_224.npz --out data/covid_cxr224/xrv_features.npz
"""
import argparse
import time

import numpy as np
import torch
import torchxrayvision as xrv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/covid_cxr224/images_224.npz")
    ap.add_argument("--out", default="data/covid_cxr224/xrv_features.npz")
    ap.add_argument("--weights", default="densenet121-res224-all")
    ap.add_argument("--bs", type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(2)
    X = np.load(a.cache)["X"]  # (N, 224, 224) uint8 in [0, 255]
    model = xrv.models.DenseNet(weights=a.weights).eval()
    feats, logits = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), a.bs):
            x = torch.from_numpy(X[i:i + a.bs].astype(np.float32) / 255.0 * 2048.0 - 1024.0).unsqueeze(1)  # xrv range
            feats.append(model.features2(x).numpy())
            logits.append(model(x).numpy())
            print(f"{i + len(x)}/{len(X)}  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(a.out, feats=np.concatenate(feats), logits=np.concatenate(logits), pathologies=np.array(model.pathologies))
    print("saved", a.out, np.concatenate(feats).shape)


if __name__ == "__main__":
    main()
