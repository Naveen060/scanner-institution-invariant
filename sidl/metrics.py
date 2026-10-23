"""Metrics: AUROC, balanced accuracy, ECE, and worst-site / cross-site gap reporting."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Macro AUROC for binary (probs: N,) or multi-class (probs: N,C)."""
    y_true = np.asarray(y_true)
    try:
        if probs.ndim == 1:
            return float(roc_auc_score(y_true, probs))
        if probs.shape[1] == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        present = np.unique(y_true)
        if len(present) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, probs, multi_class="ovr", labels=list(range(probs.shape[1]))))
    except ValueError:
        return float("nan")


def accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    pred = probs.argmax(1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    return float((pred == np.asarray(y_true)).mean())


def balanced_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    pred = probs.argmax(1) if probs.ndim == 2 else (probs > 0.5).astype(int)
    recalls = [(pred[y_true == c] == c).mean() for c in np.unique(y_true)]
    return float(np.mean(recalls))


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on top-1 confidence."""
    y_true = np.asarray(y_true)
    if probs.ndim == 1:
        probs = np.stack([1 - probs, probs], 1)
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def evaluate(y_true, probs) -> dict:
    return {
        "auroc": auroc(y_true, probs),
        "acc": accuracy(y_true, probs),
        "bal_acc": balanced_accuracy(y_true, probs),
        "ece": ece(y_true, probs),
    }


def per_site_report(y_true, probs, sites) -> dict:
    """Per-site metrics plus worst-site AUROC and max cross-site AUROC gap."""
    y_true, sites = np.asarray(y_true), np.asarray(sites)
    out = {"per_site": {}}
    for s in np.unique(sites):
        m = sites == s
        out["per_site"][str(s)] = evaluate(y_true[m], probs[m])
    aucs = [v["auroc"] for v in out["per_site"].values() if not np.isnan(v["auroc"])]
    out["worst_site_auroc"] = float(min(aucs)) if aucs else float("nan")
    out["site_auroc_gap"] = float(max(aucs) - min(aucs)) if aucs else float("nan")
    out["overall"] = evaluate(y_true, probs)
    return out
