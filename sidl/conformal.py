"""Cross-site split conformal prediction (LAC score: s = 1 - p_y).

Three calibration modes, all calibrated on SOURCE-site validation data only:
  pooled        standard split conformal; valid if the new site is exchangeable with the pooled sources.
  site_balanced weighted conformal where every source site contributes equal total weight (so large sites
                don't dominate the quantile); targets "an average source site".
  worst_site    max over sites of the per-site conformal quantile: marginal coverage >= 1-alpha holds for a
                new site whose score distribution is dominated by *some* observed site. Conservative
                (larger sets) but the right behaviour when the new site might look like the hardest one.

No mode gives a distribution-free guarantee under arbitrary site shift. We therefore report empirical
coverage on the held-out site (the quantity of interest) and the coverage gap to 1-alpha.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

MODES = ("pooled", "site_balanced", "worst_site")


def lac_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - probs[np.arange(len(y)), y]


def _quantile(scores: np.ndarray, alpha: float, weights: Optional[np.ndarray] = None) -> float:
    n = len(scores)
    if n == 0:
        return float("inf")
    order = np.argsort(scores)
    s = scores[order]
    if weights is None:
        k = math.ceil((n + 1) * (1 - alpha))
        return float("inf") if k > n else float(s[k - 1])
    w = weights[order]
    W = w.sum()
    cum = np.cumsum(w)
    target = (1 - alpha) * (W + w.mean())  # one extra unit of mass at +inf (mean weight)
    idx = np.searchsorted(cum, target, side="left")
    return float("inf") if idx >= n else float(s[idx])


def calibrate(probs: np.ndarray, y: np.ndarray, sites: np.ndarray, alpha: float = 0.1, mode: str = "pooled",
              min_site_n: int = 25) -> float:
    """min_site_n: worst_site only considers sites with at least this many validation samples (a split-conformal
    quantile from a handful of points is infinite/noisy and would make every set trivial)."""
    assert mode in MODES, mode
    sc = lac_scores(probs, np.asarray(y))
    sites = np.asarray(sites)
    if mode == "pooled":
        return _quantile(sc, alpha)
    uniq = np.unique(sites)
    if mode == "site_balanced":
        n = len(sc)
        w = np.zeros(n)
        for s in uniq:
            m = sites == s
            w[m] = n / (len(uniq) * m.sum())
        return _quantile(sc, alpha, w)
    qs = [_quantile(sc[sites == s], alpha) for s in uniq if (sites == s).sum() >= min_site_n]
    return float(max(qs)) if qs else _quantile(sc, alpha)


def prediction_sets(probs: np.ndarray, q: float) -> np.ndarray:
    """Boolean (N,C): class c is included iff 1 - p_c <= q. Never returns an empty set (adds argmax)."""
    sets = (1.0 - probs) <= q
    empty = ~sets.any(1)
    sets[empty, probs[empty].argmax(1)] = True
    return sets


def evaluate_sets(sets: np.ndarray, y: np.ndarray, sites: Optional[np.ndarray] = None) -> Dict:
    y = np.asarray(y)
    covered = sets[np.arange(len(y)), y]
    out = {"coverage": float(covered.mean()), "set_size": float(sets.sum(1).mean()),
           "singleton_rate": float((sets.sum(1) == 1).mean())}
    if sites is not None:
        out["per_site_coverage"] = {str(s): float(covered[np.asarray(sites) == s].mean()) for s in np.unique(sites)}
    return out


def conformal_report(val_probs, val_y, val_sites, test_probs, test_y, test_sites, alpha: float = 0.1) -> Dict:
    rep = {}
    for mode in MODES:
        q = calibrate(val_probs, val_y, val_sites, alpha, mode)
        ev = evaluate_sets(prediction_sets(test_probs, q), test_y, test_sites)
        ev["q"] = q
        ev["coverage_gap"] = ev["coverage"] - (1 - alpha)
        rep[mode] = ev
    return rep
