"""Test-time label-shift handling: prior estimation (EM) and label-shift-weighted conformal sets.

em_prior          Saerens et al. (2002): estimate the new site's class prior from UNLABELED predictions.
adjust_probs      posterior re-weighting p'(y|x) ∝ p(y|x) * pi_t(y)/pi_s(y).
label_shift_sets  weighted split conformal for label shift (Podkopaev & Ramdas 2021 style): for each candidate label c the test point
                  puts weight w(c)=pi_t(c)/pi_s(c) at +inf; calibration scores carry weights w(y_i).
"""
from __future__ import annotations

import numpy as np

from .conformal import lac_scores


def em_prior(probs: np.ndarray, pi_s: np.ndarray, iters: int = 200, tol: float = 1e-7) -> np.ndarray:
    pi_t = pi_s.copy()
    for _ in range(iters):
        post = probs * (pi_t / pi_s)[None, :]
        post /= post.sum(1, keepdims=True)
        new = post.mean(0)
        if np.abs(new - pi_t).max() < tol:
            pi_t = new
            break
        pi_t = new
    return pi_t


def adjust_probs(probs: np.ndarray, pi_s: np.ndarray, pi_t: np.ndarray) -> np.ndarray:
    p = probs * (pi_t / pi_s)[None, :]
    return p / p.sum(1, keepdims=True)


def label_shift_sets(val_probs, val_y, test_probs, pi_s, pi_t, alpha=0.1, w_clip=10.0):
    """Include class c iff 1 - p(c|x) <= q_c, q_c = weighted quantile of calibration scores with test mass w(c) at +inf."""
    w_cls = np.clip(pi_t / np.clip(pi_s, 1e-6, None), 1 / w_clip, w_clip)
    sc = lac_scores(val_probs, val_y)
    w = w_cls[val_y]
    order = np.argsort(sc)
    s, ww = sc[order], w[order]
    cumw = np.cumsum(ww)
    qs = []
    for c in range(len(pi_s)):
        target = (1 - alpha) * (ww.sum() + w_cls[c])
        idx = np.searchsorted(cumw, target, side="left")
        qs.append(np.inf if idx >= len(s) else s[idx])
    qs = np.array(qs)
    sets = (1.0 - test_probs) <= qs[None, :]
    empty = ~sets.any(1)
    sets[empty, test_probs[empty].argmax(1)] = True
    return sets, qs
