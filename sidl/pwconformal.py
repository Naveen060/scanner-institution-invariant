"""Conformal variants that use the label-free acquisition profile of the (unlabeled) deployment site.

profile_weighted : weighted split conformal (Tibshirani et al. 2019) where the covariate is the 5-d acquisition profile.
                   Density ratio w(x) = P(target | profile)/P(source | profile) is estimated by L2-regularized logistic regression
                   between source-validation images and the unlabeled test-site images (no test labels used).
                   Calibration scores of source samples that look like the new site (acquisition-wise) get more weight.
site_nn          : calibrate only on the k source sites whose mean profile is closest to the new site's mean profile.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .conformal import lac_scores, prediction_sets, evaluate_sets, _quantile


def density_ratio(P_src: np.ndarray, P_tgt: np.ndarray, C: float = 0.3, clip: float = 20.0):
    sc = StandardScaler().fit(np.vstack([P_src, P_tgt]))
    Xs, Xt = sc.transform(P_src), sc.transform(P_tgt)
    X = np.vstack([Xs, Xt]); d = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    clf = LogisticRegression(C=C, max_iter=2000).fit(X, d)
    prior = len(Xs) / len(Xt)
    w = lambda Z: np.clip(clf.predict_proba(Z)[:, 1] / np.clip(clf.predict_proba(Z)[:, 0], 1e-6, None) * prior, 1 / clip, clip)
    return w(Xs), w(Xt)


def weighted_quantile_with_test_mass(scores, w, w_test, alpha):
    order = np.argsort(scores)
    s, ww = scores[order], w[order]
    tot = ww.sum() + w_test
    cum = np.cumsum(ww) / tot
    idx = np.searchsorted(cum, 1 - alpha, side="left")
    return float("inf") if idx >= len(s) else float(s[idx])


def profile_weighted_sets(val_probs, val_y, P_val, test_probs, P_test, alpha=0.1, C=0.3):
    sc = lac_scores(val_probs, val_y)
    w_val, w_test = density_ratio(P_val, P_test, C)
    qs = np.array([weighted_quantile_with_test_mass(sc, w_val, wt, alpha) for wt in w_test])
    sets = (1.0 - test_probs) <= qs[:, None]
    empty = ~sets.any(1)
    sets[empty, test_probs[empty].argmax(1)] = True
    return sets, qs, float((w_val.sum() ** 2) / (w_val ** 2).sum())  # effective sample size


def site_nn_sets(val_probs, val_y, val_sites, P_val, test_probs, P_test, alpha=0.1, k=3):
    sites = np.unique(val_sites)
    mu_t = P_test.mean(0)
    sd = P_val.std(0) + 1e-9
    dist = {s: np.linalg.norm((P_val[val_sites == s].mean(0) - mu_t) / sd) for s in sites}
    near = sorted(sites, key=lambda s: dist[s])[:k]
    m = np.isin(val_sites, near)
    q = _quantile(lac_scores(val_probs[m], val_y[m]), alpha)
    return prediction_sets(test_probs, q), q
