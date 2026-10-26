"""Unit tests for the three new components (plain asserts; run: python scripts/unit_tests.py)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sidl.conformal import MODES, calibrate, conformal_report, evaluate_sets, prediction_sets  # noqa: E402
from sidl.data import SyntheticSiteDataset  # noqa: E402
from sidl.losses import class_conditional_coral  # noqa: E402
from sidl.model import SIDLNet  # noqa: E402
from sidl.siteprofile import SiteProfiles, SiteTranslator, estimate_params  # noqa: E402


def test_conformal_exchangeable():
    rng = np.random.RandomState(0)
    def draw(n):
        y = rng.randint(0, 3, n)
        logits = rng.randn(n, 3) + 2.0 * np.eye(3)[y]
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return p, y
    pv, yv = draw(4000); sv = rng.randint(0, 4, 4000)
    pt, yt = draw(20000)
    q = calibrate(pv, yv, sv, 0.1, "pooled")
    cov = evaluate_sets(prediction_sets(pt, q), yt)["coverage"]
    assert abs(cov - 0.9) < 0.02, cov
    # worst-site quantile must be >= pooled-ish and >= site_balanced? at least >= min over sites
    qs = {m: calibrate(pv, yv, sv, 0.1, m) for m in MODES}
    assert qs["worst_site"] >= min(qs["pooled"], qs["site_balanced"]) - 1e-9, qs
    print("conformal exchangeable OK", {k: round(v, 3) for k, v in qs.items()}, "coverage", round(cov, 3))


def test_conformal_site_shift_worst_site_more_conservative():
    rng = np.random.RandomState(1)
    def draw(n, sharp):
        y = rng.randint(0, 2, n)
        logits = rng.randn(n, 2) + sharp * np.eye(2)[y]
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return p, y
    ps, ys, ss = [], [], []
    for k, sharp in enumerate([3.0, 2.0, 0.5]):  # site 2 is hard
        p, y = draw(1500, sharp); ps.append(p); ys.append(y); ss.append(np.full(1500, k))
    pv, yv, sv = np.concatenate(ps), np.concatenate(ys), np.concatenate(ss)
    pt, yt = draw(5000, 0.5)  # unseen site as hard as the hardest source
    rep = conformal_report(pv, yv, sv, pt, yt, np.zeros(5000, int), 0.1)
    assert rep["worst_site"]["coverage"] >= rep["pooled"]["coverage"] - 1e-6
    assert rep["worst_site"]["coverage"] >= 0.87, rep["worst_site"]
    assert rep["pooled"]["coverage"] < rep["worst_site"]["coverage"] + 1e-6
    print("conformal shift OK", {m: (round(r["coverage"], 3), round(r["set_size"], 2)) for m, r in rep.items()})


def test_site_profiles_and_translation():
    ds = SyntheticSiteDataset(n_per_site=80, num_sites=3, size=32, seed=0, site_prior_shift=0.3)
    pr = SiteProfiles(3, 2).fit(ds, batch_size=32)
    assert pr.mu.shape == (3, 5) and pr.prior.shape == (3, 2)
    # synthetic sites differ in gain (mean intensity) -> profile means must be ordered by site
    assert pr.mu[0, 0] < pr.mu[1, 0] < pr.mu[2, 0], pr.mu[:, 0]
    # prior reweighting equalizes label priors across sites
    w = pr.prior_weights()
    eff = pr.prior * w  # P(y|s) * w[s,y] ∝ global prior
    eff = eff / eff.sum(1, keepdim=True)
    assert torch.allclose(eff, eff.mean(0, keepdim=True).expand_as(eff), atol=1e-3), eff
    # translation moves site-0 images toward the profile of a (forced) target site
    x = torch.stack([ds[i][0] for i in range(40)])  # site 0 images
    s = torch.zeros(40, dtype=torch.long)
    tr = SiteTranslator(pr, jitter=0.0, p=1.0).train()
    torch.manual_seed(0)
    out = tr(x, s)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert (out - x).abs().mean() > 1e-4  # it actually changed something
    tr.eval(); assert torch.equal(tr(x, s), x)  # identity at eval
    print("site profiles OK; site mean intensity:", [round(v, 3) for v in pr.mu[:, 0].tolist()])


def test_conditional_adversary_shapes_and_coral():
    m = SIDLNet("tiny", 2, 3, in_chans=1, pretrained=False, use_adv=True, cond_adv=True)
    x = torch.rand(8, 1, 32, 32); y = torch.randint(0, 2, (8,))
    logits, sl, z = m(x, 0.5, y)
    assert logits.shape == (8, 2) and sl.shape == (8, 3)
    assert m(x, 0.5)[1] is None  # inference without labels
    s = torch.randint(0, 3, (8,))
    assert class_conditional_coral(z, s, y, 2).ndim == 0
    sl.sum().backward()  # gradient flows through GRL
    print("conditional adversary OK")


def test_feature_heads():
    from sidl.featheads import FEAT_METHODS, predict, prior_weights, train_head
    rng = np.random.RandomState(0)
    n, d = 120, 16
    s = rng.randint(0, 3, n); y = (rng.rand(n) < np.array([0.2, 0.5, 0.8])[s]).astype(int)
    X = rng.randn(n, d).astype(np.float32); X[:, 0] += 1.5 * y
    w = prior_weights(y, s, 3).numpy()  # reweighting equalizes the label prior across sites
    prev = np.array([(w[k, 1] * (y[s == k] == 1).sum()) / (w[k, 1] * (y[s == k] == 1).sum() + w[k, 0] * (y[s == k] == 0).sum()) for k in range(3)])
    assert prev.std() < 0.08, prev
    for name, flags in FEAT_METHODS.items():
        if flags.get("center"):
            continue
        m, best = train_head(X[:90], y[:90], s[:90], X[90:], y[90:], 3, flags, 0, epochs=3)
        p = predict(m, X[90:])
        assert p.shape == (30, 2) and np.allclose(p.sum(1), 1, atol=1e-5), name
    print("feature heads OK")


if __name__ == "__main__":
    test_feature_heads()
    test_conformal_exchangeable()
    test_conformal_site_shift_worst_site_more_conservative()
    test_site_profiles_and_translation()
    test_conditional_adversary_shapes_and_coral()
    print("ALL UNIT TESTS PASSED")
