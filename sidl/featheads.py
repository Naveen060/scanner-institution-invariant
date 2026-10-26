"""Feature-space training heads on frozen encoder features.

Mirrors the methods used for the from-scratch CNN (ERM, GroupDRO, DANN, class-conditional adversary, prior reweighting,
CORAL) so the same leave-one-site-out protocol can be run on embeddings from a pretrained encoder. Only the small head is
trained; the encoder stays frozen, so a full 8-site x multi-seed sweep takes minutes on CPU.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import GroupDROState, class_conditional_coral, coral_loss
from .metrics import auroc
from .model import dann_lambda, grad_reverse

FEAT_METHODS = {
    "erm": {},
    "erm_linear": {"linear": True},
    "erm_p": {"prior": True},
    "groupdro": {"dro": True},
    "dann": {"adv": "uncond"},
    "cdann": {"adv": "cond"},
    "cdann_p": {"adv": "cond", "prior": True},
    "coral": {"coral": "pooled"},
    "cc_full": {"adv": "cond", "coral": "cond", "prior": True},  # the class-conditional recipe, minus image augmentation
    "sitecenter": {"center": True},  # TRANSDUCTIVE: subtracts a per-site feature mean, using unlabeled target features
}


class FeatNet(nn.Module):
    def __init__(self, d_in, num_classes, num_sites, proj_dim=256, drop=0.2, adv=None, linear=False):
        super().__init__()
        self.linear, self.adv, self.num_classes = linear, adv, num_classes
        if linear:
            self.cls = nn.Linear(d_in, num_classes)
            return
        self.proj = nn.Sequential(nn.Linear(d_in, proj_dim), nn.ReLU(), nn.Dropout(drop))
        self.cls = nn.Linear(proj_dim, num_classes)
        if adv:
            mk = lambda: nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, num_sites))
            if adv == "cond":  # one site discriminator per class
                self.site_heads = nn.ModuleList([mk() for _ in range(num_classes)])
            else:
                self.site_head = mk()

    def forward(self, x, lam=0.0, y=None):
        if self.linear:
            return self.cls(x), None, x
        z = self.proj(x)
        logits, site_logits = self.cls(z), None
        if self.adv:
            zr = grad_reverse(z, lam)
            if self.adv == "cond":
                if y is not None:
                    allp = torch.stack([h(zr) for h in self.site_heads], 1)
                    site_logits = allp[torch.arange(z.size(0)), y]
            else:
                site_logits = self.site_head(zr)
        return logits, site_logits, z


def prior_weights(y, s, num_sites, num_classes=2, clip=10.0):
    """w[s,y] = P_global(y) / P(y | s) (Laplace-smoothed), clipped; removes site-specific label priors."""
    cnt = np.ones((num_sites, num_classes))
    for si, yi in zip(s, y):
        cnt[si, yi] += 1
    p_sy = cnt / cnt.sum(1, keepdims=True)
    g = np.bincount(y, minlength=num_classes) + 1.0
    g = g / g.sum()
    return torch.from_numpy(np.minimum(g[None, :] / p_sy, clip)).float()


@torch.no_grad()
def predict(model, X):
    model.eval()
    return F.softmax(model(torch.from_numpy(X).float())[0], 1).numpy()


def train_head(Xtr, ytr, str_, Xva, yva, num_sites, flags, seed, epochs=30, bs=64, lr=1e-3, wd=1e-2,
               adv_weight=0.5, coral_weight=0.1, proj_dim=256, drop=0.2):
    """Trains a head; returns the model with the best source-validation AUROC (never uses the held-out site)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = FeatNet(Xtr.shape[1], 2, num_sites, proj_dim, drop, adv=flags.get("adv"), linear=flags.get("linear", False))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xt, yt, st = torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long(), torch.from_numpy(str_).long()
    pw = prior_weights(ytr, str_, num_sites) if flags.get("prior") else None
    dro = GroupDROState(num_sites) if flags.get("dro") else None
    steps_per = max(1, int(np.ceil(len(ytr) / bs)))
    total, step, best, best_state = epochs * steps_per, 0, -1.0, None
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(ytr))
        for b in range(steps_per):
            ii = torch.from_numpy(idx[b * bs:(b + 1) * bs])
            if len(ii) < 4:
                continue
            x, y, s = Xt[ii], yt[ii], st[ii]
            lam = dann_lambda(step / max(total, 1)) * adv_weight if flags.get("adv") else 0.0
            logits, site_logits, z = model(x, lam, y)
            ce = F.cross_entropy(logits, y, reduction="none")
            if dro is not None:
                loss = dro.loss(ce, s)
            elif pw is not None:
                wb = pw[s, y]
                loss = (wb * ce).sum() / wb.sum()
            else:
                loss = ce.mean()
            if site_logits is not None:
                loss = loss + F.cross_entropy(site_logits, s)
            if flags.get("coral"):
                loss = loss + coral_weight * (class_conditional_coral(z, s, y, 2) if flags["coral"] == "cond" else coral_loss(z, s))
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        va = auroc(yva, predict(model, Xva)[:, 1])
        if va == va and va > best:
            best, best_state = va, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best
