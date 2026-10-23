"""Model: backbone + classifier head + (optional) gradient-reversal site discriminator."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float = 1.0):
    return _GradReverse.apply(x, lam)


def dann_lambda(progress: float, gamma: float = 10.0) -> float:
    """Ganin et al. schedule: 0 -> 1 as training progresses (progress in [0,1])."""
    return 2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0


def build_backbone(name: str, pretrained: bool, in_chans: int):
    """Returns (module producing pooled features, feature_dim). Uses timm if available."""
    try:
        import timm
        m = timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=in_chans)
        return m, m.num_features
    except ImportError:
        pass
    import torchvision
    if name.startswith("resnet"):
        m = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
        if in_chans != 3:
            m.conv1 = nn.Conv2d(in_chans, 64, 7, 2, 3, bias=False)
        d = m.fc.in_features
        m.fc = nn.Identity()
        return m, d
    raise ValueError(f"Install timm to use backbone '{name}'")


class TinyCNN(nn.Module):
    """Small CNN for smoke tests / CPU debugging."""

    def __init__(self, in_chans=1, dim=64):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(in_chans, 16), blk(16, 32), blk(32, dim), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.num_features = dim

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    """~0.1M-parameter CNN trained from scratch (no pretrained weights are reachable offline)."""

    def __init__(self, in_chans=1, widths=(16, 32, 48, 64)):
        super().__init__()
        layers, c = [], in_chans
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2)]
            c = w
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.num_features = c

    def forward(self, x):
        return self.net(x)


class SIDLNet(nn.Module):
    def __init__(self, backbone: str, num_classes: int, num_sites: int, in_chans: int = 3,
                 pretrained: bool = True, proj_dim: int = 256, use_adv: bool = True, drop: float = 0.2,
                 cond_adv: bool = False, prof_adv: str | None = None, prof_dim: int = 5):
        super().__init__()
        self.cond_adv, self.num_classes, self.prof_adv = cond_adv, num_classes, prof_adv
        if backbone == "tiny":
            self.backbone, d = TinyCNN(in_chans), 64
        elif backbone == "small":
            self.backbone = SmallCNN(in_chans)
            d = self.backbone.num_features
        else:
            self.backbone, d = build_backbone(backbone, pretrained, in_chans)
        self.proj = nn.Sequential(nn.Linear(d, proj_dim), nn.ReLU(), nn.Dropout(drop))
        self.cls = nn.Linear(proj_dim, num_classes)
        self.use_adv = use_adv and num_sites > 1
        if self.use_adv:
            mk = lambda: nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, num_sites))
            if cond_adv:
                # one site discriminator per class: aligns P(z | y, site) instead of P(z | site),
                # so differing class priors across sites are NOT forced to look identical.
                self.site_heads = nn.ModuleList([mk() for _ in range(num_classes)])
            else:
                self.site_head = mk()

        if prof_adv:  # adversary that regresses the MEASURED acquisition profile from z (continuous, label-free, site-ID free)
            mkp = lambda: nn.Sequential(nn.Linear(proj_dim, proj_dim // 2), nn.ReLU(), nn.Linear(proj_dim // 2, prof_dim))
            self.prof_heads = nn.ModuleList([mkp() for _ in range(num_classes)]) if prof_adv == "cond" else nn.ModuleList([mkp()])

    def prof_pred(self, z, lam: float, y=None):
        zr = grad_reverse(z, lam)
        if self.prof_adv == "cond":
            allp = torch.stack([h(zr) for h in self.prof_heads], 1)  # (B,C,D)
            return allp[torch.arange(z.size(0), device=z.device), y]
        return self.prof_heads[0](zr)

    def forward(self, x, lam: float = 0.0, y=None):
        """Returns (task logits, site logits, z). With cond_adv and labels `y`, site logits come from the
        discriminator of each sample's class (B,S); without `y` (inference) site logits are None."""
        z = self.proj(self.backbone(x))
        logits = self.cls(z)
        site_logits = None
        if self.use_adv:
            zr = grad_reverse(z, lam)
            if self.cond_adv:
                if y is not None:
                    all_l = torch.stack([h(zr) for h in self.site_heads], 1)  # (B,C,S)
                    site_logits = all_l[torch.arange(z.size(0), device=z.device), y]
            else:
                site_logits = self.site_head(zr)
        return logits, site_logits, z
