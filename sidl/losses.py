"""Training objectives: ERM, DANN-style adversarial, GroupDRO, and the full SIDL loss.

SIDL = CE(task) + lam * CE(site | GRL(z)) + beta * CORAL(z across sites) + gamma * consistency(clean vs harmonized)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def coral_loss(z: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
    """Mean pairwise covariance + mean alignment distance between site feature distributions."""
    uniq = [s for s in sites.unique() if (sites == s).sum() > 1]
    if len(uniq) < 2:
        return z.new_zeros(())
    stats = []
    for s in uniq:
        zs = z[sites == s]
        mu = zs.mean(0)
        zc = zs - mu
        cov = zc.t() @ zc / (zs.size(0) - 1)
        stats.append((mu, cov))
    loss, n = z.new_zeros(()), 0
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            loss = loss + F.mse_loss(stats[i][0], stats[j][0]) + F.mse_loss(stats[i][1], stats[j][1])
            n += 1
    return loss / n


def class_conditional_coral(z: torch.Tensor, sites: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """CORAL computed within each class, so label-prior differences between sites are not penalized."""
    losses = []
    for c in range(num_classes):
        m = y == c
        if m.sum() > 3:
            l = coral_loss(z[m], sites[m])
            if l.requires_grad or l.item() > 0:
                losses.append(l)
    return torch.stack(losses).mean() if losses else z.new_zeros(())


def consistency_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between predictions on clean and harmonized views."""
    pa, pb = F.log_softmax(logits_a, 1), F.log_softmax(logits_b, 1)
    return 0.5 * (F.kl_div(pa, pb.exp(), reduction="batchmean") + F.kl_div(pb, pa.exp(), reduction="batchmean"))


class GroupDROState:
    """Online GroupDRO (Sagawa et al.): exponentiated-gradient weights over sites."""

    def __init__(self, num_sites: int, step: float = 0.01, device="cpu"):
        self.q = torch.ones(num_sites, device=device) / num_sites
        self.step = step

    def loss(self, per_sample: torch.Tensor, sites: torch.Tensor) -> torch.Tensor:
        gl = per_sample.new_zeros(self.q.numel())
        for s in sites.unique():
            gl[s] = per_sample[sites == s].mean()
        with torch.no_grad():
            present = torch.zeros_like(self.q, dtype=torch.bool)
            present[sites.unique()] = True
            self.q = self.q * torch.exp(self.step * gl.detach() * present)
            self.q = self.q / self.q.sum()
        return (self.q * gl).sum()
