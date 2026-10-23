"""Unified trainer. Methods are defined by flags so that every contribution can be ablated in isolation.

  method        harmonization   site adversary   CORAL      consistency   prior-reweight
  erm           -               -                -          -             -
  harmonize     generic         -                -          -             -
  dann          -               unconditional    -          -             -
  groupdro      (GroupDRO objective)
  sidl          generic         unconditional    pooled     yes           -          <- original recipe
  sidl_phys     site-profile    unconditional    pooled     yes           -          <- contribution (2) only
  sidl_cc       generic         class-cond.      per-class  yes           yes        <- contribution (1) only
  sidl_full     site-profile    class-cond.      per-class  yes           yes        <- (1) + (2); (3) = conformal, applied to every method
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augment import Harmonizer
from .conformal import conformal_report
from .losses import GroupDROState, class_conditional_coral, consistency_loss, coral_loss
from .metrics import evaluate, per_site_report
from .model import SIDLNet, dann_lambda
from .siteprofile import SiteProfiles, SiteTranslator, estimate_params

METHOD_FLAGS = {
    "erm": {},
    "harmonize": {"harm": "generic"},
    "phys_aug": {"harm": "phys"},
    "dann": {"adv": "uncond"},
    "cdann": {"adv": "cond"},                 # class-conditional site adversary only
    "cdann_p": {"adv": "cond", "prior": True},  # + site-prior reweighting of the task loss
    "erm_p": {"prior": True},                 # ERM + site-prior reweighting only
    "groupdro": {"dro": True},
    "sidl": {"harm": "generic", "adv": "uncond", "coral": "pooled", "cons": True},
    "sidl_phys": {"harm": "phys", "adv": "uncond", "coral": "pooled", "cons": True},
    "sidl_cc": {"harm": "generic", "adv": "cond", "coral": "cond", "cons": True, "prior": True},
    "sidl_full": {"harm": "phys", "adv": "cond", "coral": "cond", "cons": True, "prior": True},
    # profile-adversarial invariance: z must not predict the MEASURED acquisition profile (beyond the label, if cond)
    "pai": {"padv": "cond"},
    "pai_u": {"padv": "uncond"},
    "sidl_pai": {"harm": "phys", "adv": "cond", "coral": "cond", "cons": True, "prior": True, "padv": "cond"},
}
METHODS = tuple(METHOD_FLAGS)


@dataclass
class TrainConfig:
    method: str = "sidl_full"
    backbone: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 2
    in_chans: int = 3
    modality: str = "cxr"
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    wd: float = 1e-4
    adv_weight: float = 0.5     # max GRL strength
    coral_weight: float = 0.1
    cons_weight: float = 0.5
    dro_step: float = 0.01
    conformal_alpha: float = 0.1
    padv_weight: float = 1.0
    compensate: bool = False    # noise-compensated sharpness in site profiles
    aug_p: float = 0.5          # probability a sample is augmented (rest stay clean)
    aug_strength: float = 1.0   # scales generic augmentation ranges / site-profile translation interpolation
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "runs/tmp"
    amp: bool = True


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, S = [], [], []
    for x, y, s in loader:
        logits, _, _ = model(x.to(device))
        P.append(F.softmax(logits.float(), 1).cpu())
        Y.append(y)
        S.append(s)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(S).numpy()


def train_and_eval(cfg: TrainConfig, train_ds, val_ds, test_ds, num_train_sites: int,
                   site_names: Optional[list] = None, extra_eval: Optional[Dict] = None) -> dict:
    assert cfg.method in METHOD_FLAGS, cfg.method
    fl = METHOD_FLAGS[cfg.method]
    set_seed(cfg.seed)
    dev = cfg.device
    os.makedirs(cfg.out_dir, exist_ok=True)
    amp_on = cfg.amp and dev.startswith("cuda")

    adv = fl.get("adv")
    model = SIDLNet(cfg.backbone, cfg.num_classes, num_train_sites, cfg.in_chans, cfg.pretrained,
                    use_adv=adv is not None, cond_adv=(adv == "cond"), prof_adv=fl.get("padv")).to(dev)

    profiles, harm, prior_w = None, None, None
    if fl.get("harm") == "phys" or fl.get("prior") or fl.get("padv"):
        profiles = SiteProfiles(num_train_sites, cfg.num_classes, compensate=cfg.compensate).fit(train_ds, num_workers=0)
    if fl.get("harm") == "generic":
        harm = Harmonizer(p=cfg.aug_p, strength=cfg.aug_strength, modality=cfg.modality).to(dev)
    elif fl.get("harm") == "phys":
        harm = SiteTranslator(profiles, p=cfg.aug_p, alpha=cfg.aug_strength).to(dev)
    if fl.get("prior"):
        prior_w = profiles.prior_weights().to(dev)
    dro = GroupDROState(num_train_sites, cfg.dro_step, dev) if fl.get("dro") else None

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = cfg.epochs * max(len(train_ds) // cfg.batch_size, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps + 1)
    scaler = torch.amp.GradScaler(enabled=amp_on)

    mk = lambda ds, sh: DataLoader(ds, batch_size=cfg.batch_size, shuffle=sh, num_workers=cfg.num_workers,
                                   drop_last=sh, pin_memory=dev.startswith("cuda"))
    tl, vl, el = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

    best_val, best_state, step, history = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        run = {"task": 0.0, "adv": 0.0, "coral": 0.0, "cons": 0.0}
        nb = 0
        for x, y, s in tl:
            x, y, s = x.to(dev), y.to(dev), s.to(dev)
            lam = dann_lambda(step / max(steps, 1)) * cfg.adv_weight if (adv or fl.get("padv")) else 0.0
            if fl.get("harm") == "phys":
                x_in = harm(x, s)
            elif harm is not None:
                x_in = harm(x)
            else:
                x_in = x
            with torch.autocast(device_type="cuda" if dev.startswith("cuda") else "cpu", enabled=amp_on):
                logits, site_logits, z = model(x_in, lam, y)
                ce = F.cross_entropy(logits, y, reduction="none")
                if prior_w is not None:  # remove site-specific label priors from the task loss
                    wb = prior_w[s, y]
                    ce = ce * wb / wb.mean()
                task = dro.loss(ce, s) if dro is not None else ce.mean()
                loss = task
                if adv:
                    a = F.cross_entropy(site_logits.float(), s)
                    loss = loss + a  # GRL scales the reversed gradient into the encoder by lam
                    run["adv"] += a.item()
                if fl.get("padv"):
                    with torch.no_grad():
                        tgt = (estimate_params(x_in.float(), cfg.compensate)[0] - profiles.gmean.to(dev)) / profiles.gstd.to(dev)
                    pl = F.mse_loss(model.prof_pred(z, lam, y).float(), tgt)
                    loss = loss + cfg.padv_weight * pl
                    run["padv"] = run.get("padv", 0.0) + pl.item()
                if fl.get("coral"):
                    zf = z.float()
                    cor = class_conditional_coral(zf, s, y, cfg.num_classes) if fl["coral"] == "cond" else coral_loss(zf, s)
                    loss = loss + cfg.coral_weight * cor
                    run["coral"] += float(cor.detach())
                if fl.get("cons"):
                    logits_clean, _, _ = model(x, 0.0)
                    cons = consistency_loss(logits_clean.float(), logits.float())
                    loss = loss + cfg.cons_weight * cons
                    run["cons"] += cons.item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run["task"] += task.item()
            nb += 1
            step += 1

        p, yv, sv = predict(model, vl, dev)
        vm = evaluate(yv, p)
        history.append({"epoch": ep, **{k: v / max(nb, 1) for k, v in run.items()}, "val_auroc": vm["auroc"]})
        score = vm["auroc"] if not np.isnan(vm["auroc"]) else vm["acc"]
        if score > best_val:  # model selection on SOURCE-site validation only
            best_val, best_state = score, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    results = {"method": cfg.method, "best_val": best_val, "history": history}
    pt, yt, st = predict(model, el, dev)
    results["test"] = per_site_report(yt, pt, st)
    results["test_overall"] = results["test"]["overall"]
    # Contribution (3): cross-site conformal prediction, calibrated on source-site validation only
    pv, yv, sv = predict(model, vl, dev)
    results["conformal"] = conformal_report(pv, yv, sv, pt, yt, st, cfg.conformal_alpha)
    np.savez_compressed(os.path.join(cfg.out_dir, "preds.npz"), pv=pv, yv=yv, sv=sv, pt=pt, yt=yt, st=st,
                        val_rows=getattr(val_ds, "rows", np.arange(len(yv))), test_rows=getattr(test_ds, "rows", np.arange(len(yt))))
    if profiles is not None:
        results["site_profiles"] = profiles.summary()
    if extra_eval:
        results["extra"] = {}
        for name, ds in extra_eval.items():
            pe, ye, se = predict(model, mk(ds, False), dev)
            results["extra"][name] = per_site_report(ye, pe, se)
            np.savez_compressed(os.path.join(cfg.out_dir, f"extra_{name}.npz"), p=pe, y=ye, s=se,
                                rows=getattr(ds, "rows", np.arange(len(ye))))
    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    torch.save(best_state, os.path.join(cfg.out_dir, "best.pt"))
    return results
