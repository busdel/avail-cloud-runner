#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMIL and mmFormer re-implementations (tabular features) for the
missing-modality comparison, both tasks (multi / binary).

SMIL-lite: per-modality Gaussian encoders, joint posterior (product of
Gaussians) over available modalities, unified latent sample, per-modality
decoders (reconstruction of available modalities) + classifier; loss =
CE + rec + small KL.  (Following Ma et al., AAAI 2021, simplified for
tabular modalities.)

mmFormer-lite: modality tokens (learnable missing-modality tokens) +
transformer encoder + classifier; distillation from a complete-case
teacher trained inside the same fold; loss = CE + alpha * KL(student,
teacher) on complete-case subjects.  (Following Zhang et al., MICCAI 2022,
simplified.)

Usage: COMP_TASK=multi|binary python competitors2_loo.py
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from competitors_loo import (load_data, get_fold_arrays,  # noqa: E402
                             MODULES)

ROOT = None
for cand in [Path("/media/suleyman_disk/AVAIL_GPU"),
             Path("/mnt/limbic/AVAIL_GPU")]:
    if cand.exists():
        ROOT = cand
        break
OUT = ROOT / "experiments/results/competitors"
OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["HC", "MCI", "PDD"]
K = 3
SEEDS = 2
EPOCHS = 120
LR = 1e-3
WD = 1e-4
LAT = 16
D = 32
ALPHA_KD = 0.5


def make_smil(dims, k, n_mods, device):
    import torch
    import torch.nn as nn

    class SMIL(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(dims[m], 64), nn.ReLU(),
                                 nn.Linear(64, 2 * LAT)) for m in MODULES})
            self.dec = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(LAT, 64), nn.ReLU(),
                                 nn.Linear(64, dims[m])) for m in MODULES})
            self.cls = nn.Sequential(nn.Linear(LAT + n_mods, 24), nn.ReLU(),
                                     nn.Linear(24, k))

        def forward(self, X, M, train):
            mu_sum = None
            prec = None
            recs = []
            for c, m in enumerate(MODULES):
                mu, lv = self.enc[m](X[m]).chunk(2, dim=-1)
                lv = lv.clamp(-6, 6)
                prec_m = torch.exp(-lv)
                mk = M[:, c:c + 1]
                pm = prec_m * mk
                mu_sum = mu * pm if mu_sum is None else mu_sum + mu * pm
                prec = pm if prec is None else prec + pm
            prec = prec.clamp(min=1e-4)
            mu0 = mu_sum / prec
            sig0 = (1.0 / prec).sqrt()
            if train:
                z = mu0 + sig0 * torch.randn_like(sig0)
            else:
                z = mu0
            logits = self.cls(torch.cat([z, M], dim=1))
            rec = 0.0
            n_rec = 0
            for c, m in enumerate(MODULES):
                mk = M[:, c:c + 1]
                if mk.sum() > 0:
                    rec = rec + (((self.dec[m](z) - X[m]) ** 2)
                                 * mk).sum() / mk.sum()
                    n_rec += 1
            rec = rec / max(n_rec, 1)
            kl = 0.5 * (mu0 ** 2 + sig0 ** 2 - 2 * torch.log(sig0 + 1e-8))
            kl = kl.sum(1).mean()
            return logits, rec, kl

    return SMIL().to(device)


def make_mmformer(dims, k, n_mods, device):
    import torch
    import torch.nn as nn

    class MMF(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(dims[m], D), nn.ReLU())
                for m in MODULES})
            self.miss = nn.ParameterDict({
                m.replace("_", ""): nn.Parameter(torch.zeros(D))
                for m in MODULES})
            self.avail = nn.Linear(n_mods, D)
            layer = nn.TransformerEncoderLayer(
                D, nhead=4, dim_feedforward=64, dropout=0.1,
                batch_first=True, activation="gelu")
            self.tr = nn.TransformerEncoder(layer, num_layers=2)
            self.cls = nn.Sequential(nn.Linear(D, 24), nn.ReLU(),
                                     nn.Linear(24, k))

        def forward(self, X, M):
            toks = []
            for c, m in enumerate(MODULES):
                key = m.replace("_", "")
                t = self.enc[m](X[m])
                miss_tok = self.miss[key].unsqueeze(0).expand(X[m].shape[0],
                                                              -1)
                mk = M[:, c:c + 1]
                toks.append(t * mk + miss_tok * (1 - mk))
            T = torch.stack(toks, dim=1)
            T = T + self.avail(M).unsqueeze(1)
            H = self.tr(T)
            pooled = H.mean(dim=1)
            return self.cls(pooled)

    return MMF().to(device)


def run_method(variant, ids, label, feats, device, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n = len(ids)
    y = np.array([label[i] for i in ids])
    id2row = {i: r for r, i in enumerate(ids)}
    probs = np.zeros((n, K))
    for fold in range(n):
        X, M, dims = get_fold_arrays(fold, ids, label, feats, id2row)
        Xt = {m: torch.tensor(X[m], device=device) for m in MODULES}
        Mt = torch.tensor(M, device=device)
        yt = torch.tensor(y, device=device)
        tr = torch.tensor([r for r in range(n) if r != fold], device=device)
        te = torch.tensor([fold], device=device)
        if variant == "smil":
            model = make_smil(dims, K, len(MODULES), device)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            lossf = nn.CrossEntropyLoss()
            for ep in range(EPOCHS):
                model.train()
                opt.zero_grad()
                logits, rec, kl = model({m: Xt[m][tr] for m in MODULES},
                                        Mt[tr], True)
                loss = lossf(logits, yt[tr]) + 1.0 * rec + 1e-3 * kl
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                logits, _, _ = model({m: Xt[m][te] for m in MODULES},
                                     Mt[te], False)
                probs[fold] = torch.softmax(logits, 1)[0].cpu().numpy()
        else:  # mmformer
            complete = (Mt[tr].sum(1) == len(MODULES))
            teacher = None
            if complete.sum() >= 5:
                teacher = make_mmformer(dims, K, len(MODULES), device)
                topt = torch.optim.Adam(teacher.parameters(), lr=LR,
                                        weight_decay=WD)
                lossf = nn.CrossEntropyLoss()
                idx = tr[complete]
                for ep in range(EPOCHS):
                    teacher.train()
                    topt.zero_grad()
                    lg = teacher({m: Xt[m][idx] for m in MODULES}, Mt[idx])
                    loss = lossf(lg, yt[idx])
                    loss.backward()
                    topt.step()
                for p in teacher.parameters():
                    p.requires_grad_(False)
            model = make_mmformer(dims, K, len(MODULES), device)
            opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            lossf = nn.CrossEntropyLoss()
            kldiv = nn.KLDivLoss(reduction="batchmean")
            for ep in range(EPOCHS):
                model.train()
                opt.zero_grad()
                lg = model({m: Xt[m][tr] for m in MODULES}, Mt[tr])
                loss = lossf(lg, yt[tr])
                if teacher is not None and complete.sum() > 0:
                    idx = tr[complete]
                    with torch.no_grad():
                        tl = teacher({m: Xt[m][idx] for m in MODULES},
                                     Mt[idx])
                        tsoft = torch.softmax(tl, 1)
                    ssoft = torch.log_softmax(
                        model({m: Xt[m][idx] for m in MODULES}, Mt[idx]), 1)
                    loss = loss + ALPHA_KD * kldiv(ssoft, tsoft)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                lg = model({m: Xt[m][te] for m in MODULES}, Mt[te])
                probs[fold] = torch.softmax(lg, 1)[0].cpu().numpy()
        if (fold + 1) % 25 == 0:
            print(f"   [{variant} s{seed}] fold {fold+1}/{n}", flush=True)
    return probs


def main():
    global K, CLASSES, EPOCHS
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[comp2] device={device}", flush=True)
    ids, label, feats = load_data()
    task = os.environ.get("COMP_TASK", "multi")
    if task == "binary":
        ids = [i for i in ids if label[i] in (0, 2)]
        label = {i: (0 if label[i] == 0 else 1) for i in ids}
        K = 2
        CLASSES = ["HC", "PDD"]
        print(f"[comp2] BINARY task: HC vs PDD, n={len(ids)}", flush=True)
    lim = int(os.environ.get("COMP_FOLDS", "0"))
    if lim:
        ids = ids[:lim]
    EPOCHS = int(os.environ.get("COMP_EPOCHS", str(EPOCHS)))
    print(f"[comp2] n={len(ids)} epochs={EPOCHS} K={K}", flush=True)
    y = np.array([label[i] for i in ids])
    from avail_ai.learners import macro_auc
    from sklearn.metrics import roc_auc_score
    rows = []
    preds = {}
    for variant in ["smil", "mmformer"]:
        P = np.zeros((len(ids), K))
        for s in range(SEEDS):
            P += run_method(variant, ids, label, feats, device, 2000 + s)
            print(f"[comp2] {variant} seed {s} done", flush=True)
        P /= SEEDS
        preds[variant] = P
        row = {"method": variant, "macro_auc": macro_auc(y, P)}
        for c, cn in enumerate(CLASSES):
            row[f"auc_{cn}"] = float(roc_auc_score((y == c).astype(int),
                                                   P[:, c]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"competitors2_loo_results_{task}.tsv",
                                  sep="\t", index=False)
        print(f"[comp2] {variant}: macro={row['macro_auc']:.3f}", flush=True)
    np.savez_compressed(OUT / f"competitors2_loo_predictions_{task}.npz",
                        ids=np.array(ids), y=y, **preds)
    print("[comp2] DONE", flush=True)


if __name__ == "__main__":
    main()
