#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Competitor missing-modality architectures on our cohort (5-module setup).

Evaluated under the same leak-free leave-one-out protocol as the AVAIL
ladder: per-fold preprocessing fitted on training subjects only; all 150
subjects included via availability masks.

Methods (re-implemented following the original descriptions):
  early_fusion  : zero-filled concatenation + availability flags -> MLP
  heimis        : per-modality encoders + masked mean pooling (Havaei 2016)
  moddrop       : HeMIS encoder + training-time modality dropout p=0.3
                  (Neverova 2016)
  tmc           : per-modality evidential heads + Dirichlet fusion
                  (Han 2021, simplified)

Outputs:
  competitors_loo_results.tsv   macro/per-class AUC per method
  competitors_loo_predictions.npz
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = None
for cand in [Path("/media/suleyman_disk/AVAIL_GPU"),
             Path("/mnt/limbic/AVAIL_GPU")]:
    if cand.exists():
        ROOT = cand
        break
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import preproc, apply_preproc  # noqa: E402
from avail_ai.learners import macro_auc  # noqa: E402

DATA = None
for cand in [ROOT / "experiments_vox/results/ladder_generic",
             ROOT / "experiments/results/ladder_generic"]:
    if cand.exists():
        DATA = cand
        break
OUT = ROOT / "experiments/results/competitors"
OUT.mkdir(parents=True, exist_ok=True)

MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MICRO = ["stool16S", "stoolSG", "saliva16S", "salivaSG"]
CLASSES = ["HC", "MCI", "PDD"]
K = 3
SEEDS = 2
EPOCHS = 120
LR = 1e-3
WD = 1e-4
EMB = 32

def load_data():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    ids = meta["SampleID"].tolist()
    label = {r.SampleID: CLASSES.index(r.Group) for r in meta.itertuples()}
    feats = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t")
        t["SampleID"] = t["SampleID"].astype(str).str.strip()
        feats[m] = t.set_index("SampleID")
    return ids, label, feats


def fold_preproc(feats, ids, train_ids):
    """Fit per-modality preprocessing on train subjects only; return
    transformed matrices for ALL ids (train+test) per modality."""
    out = {}
    for m in MODULES:
        F = feats[m]
        avail_tr = [i for i in train_ids if i in F.index]
        Xtr = F.loc[avail_tr].values.astype(float)
        ytr = np.zeros(len(avail_tr), dtype=int)
        Ctr = np.zeros((len(avail_tr), 3))
        is_micro = m in MICRO
        try:
            sel, sel2, res, sc, Xt = preproc(Xtr, ytr, Ctr, is_micro,
                                             residualize=False)
        except Exception:
            # fallback: variance top-20, standardized
            d = Xtr.shape[1]
            top = np.argsort(Xtr.var(axis=0))[::-1][:min(20, d)]
            Xt = Xtr[:, top]
            Xt = (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9)
            sel, sel2, res = ("var", top), None, None
            sc = None
        rows, keep = [], []
        for i in ids:
            if i in F.index:
                x = F.loc[i].values.astype(float)[None, :]
                try:
                    xt = apply_preproc(sel, sel2, res, sc, x, np.zeros((1, 3)))
                except Exception:
                    xt = x
                rows.append(np.nan_to_num(xt.ravel()))
                keep.append(True)
            else:
                rows.append(None)
                keep.append(False)
        out[m] = (rows, keep)
    return out


def build_arrays(trans, ids, id2row):
    """Padded per-modality matrices + mask for the given subject order."""
    dims = {}
    for m in MODULES:
        rows, keep = trans[m]
        d = None
        for r in rows:
            if r is not None:
                d = r.shape[0]
                break
        dims[m] = d or 1
    n = len(ids)
    X = {m: np.zeros((n, dims[m]), dtype=np.float32) for m in MODULES}
    M = np.zeros((n, len(MODULES)), dtype=np.float32)
    for r, i in enumerate(ids):
        for c, m in enumerate(MODULES):
            rows, keep = trans[m]
            x = rows[id2row[i]]
            if x is not None:
                X[m][r, :x.shape[0]] = x
                M[r, c] = 1.0
    return X, M, dims


_FOLD_CACHE = {}


def get_fold_arrays(fold, ids, label, feats, id2row):
    if fold not in _FOLD_CACHE:
        train_ids = [i for i in ids if i != ids[fold]]
        trans = fold_preproc(feats, ids, train_ids)
        _FOLD_CACHE[fold] = build_arrays(trans, ids, id2row)
    return _FOLD_CACHE[fold]


def run_method(variant, ids, label, feats, device, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n = len(ids)
    y = np.array([label[i] for i in ids])
    id2row = {i: r for r, i in enumerate(ids)}
    probs = np.zeros((n, K))
    for fold in range(n):
        te_id = ids[fold]
        X, M, dims = get_fold_arrays(fold, ids, label, feats, id2row)
        Xt = {m: torch.tensor(X[m], device=device) for m in MODULES}
        Mt = torch.tensor(M, device=device)
        yt = torch.tensor(y, device=device)
        tr = torch.tensor([r for r in range(n) if r != fold], device=device)
        te = torch.tensor([fold], device=device)

        if variant == "early_fusion":
            dim = int(Mt.sum(1).max() * 0) + sum(dims.values())
            feat = torch.cat([Xt[m] for m in MODULES], dim=1)
            feat = torch.cat([feat, Mt], dim=1)
            layers = [feat.shape[1], 128, 64, K]
            model = nn.Sequential(
                nn.Linear(layers[0], layers[1]), nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(layers[1], layers[2]), nn.ReLU(),
                nn.Linear(layers[2], layers[3])).to(device)

            def forward(idx, train_mode):
                return model(feat[idx])
        else:
            enc = {}
            for m in MODULES:
                enc[m] = nn.Sequential(
                    nn.Linear(dims[m], EMB), nn.ReLU()).to(device)
            if variant == "tmc":
                ev = {}
                for m in MODULES:
                    ev[m] = nn.Sequential(
                        nn.Linear(EMB, EMB), nn.ReLU(),
                        nn.Linear(EMB, K), nn.Softplus()).to(device)
                head = None
            else:
                head = nn.Sequential(
                    nn.Linear(EMB + len(MODULES), 24), nn.ReLU(),
                    nn.Linear(24, K)).to(device)

            def forward(idx, train_mode):
                hs, ms = [], []
                for c, m in enumerate(MODULES):
                    x = Xt[m][idx]
                    mask = Mt[idx][:, c:c + 1]
                    if variant == "moddrop" and train_mode:
                        keep = (torch.rand_like(mask) > 0.3).float()
                        mask = mask * keep
                    hs.append(enc[m](x) * mask)
                    ms.append(mask)
                H = torch.stack(hs, 0).sum(0)
                MS = torch.stack(ms, 0).sum(0).clamp(min=1.0)
                Hm = H / MS
                if variant == "tmc":
                    alpha = 1.0
                    for c, m in enumerate(MODULES):
                        e = ev[m](Hm) + 1.0
                        alpha = alpha + e * ms[c]
                    p = alpha / alpha.sum(1, keepdim=True)
                    return torch.log(p + 1e-8)
                z = torch.cat([Hm, Mt[idx]], dim=1)
                return head(z)

        if variant == "early_fusion":
            module_list = [model]
        elif variant == "tmc":
            module_list = list(enc.values()) + [ev[m] for m in MODULES]
        else:
            module_list = list(enc.values()) + [head]
        params = []
        for mod in module_list:
            params += list(mod.parameters())
        opt = torch.optim.Adam(params, lr=LR, weight_decay=WD)
        lossf = nn.CrossEntropyLoss()
        for ep in range(EPOCHS):
            for mod in module_list:
                mod.train()
            opt.zero_grad()
            logits = forward(tr, True)
            loss = lossf(logits, yt[tr])
            loss.backward()
            opt.step()
        for mod in module_list:
            mod.eval()
        with torch.no_grad():
            logits = forward(te, False)
            probs[fold] = torch.softmax(logits, 1)[0].cpu().numpy()
        if (fold + 1) % 25 == 0:
            print(f"   [{variant} s{seed}] fold {fold+1}/{n}", flush=True)
    return probs


def main():
    import os
    import torch
    global EPOCHS, K, CLASSES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[comp] device={device}", flush=True)
    ids, label, feats = load_data()
    task = os.environ.get("COMP_TASK", "multi")
    if task == "binary":
        ids = [i for i in ids if label[i] in (0, 2)]
        label = {i: (0 if label[i] == 0 else 1) for i in ids}
        K = 2
        CLASSES = ["HC", "PDD"]
        print(f"[comp] BINARY task: HC vs PDD, n={len(ids)}", flush=True)
    lim = int(os.environ.get("COMP_FOLDS", "0"))
    if lim:
        ids = ids[:lim]
        print(f"[comp] SMOKE MODE: {lim} subjects", flush=True)
    EPOCHS = int(os.environ.get("COMP_EPOCHS", str(EPOCHS)))
    print(f"[comp] n={len(ids)} epochs={EPOCHS} K={K}", flush=True)
    y = np.array([label[i] for i in ids])
    methods = ["early_fusion", "heimis", "moddrop", "tmc"]
    rows = []
    all_preds = {}
    for variant in methods:
        P = np.zeros((len(ids), K))
        for s in range(SEEDS):
            P += run_method(variant, ids, label, feats, device, 1000 + s)
            print(f"[comp] {variant} seed {s} done", flush=True)
        P /= SEEDS
        all_preds[variant] = P
        row = {"method": variant, "macro_auc": macro_auc(y, P)}
        from sklearn.metrics import roc_auc_score
        for c, cn in enumerate(CLASSES):
            row[f"auc_{cn}"] = float(roc_auc_score((y == c).astype(int),
                                                   P[:, c]))
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"competitors_loo_results_{task}.tsv",
                                  sep="\t", index=False)
        print(f"[comp] {variant}: macro={row['macro_auc']:.3f}", flush=True)
    np.savez_compressed(OUT / f"competitors_loo_predictions_{task}.npz",
                        ids=np.array(ids), y=y, **all_preds)
    print("[comp] DONE", flush=True)


if __name__ == "__main__":
    main()
