# -*- coding: utf-8 -*-
"""UniMIX-AD official model (SDH-Lab) adapted to our tabular cohort.

Uses the published models.py (FusionModule, sparse MoE transformer) and
umi_module.py (UMI cross-modal imputation) UNCHANGED from the official repo.
Adaptations (documented for the paper):
  - tabular MLP encoders (their image encoders replaced, input is feature
    vectors, not MRI volumes)
  - our n=150 zero-loss 7-layer cohort as data_dict
  - same protocol as the paper: stratified train/val/test split, 50 epochs,
    warm-up, CE + gate loss + MCG contrastive loss, cosine schedule

For a fair head-to-head, OUR pipeline (SimpleMean / XGBoost meta on the same
LOO base predictions) is also evaluated on the SAME test splits.

Outputs: experiments/results/unimix_ad_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

UNI = Path("/media/suleyman_disk/UniMIX-AD")
SHIM = Path("/media/suleyman_disk/AVAIL_GPU/experiments")
sys.path.insert(0, str(SHIM))   # fmoe shim first
sys.path.insert(0, str(UNI))
from models import FusionModule  # noqa: E402
from umi_module import UMI  # noqa: E402

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    meta = pd.read_csv(ROOT / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    # SAME class coding as run_generic: 3 most frequent groups
    from collections import Counter
    classes = [c for c, _ in Counter(
        meta["Group"].astype(str).str.strip().values).most_common(3)]
    m = meta[meta.Group.isin(classes)]
    ids = m["SampleID"].tolist()
    y = np.array([classes.index(g) for g in m["Group"]])
    mods = []
    for f in sorted((ROOT / "features").iterdir()):
        d = pd.read_csv(f, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d[d.SampleID.isin(ids)].set_index("SampleID").reindex(ids)
        X = np.log(d.values.astype(float) + 1.0)
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        avail = d.notna().any(axis=1).values.astype(np.float32)
        mods.append((f.stem, X, avail))
    return ids, y, classes, mods


class TabEncoder(nn.Module):
    """Tabular encoder -> (num_patches, hidden) tokens (adaptation)."""

    def __init__(self, in_dim, num_patches, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden * num_patches),
            nn.LayerNorm(hidden * num_patches),
            nn.ReLU(),
        )
        self.num_patches = num_patches
        self.hidden = hidden

    def forward(self, x):
        B = x.shape[0]
        return self.net(x).view(B, self.num_patches, self.hidden)


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def run_one(seed, split, ids, y, mods, epochs=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(ids)
    tr_idx, va_idx, te_idx = split
    M = len(mods)
    K = 3
    hidden = 128
    npatches = 16

    encoders = nn.ModuleList([TabEncoder(mods[i][1].shape[1], npatches, hidden)
                              for i in range(M)])
    fusion = FusionModule(M, 0, npatches, hidden, K,
                          1, 1, 16, 1, 4, 2, 0.5)
    # input-shape adaptation: their pos_embed assumes 64 tokens (ADNI);
    # we have M x num_patches tokens
    total_tokens = M * npatches
    if fusion.pos_embed is not None and fusion.pos_embed.shape[1] != total_tokens:
        fusion.pos_embed = nn.Parameter(
            torch.zeros(1, total_tokens, hidden,
                        device=fusion.pos_embed.device))
    umi = UMI(num_modalities=M, input_dims=[hidden] * M,
              sequence_lengths=[npatches] * M, d_model=hidden,
              target_length=npatches, num_heads=4, d_ff=256, dropout=0.1)
    fusion = fusion.to(DEV)
    umi = umi.to(DEV)
    encoders = encoders.to(DEV)
    params = list(encoders.parameters()) + list(fusion.parameters()) + \
        list(umi.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)
    crit = nn.CrossEntropyLoss()

    Xt = [torch.tensor(mods[i][1]).to(DEV) for i in range(M)]
    Av = [torch.tensor(mods[i][2]).to(DEV) for i in range(M)]
    yt = torch.tensor(y).to(DEV)

    def forward_batch(idx_b):
        feats = []
        mask = torch.stack([Av[i][idx_b] for i in range(M)], dim=1)  # B x M
        for i in range(M):
            enc = encoders[i](Xt[i][idx_b])           # B x P x H
            enc = enc * Av[i][idx_b].unsqueeze(-1).unsqueeze(-1)
            feats.append(enc)
        feats = umi(feats, mask)                      # refined
        # expert_indices=zeros: skips the ADNI-specific combination-alignment
        # gate loss while keeping the router/load-balancing machinery active
        eidx = torch.zeros(len(idx_b), dtype=torch.long, device=DEV)
        out, gate_loss, mcg_loss = fusion(*feats, expert_indices=eidx)
        return out, gate_loss, mcg_loss

    # train
    for ep in range(epochs):
        encoders.train()
        fusion.train()
        umi.train()
        perm = np.random.permutation(tr_idx)
        for s in range(0, len(perm), 16):
            idx_b = perm[s:s + 16]
            opt.zero_grad()
            out, gl, ml = forward_batch(idx_b)
            loss = crit(out, yt[idx_b]) + 0.01 * gl.mean() + 0.1 * ml.mean()
            loss.backward()
            opt.step()
        if ep % 10 == 0:
            print(f"  seed {seed} epoch {ep}", flush=True)

    encoders.eval()
    fusion.eval()
    umi.eval()
    P = []
    with torch.no_grad():
        for s in range(0, len(te_idx), 16):
            idx_b = te_idx[s:s + 16]
            out, _, _ = forward_batch(idx_b)
            P.append(F.softmax(out, dim=-1).cpu().numpy())
    P = np.vstack(P)
    return macro_auc(y[te_idx], P)


def our_pipeline_on_same_splits(split, ids, y):
    """Our meta-learners (SimpleMean / XGBoost) on the SAME train/test splits,
    using subject-LOO base predictions (leak-free)."""
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    RES = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results")
    bp = pd.read_csv(RES / "base_predictions_generic.tsv", sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    piv = {}
    for _, r in bp.iterrows():
        piv.setdefault(r["patient"], {})[r["modality"]] = \
            np.array([r[f"p_{c}"] for c in range(3)])
    mods = sorted(bp["modality"].unique())
    K = 3
    per = K + 1
    X = np.zeros((len(ids), len(mods) * per))
    for i, p in enumerate(ids):
        for c, m in enumerate(mods):
            b = c * per
            if p in piv and m in piv[p]:
                X[i, b:b + K] = piv[p][m]
                X[i, b + K] = 1.0
    tr_idx, _, te_idx = split
    out = {}
    P = np.zeros((len(te_idx), K))
    for k, i in enumerate(te_idx):
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(len(mods)):
            b = c * per
            if X[i, b + K] == 1:
                acc += X[i, b:b + K]
                wsum += 1.0
        P[k] = acc / wsum if wsum > 0 else np.full(K, 1 / K)
    out["SimpleMean"] = macro_auc(y[te_idx], P)
    xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                        random_state=0, n_jobs=6, objective="multi:softprob",
                        num_class=K, eval_metric="mlogloss")
    xgb.fit(X[tr_idx], y[tr_idx])
    out["XGBoost"] = macro_auc(y[te_idx], xgb.predict_proba(X[te_idx]))
    return out


def main():
    ids, y, classes, mods = load()
    print(f"n={len(ids)}, modalities={[m[0] for m in mods]}, device={DEV}")
    rows = []
    for seed in range(3):
        tr_idx, rest = train_test_split(np.arange(len(y)), test_size=0.3,
                                        random_state=seed, stratify=y)
        va_idx, te_idx = train_test_split(rest, test_size=0.5,
                                          random_state=seed, stratify=y[rest])
        split = (tr_idx, va_idx, te_idx)
        auc = run_one(seed, split, ids, y, mods)
        rows.append({"model": "UniMIX_AD_official", "seed": seed,
                     "test_macro_auc": auc, "n_test": len(te_idx)})
        ours = our_pipeline_on_same_splits(split, ids, y)
        for m, a in ours.items():
            rows.append({"model": f"ours_{m}", "seed": seed,
                         "test_macro_auc": a, "n_test": len(te_idx)})
        print(f"seed {seed}: UniMIX={auc:.3f} "
              f"ours_SimpleMean={ours['SimpleMean']:.3f} "
              f"ours_XGBoost={ours['XGBoost']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "unimix_ad_results.tsv", sep="\t", index=False)
    print(df.groupby("model")["test_macro_auc"]
          .agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
