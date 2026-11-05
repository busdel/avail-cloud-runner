# -*- coding: utf-8 -*-
"""Graph-MoE-inspired baseline (Taming arbitrary modality missingness and
imbalance, AD diagnosis) re-implemented for our cohort.

  - per-modality expert MLPs on log(x+1) raw features
  - patient kNN graph built in the complete meta-feature space
  - graph convolution lets missing-modality subjects borrow signal
  - availability-aware MoE gating + class-weighted loss (imbalance)

Same LOO protocol, same n=150 zero-loss cohort, 3 seeds.
Outputs: experiments/results/graph_moe_baseline_results.tsv + predictions
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load():
    meta = pd.read_csv(ROOT / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    classes = ["HC", "MCI", "PDD"]
    m = meta[meta.Group.isin(classes)]
    ids = m["SampleID"].tolist()
    y = np.array([classes.index(g) for g in m["Group"]])
    K = 3
    mods = []
    for f in sorted((ROOT / "features").iterdir()):
        d = pd.read_csv(f, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d[d.SampleID.isin(ids)].set_index("SampleID").reindex(ids)
        X = np.log(d.values.astype(float) + 1.0)
        mask = d.notna().values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)
        mods.append((f.stem, X, mask))
    return ids, y, K, mods


class Expert(nn.Module):
    def __init__(self, d_in, d_h, K):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(d_h, K))

    def forward(self, x):
        return self.net(x)


class GraphMoE(nn.Module):
    def __init__(self, n_mods, dims, d_h=64, K=3, knn=8):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d, d_h, K) for d in dims])
        self.gate = nn.Sequential(nn.Linear(n_mods * K + n_mods, 64),
                                  nn.ReLU(), nn.Linear(64, n_mods))
        self.graph = nn.Linear(n_mods * K, n_mods * K, bias=False)
        self.knn = knn
        self.n_mods = n_mods
        self.K = K

    def forward(self, Xs, masks, A):
        # per-modality expert logits (missing -> 0 vector)
        logits = []
        Lmask = []
        for e, x, m in zip(self.experts, Xs, masks):
            logits.append(e(x))
            # subject-level availability = any observed feature in the layer
            Lmask.append((m.sum(dim=1) > 0).float())
        L = torch.stack(logits, dim=1)          # B x M x K
        Lmask = torch.stack(Lmask, dim=1)       # B x M
        L = L * Lmask.unsqueeze(-1)
        P = F.softmax(L, dim=-1)                # B x M x K
        pvec = P.reshape(P.shape[0], -1)        # B x (M*K)
        # graph refinement: borrow from neighbours in probability space
        pgraph = F.relu(self.graph(pvec))
        pgraph = A @ pgraph                       # B x (M*K) (row-norm A)
        pgraph = pgraph.reshape(P.shape[0], self.n_mods, self.K)
        # gating (availability-aware)
        g = self.gate(torch.cat([pvec, Lmask], dim=-1))
        g = torch.softmax(g, dim=-1) * Lmask
        g = g / g.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = (g.unsqueeze(-1) * (P + pgraph)).sum(dim=1)  # B x K
        return out


def kNN_graph(P):
    """Row-normalized kNN adjacency from meta-feature probability vectors."""
    from sklearn.neighbors import NearestNeighbors
    nn_ = NearestNeighbors(n_neighbors=9)
    nn_.fit(P)
    _, idx = nn_.kneighbors(P)
    n = P.shape[0]
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in idx[i, 1:]:
            A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1, keepdims=True) + 1e-6
    A = A / deg
    return A


def train_model(model, Xs, masks, A, y, K, epochs=300, patience=30, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = [torch.tensor(x).to(DEV) for x in Xs]
    Mt = [torch.tensor(m).to(DEV) for m in masks]
    At = torch.tensor(A).to(DEV)
    yt = torch.tensor(y).to(DEV)
    counts = np.bincount(y, minlength=K).astype(float)
    w = torch.tensor(len(y) / (K * counts), dtype=torch.float32).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=w)
    best = np.inf
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(Xt, Mt, At), yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            vl = loss.item()
        if vl < best - 1e-5:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def main():
    ids, y, K, mods = load()
    n = len(y)
    print(f"n={n}, modalities={[m[0] for m in mods]}, device={DEV}")

    # base predictions for graph space (LOO RF per modality, leak-free)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    piv = {}
    mod_names = []
    for mname, X, mask in mods:
        mod_names.append(mname)
        sub = np.where(mask.mean(axis=1) > 0)[0]
        sids = [ids[i] for i in sub]
        if len(sids) < 10:
            continue
        Xs_ = X[sub]
        ys_ = y[sub]
        P = np.zeros((len(sids), K))
        for k in range(len(sids)):
            tr = [t for t in range(len(sids)) if t != k]
            rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                        n_jobs=6)
            rf.fit(Xs_[tr], ys_[tr])
            P[k] = rf.predict_proba(Xs_[k:k + 1])[0]
        for p, pv in zip(sids, P):
            piv.setdefault(p, {})[mname] = pv
    Pmat = np.zeros((n, len(mods) * K))
    Fmat = np.zeros((n, len(mods)))
    for i, p in enumerate(ids):
        for mi, mname in enumerate(mod_names):
            if p in piv and mname in piv[p]:
                Pmat[i, mi * K:(mi + 1) * K] = piv[p][mname]
    for mi, (mname, X, mask) in enumerate(mods):
        Fmat[:, mi] = mask.mean(axis=1)
    A = kNN_graph(np.hstack([Pmat, Fmat]))

    Xs_all = [torch.tensor(m[1]).to(DEV) for m in mods]
    Masks = [torch.tensor(m[2]).to(DEV) for m in mods]
    At = torch.tensor(A).to(DEV)
    yt = torch.tensor(y).to(DEV)

    P = np.zeros((n, K))
    for k in range(n):
        tr = [t for t in range(n) if t != k]
        Xtr = [x[tr] for x in Xs_all]
        Mtr = [m[tr] for m in Masks]
        Atr = A[np.ix_(tr, tr)]
        Atr = Atr / Atr.sum(axis=1, keepdims=True).clip(min=1e-9)
        seeds = []
        for s in range(3):
            model = GraphMoE(len(mods), [m[1].shape[1] for m in mods],
                             K=K).to(DEV)
            model = train_model(model, Xtr, Mtr, Atr, y[tr], K, seed=s)
            model.eval()
            with torch.no_grad():
                out = model([x[k:k + 1] for x in Xs_all],
                            [m[k:k + 1] for m in Masks],
                            torch.tensor(A[k:k + 1]).to(DEV))
                seeds.append(F.softmax(out, dim=-1).cpu().numpy()[0])
        P[k] = np.mean(seeds, axis=0)
        if (k + 1) % 30 == 0:
            print(f"  LOO {k + 1}/{n}", flush=True)

    aucs = []
    for c in range(K):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    auc = float(np.mean(aucs))
    print(f"graph-MoE macro-AUC = {auc:.3f}")
    pred = pd.DataFrame({"patient": ids, "y": y})
    for c, cls in enumerate(["HC", "MCI", "PDD"]):
        pred[f"graphMoE_{cls}"] = P[:, c]
    pred.to_csv(OUT / "graph_moe_predictions.tsv", sep="\t", index=False)
    with open(OUT / "graph_moe_baseline_results.tsv", "w") as f:
        f.write("model\tmacro_auc\tn\n")
        f.write(f"graph_moe_inspired\t{auc:.6f}\t{n}\n")
    print("done")


if __name__ == "__main__":
    main()
