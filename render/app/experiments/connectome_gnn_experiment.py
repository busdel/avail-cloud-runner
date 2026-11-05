# -*- coding: utf-8 -*-
"""Experiment (a): connectome GCN on the fMRI network summary.

Nodes = 36 subnetworks (from fc_within/between summaries), edges = between-
network |FC|, self weights = within-network FC. Node features = coarse
network one-hot + degree + strength. A 2-layer GCN with mean readout is
trained leave-one-out (HC/MCI/PDD) and compared against the static RF baseline.

Outputs: experiments/results/connectome_gcn_results.tsv + loo predictions.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
W_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_within_network_summary.csv"
B_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_between_network_summary.csv"
Y_F = ROOT / "fMRI correlation and feature selection/y_data.xlsx"
S_F = ROOT / "fMRI correlation and feature selection/selected_feature_data.xlsx"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COARSE = {
    "Default": 0, "Dorsal Attention": 1, "Frontoparietal Control": 2,
    "Limbic": 3, "Salience Ventral Attention": 4, "Somatomotor": 5,
    "Temporoparietal": 6, "Visual": 7,
}


def coarse_of(name):
    for k in COARSE:
        if name.startswith(k):
            return COARSE[k]
    return len(COARSE)


class GCN(nn.Module):
    def __init__(self, in_f, h, K):
        super().__init__()
        self.w1 = nn.Linear(in_f, h)
        self.w2 = nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(h, 24), nn.ReLU(), nn.Linear(24, K))

    def forward(self, A, X):
        if A.dim() == 2:
            A = A.unsqueeze(0)
            X = X.unsqueeze(0)
        H = F.relu(self.w1(torch.bmm(A, X)))
        H = F.relu(self.w2(torch.bmm(A, H)))
        return self.head(H.mean(dim=1))


def build_graphs():
    w = pd.read_csv(W_F)
    b = pd.read_csv(B_F)
    y = pd.read_excel(Y_F)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    gmap = {"control": "HC", "mci": "MCI", "dementia": "PDD"}
    y["cls"] = y["Group"].map(gmap)
    labels = dict(zip(y["PatientID"], y["cls"]))
    w["patient"] = w["patient"].astype(str).str.strip()
    b["patient"] = b["patient"].astype(str).str.strip()
    common = sorted(set(w.patient) & set(b.patient) & set(labels))
    common = [p for p in common if labels[p] in ("HC", "MCI", "PDD")]

    wnodes = [c.replace("within_", "") for c in w.columns[1:]
    if not w[c].isna().all()]
    nodes = sorted(set(wnodes) | {c.split("__")[0] for c in b.columns[1:]}
                   | {c.split("__")[1] for c in b.columns[1:]})
    nid = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    graphs = []
    for p in common:
        A = np.zeros((n, n), dtype=np.float32)
        rw = w[w.patient == p].iloc[0]
        for c in w.columns[1:]:
            v = rw[c]
            if pd.isna(v):
                continue
            node = c.replace("within_", "")
            A[nid[node], nid[node]] = abs(float(v))
        rb = b[b.patient == p].iloc[0]
        for c in b.columns[1:]:
            v = rb[c]
            if pd.isna(v):
                continue
            a, b_ = c.split("__")
            A[nid[a], nid[b_]] = A[nid[b_], nid[a]] = abs(float(v))
        # normalize adjacency (GCN-style)
        deg = A.sum(axis=1) + 1e-6
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        An = D_inv_sqrt @ A @ D_inv_sqrt
        # node features: coarse one-hot + degree + strength
        X = np.zeros((n, len(COARSE) + 2), dtype=np.float32)
        for node in nodes:
            i = nid[node]
            X[i, coarse_of(node)] = 1.0
            X[i, len(COARSE)] = deg[i]
            X[i, len(COARSE) + 1] = A[i].sum()
        graphs.append((torch.tensor(An).to(DEV), torch.tensor(X).to(DEV)))
    yv = torch.tensor([["HC", "MCI", "PDD"].index(labels[p]) for p in common])
    return graphs, yv, common


def train_one(tr, model, opt, epochs=300, patience=30):
    best = np.inf
    bad = 0
    best_state = None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        outs = [model(graphs[i][0], graphs[i][1]) for i in tr]
        logits = torch.stack(outs).squeeze(1)
        loss = F.cross_entropy(logits, yv[tr].to(DEV))
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
    global graphs, yv
    graphs, yv, common = build_graphs()
    n = len(yv)
    print(f"subjects: {n}, nodes: {graphs[0][0].shape[0]}, "
          f"device: {DEV}")

    from sklearn.metrics import roc_auc_score

    def macro_auc(yv_np, P):
        aucs = []
        for c in range(3):
            yc = (yv_np == c).astype(int)
            if len(np.unique(yc)) == 2:
                aucs.append(roc_auc_score(yc, P[:, c]))
        return float(np.mean(aucs))

    P = np.zeros((n, 3))
    for k in range(n):
        tr = [t for t in range(n) if t != k]
        seeds = []
        for s in range(5):
            torch.manual_seed(42 + s)
            np.random.seed(42 + s)
            model = GCN(len(COARSE) + 2, 32, 3).to(DEV)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
            model = train_one(tr, model, opt)
            model.eval()
            with torch.no_grad():
                seeds.append(F.softmax(model(graphs[k][0], graphs[k][1]), dim=-1)
                             .cpu().numpy()[0])
        P[k] = np.mean(seeds, axis=0)
        if (k + 1) % 20 == 0:
            print(f"  LOO {k+1}/{n}")
    auc_gcn = macro_auc(yv.numpy(), P)
    pred = pd.DataFrame({"patient": common, "y": yv.numpy()})
    for c, cls in enumerate(["HC", "MCI", "PDD"]):
        pred[f"GCN_{cls}"] = P[:, c]
    pred.to_csv(OUT / "connectome_gcn_predictions.tsv", sep="\t", index=False)

    # static RF baseline (same subjects)
    sf = pd.read_excel(S_F)
    sf["PatientID"] = sf["PatientID"].astype(str).str.strip()
    sf = sf[sf.PatientID.isin(common)].set_index("PatientID").loc[common]
    Xs = sf.values.astype(float)
    from sklearn.ensemble import RandomForestClassifier
    Prf = np.zeros((n, 3))
    for k in range(n):
        tr = [t for t in range(n) if t != k]
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(Xs[tr], yv.numpy()[tr])
        Prf[k] = rf.predict_proba(Xs[k:k + 1])[0]
    auc_rf = macro_auc(yv.numpy(), Prf)

    rows = [{"model": "static_RF_baseline", "macro_auc": auc_rf},
            {"model": "connectome_GCN", "macro_auc": auc_gcn}]
    pd.DataFrame(rows).to_csv(OUT / "connectome_gcn_results.tsv",
                              sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
