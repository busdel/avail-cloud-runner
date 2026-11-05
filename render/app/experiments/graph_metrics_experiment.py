# -*- coding: utf-8 -*-
"""Experiment (b): graph-theory features of the fMRI network summary.

Builds one weighted graph per subject from fc_within/&between_network_summary
(19 subnetwork nodes, 172 weighted edges) and derives standard graph metrics
(strength, clustering, efficiency, modularity, participation, small-world
sigma...). Then compares LOO Random-Forest classification:

  A) static selected features (baseline)
  B) graph metrics only
  C) combined

Outputs: experiments/results/graph_metrics_results.tsv + *_features.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
OUT.mkdir(parents=True, exist_ok=True)

W_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_within_network_summary.csv"
B_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_between_network_summary.csv"
Y_F = ROOT / "fMRI correlation and feature selection/y_data.xlsx"
S_F = ROOT / "fMRI correlation and feature selection/selected_feature_data.xlsx"


def load_labels():
    y = pd.read_excel(Y_F)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    gmap = {"control": "HC", "mci": "MCI", "dementia": "PDD"}
    y["cls"] = y["Group"].map(gmap)
    return dict(zip(y["PatientID"], y["cls"]))


def graph_metrics(row_w, row_b):
    G = nx.Graph()
    for col, v in row_w.items():
        if col == "patient" or pd.isna(v):
            continue
        node = col.replace("within_", "")
        G.add_node(node, within=float(v))
    for col, v in row_b.items():
        if col == "patient" or pd.isna(v):
            continue
        a, b = col.split("__")
        if not G.has_node(a):
            G.add_node(a, within=0.0)
        if not G.has_node(b):
            G.add_node(b, within=0.0)
        G.add_edge(a, b, weight=abs(float(v)))
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n < 3 or m < 1:
        return None

    # weighted metrics on |w|
    w_adj = nx.to_numpy_array(G, weight="weight")
    np.fill_diagonal(w_adj, [abs(G.nodes[k].get("within", 0.0)) for k in G.nodes()])
    A = w_adj
    A = A / (A.max() + 1e-9)
    strengths = A.sum(axis=1)
    deg = (A > 1e-6).sum(axis=1).astype(float)
    k_avg = deg.mean()
    density = m / (n * (n - 1) / 2)

    clust = []
    for i in range(n):
        nb = np.where(A[i] > 1e-6)[0]
        if len(nb) < 2:
            clust.append(0.0)
            continue
        tri = 0.0
        for j in nb:
            for k in nb:
                if j < k:
                    tri += (A[i, j] * A[i, k] * A[j, k]) ** (1 / 3)
        clust.append(tri / (len(nb) * (len(nb) - 1) / 2))
    clust = np.array(clust)
    C = clust.mean()

    # efficiency on distance graph
    D = 1.0 / (A + 1e-4)
    np.fill_diagonal(D, 0)
    eff = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                d = nx.dijkstra_path_length(
                    nx.from_numpy_array(D), i, j, weight="weight")
                eff.append(1.0 / max(d, 1e-6))
            except Exception:
                pass
    Eglob = np.mean(eff) if eff else 0.0

    # modularity (greedy) + participation
    Q = 0.0
    part = np.zeros(n)
    try:
        comms = nx.algorithms.community.greedy_modularity_communities(
            G, weight="weight")
        Q = nx.algorithms.community.modularity(G, comms, weight="weight")
        memb = {}
        for ci, cset in enumerate(comms):
            for node in cset:
                memb[node] = ci
        for i, node in enumerate(G.nodes()):
            ki = strengths[i]
            if ki <= 1e-9:
                continue
            s = 0.0
            for c in set(memb.values()):
                kis = strengths[[j for j, nd in enumerate(G.nodes())
                                 if memb[nd] == c]].sum()
                s += (kis / ki) ** 2
            part[i] = 1.0 - s
    except Exception:
        pass

    # small-world sigma (random-graph approximation)
    L = None
    try:
        L = nx.average_shortest_path_length(
            nx.from_numpy_array(D), weight="weight")
    except Exception:
        L = np.log(n) / np.log(k_avg + 1)
    Lr = np.log(n) / np.log(k_avg + 1e-9)
    Cr = k_avg / n
    sigma = (C / max(Cr, 1e-9)) / (max(L, 1e-9) / max(Lr, 1e-9))

    return {
        "strength_mean": strengths.mean(),
        "strength_sd": strengths.std(),
        "clust_mean": C,
        "clust_sd": clust.std(),
        "efficiency": Eglob,
        "modularity": Q,
        "participation_mean": part.mean(),
        "participation_max": part.max(),
        "smallworld_sigma": sigma,
        "density": density,
        "mean_weight": A.mean(),
        "n_communities": len(comms) if 'comms' in dir() else np.nan,
    }


def main():
    w = pd.read_csv(W_F)
    b = pd.read_csv(B_F)
    labels = load_labels()
    w["patient"] = w["patient"].astype(str).str.strip()
    b["patient"] = b["patient"].astype(str).str.strip()
    common = sorted(set(w.patient) & set(b.patient) & set(labels))
    common = [p for p in common if labels[p] in ("HC", "MCI", "PDD")]
    print(f"subjects with labels: {len(common)}")

    feats = {}
    for p in common:
        rw = w[w.patient == p].iloc[0]
        rb = b[b.patient == p].iloc[0]
        m = graph_metrics(rw, rb)
        if m:
            feats[p] = m
    cols = sorted(next(iter(feats.values())).keys())
    Xg = np.vstack([[feats[p][c] for c in cols] for p in common])
    y = np.array([["HC", "MCI", "PDD"].index(labels[p]) for p in common])
    gdf = pd.DataFrame(Xg, columns=cols)
    gdf.insert(0, "patient", common)
    gdf.to_csv(OUT / "graph_metrics_features.tsv", sep="\t", index=False)
    print(f"graph features: {Xg.shape}")

    # static baseline
    sf = pd.read_excel(S_F)
    sf["PatientID"] = sf["PatientID"].astype(str).str.strip()
    sf = sf[sf.PatientID.isin(common)].set_index("PatientID").loc[common]
    Xs = sf.values.astype(float)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    def macro_auc(yv, P):
        K = P.shape[1]
        aucs = []
        for c in range(K):
            yc = (yv == c).astype(int)
            if len(np.unique(yc)) == 2:
                aucs.append(roc_auc_score(yc, P[:, c]))
        return float(np.mean(aucs))

    def loo_auc(X):
        P = np.zeros((len(y), 3))
        for k in range(len(y)):
            tr = [t for t in range(len(y)) if t != k]
            rf = RandomForestClassifier(n_estimators=300, random_state=42,
                                        n_jobs=-1)
            rf.fit(X[tr], y[tr])
            P[k] = rf.predict_proba(X[k:k + 1])[0]
        return macro_auc(y, P)

    auc_static = loo_auc(Xs)
    auc_graph = loo_auc(Xg)
    auc_comb = loo_auc(np.hstack([Xs, Xg]))
    rows = [{"features": "static_selected", "macro_auc": auc_static},
            {"features": "graph_metrics", "macro_auc": auc_graph},
            {"features": "combined", "macro_auc": auc_comb}]
    pd.DataFrame(rows).to_csv(OUT / "graph_metrics_results.tsv",
                              sep="\t", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
