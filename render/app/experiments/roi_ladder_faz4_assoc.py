#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 4: ROI x cognition association analyses (83 fMRI patients).

1. ROI-level mass-univariate: mean edge strength per ROI x cognitive score
   (partial Spearman, age-adjusted) with BH-FDR across 100 ROIs.
2. Edge-level: 4950 edges x cognition, age-adjusted partial Spearman, BH-FDR.
3. Network x network: 19x19 block summaries (mean rho and fraction of
   significant edges per block).
4. NBS-style: threshold rho (|rho|>=0.35), connected components, 1000
   permutations for component-size null.
5. Behavioral PLS: edges x cognitive battery, SVD, 500 permutations of
   explained covariance.

Outputs: experiments/results/roi_ladder/assoc/
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx

ROOT = Path("/mnt/limbic/AVAIL_GPU")
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/roi_ladder/assoc"
OUT.mkdir(parents=True, exist_ok=True)

Y_F = BENCH / "fMRI correlation and feature selection/y_data.xlsx"
META_F = BENCH / "busra_pd_multisite/metadata.txt"
MAP_F = BENCH / "yeni_eklenenler/paper_pipeline/metadata_fmr/roi_network_map.csv"
ROI_XLSX = Path("/home/suleyman/Desktop/fMRI datasets/"
                "partial_corr_reshaped_Spe21-23.xlsx")
CANON = "/tmp/opencode/canonical_labels_100.npy"

COGNITIVE = ["MMSE", "Categorical_Fluency_", "Phonemic_Fluency",
             "OVMPT_Coding", "OVMPT_Recalling", "DS_Forward", "DS_Backward",
             "Benton_Face_Recognition_Test", "Clock_Drawing_Test", "CDRS"]
RNG_SEED = 42


def load_roi():
    y = pd.read_excel(Y_F).sort_values("Sort Number").reset_index(drop=True)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    xl = pd.ExcelFile(ROI_XLSX)
    tri = np.triu_indices(100, k=1)
    M = np.stack([xl.parse(s, header=None).values.astype(float)[tri]
                  for s in xl.sheet_names])
    return M, y


def load_cog(patients):
    meta = pd.read_csv(META_F, sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    meta = meta.set_index("SampleID").loc[patients]
    age = meta["Age"].astype(float).values
    cog = {}
    for c in COGNITIVE:
        v = pd.to_numeric(meta[c], errors="coerce").values.astype(float)
        cog[c] = v
    return age, cog


def partial_spearman(x, y, z):
    """Partial Spearman of x|y given z (rank-based, via residuals)."""
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    rx = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ry = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r, p = stats.spearmanr(rx, ry)
    return r, p


def vec_partial_spearman(X, y, z):
    """Vectorized partial Spearman: X (n,p), y (n,), z (n,) -> rho (p,)."""
    rx = np.apply_along_axis(stats.rankdata, 0, X)
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    b = np.polyfit(rz, rx, 1)
    rx = rx - (b[0][None, :] * rz[:, None] + b[1][None, :])
    by = np.polyfit(rz, ry, 1)
    ry = ry - (by[0] * rz + by[1])
    rx = rx - rx.mean(axis=0, keepdims=True)
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum()) + 1e-12
    return (rx * ry[:, None]).sum(axis=0) / denom


def precompute_residuals(X, z):
    """Rank + age-residualize the columns of X once."""
    rx = np.apply_along_axis(stats.rankdata, 0, X)
    rz = stats.rankdata(z)
    b = np.polyfit(rz, rx, 1)
    rx = rx - (b[0][None, :] * rz[:, None] + b[1][None, :])
    rx = rx - rx.mean(axis=0, keepdims=True)
    return rx


def corr_with_perm(rx, y, z):
    """rho of rx columns vs y ranks (age-residualized) -> (p,)."""
    ry = stats.rankdata(y)
    rz = stats.rankdata(z)
    by = np.polyfit(rz, ry, 1)
    ry = ry - (by[0] * rz + by[1])
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum(axis=0) * (ry ** 2).sum()) + 1e-12
    return (rx * ry[:, None]).sum(axis=0) / denom


def p_from_rho(rho, n):
    """Two-sided p for Spearman rho (t-approx)."""
    t = rho * np.sqrt((n - 2) / np.clip(1 - rho ** 2, 1e-12, None))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def main():
    M, y = load_roi()
    n, n_edges = M.shape
    patients = y["PatientID"].tolist()
    age, cog = load_cog(patients)
    labels = np.load(CANON, allow_pickle=True)
    nets = sorted(set(labels))
    net_idx = {n_: np.where(labels == n_)[0] for n_ in nets}
    tri = np.triu_indices(100, k=1)
    edge_net = np.array([(labels[i], labels[j]) for i, j in zip(*tri)])

    rng = np.random.default_rng(RNG_SEED)
    results = {}

    # ---- 1. ROI-level mass-univariate ----
    print("[faz4] ROI-level mass-univariate...", flush=True)
    roi_strength = np.zeros((n, 100))
    for r in range(100):
        idx = np.where((tri[0] == r) | (tri[1] == r))[0]
        roi_strength[:, r] = np.abs(M[:, idx]).mean(axis=1)
    roi_rows = []
    for cog_name, v in cog.items():
        ok = np.isfinite(v)
        if ok.sum() < 15:
            continue
        rho = np.zeros(100)
        pv = np.zeros(100)
        for r in range(100):
            rho[r], pv[r] = partial_spearman(roi_strength[ok, r], v[ok],
                                             age[ok])
        rej = multipletests(pv, method="fdr_bh")[1]
        for r in range(100):
            roi_rows.append({"score": cog_name, "roi": r,
                             "network": labels[r], "rho": rho[r],
                             "p": pv[r], "p_bh": rej[r],
                             "sig": bool(rej[r] < 0.05)})
    roi_df = pd.DataFrame(roi_rows)
    roi_df.to_csv(OUT / "roi_mass_univariate.tsv", sep="\t", index=False)
    results["roi"] = roi_df
    print(f"   ROI hits (BH<0.05): "
          f"{roi_df[roi_df['sig']].groupby('score').size().to_dict()}",
          flush=True)

    # ---- 2. edge-level ----
    print("[faz4] edge-level mass-univariate...", flush=True)
    edge_rows = []
    edge_mats = {}
    for cog_name, v in cog.items():
        ok = np.isfinite(v)
        if ok.sum() < 15:
            continue
        rho = vec_partial_spearman(M[ok], v[ok], age[ok])
        pv = p_from_rho(rho, ok.sum())
        rej = multipletests(pv, method="fdr_bh")[1]
        edge_mats[cog_name] = (rho, pv, rej)
        for e in range(n_edges):
            edge_rows.append({"score": cog_name, "edge": e, "roi_i": tri[0][e],
                              "roi_j": tri[1][e], "net_i": edge_net[e][0],
                              "net_j": edge_net[e][1], "rho": rho[e],
                              "p": pv[e], "p_bh": rej[e],
                              "sig": bool(rej[e] < 0.05)})
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(OUT / "edge_mass_univariate.tsv.gz", sep="\t", index=False,
                   compression="gzip")
    hits = edge_df[edge_df["sig"]]
    print("   edge hits per score:", hits.groupby("score").size().to_dict(),
          flush=True)
    top = edge_df.nlargest(40, "rho")
    top.to_csv(OUT / "edge_top40.tsv", sep="\t", index=False)

    # ---- 3. network x network block summary ----
    print("[faz4] network x network summaries...", flush=True)
    for cog_name, (rho, pv, rej) in edge_mats.items():
        blk_rho = pd.DataFrame(index=nets, columns=nets, dtype=float)
        blk_frac = pd.DataFrame(index=nets, columns=nets, dtype=float)
        blk_n = pd.DataFrame(index=nets, columns=nets, dtype=int)
        for a in nets:
            for b in nets:
                idx = np.where((edge_net[:, 0] == a) & (edge_net[:, 1] == b))[0]
                if len(idx) == 0:
                    idx = np.where((edge_net[:, 0] == b)
                                   & (edge_net[:, 1] == a))[0]
                if len(idx) == 0:
                    continue
                blk_rho.loc[a, b] = rho[idx].mean()
                blk_frac.loc[a, b] = rej[idx].mean()
                blk_n.loc[a, b] = len(idx)
        blk_rho.to_csv(OUT / f"network_map_rho_{cog_name}.tsv", sep="\t")
        blk_frac.to_csv(OUT / f"network_map_fracsig_{cog_name}.tsv", sep="\t")

    # ---- 4. NBS-style ----
    print("[faz4] NBS-style components...", flush=True)
    nbs_rows = []
    for cog_name, (rho, pv, rej) in edge_mats.items():
        ok = np.isfinite(cog[cog_name])
        v = cog[cog_name][ok]
        a = age[ok]
        adj = np.zeros((100, 100))
        mask = np.abs(rho) >= 0.35
        for e in np.where(mask)[0]:
            adj[tri[0][e], tri[1][e]] = 1
            adj[tri[1][e], tri[0][e]] = 1
        G = nx.from_numpy_array(adj)
        comps = [c for c in nx.connected_components(G) if len(c) > 1]
        obs_sizes = sorted([len(c) for c in comps], reverse=True)
        # permutation null: shuffle cognition (precomputed edge residuals)
        max_sizes = []
        n_perm = 1000
        Xok = M[ok]
        RX = precompute_residuals(Xok, a)
        for _ in range(n_perm):
            perm = rng.permutation(v)
            rp = corr_with_perm(RX, perm, a)
            maskp = np.abs(rp) >= 0.35
            adjp = np.zeros((100, 100))
            for e in np.where(maskp)[0]:
                adjp[tri[0][e], tri[1][e]] = 1
                adjp[tri[1][e], tri[0][e]] = 1
            Gp = nx.from_numpy_array(adjp)
            sizes = [len(c) for c in nx.connected_components(Gp) if len(c) > 1]
            max_sizes.append(max(sizes) if sizes else 0)
        max_sizes = np.array(max_sizes)
        for k, s in enumerate(obs_sizes):
            p_emp = float((max_sizes >= s).mean())
            nbs_rows.append({"score": cog_name, "component": k,
                             "size": s, "p_perm": p_emp})
    nbs_df = pd.DataFrame(nbs_rows)
    nbs_df.to_csv(OUT / "nbs_components.tsv", sep="\t", index=False)
    print(nbs_df.to_string(), flush=True)

    # ---- 5. behavioral PLS ----
    print("[faz4] behavioral PLS...", flush=True)
    Ycols = [c for c in ["MMSE", "Categorical_Fluency_", "Phonemic_Fluency",
                         "OVMPT_Coding", "OVMPT_Recalling", "DS_Forward",
                         "DS_Backward", "Benton_Face_Recognition_Test",
                         "Clock_Drawing_Test"]
             if c in cog]
    Ymat = np.column_stack([cog[c] for c in Ycols])
    ok = np.isfinite(Ymat).all(axis=1)
    X = M[ok]
    Y = Ymat[ok]
    Xz = stats.zscore(X, axis=0)
    Yz = stats.zscore(Y, axis=0)
    C = Xz.T @ Yz / (Xz.shape[0] - 1)
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    obs_cov = (s ** 2).sum()
    perm_covs = []
    for _ in range(500):
        Yp = Yz[rng.permutation(Yz.shape[0])]
        Cp = Xz.T @ Yp / (Xz.shape[0] - 1)
        perm_covs.append((np.linalg.svd(Cp, compute_uv=False) ** 2).sum())
    perm_covs = np.array(perm_covs)
    p_cov = float((perm_covs >= obs_cov).mean())
    lv1 = U[:, 0]
    sal = np.abs(lv1)
    pls_df = pd.DataFrame({
        "score": Ycols, "loading_lv1": Vt[0],
    })
    pls_df.to_csv(OUT / "pls_loadings.tsv", sep="\t", index=False)
    pls_edges = pd.DataFrame({
        "edge": np.arange(n_edges), "roi_i": tri[0], "roi_j": tri[1],
        "net_i": edge_net[:, 0], "net_j": edge_net[:, 1],
        "salience_lv1": sal})
    pls_edges.to_csv(OUT / "pls_edge_salience.tsv.gz", sep="\t", index=False,
                     compression="gzip")
    with open(OUT / "pls_summary.txt", "w") as f:
        f.write(f"n subjects (complete battery): {ok.sum()}\n")
        f.write(f"LV1 explains {100 * s[0] ** 2 / obs_cov:.1f}% of "
                f"cross-block covariance\n")
        f.write(f"permutation p (total covariance): {p_cov}\n")
        f.write("score loadings LV1:\n")
        for c, l in zip(Ycols, Vt[0]):
            f.write(f"  {c:30s} {l:+.3f}\n")
    print(f"   n={ok.sum()}, LV1 {100 * s[0] ** 2 / obs_cov:.1f}% cov, "
          f"p={p_cov}", flush=True)
    print("[faz4] DONE.", flush=True)


if __name__ == "__main__":
    main()
