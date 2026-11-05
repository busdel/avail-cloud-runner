#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 4b: Network-level visualization of ROI x cognition associations.

Outputs (assoc/):
  network_involvement.tsv        per 7-network significant-edge involvement
  figures/heatmap_<score>.png/pdf    19x19 and 7x7 network maps
  figures/brain_<score>.png/pdf      glass-brain connectome of significant
                                     edges (Schaefer-100, node color = Yeo-7)
  nbs_network_composition.tsv        network composition of NBS components
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

ROOT = Path("/mnt/limbic/AVAIL_GPU")
ASSOC = ROOT / "experiments/results/roi_ladder/assoc"
FIGD = ASSOC / "figures"
FIGD.mkdir(parents=True, exist_ok=True)

COARSE = {
    "Default A": "DMN", "Default B": "DMN", "Default C": "DMN",
    "Dorsal Attention A": "DAN", "Dorsal Attention B": "DAN",
    "Frontoparietal Control A": "FPN", "Frontoparietal Control B": "FPN",
    "Frontoparietal Control C": "FPN",
    "Limbic": "LIM", "Limbic Orbitofrontal": "LIM",
    "Limbic Temporopolar": "LIM",
    "Salience Ventral Attention A": "SAL", "Salience Ventral Attention B": "SAL",
    "Somatomotor A": "SMN", "Somatomotor B": "SMN",
    "Somatomotor B Auditory": "SMN",
    "Temporoparietal": "TPN",
    "Visual Central": "VIS", "Visual Peripheral": "VIS",
}
COARSE_ORDER = ["VIS", "SMN", "DAN", "SAL", "LIM", "FPN", "DMN", "TPN"]
COARSE_COLORS = {
    "VIS": "#781286", "SMN": "#4682B4", "DAN": "#00760E",
    "SAL": "#C43AFA", "LIM": "#E69422", "FPN": "#CD3E4F",
    "DMN": "#DC050C", "TPN": "#8B8B8B"}


def load_labels():
    from nilearn.datasets import fetch_atlas_schaefer_2018
    d = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
    names = d.labels[1:]

    def canon(n):
        if "VisCent" in n:
            return "Visual Central"
        if "VisPeri" in n:
            return "Visual Peripheral"
        if "SomMotA" in n:
            return "Somatomotor A"
        if "SomMotB_Aud" in n:
            return "Somatomotor B Auditory"
        if "SomMotB" in n:
            return "Somatomotor B"
        if "DorsAttnA" in n:
            return "Dorsal Attention A"
        if "DorsAttnB" in n:
            return "Dorsal Attention B"
        if "SalVentAttnA" in n:
            return "Salience Ventral Attention A"
        if "SalVentAttnB" in n:
            return "Salience Ventral Attention B"
        if "LimbicA" in n:
            return "Limbic Temporopolar"
        if "LimbicB" in n:
            return "Limbic Orbitofrontal"
        if "ContA" in n:
            return "Frontoparietal Control A"
        if "ContB" in n:
            return "Frontoparietal Control B"
        if "ContC" in n:
            return "Frontoparietal Control C"
        if "DefaultA" in n:
            return "Default A"
        if "DefaultB" in n:
            return "Default B"
        if "DefaultC" in n:
            return "Default C"
        if "TempPar" in n:
            return "Temporoparietal"
        return "??"

    return [canon(n) for n in names]


def coarse_heatmap(edge_df, score, ax_rho, ax_frac, seven=True):
    sub = edge_df[edge_df["score"] == score].copy()
    sub["coarse_i"] = sub["net_i"].map(COARSE)
    sub["coarse_j"] = sub["net_j"].map(COARSE)
    sub = sub.dropna(subset=["coarse_i", "coarse_j"])
    nets = COARSE_ORDER
    rho = pd.DataFrame(0.0, index=nets, columns=nets)
    frac = pd.DataFrame(0.0, index=nets, columns=nets)
    cnt = pd.DataFrame(0, index=nets, columns=nets)
    for (a, b), g in sub.groupby(["coarse_i", "coarse_j"]):
        rho.loc[a, b] = rho.loc[b, a] = g["rho"].mean()
        frac.loc[a, b] = frac.loc[b, a] = g["sig"].mean()
        cnt.loc[a, b] = cnt.loc[b, a] = len(g)
    vmax = max(0.1, np.abs(rho.values).max())
    im = ax_rho.imshow(rho.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax_rho.set_xticks(range(len(nets)))
    ax_rho.set_xticklabels(nets, rotation=45, fontsize=8)
    ax_rho.set_yticks(range(len(nets)))
    ax_rho.set_yticklabels(nets, fontsize=8)
    ax_rho.set_title(f"mean rho ({score})", fontsize=10)
    plt.colorbar(im, ax=ax_rho, fraction=0.046)
    im2 = ax_frac.imshow(frac.values, cmap="YlOrRd", vmin=0, vmax=1)
    ax_frac.set_xticks(range(len(nets)))
    ax_frac.set_xticklabels(nets, rotation=45, fontsize=8)
    ax_frac.set_yticks(range(len(nets)))
    ax_frac.set_yticklabels(nets, fontsize=8)
    ax_frac.set_title(f"frac significant (BH-FDR)", fontsize=10)
    plt.colorbar(im2, ax=ax_frac, fraction=0.046)
    return cnt


def brain_glass(edge_df, score, labels, max_edges=120):
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn import plotting, image
    d = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
    coords = plotting.find_parcellation_cut_coords(image.load_img(d.maps))
    sub = edge_df[edge_df["score"] == score].copy()
    sig = sub[sub["sig"]].nlargest(max_edges, "rho")
    if len(sig) == 0:
        sig = sub.nlargest(max_edges, "rho")
    adj = np.zeros((100, 100))
    for _, r in sig.iterrows():
        i, j = int(r["roi_i"]), int(r["roi_j"])
        adj[i, j] = abs(r["rho"])
        adj[j, i] = abs(r["rho"])
    node_colors = [COARSE_COLORS[COARSE[l]] for l in labels]
    pos_color = [COARSE_COLORS[COARSE[l]] for l in labels]
    neg = sub[sub["sig"] & (sub["rho"] < 0)]
    fig = plt.figure(figsize=(9, 9))
    plotting.plot_connectome(adj, coords, node_color=node_colors,
                             edge_threshold=0.30, edge_cmap="Reds",
                             node_size=12, colorbar=False, figure=fig,
                             display_mode="lyrz",
                             title=f"{score}: significant edges (n={len(sig)})")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=c, label=n) for n, c in
               COARSE_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    return fig


def main():
    edge = pd.read_csv(ASSOC / "edge_mass_univariate.tsv.gz", sep="\t",
                       compression="gzip")
    labels = load_labels()
    assert len(labels) == 100 and "??" not in labels
    roi = pd.read_csv(ASSOC / "roi_mass_univariate.tsv", sep="\t")
    roi["coarse"] = roi["network"].map(COARSE)
    nbs = pd.read_csv(ASSOC / "nbs_components.tsv", sep="\t")

    # ---- network involvement summary ----
    rows = []
    for score, g in edge.groupby("score"):
        g = g.copy()
        g["ci"] = g["net_i"].map(COARSE)
        g["cj"] = g["net_j"].map(COARSE)
        g = g.dropna(subset=["ci", "cj"])
        for n_ in COARSE_ORDER:
            sel = g[(g["ci"] == n_) | (g["cj"] == n_)]
            rows.append({"score": score, "network": n_,
                         "n_edges": len(sel),
                         "n_sig": int(sel["sig"].sum()),
                         "frac_sig": float(sel["sig"].mean()),
                         "mean_rho": float(sel["rho"].mean())})
    invol = pd.DataFrame(rows)
    invol.to_csv(ASSOC / "network_involvement.tsv", sep="\t", index=False)

    # bar plot of involvement
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, score in zip(axes.ravel(), invol["score"].unique()):
        g = invol[invol["score"] == score]
        ax.bar(g["network"], g["n_sig"], color=[COARSE_COLORS[n]
                for n in g["network"]])
        ax.set_title(score, fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes.ravel()[len(invol["score"].unique()):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGD / "network_involvement.png", dpi=300)
    fig.savefig(FIGD / "network_involvement.pdf")
    plt.close(fig)
    print("[faz4b] network_involvement written", flush=True)

    # ---- heatmaps per score ----
    scores = edge["score"].unique()
    for score in scores:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        coarse_heatmap(edge, score, axes[0], axes[1])
        fig.suptitle(f"{score} - ROI edge associations by Yeo-7 network",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        safe = score.replace("/", "_").replace(" ", "_")
        fig.savefig(FIGD / f"heatmap_{safe}.png", dpi=300)
        fig.savefig(FIGD / f"heatmap_{safe}.pdf")
        plt.close(fig)
    print(f"[faz4b] {len(scores)} heatmaps written", flush=True)

    # ---- NBS composition ----
    comp_rows = []
    tri = np.triu_indices(100, k=1)
    roi_map = {r["roi"]: r["coarse"] for _, r in roi.iterrows()}
    for score in nbs["score"].unique():
        sub = edge[edge["score"] == score]
        rho = np.zeros(len(sub))
        rho[:] = sub["rho"].values if len(sub) == len(tri[0]) else 0
        if len(sub) != len(tri[0]):
            m_ = {}
            for _, r in sub.iterrows():
                m_[r["edge"]] = r["rho"]
            rho = np.array([m_.get(e, 0.0) for e in range(len(tri[0]))])
        import networkx as nx
        adj = np.zeros((100, 100))
        for e in np.where(np.abs(rho) >= 0.35)[0]:
            adj[tri[0][e], tri[1][e]] = 1
            adj[tri[1][e], tri[0][e]] = 1
        G = nx.from_numpy_array(adj)
        for c in sorted([c for c in nx.connected_components(G) if len(c) > 1],
                        key=len, reverse=True)[:3]:
            from collections import Counter
            cnt = Counter(roi_map[n] for n in c)
            for net_, k_ in cnt.items():
                comp_rows.append({"score": score, "size": len(c),
                                  "network": net_, "n_nodes": k_})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(ASSOC / "nbs_network_composition.tsv", sep="\t", index=False)
    print("[faz4b] nbs composition written", flush=True)

    # ---- glass brains ----
    for score in ["MMSE", "CDRS", "OVMPT_Coding", "Clock_Drawing_Test",
                  "Categorical_Fluency_"]:
        fig = brain_glass(edge, score, labels)
        safe = score.replace("/", "_").replace(" ", "_")
        fig.savefig(FIGD / f"brain_{safe}.png", dpi=300,
                    bbox_inches="tight")
        fig.savefig(FIGD / f"brain_{safe}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"[faz4b] brain glass {score} written", flush=True)
    print("[faz4b] DONE.", flush=True)


if __name__ == "__main__":
    main()
