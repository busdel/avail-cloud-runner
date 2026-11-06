#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-omics availability and network figures for AVAIL.

Figure 1: sample loss / availability (UpSet + coverage + complete-case).
Figure 2: multi-omics network overview (threshold, modules, topology, networks).
Figure 3: key subnetwork with central nodes.

Designed as a reference-style figure suite; raw data are read from a generic
AVAIL data root and all thresholds/counts are written to provenance JSON.
"""
import json, math
from itertools import combinations
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from scipy.spatial import ConvexHull

CPAL = ["#2E5A87", "#5B9279", "#C77D9D", "#B0A7C9", "#3D2E5F",
        "#4393C3", "#4DAC26", "#D6604D", "#8C6D31", "#666666", "#762A83"]
POS, NEG = "#2E5A87", "#B4525C"
INK, MUTED, GRID = "#2F3A44", "#6B7A88", "#DDE5EC"


def style():
    mpl.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 600,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
        "axes.titleweight": "bold", "axes.edgecolor": "#AEB9C2",
        "axes.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel(ax, letter, x=-0.12, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="bottom", ha="left")


def save(fig, base):
    base = Path(base); base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    print("[fig]", base.with_suffix(".pdf"), flush=True)


def load(root):
    root = Path(root)
    meta = pd.read_csv(root / "metadata.tsv", sep="\t", dtype=str)
    ids = meta["SampleID"].astype(str).str.strip().tolist()
    feats = {}
    for p in sorted((root / "features").glob("*.tsv")):
        d = pd.read_csv(p, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d.set_index("SampleID")
        feats[p.stem] = d
    groups = dict(zip(ids, meta["Group"].astype(str)))
    return ids, groups, feats


def availability(ids, feats):
    A = pd.DataFrame(0, index=ids, columns=sorted(feats), dtype=int)
    for m, d in feats.items():
        present = set(d.dropna(how="all").index.astype(str))
        A[m] = [1 if i in present else 0 for i in ids]
    return A


def clr_corr(X):
    Z = np.log(np.asarray(X, dtype=float) + 0.5)
    Z -= Z.mean(axis=1, keepdims=True)
    return pd.DataFrame(Z, columns=X.columns).corr()


def graph_from(X, method, threshold):
    corr = clr_corr(X) if method == "sparcc" else X.corr(method="spearman")
    G = nx.Graph(); G.add_nodes_from(map(str, corr.columns))
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            r = corr.loc[a, b]
            if np.isfinite(r) and abs(r) >= threshold:
                G.add_edge(str(a), str(b), weight=float(abs(r)), sign=float(r))
    return G


def pick_threshold(X, method):
    grid = np.round(np.arange(0.20, 0.81, 0.05), 2)
    rows = []
    for th in grid:
        G = graph_from(X, method, float(th))
        if G.number_of_edges() < 5:
            continue
        comms = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        mod = nx.community.modularity(G, comms, weight="weight") if comms else 0
        rows.append((float(th), float(mod), G.number_of_edges(), len(comms)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def draw_network(ax, G, pos, node_mod, sizes, labels, hubs=None):
    for a, b, d in G.edges(data=True):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=POS if d["sign"] >= 0 else NEG,
                lw=0.4 + 1.6 * d["weight"],
                ls="-" if d["sign"] >= 0 else "--",
                alpha=0.25 + 0.35 * d["weight"], zorder=1)
    colors = [CPAL[node_mod.get(n, 0) % len(CPAL)] for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes,
                           node_color=colors, edgecolors="white",
                           linewidths=0.5, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=4.8,
                            font_color=INK)
    for t in ax.texts:
        t.set_clip_on(False)
    for n in hubs or []:
        ax.scatter([pos[n][0]], [pos[n][1]], s=110, facecolors="none",
                   edgecolors="#B4525C", linewidths=1.2, zorder=4)
    ax.set_axis_off()


def fig_availability(ids, groups, A, out, prov):
    fig = plt.figure(figsize=(183 / 25.4, 82 / 25.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.0])
    # a: sample loss
    ax = fig.add_subplot(gs[0, 0])
    n = len(ids); atleast = int((A.sum(axis=1) > 0).sum())
    complete = int((A.sum(axis=1) == A.shape[1]).sum())
    vals = [n, atleast, complete]
    ax.bar(["all", "≥1 modality", "complete"], vals,
           color=[CPAL[9], CPAL[0], CPAL[2]], width=0.65,
           edgecolor="white", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, str(v), ha="center", va="bottom", fontsize=6)
    ax.set_ylabel("subjects"); ax.set_title("Sample retention", loc="left")
    panel(ax, "a")
    # b: UpSet-style intersections
    ax = fig.add_subplot(gs[0, 1])
    mods = list(A.columns)
    combos = []
    for r in range(1, len(mods) + 1):
        for c in combinations(range(len(mods)), r):
            mask = np.ones(len(A), dtype=bool)
            for j, m in enumerate(mods):
                mask &= (A[m].values == (1 if j in c else 0))
            combos.append((tuple(c), int(mask.sum())))
    combos.sort(key=lambda x: x[1], reverse=True)
    top = combos[:12]
    x = np.arange(len(top))
    ax.bar(x, [v for _, v in top], color=CPAL[0], width=0.65,
           edgecolor="white", linewidth=0.4)
    for i, (_, v) in enumerate(top):
        ax.text(i, v + 0.7, str(v), ha="center", va="bottom", fontsize=5)
    ax.set_xticks([]); ax.set_ylabel("subjects")
    ax.set_title("Availability intersections", loc="left")
    ax.set_ylim(0, max(v for _, v in top) * 1.18)
    # dot matrix below
    y0 = -max(v for _, v in top) * 0.30
    for i, (c, _) in enumerate(top):
        for j in range(len(mods)):
            on = j in c
            ax.scatter(i, y0 - j, s=13, color=INK if on else "#DCE3E8",
                       edgecolor="white", linewidth=0.3, clip_on=False)
    ax.set_yticks([]); ax.set_ylim(y0 - len(mods), max(v for _, v in top) * 1.2)
    for j, m in enumerate(mods):
        ax.text(-0.7, y0 - j, m, ha="right", va="center", fontsize=5,
                color=INK, clip_on=False)
    panel(ax, "b")
    # c: coverage by group
    ax = fig.add_subplot(gs[0, 2])
    order = [g for g in ("HC", "MCI", "PDD") if g in set(groups.values())]
    gser = order + [g for g in sorted(set(groups.values())) if g not in order]
    x = np.arange(len(gser)); width = 0.8 / len(mods)
    for j, m in enumerate(mods):
        vals = [A.loc[[i for i in ids if groups[i] == g], m].mean() for g in gser]
        ax.bar(x + (j - (len(mods)-1)/2) * width, vals, width,
               color=CPAL[j], label=m, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(gser)
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction available")
    ax.set_title("Coverage by group", loc="left")
    ax.legend(fontsize=5, ncol=1, loc="center left",
              bbox_to_anchor=(1.01, 0.5), frameon=False)
    panel(ax, "c")
    fig.suptitle("Multi-omics availability — sample loss across layers",
                 x=0.01, ha="left", fontsize=8, fontweight="bold")
    save(fig, out)
    prov["availability"] = {
        "n": n, "at_least_one": atleast, "complete_case": complete,
        "missing_everywhere": n - atleast,
        "intersections": [{"sets": [mods[j] for j in c], "n": v}
                          for c, v in top],
    }


def fig_networks(ids, feats, out, prov):
    """Per-layer network structure without hairballs: threshold scan, module
    size distribution, topology heatmap, and one clean key-module network per
    layer."""
    tables = {}
    for m, d in feats.items():
        d = d.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        d = d.loc[:, d.var().sort_values(ascending=False).head(45).index]
        tables[m] = d
    methods = {m: ("sparcc" if "16S" in m else "spearman") for m in tables}
    scans = {m: pick_threshold(tables[m], methods[m]) for m in tables}
    mods = sorted(tables)
    chosen = {m: scans[m][0][0] for m in mods}
    graphs, modules, metrics = {}, {}, {}
    for m in mods:
        G = graph_from(tables[m], methods[m], chosen[m])
        graphs[m] = G
        allc = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        cm = [c for c in allc if len(c) >= 3]
        modules[m] = sorted(cm, key=len, reverse=True)
        metrics[m] = {
            "density": nx.density(G),
            "clustering": nx.average_clustering(G, weight="weight"),
            "modularity": nx.community.modularity(G, allc, weight="weight") if allc else 0,
            "mean degree": float(np.mean([d for _, d in G.degree()])) if G.number_of_nodes() else 0,
            "diameter": float(nx.diameter(G)) if nx.is_connected(G) else np.nan,
        }
    fig = plt.figure(figsize=(183 / 25.4, 145 / 25.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.6])
    # a threshold scan with selected threshold
    ax = fig.add_subplot(gs[0, 0])
    for j, m in enumerate(mods):
        rows = scans[m]
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "-o", ms=2.2,
                lw=0.9, color=CPAL[j], label=m)
        ax.scatter([chosen[m]], [dict((r[0], r[1]) for r in rows)[chosen[m]]],
                   s=28, color=CPAL[j], edgecolor="white", zorder=4)
    ax.set_xlabel("|correlation| threshold"); ax.set_ylabel("modularity")
    ax.set_title("Threshold scan", loc="left"); panel(ax, "a")
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    # b module size distribution (strip + median)
    ax = fig.add_subplot(gs[0, 1])
    for j, m in enumerate(mods):
        sizes = [len(c) for c in modules[m][:8]]
        if sizes:
            ax.scatter(np.full(len(sizes), j), sizes, s=16, color=CPAL[j],
                       edgecolor="white", linewidth=0.3, zorder=3)
            ax.plot([j - .22, j + .22], [np.median(sizes)] * 2,
                    color=INK, lw=1.1)
    ax.set_xticks(range(len(mods))); ax.set_xticklabels(mods, rotation=20, ha="right")
    ax.set_ylabel("module size (nodes)")
    ax.set_title("Module size distribution", loc="left"); panel(ax, "b")
    # c topology heatmap
    ax = fig.add_subplot(gs[0, 2])
    mets = ["density", "clustering", "modularity", "mean degree", "diameter"]
    V = np.array([[metrics[m][k] for k in mets] for m in mods], dtype=float)
    Vn = np.zeros_like(V)
    for j in range(V.shape[1]):
        col = V[:, j]; lo, hi = np.nanmin(col), np.nanmax(col)
        Vn[:, j] = (col - lo) / (hi - lo) if hi > lo else 0.5
    im = ax.imshow(Vn, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v = V[i, j]
            ax.text(j, i, "NA" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=5,
                    color="white" if Vn[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(mets))); ax.set_xticklabels(mets, rotation=35, ha="right")
    ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
    ax.set_title("Topology (normalized)", loc="left"); panel(ax, "c")
    # d-f one clean key module per modality
    for j, m in enumerate(mods[:3]):
        ax = fig.add_subplot(gs[1, j])
        if not modules[m]:
            ax.set_axis_off(); continue
        comp = modules[m][0]
        deg = {n: graphs[m].degree(n, weight="weight") for n in comp}
        keep = sorted(comp, key=lambda n: deg[n], reverse=True)[:12]
        sub = graphs[m].subgraph(keep).copy()
        pos = nx.spring_layout(sub, seed=51 + j, k=1.55 / math.sqrt(len(sub)),
                               iterations=240, weight="weight")
        xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
        ax.set_xlim(min(xs)-.18, max(xs)+.18); ax.set_ylim(min(ys)-.18, max(ys)+.18)
        for a, b, d in sub.edges(data=True):
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                    color=POS if d["sign"] >= 0 else NEG,
                    lw=0.45 + 1.8*d["weight"],
                    ls="-" if d["sign"] >= 0 else "--",
                    alpha=0.35 + 0.35*d["weight"], zorder=1)
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=[24+30*deg[n] for n in sub],
                               node_color=CPAL[j], edgecolors="white", linewidths=0.6)
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=4.8, font_color=INK,
                                labels={n: (n if len(str(n)) <= 24 else str(n)[:22]+"…")
                                        for n in sub.nodes})
        for t in ax.texts: t.set_clip_on(False)
        for n in sorted(sub.nodes, key=lambda n: deg[n], reverse=True)[:2]:
            ax.scatter([pos[n][0]], [pos[n][1]], s=120, facecolors="none",
                       edgecolors="#B4525C", linewidths=1.3, zorder=4)
        ax.set_axis_off()
        ax.set_title(f"{m} — key module", loc="left"); panel(ax, chr(100+j))
    fig.suptitle("Multi-omics networks — per-layer structure",
                 x=0.01, ha="left", fontsize=8, fontweight="bold")
    save(fig, out)
    prov["networks"] = {
        "thresholds": chosen, "methods": methods,
        "module_sizes": {m: [len(c) for c in modules[m]] for m in mods},
        "topology": {m: metrics[m] for m in mods},
    }


def fig_key(ids, feats, out, prov):
    """Key subnetwork and central nodes; selects the densest module across
    all layers (not just the largest one)."""
    candidates = []
    for m, d in feats.items():
        X = d.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X = X.loc[:, X.var().sort_values(ascending=False).head(45).index]
        method = "sparcc" if "16S" in m else "spearman"
        scans = pick_threshold(X, method)
        th = scans[0][0]
        G = graph_from(X, method, th)
        allc = list(nx.community.greedy_modularity_communities(G, weight="weight"))
        for comp in allc:
            if len(comp) < 5:
                continue
            sub = G.subgraph(comp)
            candidates.append((sub.number_of_edges() * len(comp), m, method,
                               th, comp, G))
    if not candidates:
        raise ValueError("no module with >=5 nodes")
    _, m, method, th, comp, G = max(candidates, key=lambda x: x[0])
    deg = {n: G.degree(n, weight="weight") for n in comp}
    btw = nx.betweenness_centrality(G.subgraph(comp), weight="weight")
    pr = nx.pagerank(G.subgraph(comp), weight="weight")
    keep = sorted(comp, key=lambda n: deg[n], reverse=True)[:15]
    sub = G.subgraph(keep).copy()
    central = sorted(sub.nodes, key=lambda n: deg[n] * (1 + btw.get(n, 0) + pr.get(n, 0)),
                     reverse=True)[:5]
    pos = nx.spring_layout(sub, seed=41, k=1.6/math.sqrt(len(sub)),
                           iterations=260, weight="weight")
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    fig = plt.figure(figsize=(183 / 25.4, 72 / 25.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(min(xs)-.18, max(xs)+.18); ax.set_ylim(min(ys)-.18, max(ys)+.18)
    for a, b, d in sub.edges(data=True):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=POS if d["sign"] >= 0 else NEG,
                lw=0.5 + 1.8*d["weight"], ls="-" if d["sign"] >= 0 else "--",
                alpha=0.35 + 0.35*d["weight"], zorder=1)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=[26+34*deg[n] for n in sub],
                           node_color=CPAL[0], edgecolors="white", linewidths=0.6)
    cid = {n: str(i + 1) for i, n in enumerate(central)}
    labels = {n: cid[n] for n in central}
    nx.draw_networkx_labels(sub, pos, labels=labels, ax=ax, font_size=7,
                            font_color="#B4525C", font_weight="bold")
    for n in central:
        ax.scatter([pos[n][0]], [pos[n][1]], s=150, facecolors="none",
                   edgecolors="#B4525C", linewidths=1.4, zorder=4)
        for txt in ax.texts:
            if txt.get_text() == cid[n]:
                txt.set_bbox(dict(boxstyle="circle,pad=0.22", fc="white",
                                  ec="#B4525C", lw=0.6, alpha=0.95))
    for t in ax.texts: t.set_clip_on(False)
    ax.set_axis_off()
    ax.set_title(f"Key subnetwork — {m} (threshold {th:.2f})", loc="left")
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    metrics = {"degree": [deg[n] for n in central],
               "betweenness": [btw.get(n, 0) for n in central],
               "pagerank": [pr.get(n, 0) for n in central]}
    # scale each metric to 0-1 across the central nodes for a shared axis
    for key in metrics:
        v = np.array(metrics[key], dtype=float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        metrics[key] = ((v - lo) / (hi - lo)).tolist() if hi > lo else [0.5]*len(v)
    xs = np.arange(len(central)); width = 0.26
    for j, (name, vals) in enumerate(metrics.items()):
        ax.bar(xs + (j-1)*width, vals, width, color=CPAL[j], label=name,
               edgecolor="white", linewidth=0.3)
    xt = [f"{i+1} {n}" for i, n in enumerate(central)]
    ax.set_xticks(xs); ax.set_xticklabels(xt, rotation=40, ha="right", fontsize=4.8)
    ax.set_ylabel("scaled score (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Central nodes", loc="left")
    ax.legend(fontsize=5, frameon=False)
    panel(ax, "b")
    fig.suptitle("Central nodes in a key multi-omics subnetwork",
                 x=0.01, ha="left", fontsize=8, fontweight="bold")
    save(fig, out)
    prov["key_subnetwork"] = {
        "modality": m, "method": method, "threshold": th,
        "module_nodes": len(comp), "shown_nodes": len(sub),
        "central": central,
        "scores": {n: {"degree": deg[n], "betweenness": btw.get(n, 0),
                       "pagerank": pr.get(n, 0)} for n in central},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    style()
    ids, groups, feats = load(args.data_root)
    A = availability(ids, feats)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prov = {"data_root": str(args.data_root), "n_metadata": len(ids),
            "modalities": {m: int(d.dropna(how="all").shape[0])
                           for m, d in feats.items()},
            "note": "research figures; no publisher-compliance claim"}
    fig_availability(ids, groups, A, out / "fig1_availability", prov)
    fig_networks(ids, feats, out / "fig2_multomics_networks", prov)
    fig_key(ids, feats, out / "fig3_key_subnetwork", prov)
    (out / "figure_provenance.json").write_text(json.dumps(prov, indent=2))
    print("done", out)


if __name__ == "__main__":
    main()
