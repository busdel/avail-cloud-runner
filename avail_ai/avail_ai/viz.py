# -*- coding: utf-8 -*-
"""Soft-palette visualization toolkit for AVAIL results.

Three high-value figures, usable from Python or the CLI:

    from avail_omics import viz
    viz.plot_network("results", data_root="mydata", out_path="figs/network")
    viz.plot_roc_calibration("results", out_path="figs/roc")
    viz.plot_availability("mydata", out_path="figs/availability")

Colors are deliberately muted so the figures remain readable in print and
for colour-vision deficiencies. No result values are invented: each function
reads the documented TSV outputs (or the generic data layout) and fails with a
clear message when the required input is absent.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

SOFT = ["#7A9EAF", "#A8C0B0", "#D7A9B1", "#B7ACD6", "#C9B99B",
        "#8FB6B0", "#D6C2A8", "#9AA9B8", "#B7C9A8", "#C4A8B8"]
POS = "#6F9BAE"
NEG = "#C98F9B"
GRID = "#DDE5EC"
INK = "#2F3A44"
MUTED = "#6B7A88"


def set_soft_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
        "axes.edgecolor": "#B9C4CE", "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.fontsize": 9, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig, out_path):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", dpi=400)
    plt.close(fig)
    print(f"  + {p}.pdf / {p}.png", flush=True)


def _read(out, *names):
    for n in names:
        q = Path(out) / n
        if q.exists():
            return pd.read_csv(q, sep="\t")
    return None


def _feature_tables(data_root):
    root = Path(data_root)
    feats = {}
    fdir = root / "features"
    if fdir.exists():
        for p in sorted(fdir.glob("*.tsv")):
            try:
                d = pd.read_csv(p, sep="\t")
                d["SampleID"] = d["SampleID"].astype(str).str.strip()
                feats[p.stem] = d.set_index("SampleID")
            except Exception:
                continue
    return feats


def _match(cols, name):
    s = str(name).strip()
    if s in cols:
        return s
    low = {str(c).strip().lower(): c for c in cols}
    return low.get(s.lower())


def _pack_layout(G, seed=7):
    """Spring layout per connected component, packed on a grid."""
    import math
    comps = sorted(__import__("networkx").connected_components(G),
                   key=len, reverse=True)
    if len(comps) <= 1:
        return __import__("networkx").spring_layout(
            G, seed=seed, k=1.0 / max(1, math.sqrt(len(G))),
            iterations=140, weight="weight")
    ncol = max(1, int(math.ceil(math.sqrt(len(comps)))))
    pos = {}
    for i, comp in enumerate(comps):
        sub = G.subgraph(comp)
        p = __import__("networkx").spring_layout(
            sub, seed=seed + i, k=1.0 / max(1, math.sqrt(len(sub))),
            iterations=100, weight="weight")
        xs = [v[0] for v in p.values()] or [0]
        ys = [v[1] for v in p.values()] or [0]
        w = max(1e-6, max(xs) - min(xs)); h = max(1e-6, max(ys) - min(ys))
        scale = 0.75 / max(w, h)
        col, row = i % ncol, i // ncol
        offx, offy = col * 1.05, -row * 1.05
        for n, (x, y) in p.items():
            pos[n] = ((x - min(xs)) * scale + offx,
                      (y - min(ys)) * scale + offy)
    return pos


def plot_network(out, data_root=None, out_path=None, top_n=24,
                 threshold=0.30):
    """Feature network from the top biomarkers.

    With ``data_root``: Spearman correlation between the selected features,
    edges with |r| >= threshold. Without it: a modality-feature bipartite
    graph weighted by biomarker importance (still useful, clearly labelled).
    """
    import networkx as nx
    set_soft_style()
    bio = _read(out, "biomarkers_top20_per_modality.tsv")
    if bio is None or not {"feature", "modality", "importance"}.issubset(bio.columns):
        raise FileNotFoundError(
            "biomarkers_top20_per_modality.tsv with feature/modality/importance "
            "is required for the network plot")
    bio = bio.sort_values("importance", ascending=False).head(int(top_n)).copy()
    bio["feature"] = bio["feature"].astype(str)
    bio["modality"] = bio["modality"].astype(str)
    # disambiguate features shared across modalities
    dup = bio["feature"].duplicated(keep=False)
    bio["node"] = np.where(dup, bio["modality"] + ": " + bio["feature"],
                           bio["feature"])
    bio = bio.drop_duplicates("node")
    nodes = bio.set_index("node")[["feature", "modality", "importance"]].to_dict("index")
    feats = _feature_tables(data_root) if data_root else {}
    G = nx.Graph()
    for name, meta in nodes.items():
        G.add_node(name, modality=str(meta["modality"]),
                   importance=float(meta["importance"]))
    real_edges = 0
    if feats:
        rows = []
        for node, meta in nodes.items():
            for m, df in feats.items():
                c = _match(df.columns, meta["feature"])
                if c is not None:
                    rows.append(df[c].astype(float).rename(node))
                    break
        if len(rows) >= 2:
            X = pd.concat(rows, axis=1)
            corr = X.corr(method="spearman")
            for i, a in enumerate(corr.columns):
                for b in corr.columns[i + 1:]:
                    r = corr.loc[a, b]
                    if np.isfinite(r) and abs(r) >= threshold:
                        G.add_edge(a, b, weight=float(abs(r)), sign=float(r))
                        real_edges += 1
    if real_edges == 0:
        for name, meta in nodes.items():
            G.add_node(str(meta["modality"]), modality=str(meta["modality"]),
                       importance=0.0, modality_node=True)
            G.add_edge(name, str(meta["modality"]),
                       weight=max(0.05, float(meta["importance"])),
                       sign=1.0)
        title = "Top biomarkers by modality (feature values unavailable)"
    else:
        title = f"Top biomarker network (Spearman |r| >= {threshold:g})"
    mods = sorted({d.get("modality", "?") for _, d in G.nodes(data=True)
                   if not d.get("modality_node")})
    cmap = {m: SOFT[i % len(SOFT)] for i, m in enumerate(mods)}
    pos = _pack_layout(G, seed=7)
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for a, b, d in G.edges(data=True):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=POS if d.get("sign", 1) >= 0 else NEG,
                alpha=0.18 + 0.45 * min(1.0, d.get("weight", 0.2)),
                linewidth=0.6 + 2.2 * min(1.0, d.get("weight", 0.2)),
                zorder=1)
    sizes = []
    colors = []
    for n, d in G.nodes(data=True):
        if d.get("modality_node"):
            sizes.append(420)
            colors.append(cmap.get(d.get("modality", "?"), "#D9E2E9"))
        else:
            sizes.append(90 + 900 * min(0.12, float(d.get("importance", 0.01))))
            colors.append(cmap.get(d.get("modality", "?"), SOFT[0]))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           edgecolors="white", linewidths=1.1, alpha=0.95)
    top_labels = sorted(
        [n for n, d in G.nodes(data=True) if not d.get("modality_node")],
        key=lambda n: G.nodes[n].get("importance", 0), reverse=True)[:14]
    labels = {n: (n if len(n) <= 30 else n[:28] + "…") for n in top_labels}
    labels.update({n: n for n, d in G.nodes(data=True) if d.get("modality_node")})
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5,
                            font_color=INK)
    ax.set_title(title, loc="left", color=INK)
    ax.set_axis_off()
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor=cmap[m], markeredgecolor="white",
                          label=m) for m in mods]
    if handles:
        ax.legend(handles=handles, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), title="modality",
                  title_fontsize=8, fontsize=7, frameon=False)
    _save(fig, out_path or Path(out) / "viz" / "network")


def plot_roc_calibration(out, out_path=None, model=None, n_bins=8):
    """One-vs-rest ROC and reliability curves from loo_predictions.tsv."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.calibration import calibration_curve
    set_soft_style()
    loo = _read(out, "loo_predictions.tsv")
    if loo is None:
        raise FileNotFoundError("loo_predictions.tsv is required for ROC/calibration")
    if "y" not in loo.columns:
        raise ValueError("loo_predictions.tsv must contain a 'y' column")
    # model names are the prefixes before _<class>
    suffixes = {}
    for c in loo.columns:
        if "_" in c and c not in ("patient", "y"):
            pre, cls = c.rsplit("_", 1)
            suffixes.setdefault(pre, []).append(cls)
    if not suffixes:
        raise ValueError("no <model>_<class> probability columns found")
    if model is None:
        meta = _read(out, "meta_benchmark_3class_results.tsv",
                     "meta_benchmark_results.tsv")
        model = None
        if meta is not None and "macro_auc" in meta.columns:
            sub = meta[meta.get("subgroup", "all") == "all"] if "subgroup" in meta.columns else meta
            model = str(sub.sort_values("macro_auc", ascending=False).iloc[0]["model"])
        model = model if model in suffixes else sorted(suffixes)[0]
    if model not in suffixes:
        raise ValueError(f"model {model!r} not found in loo_predictions.tsv")
    classes = suffixes[model]
    y_raw = loo["y"].values
    uniq = sorted(pd.unique(y_raw))
    # map integer codes to class labels in column order; otherwise use labels
    if all(isinstance(v, (int, np.integer)) for v in uniq) and \
            set(int(v) for v in uniq) <= set(range(len(classes))):
        y_idx = y_raw.astype(int)
    else:
        order = {v: i for i, v in enumerate(uniq)}
        y_idx = np.array([order[v] for v in y_raw])
        if len(uniq) != len(classes):
            classes = [str(v) for v in uniq]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    for k, cls in enumerate(classes):
        col = f"{model}_{cls}"
        if col not in loo.columns:
            continue
        prob = pd.to_numeric(loo[col], errors="coerce").values
        ok = np.isfinite(prob) & (y_idx == k)
        neg = np.isfinite(prob) & (y_idx != k)
        if ok.sum() == 0 or neg.sum() == 0:
            continue
        y_bin = (y_idx == k).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, prob)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=SOFT[k % len(SOFT)], lw=1.8,
                label=f"{cls} (AUC {a:.2f})")
        ax = axes[1]
        try:
            frac, mean = calibration_curve(y_bin, prob, n_bins=n_bins,
                                           strategy="quantile")
            ax.plot(mean, frac, marker="o", ms=3.5,
                    color=SOFT[k % len(SOFT)], lw=1.4, label=str(cls))
        except Exception:
            pass
        ax = axes[0]
    axes[0].plot([0, 1], [0, 1], ls="--", color="#AEB9C2", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC — {model}", loc="left")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot([0, 1], [0, 1], ls="--", color="#AEB9C2", lw=1)
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed frequency")
    axes[1].set_title("Reliability", loc="left")
    axes[1].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    _save(fig, out_path or Path(out) / "viz" / "roc_calibration")


def plot_availability(data_root, out_path=None, group_col=None):
    """Availability heatmap, group coverage and pattern frequencies."""
    set_soft_style()
    root = Path(data_root)
    meta_path = root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.tsv is required for the availability plot")
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.suffix == ".tsv" else ",",
                       dtype=str)
    id_col = "SampleID" if "SampleID" in meta.columns else ("sample" if "sample" in meta.columns else meta.columns[0])
    gcol = group_col or ("Group" if "Group" in meta.columns else ("group" if "group" in meta.columns else None))
    ids = meta[id_col].astype(str).str.strip().tolist()
    groups = meta[gcol].astype(str).tolist() if gcol else ["all"] * len(ids)
    feats = _feature_tables(root)
    if not feats:
        raise FileNotFoundError("features/*.tsv are required for the availability plot")
    mods = sorted(feats)
    A = np.zeros((len(ids), len(mods)), dtype=int)
    for j, m in enumerate(mods):
        df = feats[m]
        present = set(df.dropna(how="all").index.astype(str))
        A[:, j] = [1 if s in present else 0 for s in ids]
    order = np.lexsort((A.sum(1), np.array(groups),))
    A = A[order]; ids = [ids[i] for i in order]; groups = [groups[i] for i in order]
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1.0], height_ratios=[2.2, 1.0],
                          wspace=0.28, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    from matplotlib.colors import ListedColormap
    ax.imshow(A, aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#EEF2F5", SOFT[0]]), vmin=0, vmax=1)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels(mods, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("subjects (sorted by group, then coverage)")
    ax.set_yticks([])
    # group separators
    start = 0
    for g in dict.fromkeys(groups):
        end = start + groups.count(g)
        if start:
            ax.axhline(start - 0.5, color=INK, lw=0.8, alpha=0.5)
        ax.text(-0.6, (start + end - 1) / 2, str(g), rotation=90,
                va="center", ha="right", fontsize=8, color=MUTED)
        start = end
    ax.set_title("Modality availability per subject  (filled = available)", loc="left")
    ax2 = fig.add_subplot(gs[0, 1])
    gser = pd.Series(groups).value_counts()
    x = np.arange(len(gser))
    width = 0.8 / max(1, len(mods))
    for j, m in enumerate(mods):
        vals = np.array([A[np.array(groups) == g, j].mean()
                         for g in gser.index])
        ax2.bar(x + (j - (len(mods) - 1) / 2) * width, vals, width,
                color=SOFT[j % len(SOFT)], label=m, edgecolor="white",
                linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(gser.index, rotation=0, fontsize=8)
    ax2.set_ylabel("fraction available")
    ax2.set_title("Coverage by group", loc="left")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=6, ncol=2, loc="upper left", frameon=False)
    ax3 = fig.add_subplot(gs[1, 1])
    pat = pd.Series(["".join(map(str, row)) for row in A]).value_counts().head(12)
    ax3.barh(range(len(pat))[::-1], pat.values, color=SOFT[1], height=0.65)
    ax3.set_yticks(range(len(pat))[::-1])
    ax3.set_yticklabels(pat.index, fontsize=7)
    ax3.set_xlabel("subjects")
    ax3.set_title("Top availability patterns", loc="left")
    for ax in (ax2, ax3):
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    _save(fig, out_path or root / "viz" / "availability")


__all__ = ["SOFT", "set_soft_style", "plot_network",
           "plot_roc_calibration", "plot_availability"]


def plot_umap(data_root, out_path=None, group_col=None, modality=None,
              n_neighbors=15, min_dist=0.10, random_state=0):
    """2-D UMAP of the concatenated feature space (PCA fallback if umap-learn
    is not installed). Points are coloured by group with the soft palette."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    set_soft_style()
    root = Path(data_root)
    meta_path = root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError("metadata.tsv is required for the UMAP plot")
    meta = pd.read_csv(meta_path, sep="\t" if meta_path.suffix == ".tsv" else ",",
                       dtype=str)
    id_col = "SampleID" if "SampleID" in meta.columns else meta.columns[0]
    gcol = group_col or ("Group" if "Group" in meta.columns else ("group" if "group" in meta.columns else None))
    ids = meta[id_col].astype(str).str.strip().tolist()
    groups = meta[gcol].astype(str).tolist() if gcol else ["all"] * len(ids)
    feats = _feature_tables(root)
    if modality:
        feats = {k: v for k, v in feats.items() if str(k) == str(modality)}
    if not feats:
        raise FileNotFoundError("features/*.tsv are required for the UMAP plot")
    parts = []
    for m, df in sorted(feats.items()):
        d = df.reindex(ids)
        d = d.apply(pd.to_numeric, errors="coerce")
        d = d.loc[:, d.notna().any(axis=0)]
        if d.shape[1]:
            parts.append(d.add_prefix(f"{m}::"))
    if not parts:
        raise ValueError("no numeric features found")
    X = pd.concat(parts, axis=1)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    Xs = StandardScaler().fit_transform(X.values)
    try:
        import umap  # type: ignore
        emb = umap.UMAP(n_components=2, n_neighbors=int(n_neighbors),
                        min_dist=float(min_dist), random_state=random_state,
                        n_jobs=1).fit_transform(Xs)
        method = "UMAP"
    except Exception:
        emb = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
        method = "PCA (install avail-omics[viz] for UMAP)"
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ug = list(dict.fromkeys(groups))
    for i, g in enumerate(ug):
        idx = [j for j, x in enumerate(groups) if x == g]
        ax.scatter(emb[idx, 0], emb[idx, 1], s=28, alpha=0.85,
                   color=SOFT[i % len(SOFT)], edgecolor="white",
                   linewidth=0.4, label=str(g))
    short = method.split(" (")[0]
    ax.set_xlabel(f"{short}-1")
    ax.set_ylabel(f"{short}-2")
    ax.set_title(f"Feature embedding ({short})", loc="left")
    if "install" in method:
        ax.text(0.99, 0.01, "install avail-omics[viz] for UMAP",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=MUTED)
    ax.legend(title=gcol or "group", loc="best")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _save(fig, out_path or root / "viz" / "umap")


def plot_volcano(out, out_path=None, comparison=None, modality=None,
                 q_thresh=0.05, fc_thresh=1.0):
    """Volcano plot from diff_abundance_*.tsv (log2 fold change vs q-value)."""
    set_soft_style()
    out = Path(out)
    if modality:
        cand = out / f"diff_abundance_{modality}.tsv"
        candidates = [cand] if cand.exists() else []
    else:
        candidates = [out / "diff_abundance_all.tsv"] + sorted(out.glob("diff_abundance_*.tsv"))
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("diff_abundance_*.tsv is required for the volcano plot")
    d = pd.read_csv(path, sep="\t")
    need = {"feature", "log2fc"}
    if not need.issubset(d.columns):
        raise ValueError(f"{path.name} needs feature and log2fc columns")
    if comparison is not None and "comparison" in d.columns:
        d = d[d["comparison"].astype(str) == str(comparison)]
    elif "comparison" in d.columns and d["comparison"].nunique() > 1:
        first = d["comparison"].astype(str).iloc[0]
        d = d[d["comparison"].astype(str) == first]
    pcol = "q_value" if "q_value" in d.columns else ("p_value" if "p_value" in d.columns else None)
    if pcol is None:
        raise ValueError("diff_abundance file needs q_value or p_value")
    x = pd.to_numeric(d["log2fc"], errors="coerce")
    p = pd.to_numeric(d[pcol], errors="coerce").clip(lower=1e-300)
    y = -np.log10(p)
    ok = np.isfinite(x) & np.isfinite(y)
    d = d[ok].copy(); x = x[ok]; y = y[ok]
    sig = (p[ok].values <= q_thresh) & (np.abs(x.values) >= fc_thresh)
    up = sig & (x.values > 0)
    down = sig & (x.values < 0)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(x[~sig], y[~sig], s=14, color="#C9D2D9", alpha=0.65,
               edgecolor="white", linewidth=0.3, label="not significant")
    ax.scatter(x[up], y[up], s=22, color=NEG, alpha=0.9,
               edgecolor="white", linewidth=0.4, label="up")
    ax.scatter(x[down], y[down], s=22, color=POS, alpha=0.9,
               edgecolor="white", linewidth=0.4, label="down")
    ax.axhline(-np.log10(q_thresh), color=MUTED, ls="--", lw=0.9)
    for v in (-fc_thresh, fc_thresh):
        ax.axvline(v, color=MUTED, ls="--", lw=0.9)
    score = y.values * (np.abs(x.values) + 1.0)
    top = d.assign(_y=y.values, _score=score).sort_values(
        "_score", ascending=False).head(6)
    xr = max(1e-6, float(np.nanmax(np.abs(x.values))))
    yr = max(1e-6, float(np.nanmax(y.values)))
    for i, (_, r) in enumerate(top.iterrows()):
        xv = float(r["log2fc"]); yv = float(r["_y"])
        xo = 0.015 * xr * (1 if xv >= 0 else -1)
        yo = 0.035 * yr * (i % 3 + 1)
        ax.annotate(str(r["feature"])[:26], (xv, yv),
                    xytext=(xv + xo, yv + yo), textcoords="data",
                    fontsize=6.5, color=INK,
                    ha="left" if xv >= 0 else "right")
    ax.set_xlabel("log2 fold change")
    ax.set_ylabel(f"-log10({pcol})")
    comp = ""
    if "comparison" in d.columns and d["comparison"].nunique():
        comp = f" — {d['comparison'].astype(str).iloc[0]}"
    ax.set_title(f"Volcano{comp}", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _save(fig, out_path or out / "viz" / "volcano")


def plot_forest(out, out_path=None, table=None, label_col=None,
                effect_col=None, lo_col=None, hi_col=None, title=None):
    """Forest plot for a table with effect + 95% CI columns.

    Accepted column aliases include label/competitor/feature/model/term,
    effect/delta/log2fc/estimate/mean, lo95/ci_low/lower/ci_lo and
    hi95/ci_high/upper/ci_hi.
    """
    set_soft_style()
    out = Path(out)
    path = Path(table) if table else None
    if path is None or not path.exists():
        for name in ("forest.tsv", "effect_sizes.tsv", "effects.tsv",
                     "TableS5_delta_ci_multi.tsv", "TableS5_delta_ci_binary.tsv"):
            q = out / name
            if q.exists():
                path = q
                break
    if path is None or not path.exists():
        raise FileNotFoundError(
            "forest plot needs a table with effect and 95% CI columns "
            "(e.g. forest.tsv / TableS5_delta_ci_*.tsv)")
    d = pd.read_csv(path, sep="\t")
    def pick(*names):
        for n in names:
            if n in d.columns:
                return n
        return None
    lab = label_col or pick("label", "competitor", "feature", "model", "term")
    eff = effect_col or pick("delta", "effect", "log2fc", "estimate", "mean")
    lo = lo_col or pick("lo95", "ci_low", "lower", "ci_lo")
    hi = hi_col or pick("hi95", "ci_high", "upper", "ci_hi")
    if not all((lab, eff, lo, hi)):
        raise ValueError(f"{path.name} needs label/effect/lo/hi columns; "
                         f"found {list(d.columns)}")
    d = d[[lab, eff, lo, hi]].dropna()
    d[eff] = pd.to_numeric(d[eff], errors="coerce")
    d[lo] = pd.to_numeric(d[lo], errors="coerce")
    d[hi] = pd.to_numeric(d[hi], errors="coerce")
    d = d.dropna().sort_values(eff)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.2, max(3.2, 0.32 * len(d) + 1.4)))
    ax.axvline(0, color=MUTED, lw=1, ls="--")
    for i, (_, r) in enumerate(d.iterrows()):
        c = NEG if r[eff] > 0 else POS
        ax.plot([r[lo], r[hi]], [i, i], color=c, lw=1.8, solid_capstyle="round")
        ax.plot(r[eff], i, "o", color=c, ms=6, markeredgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([str(v)[:38] for v in d[lab]], fontsize=8)
    ax.set_xlabel("effect (95% CI)")
    ax.set_title(title or f"Forest — {path.stem}", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    _save(fig, out_path or out / "viz" / "forest")


__all__ = ["SOFT", "set_soft_style", "plot_network",
           "plot_roc_calibration", "plot_availability",
           "plot_umap", "plot_volcano", "plot_forest"]
