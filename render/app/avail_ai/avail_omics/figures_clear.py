# -*- coding: utf-8 -*-
"""Clear, web-friendly figures: one message per panel, honest encodings.

Rules kept: axes include zero, value labels on every mark, chance line
explicit, n annotated, color is redundant with labels/markers, PNG only
(manuscript figures remain in figures_publication.py)."""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN, VERM, SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9"
GRID, EDGE, MUTED = "#E6ECF2", "#C9D3DE", "#7D8B9A"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "axes.edgecolor": EDGE,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 11, "ytick.labelsize": 11.5,
    "axes.unicode_minus": False,
})


def _save(fig, out_dir, name, source=None):
    # provenance is carried by the UI captions and the downloadable tables
    fig.savefig(Path(out_dir) / f"{name}.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  + {name}.png", flush=True)


def _lolli(ax, labels, values, colors, labels_right=True, fmt="{:.3f}"):
    y = np.arange(len(labels))
    for i, (v, c) in enumerate(zip(values, colors)):
        ax.plot([0, v], [i, i], color=GRID, lw=3, solid_capstyle="round",
                zorder=1)
        ax.scatter(v, i, s=190, color=c, edgecolor="white", linewidth=1.6,
                   zorder=3)
        if labels_right:
            ax.text(v + 0.02, i, fmt.format(v), va="center", fontsize=10.5,
                    fontweight="bold", color="#24313F")


def _classes(out_dir):
    p = Path(out_dir) / "subject_classes.tsv"
    if not p.exists():
        return None
    d = pd.read_csv(p, sep="\t")
    return d if {"patient", "class"}.issubset(d.columns) else None


def _counts(d):
    if d is None:
        return ""
    c = d["class"].value_counts()
    order = [g for g in ["HC", "MCI", "PDD"] if g in c]
    other = [g for g in c.index if g not in order]
    parts = [f"{g} {int(c[g])}" for g in order + other]
    return " · ".join(parts)


def fig_model_performance(out_dir):
    p = Path(out_dir) / "meta_benchmark_results.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    d = d[d.subgroup == "all"] if "subgroup" in d.columns else d
    d = d.sort_values("macro_auc")
    n = int(d.n.iloc[0]) if "n" in d.columns else len(d)
    fig, ax = plt.subplots(figsize=(7.8, 0.55 * len(d) + 1.7),
                           layout="constrained")
    cols = [BLUE] * len(d)
    cols[-1] = ORANGE
    _lolli(ax, d.model.tolist(), d.macro_auc.tolist(), cols)
    ax.set_yticks(range(len(d)), d.model.tolist())
    ax.axvline(0.5, color=MUTED, ls=(0, (4, 3)), lw=1.2)
    ax.set_ylim(-0.75, len(d) - 0.35)
    ax.text(0.507, -0.55, "chance (0.5)", fontsize=9.5, color=MUTED)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Macro-AUC, leave-one-out  (0 = always wrong, 1 = perfect)")
    grp = _counts(_classes(out_dir))
    ax.set_title(f"Model performance  ·  n = {n}" +
                 (f"  ({grp})" if grp else ""), pad=12, fontsize=13)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig1_model_performance", "meta_benchmark_results.tsv")


def fig_modality_coverage(out_dir):
    p = Path(out_dir) / "base_predictions_generic.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    if "patient" not in d.columns or "modality" not in d.columns:
        return
    sc = _classes(out_dir)
    mods = sorted(d.modality.dropna().unique())
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(mods) + 1.9),
                           layout="constrained")
    if sc is not None:
        m = d.merge(sc[["patient", "class"]], on="patient", how="left")
        groups = [g for g in ["HC", "MCI", "PDD"] if g in set(m["class"])]
        colors = {"HC": BLUE, "MCI": ORANGE, "PDD": GREEN}
        h = 0.8 / max(1, len(groups))
        for i, mod in enumerate(mods):
            for k, g in enumerate(groups):
                sub = m[(m.modality == mod) & (m["class"] == g)]
                base = m[m["class"] == g]
                pct = 100 * sub.patient.nunique() / max(1, base.patient.nunique())
                y = i - 0.4 + h * (k + 0.5)
                ax.barh(y, pct, height=h * 0.9, color=colors[g],
                        edgecolor="white", linewidth=.8)
                ax.text(pct + 1.2, y, f"{pct:.0f}%", va="center", fontsize=8.5)
        ax.set_yticks(range(len(mods)), mods)
        ax.set_title(f"Modality coverage by group  ·  {_counts(sc)}", pad=12,
                     fontsize=13)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[g]) for g in groups]
        fig.legend(handles, groups, loc="outside lower center",
                   ncol=len(groups), frameon=False, fontsize=9)
    else:
        n = d.patient.nunique()
        cov = d.groupby("modality").patient.nunique().sort_values()
        pct = 100 * cov / n
        bars = ax.barh(pct.index, pct.values, color=SKY, height=0.6)
        for b, v in zip(bars, pct.values):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%",
                    va="center", fontsize=10)
        ax.set_xlim(0, 108)
        ax.set_title(f"Modality coverage  ·  n = {n} subjects", pad=12,
                     fontsize=13)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of subjects with the modality")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig2_modality_coverage", "base_predictions_generic.tsv")


def fig_top_biomarkers(out_dir):
    p = Path(out_dir) / "biomarkers_top20_per_modality.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    if "modality" not in d.columns or "feature" not in d.columns:
        return
    mods = list(d.modality.dropna().unique())
    ncol = min(2, len(mods))
    nrow = int(np.ceil(len(mods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.6, 3.6 * nrow),
                             layout="constrained", squeeze=False)
    for k, m in enumerate(mods):
        ax = axes[k // ncol][k % ncol]
        sub = d[d.modality == m].nlargest(8, "importance").iloc[::-1]
        if "spearman" in sub.columns:
            cols = [GREEN if s >= 0 else VERM for s in sub.spearman]
        else:
            cols = [BLUE] * len(sub)
        _lolli(ax, sub.feature.astype(str).tolist(),
               sub.importance.tolist(), cols, fmt="{:.3f}")
        if len(sub) and max(sub.importance) > 0:
            ax.set_xlim(0, max(sub.importance) * 1.35)
        ax.set_yticks(range(len(sub)), sub.feature.astype(str).tolist())
        ax.tick_params(axis="y", labelsize=9.5)
        ax.set_title(f"{m}  ·  top 8 by attribution", fontsize=12.5, pad=8)
        ax.set_xlabel("Attribution importance (model)")
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
        if k == 0:
            h_pos = plt.Line2D([], [], marker="o", ls="", color=GREEN,
                               markersize=9, label="higher with severity")
            h_neg = plt.Line2D([], [], marker="o", ls="", color=VERM,
                               markersize=9, label="lower with severity")
            ax.legend(handles=[h_pos, h_neg], fontsize=9, frameon=False,
                      loc="lower right")
    for k in range(len(mods), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Which features drive the model  ·  color = direction of "
                 "the Spearman correlation", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "fig3_top_biomarkers")  # provenance lives in the UI caption


def fig_subgroup_comparison(out_dir):
    """Availability-stratified macro-AUC per model (slope chart)."""
    p = Path(out_dir) / "meta_benchmark_results.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    if "subgroup" not in d.columns:
        return
    order = [g for g in ["all", "complete", "ge3mod"] if g in set(d.subgroup)]
    if len(order) < 2:
        return
    piv = d[d.subgroup.isin(order)].pivot_table(
        index="model", columns="subgroup", values="macro_auc")
    cols = [c for c in order if c in piv.columns]
    piv = piv.dropna(subset=cols[:1]).sort_values(cols[0])
    n_map = (d[d.subgroup.isin(order)].groupby("subgroup").n.max().to_dict())
    fig, ax = plt.subplots(figsize=(8.4, 5.2), layout="constrained")
    xs = np.arange(len(cols))
    palette = [BLUE, ORANGE, GREEN, VERM, SKY, "#CC79A7", "#F0E442", "#666666"]
    for i, (model, row) in enumerate(piv.iterrows()):
        c = palette[i % len(palette)]
        y = [row.get(k, np.nan) for k in cols]
        ax.plot(xs, y, "-o", color=c, lw=2, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
    # direct labels at the right edge, spaced to avoid overlap
    last = piv[cols[-1]].values
    order_idx = np.argsort(last)[::-1]
    ypos = list(last[order_idx])
    min_gap = 0.022
    for i in range(1, len(ypos)):
        if ypos[i - 1] - ypos[i] < min_gap:
            ypos[i] = ypos[i - 1] - min_gap
    label_y = {piv.index[order_idx[i]]: ypos[i] for i in range(len(ypos))}
    for i, (model, row) in enumerate(piv.iterrows()):
        ax.text(xs[-1] + 0.08, label_y[model], model, fontsize=9.5,
                va="center", color=palette[i % len(palette)],
                fontweight="bold")
    ax.set_xticks(xs, [f"{c}\n(n = {int(n_map.get(c, 0))})" for c in cols])
    ax.set_xlim(-0.25, len(cols) - 1 + 1.55)
    ax.set_ylabel("Macro-AUC")
    ax.set_title("Does performance survive missing modalities?", pad=12)
    ax.axhline(0.5, color=MUTED, ls=(0, (4, 3)), lw=1)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "fig4_subgroup_comparison", "meta_benchmark_results.tsv")


def fig_availability_pattern(out_dir):
    """Subject x modality matrix with a class stripe (HC/MCI/PDD)."""
    p = Path(out_dir) / "base_predictions_generic.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    if "patient" not in d.columns or "modality" not in d.columns:
        return
    mat = (d.assign(v=1).pivot_table(index="patient", columns="modality",
                                     values="v", aggfunc="max")
           .fillna(0))
    order = mat.sum(axis=1).sort_values(ascending=False).index
    mat = mat.loc[order]
    sc = _classes(out_dir)
    cmap_cls = matplotlib.colors.ListedColormap([BLUE, ORANGE, GREEN])
    cls_colors = {"HC": BLUE, "MCI": ORANGE, "PDD": GREEN}
    if sc is not None:
        m = sc.set_index("patient").reindex(mat.index)
        code = {"HC": 0, "MCI": 1, "PDD": 2}
        codes = m["class"].map(code).fillna(-1).values
        fig, (axc, ax) = plt.subplots(
            1, 2, figsize=(8.8, 5.4), layout="constrained",
            gridspec_kw={"width_ratios": [0.035, 0.965]})
        axc.imshow(codes[:, None], aspect="auto", interpolation="nearest",
                   cmap=cmap_cls, vmin=0, vmax=2)
        axc.set_xticks([]); axc.set_yticks([])
        axc.set_ylabel("class", fontsize=9)
        groups = [g for g in ["HC", "MCI", "PDD"] if g in set(m["class"])]
        handles = [plt.Rectangle((0, 0), 1, 1, color=cls_colors[g])
                   for g in groups]
        fig.legend(handles, groups, loc="outside lower center",
                   ncol=len(groups), frameon=False, fontsize=9)
        title = f"Availability is structural  ·  {_counts(sc)}"
    else:
        fig, ax = plt.subplots(figsize=(8.0, 5.4), layout="constrained")
        title = f"Availability is structural  ·  n = {mat.shape[0]} subjects"
    ax.imshow(mat.values, aspect="auto", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap(["#E8EDF3", GREEN]),
              vmin=0, vmax=1)
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=18, ha="right",
                  fontsize=9.5)
    ax.set_yticks([])
    for j in range(mat.shape[1]):
        pct = 100 * mat.iloc[:, j].mean()
        ax.text(j, -0.9, f"{pct:.0f}%", ha="center", va="bottom",
                fontsize=9.5, color=MUTED)
    ax.set_title(title, pad=16, fontsize=13)
    _save(fig, out_dir, "fig5_availability_pattern",
          "base_predictions_generic.tsv")


def fig_diff_abundance(out_dir):
    """Volcano panels per comparison (HC_vs_MCI, HC_vs_PDD, MCI_vs_PDD)."""
    p = Path(out_dir) / "diff_abundance_all.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    need = {"log2fc", "q_value", "feature", "modality", "comparison"}
    if not need.issubset(d.columns):
        return
    d = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log2fc", "q_value"])
    d = d[d.q_value > 0]
    if not len(d):
        return
    comps = [c for c in ["HC_vs_PDD", "MCI_vs_PDD", "HC_vs_MCI"]
             if c in set(d.comparison)]
    if not comps:
        return
    mods = list(d.modality.dropna().unique())
    cols = [BLUE, ORANGE, GREEN, VERM, SKY]
    cmap_mod = {m: cols[i % len(cols)] for i, m in enumerate(mods)}
    fig, axes = plt.subplots(1, len(comps), figsize=(4.4 * len(comps), 4.8),
                             layout="constrained", squeeze=False,
                             sharey=True)
    for k, comp in enumerate(comps):
        ax = axes[0][k]
        sub = d[d.comparison == comp]
        y = -np.log10(sub.q_value.clip(lower=1e-300))
        for m in mods:
            sel = sub.modality == m
            if sel.any():
                ax.scatter(sub.log2fc[sel], y[sel], s=26,
                           color=cmap_mod[m], alpha=.72,
                           edgecolor="white", linewidth=.4)
        ax.axhline(-np.log10(0.05), color=MUTED, ls=(0, (4, 3)), lw=1)
        ax.set_title(comp, fontsize=12.5, pad=26)
        if len(sub):
            top = sub.assign(yv=y).nlargest(1, "yv").iloc[0]
            lab = (str(top.feature)[:30] + "…") if len(str(top.feature)) > 30 \
                else str(top.feature)
            ax.text(0.5, 1.015, f"top: {lab} (q={top.q_value:.1e})",
                    transform=ax.transAxes, fontsize=8.5,
                    color="#24313F", ha="center", va="bottom")
        ax.set_xlabel("log2 fold-change")
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("-log10 q (BH-FDR)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap_mod[m],
                          markersize=7, label=str(m)) for m in mods]
    fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(mods)),
               frameon=False, fontsize=9)
    fig.suptitle("Which features differ between groups?  ·  per comparison",
                 fontsize=14, fontweight="bold")
    _save(fig, out_dir, "fig6_diff_abundance", "diff_abundance_all.tsv")


def fig_per_class_auc(out_dir):
    p = Path(out_dir) / "per_class_auc.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    cls = [c for c in d.columns if c != "model"]
    if not cls:
        return
    mat = d[cls].values.astype(float)
    fig, ax = plt.subplots(figsize=(1.15 * len(cls) + 3.4,
                                    0.48 * len(d) + 2.2),
                           layout="constrained")
    im = ax.imshow(mat, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9.5,
                        color="white" if v < 0.75 else "black")
    ax.set_xticks(range(len(cls)), cls)
    ax.set_yticks(range(len(d)), d.model)
    ax.set_title("Per-class performance  ·  one-vs-rest AUC", pad=12,
                 fontsize=13)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("AUC (0.5 = chance)")
    _save(fig, out_dir, "fig7_per_class_auc", "per_class_auc.tsv")


def fig_group_profiles(out_dir):
    """Raw values of top differential features, per group (HC/MCI/PDD)."""
    p = Path(out_dir) / "top_features_long.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    if not {"feature", "modality", "value", "class"}.issubset(d.columns):
        return
    feats = (d.drop_duplicates(["modality", "feature"])
             [["modality", "feature", "q_value", "comparison"]])
    feats = feats.sort_values("q_value").head(9)
    if not len(feats):
        return
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.4, 3.7 * nrow),
                             layout="constrained", squeeze=False)
    rng = np.random.default_rng(7)
    order = ["HC", "MCI", "PDD"]
    colors = {"HC": BLUE, "MCI": ORANGE, "PDD": GREEN}
    for k, (_, f) in enumerate(feats.iterrows()):
        ax = axes[k // ncol][k % ncol]
        sub = d[(d.modality == f.modality) & (d.feature == f.feature)]
        groups = [g for g in order if (sub["class"] == g).any()]
        data = [sub.loc[sub["class"] == g, "value"].values for g in groups]
        bp = ax.boxplot(data, positions=range(len(groups)), widths=.52,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="#1B2735", lw=1.6))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(colors[g]); patch.set_alpha(.25)
            patch.set_edgecolor(colors[g])
        for i, g in enumerate(groups):
            vals = sub.loc[sub["class"] == g, "value"].values
            xj = i + rng.uniform(-0.17, 0.17, len(vals))
            ax.scatter(xj, vals, s=16, color=colors[g], alpha=.85,
                       edgecolor="white", linewidth=.4, zorder=3)
        positive = sub.value.dropna() > 0
        span = (sub.value.max() / max(sub.value[sub.value > 0].min(), 1e-12)
                if positive.all() and (sub.value > 0).any() else 0)
        if span > 100:
            ax.set_yscale("log")
            ax.set_ylabel("value (log scale)", fontsize=10)
        else:
            ax.set_ylabel("value", fontsize=10)
        ax.set_xticks(range(len(groups)), groups)
        import textwrap as _tw
        lab = _tw.fill(str(f.feature), 30)
        ax.set_title(f"{lab}\n{f.comparison} · q={f.q_value:.1e}",
                     fontsize=9.6, pad=8)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    for k in range(len(feats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Top differential features by group  ·  individual subjects "
                 "shown", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "fig8_group_profiles", "top_features_long.tsv")


def fig_effect_sizes(out_dir):
    """Signed effect sizes of the top differential features per comparison."""
    p = Path(out_dir) / "diff_abundance_all.tsv"
    if not p.exists():
        return
    d = pd.read_csv(p, sep="\t")
    need = {"log2fc", "q_value", "feature", "comparison"}
    if not need.issubset(d.columns):
        return
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["log2fc", "q_value"])
    comps = [c for c in ["HC_vs_PDD", "MCI_vs_PDD", "HC_vs_MCI"]
             if c in set(d.comparison)]
    if not comps:
        return
    fig, axes = plt.subplots(1, len(comps), figsize=(4.6 * len(comps), 4.9),
                             layout="constrained", squeeze=False,
                             sharex=False)
    for k, comp in enumerate(comps):
        ax = axes[0][k]
        sub = d[d.comparison == comp].copy()
        sub["score"] = -np.log10(sub.q_value.clip(lower=1e-300))
        sub = sub.nlargest(8, "score").sort_values("log2fc")
        y = np.arange(len(sub))
        cols = [GREEN if v >= 0 else VERM for v in sub.log2fc]
        ax.barh(y, sub.log2fc, color=cols, height=.62, edgecolor="white",
                linewidth=.8)
        ax.axvline(0, color="#4A5A6A", lw=1)
        lo, hi = ax.get_xlim()
        rng_ = hi - lo
        ax.set_xlim(lo - 0.02 * rng_, hi + 0.34 * rng_)
        qx = hi + 0.03 * rng_
        ax.text(qx, len(sub) - 0.25, "q (BH-FDR)", fontsize=8,
                color=MUTED, va="bottom")
        for yi, (_, rr) in zip(y, sub.iterrows()):
            ax.text(qx, yi, f"{rr.q_value:.1e}", va="center", ha="left",
                    fontsize=7.8, color="#4A5A6A")
        labs = [(str(f)[:26] + "…") if len(str(f)) > 26 else str(f)
                for f in sub.feature]
        ax.set_yticks(y, labs)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_title(comp, fontsize=12, pad=10)
        ax.set_xlabel("log2 fold-change (signed)")
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Effect sizes of the top differential features",
                 fontsize=14, fontweight="bold")
    _save(fig, out_dir, "fig9_effect_sizes", "diff_abundance_all.tsv")


def make_clear_figures(out_dir):
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    fig_model_performance(out_dir)
    fig_modality_coverage(out_dir)
    fig_top_biomarkers(out_dir)
    fig_subgroup_comparison(out_dir)
    fig_availability_pattern(out_dir)
    fig_diff_abundance(out_dir)
    fig_per_class_auc(out_dir)
    fig_group_profiles(out_dir)
    fig_effect_sizes(out_dir)
