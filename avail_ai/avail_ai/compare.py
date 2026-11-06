# -*- coding: utf-8 -*-
"""Cross-cohort comparison of AVAIL meta-benchmark results.

The function is deliberately generic: it reads one or more results folders
(``meta_benchmark_results.tsv`` or ``meta_benchmark_3class_results.tsv``)
and writes a machine-readable summary plus a two-panel scientific figure.
No cohort names or manuscript-specific labels are hard-coded.
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CPAL = ["#2E5A87", "#5B9279", "#C77D9D", "#B0A7C9", "#3D2E5F",
        "#4393C3", "#4DAC26", "#D6604D", "#8C6D31", "#666666"]

_COMPLETE_KEYS = ("complete", "complete5", "complete4", "complete3",
                  "complete2")


def _read_results(results_dir):
    d = Path(results_dir)
    for name in ("meta_benchmark_results.tsv",
                 "meta_benchmark_3class_results.tsv"):
        p = d / name
        if p.exists():
            return pd.read_csv(p, sep="\t")
    raise FileNotFoundError(
        f"no meta_benchmark_results.tsv or meta_benchmark_3class_results.tsv "
        f"under {d}")


def _complete_subgroup(df):
    present = set(df["subgroup"].astype(str))
    for key in _COMPLETE_KEYS:
        if key in present:
            return key
    return "all"


def _summary(df, subgroup, exclude_gnn=True):
    s = df[df.subgroup.astype(str) == subgroup].copy()
    if exclude_gnn:
        s = s[~s.model.str.contains("GNN", case=False, na=False)]
    if s.empty:
        return dict(n=0, mean=np.nan, lo=np.nan, hi=np.nan, best=np.nan,
                    best_model="")
    best = s.sort_values("macro_auc", ascending=False).iloc[0]
    return dict(n=int(s.n.max()), mean=float(s.macro_auc.mean()),
                lo=float(s.macro_auc.min()), hi=float(s.macro_auc.max()),
                best=float(best.macro_auc), best_model=str(best.model))


def compare_results(entries, out_dir, exclude_gnn=True, title=None):
    """Compare AVAIL results across cohorts.

    Parameters
    ----------
    entries : list of dict
        Each entry has ``label`` and ``results`` (path to a run directory).
    out_dir : str or Path
        Directory for ``cohort_comparison.tsv`` and figure files.
    exclude_gnn : bool
        Exclude GNN learners from the summary (default True, matching the
        manuscript convention).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in entries:
        label = str(e["label"])
        df = _read_results(e["results"])
        if e.get("complete_results"):
            cdf = _read_results(e["complete_results"])
            comp = "all"
        else:
            cdf = df
            comp = _complete_subgroup(df)
        a = _summary(df, "all", exclude_gnn=exclude_gnn)
        c = _summary(cdf, comp, exclude_gnn=exclude_gnn)
        rows.append({
            "cohort": label,
            "results_dir": str(e["results"]),
            "complete_results_dir": str(e.get("complete_results") or ""),
            "complete_subgroup": comp,
            "n_all": a["n"], "n_complete": c["n"],
            "mean_all": a["mean"], "min_all": a["lo"], "max_all": a["hi"],
            "best_all": a["best"], "best_model_all": a["best_model"],
            "mean_complete": c["mean"], "min_complete": c["lo"],
            "max_complete": c["hi"], "best_complete": c["best"],
            "best_model_complete": c["best_model"],
        })
    data = pd.DataFrame(rows)
    data.to_csv(out / "cohort_comparison.tsv", sep="\t", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return data

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, max(2.4, 0.42 * len(data) + 1.2)),
        gridspec_kw={"width_ratios": [1.0, 1.15]})
    yy = np.arange(len(data))[::-1]

    # Panel a: complete-case coverage
    for y, (_, r) in zip(yy, data.iterrows()):
        frac = r.n_complete / r.n_all if r.n_all else np.nan
        ax1.barh(y, frac, height=0.56, color=CPAL[1], edgecolor="white",
                 linewidth=0.5)
        ax1.barh(y, 1 - frac, left=frac, height=0.56, color="#E8E8E8",
                 edgecolor="white", linewidth=0.5, hatch="///")
        ax1.text(1.02, y, f"{int(r.n_complete)}/{int(r.n_all)}", va="center",
                 ha="left", fontsize=6.2, color="#333333")
    ax1.set_yticks(yy)
    ax1.set_yticklabels(data.cohort)
    ax1.set_xlim(0, 1.20)
    ax1.set_xticks([0, .25, .5, .75, 1.0])
    ax1.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax1.set_xlabel("Complete-case retention")
    ax1.set_title("a  Complete-case coverage", loc="left", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel b: mean macro-AUC with model-ensemble range
    for y, (_, r) in zip(yy, data.iterrows()):
        for mean, lo, hi, color, offset in [
                (r.mean_all, r.min_all, r.max_all, CPAL[0], 0.09),
                (r.mean_complete, r.min_complete, r.max_complete, CPAL[3], -0.09)]:
            if np.isnan(mean):
                continue
            ax2.plot([lo, hi], [y + offset, y + offset], color=color,
                     lw=1.0, alpha=0.55, zorder=1)
            ax2.plot(mean, y + offset, "o", ms=3.4, color=color,
                     mec="white", mew=0.45, zorder=3)
    ax2.axvline(0.5, color="#999999", lw=0.7, ls=(0, (2, 2)), zorder=0)
    ax2.set_yticks(yy)
    ax2.set_yticklabels(data.cohort)
    ax2.set_xlim(0.30, 1.02)
    ax2.set_xlabel("Macro-AUC (mean; range across meta-learners)")
    ax2.set_title("b  Macro-AUC by cohort", loc="left", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CPAL[0],
               markeredgecolor="white", markersize=4.5, label="all available"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CPAL[3],
               markeredgecolor="white", markersize=4.5, label="complete-case"),
    ], loc="lower left", frameon=False, handlelength=1.2,
        borderaxespad=0.2)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=8, fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png", "tiff"):
        fig.savefig(out / f"cohort_comparison.{ext}", bbox_inches="tight",
                    facecolor="white", dpi=600)
    return data
