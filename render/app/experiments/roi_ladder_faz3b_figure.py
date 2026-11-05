#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 3b: ladder figure + delta table from ladder_summary.tsv.

Outputs (experiments/results/roi_ladder/):
  ladder_figure.png / .pdf    macro-AUC vs rung per arm (7 models + mean)
  ladder_delta.tsv            avail - complete per rung per model
  ladder_coverage.tsv         n per rung per arm
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/mnt/limbic/AVAIL_GPU/experiments/results/roi_ladder")
SUM = OUT / "ladder_summary.tsv"

COLORS = {
    "SimpleMean": "#9aa0a6", "WSimpleMean": "#5f6368",
    "ElasticNetLR": "#1a73e8", "XGBoost": "#188038",
    "MLP": "#d93025", "TabPFN": "#f29900", "TabICL": "#7b1fa2",
}


def main():
    s = pd.read_csv(SUM, sep="\t")
    if s.empty:
        print("[faz3b] no ladder_summary yet")
        return
    s["modules"] = s["modules"].astype(str)
    arm_style = {"complete": ("#b00020", "--", "complete-case"),
                 "avail": ("#1a73e8", "-", "AVAIL (n=150)")}
    mod_labels = ["fMRI", "+stool16S", "+stoolSG", "+saliva16S", "+salivaSG"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rungs = sorted(s["rung"].unique())
    for arm in ["complete", "avail"]:
        sub = s[s["arm"] == arm]
        color, ls, lbl = arm_style[arm]
        for model in s["model"].unique():
            m = sub[sub["model"] == model].sort_values("rung")
            ax.plot(m["rung"], m["macro_auc"], ls=ls, lw=1.1,
                    color=COLORS.get(model, "#888"),
                    alpha=0.35, zorder=1)
        mean = sub.groupby("rung")["macro_auc"].mean()
        ax.plot(mean.index, mean.values, ls=ls, lw=2.6, color=color,
                marker="o", ms=6, zorder=3,
                label=f"{lbl} (mean of 7 models)")
    ax.set_xticks(rungs)
    ax.set_xticklabels(mod_labels, fontsize=9)
    ax.set_xlabel("Modules added (cumulative ladder)", fontsize=11)
    ax.set_ylabel("3-class macro-AUC (LOO)", fontsize=11)
    ax.set_ylim(0.35, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Module ladder: complete-case vs AVAIL (zero sample loss)",
                 fontsize=12)
    ax2 = ax.twinx()
    cov = s[s["arm"] == "avail"].groupby("rung")["n"].first()
    comp_n = s[s["arm"] == "complete"].groupby("rung")["n"].first()
    ax2.plot(rungs, cov.values, ls=":", color="#1a73e8", lw=1.2, alpha=0.7)
    ax2.plot(rungs, comp_n.values, ls=":", color="#b00020", lw=1.2, alpha=0.7)
    ax2.set_ylabel("n (coverage)", fontsize=11)
    for r, n_ in comp_n.items():
        ax2.annotate(str(n_), (r, n_), textcoords="offset points",
                     xytext=(2, -12), fontsize=8, color="#b00020")
    for r, n_ in cov.items():
        ax2.annotate(str(n_), (r, n_), textcoords="offset points",
                     xytext=(-14, 4), fontsize=8, color="#1a73e8")
    fig.tight_layout()
    fig.savefig(OUT / "ladder_figure.png", dpi=300)
    fig.savefig(OUT / "ladder_figure.pdf")
    plt.close(fig)
    print("[faz3b] ladder_figure written")

    delta_rows = []
    for rung in rungs:
        for model in s["model"].unique():
            a = s[(s["rung"] == rung) & (s["arm"] == "avail")
                  & (s["model"] == model)]["macro_auc"]
            c = s[(s["rung"] == rung) & (s["arm"] == "complete")
                  & (s["model"] == model)]["macro_auc"]
            if len(a) and len(c):
                delta_rows.append({"rung": rung, "model": model,
                                   "avail": a.iloc[0], "complete": c.iloc[0],
                                   "delta": a.iloc[0] - c.iloc[0]})
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "ladder_delta.tsv", sep="\t", index=False)
    piv = delta.pivot_table(index="model", columns="rung", values="delta")
    print(piv.round(3).to_string())
    cov_df = pd.DataFrame({"rung": rungs,
                           "n_complete": comp_n.values,
                           "n_avail": cov.values,
                           "modules": mod_labels})
    cov_df.to_csv(OUT / "ladder_coverage.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
