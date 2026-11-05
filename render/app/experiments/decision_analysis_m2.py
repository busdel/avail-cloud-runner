#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision-theoretic layer (Paper 2, workstream B).
Net-benefit decision curves for the rung-5 AVAIL vs complete-case arms,
two clinical targets (HC vs PDD; MCI vs rest), zoo-mean bands, decision
impact table at clinically relevant thresholds.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE = Path("/mnt/limbic/AVAIL_GPU")
PRED = BASE / "experiments/results/roi_ladder/predictions"
OUT = BASE / "experiments/results/factorial"
OUT.mkdir(parents=True, exist_ok=True)
FIGS = BASE / "revizyon/figür_tablolari"
SUPP = BASE / "revizyon/makaleler/supplement"

_fonts = [str(p) for p in
          (Path("/usr/share/fonts/truetype/liberation").glob("*.ttf"))]
for f in _fonts:
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({"font.family": "Liberation Sans" if _fonts else "DejaVu Sans",
                     "font.size": 7, "pdf.fonttype": 42})

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "TabPFN", "TabICL"]
C_AVAIL = "#1F7A6D"
C_CC = "#C74B4A"
THR = np.linspace(0.05, 0.95, 91)


def load(fname):
    return np.load(PRED / fname)


def binary_probs(z, model, target):
    P = z[model]
    if target == "HC-PDD":
        sel = np.isin(z["y"], [0, 2])
        p = P[sel][:, 2] / (P[sel][:, 0] + P[sel][:, 2] + 1e-12)
        yb = (z["y"][sel] == 2).astype(int)
    else:  # MCI vs rest
        sel = np.ones(len(z["y"]), dtype=bool)
        p = P[sel][:, 1]
        yb = (z["y"][sel] == 1).astype(int)
    return yb, p


def net_benefit(yb, p, thr):
    nb = []
    n = len(yb)
    for t in thr:
        pos = p >= t
        tp = np.sum(pos & (yb == 1))
        fp = np.sum(pos & (yb == 0))
        nb.append(tp / n - fp / n * t / (1 - t))
    return np.array(nb)


def curves(z, target):
    out = {}
    for m in MODELS:
        yb, p = binary_probs(z, m, target)
        out[m] = net_benefit(yb, p, THR)
    return out


za = load("rung5_avail.npz")
zc = load("rung5_complete.npz")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
rows = []
for ax, target in zip(axes, ["HC-PDD", "MCI-rest"]):
    for z, arm, color in [(za, "AVAIL", C_AVAIL), (zc, "complete", C_CC)]:
        cs = curves(z, target)
        M = np.vstack([cs[m] for m in MODELS])
        mean = M.mean(axis=0)
        ax.fill_between(THR, M.min(axis=0), M.max(axis=0),
                        color=color, alpha=0.12, lw=0)
        ax.plot(THR, mean, color=color, lw=1.6,
                label=f"{arm} (n={len(z['y'])}, zoo mean)")
    yb, p = binary_probs(za, "SimpleMean", target)
    prev = yb.mean()
    ax.plot(THR, prev - (1 - prev) * THR / (1 - THR), "k--", lw=1.0,
            label="treat all")
    ax.plot(THR, np.zeros_like(THR), "k:", lw=1.0, label="treat none")
    ax.set_xlabel("threshold probability")
    ax.set_title(f"{target}")
    ax.set_ylim(-0.15, max(0.65, prev + 0.1))
    ax.grid(alpha=0.15, lw=0.4)
    if target == "HC-PDD":
        ax.set_ylabel("net benefit")
        ax.legend(fontsize=5.8, loc="upper right", frameon=False)
    for t in [0.2, 0.3, 0.4, 0.5]:
        i = int(np.argmin(np.abs(THR - t)))
        nb_a = np.mean([curves(za, target)[m][i] for m in MODELS])
        nb_c = np.mean([curves(zc, target)[m][i] for m in MODELS])
        per_model = [curves(za, target)[m][i] - curves(zc, target)[m][i]
                     for m in MODELS]
        rows.append({"target": target, "threshold": t,
                     "nb_avail_mean": nb_a, "nb_complete_mean": nb_c,
                     "delta_nb_mean": nb_a - nb_c,
                     "delta_nb_min": np.min(per_model),
                     "delta_nb_max": np.max(per_model),
                     "extra_net_detections_per_1000": 1000 * (nb_a - nb_c)})
fig.tight_layout()
for ext, dpi in [("png", 600), ("pdf", 300)]:
    fig.savefig(FIGS / f"figM2_3_decision_curves.{ext}", dpi=dpi,
                bbox_inches="tight")
plt.close(fig)

imp = pd.DataFrame(rows)
imp.to_csv(OUT / "decision_impact.tsv", sep="\t", index=False)
print(imp.round(4).to_string(index=False))

md = ["# Table S13. Decision impact of the flag representation "
      "(rung 5, HC-PDD and MCI-vs-rest)",
      "",
      "Net benefit (NB) differences between the AVAIL arm (n=150) and the "
      "matched complete-case arm (n=50) at clinically relevant decision "
      "thresholds; zoo mean over seven meta-learners, with per-model range. "
      "`extra_net_detections_per_1000` = 1000 x mean delta-NB.",
      "",
      imp.round(3).to_markdown(index=False), ""]
(SUPP / "TableS13_decision_impact.md").write_text("\n".join(md))
print("saved figure + table")
