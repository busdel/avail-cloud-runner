#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decision-curve (net-benefit) module for the web comparison suite.

Reads a finished generic job's loo_predictions.tsv (columns: patient, y,
"{Model}_{Class}" probability blocks) and produces net-benefit curves for
the extreme-class dichotomy (first vs last class) and, when K >= 3, for the
middle classes one-vs-rest. Writes decision_curves.png/pdf and
decision_impact.tsv into --out.
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "MoE", "GNN", "Transformer", "TabNet", "TabPFN", "TabICL"]
THR = np.linspace(0.05, 0.95, 91)


def parse_blocks(df):
    cols = [c for c in df.columns if c not in ("patient", "y")]
    blocks = {}
    for m in MODELS:
        mc = [c for c in cols if c.startswith(m + "_")]
        if mc:
            blocks[m] = mc
    if not blocks:
        raise SystemExit("no model probability blocks found in loo file")
    ref = blocks[list(blocks)[0]]
    labels = [c.split("_", 1)[1] for c in ref]
    return blocks, labels


def net_benefit(yb, p, thr):
    n = len(yb)
    out = []
    for t in thr:
        pos = p >= t
        tp = np.sum(pos & (yb == 1))
        fp = np.sum(pos & (yb == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.loo, sep="\t")
    if "y" not in df.columns:
        raise SystemExit("loo file has no y column")
    y = df["y"].values
    blocks, labels = parse_blocks(df)
    K = len(labels)
    codes = sorted(set(y.tolist()))

    targets = []
    if K == 2:
        targets.append((f"{labels[0]} vs {labels[1]}", 0, 1))
    else:
        targets.append((f"{labels[0]} vs {labels[-1]}", 0, K - 1))
        for mid in range(1, K - 1):
            targets.append((f"{labels[mid]} vs rest", mid, None))

    fig, axes = plt.subplots(1, len(targets), figsize=(4.1 * len(targets), 3.2),
                             squeeze=False)
    rows = []
    for ax, (name, a, b) in zip(axes[0], targets):
        if b is None:
            sel = np.ones(len(y), dtype=bool)
            yb = (y == a).astype(int)
            prob = lambda m: df[blocks[m][a]].values  # noqa: E731
        else:
            sel = np.isin(y, [a, b])
            yb = (y[sel] == b).astype(int)
            prob = lambda m: (df[blocks[m][b]].values[sel] /
                              (df[blocks[m][a]].values[sel] +
                               df[blocks[m][b]].values[sel] + 1e-12))  # noqa: E731
        curves = {}
        for m in blocks:
            p = prob(m)
            if np.all(np.isfinite(p)):
                curves[m] = net_benefit(yb, p, THR)
        if not curves:
            continue
        M = np.vstack(list(curves.values()))
        ax.plot(THR, M.mean(axis=0), color="#1F7A6D", lw=1.6,
                label=f"zoo mean (n={len(yb)})")
        ax.fill_between(THR, M.min(axis=0), M.max(axis=0),
                        color="#1F7A6D", alpha=0.13, lw=0)
        prev = yb.mean()
        ax.plot(THR, prev - (1 - prev) * THR / (1 - THR), "k--", lw=1.0,
                label="treat all")
        ax.plot(THR, np.zeros_like(THR), "k:", lw=1.0, label="treat none")
        ax.set_title(name)
        ax.set_xlabel("threshold probability")
        ax.set_ylim(-0.15, max(0.7, prev + 0.1))
        ax.grid(alpha=0.15, lw=0.4)
        ax.legend(fontsize=5.8, loc="upper right", frameon=False)
        for t in (0.2, 0.3, 0.4, 0.5):
            i = int(np.argmin(np.abs(THR - t)))
            nbv = np.array([c[i] for c in curves.values()])
            rows.append({"target": name, "threshold": t,
                         "nb_mean": nbv.mean(), "nb_min": nbv.min(),
                         "nb_max": nbv.max(), "n_models": len(nbv)})
    axes[0][0].set_ylabel("net benefit")
    fig.tight_layout()
    fig.savefig(out / "decision_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "decision_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    imp = pd.DataFrame(rows)
    imp.to_csv(out / "decision_impact.tsv", sep="\t", index=False)
    print("[decision] targets:", [t[0] for t in targets])
    print(imp.round(4).to_string(index=False))
    print("[decision] saved to", out)


if __name__ == "__main__":
    main()
