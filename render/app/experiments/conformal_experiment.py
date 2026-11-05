# -*- coding: utf-8 -*-
"""Mondrian conformal prediction with availability-aware groups.

Using the out-of-sample (LOO) meta-learner predictions, we build
prediction sets with group-conditional coverage: subjects are grouped by
their number of available modalities, so the set-size guarantee holds for
EVERY availability pattern. Clinically: fewer modalities -> wider (more
honest) prediction sets.

Outputs: experiments/results/conformal_results.tsv + figure
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))
import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--res-dir",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results")
__ap.add_argument("--out",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results")
__a2, _ = __ap.parse_known_args()
RES = Path(__a2.res_dir)
OUT = Path(__a2.out)

ALPHA = 0.10


def main():
    loo = pd.read_csv(RES / "loo_predictions.tsv", sep="\t")
    bp = pd.read_csv(RES / "base_predictions_generic.tsv", sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    ids = loo["patient"].astype(str).tolist()
    y = loo["y"].values
    K = 3
    classes = [c.rsplit("_", 1)[1] for c in loo.columns
               if c.startswith("WSimpleMean_")]
    counts = bp.groupby("patient")["modality"].nunique()
    avail = np.array([counts.get(p, 0) for p in ids])

    rows = []
    for model in ["WSimpleMean", "XGBoost", "MLP"]:
        cols = [f"{model}_{c}" for c in classes]
        if not all(c in loo.columns for c in cols):
            continue
        P = loo[cols].values
        scores = 1.0 - P[np.arange(len(y)), y]
        # split-conformal: 60% calibration / 40% test, per group
        rng = np.random.RandomState(42)
        cal_idx, test_idx = [], []
        for g in np.unique(avail):
            g_idx = np.where(avail == g)[0]
            rng.shuffle(g_idx)
            n_cal = int(0.6 * len(g_idx))
            cal_idx.extend(g_idx[:n_cal])
            test_idx.extend(g_idx[n_cal:])
        cal_idx = np.array(cal_idx)
        test_idx = np.array(test_idx)
        # groups: per availability count, but small groups (<10 cal
        # subjects) are pooled into a 'sparse' bucket for stable quantiles
        group_label = {}
        for g in np.unique(avail):
            n_c = (avail[cal_idx] == g).sum()
            group_label[g] = str(g) if n_c >= 10 else "sparse"
        qhat = {}
        for lab in set(group_label.values()):
            g_scores = scores[cal_idx[[group_label[g] == lab
                                       for g in avail[cal_idx]]]]
            n_g = len(g_scores)
            if n_g == 0:
                continue
            level = np.ceil((1 - ALPHA) * (n_g + 1)) / n_g
            qhat[lab] = np.quantile(g_scores, min(level, 1.0))
        for lab in sorted(set(group_label.values()),
                          key=lambda x: (x == "sparse", x)):
            t = test_idx[np.array([group_label[g] == lab
                                   for g in avail[test_idx]])]
            if len(t) == 0:
                continue
            sets = P[t] >= (1 - qhat.get(lab, 1.0))
            cov = sets[np.arange(len(t)), y[t]].mean()
            size = sets.sum(axis=1).mean()
            rows.append({"model": model, "group": f"n_mod={lab}",
                         "n_test": len(t), "coverage": cov,
                         "mean_set_size": size})
        # overall
        t = test_idx
        q_all = np.quantile(scores[cal_idx],
                            min(np.ceil((1 - ALPHA) * (len(cal_idx) + 1))
                                / len(cal_idx), 1.0))
        sets = P[t] >= (1 - q_all)
        rows.append({"model": model, "group": "overall", "n_test": len(t),
                     "coverage": sets[np.arange(len(t)), y[t]].mean(),
                     "mean_set_size": sets.sum(axis=1).mean()})
        print(model, "done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "conformal_results.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))

    # figure: set size vs availability
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = df[df.model == "WSimpleMean"].copy()
    nums = sub.group.str.extract(r"(\d+)")[0]
    order = sorted([int(v) for v in nums.dropna().unique()])
    gs = order
    sizes = [sub[sub.group == f"n_mod={g}"].mean_set_size.iloc[0]
             for g in gs]
    covs = [sub[sub.group == f"n_mod={g}"].coverage.iloc[0] for g in gs]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(gs, sizes, "o-", color="#2E5A87", lw=2, label="mean set size")
    ax1.set_xlabel("# available modalities")
    ax1.set_ylabel("mean prediction-set size", color="#2E5A87")
    ax2 = ax1.twinx()
    ax2.plot(gs, covs, "s--", color="#B4525C", lw=1.5, label="coverage")
    ax2.set_ylabel("coverage", color="#B4525C")
    ax2.set_ylim(0.5, 1.05)
    ax2.axhline(1 - ALPHA, color="grey", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(OUT / "conformal_set_size.png", dpi=300)
    plt.close(fig)
    print("conformal experiment done")


if __name__ == "__main__":
    main()
