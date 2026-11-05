# -*- coding: utf-8 -*-
"""Robustness: bootstrap CIs + paired bootstrap/deLong-style model comparisons.

Reads a results dir (loo_predictions.tsv + meta_benchmark_results.tsv):
  - 2000 bootstrap resamples of SUBJECTS -> 95% CI per model (macro-AUC)
  - pairwise model comparison: DeLong AUC variance (multi-class: average of
    one-vs-rest DeLong) -> p-values between models
Outputs: experiments/results/robustness_<dir>.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def delong_var(y, p):
    """DeLong variance of one-vs-rest AUC (binormal approximation)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    n1, n0 = len(pos), len(neg)
    if n1 < 2 or n0 < 2:
        return np.nan
    # rank-based formula
    X = np.concatenate([pos, neg])
    R = pd.Series(X).rank().values
    V10 = ((R[y == 1].sum() / n1 - (n1 + 1) / 2)
           / n0)  # Mann-Whitney U estimate
    auc = V10
    # structural components (Hanley-McNeil via ranks)
    V01 = ((R[y == 0].sum() / n0 - (n0 + 1) / 2) / n1)
    Q1 = ((R[y == 1] ** 2).sum() / n1) / ((n0 + 1) ** 2)
    Q2 = ((R[y == 0] ** 2).sum() / n0) / ((n1 + 1) ** 2)
    var = (auc * (1 - auc) +
           (n1 - 1) * (Q1 - auc ** 2) +
           (n0 - 1) * (Q2 - auc ** 2)) / (n1 * n0)
    return var, auc


def delong_test(y, pA, pB):
    K = pA.shape[1]
    vars_sum = np.zeros((2, 2))
    diff = 0.0
    for c in range(K):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) < 2:
            continue
        va, aa = delong_var(yc, pA[:, c])
        vb, bb = delong_var(yc, pB[:, c])
        if np.isnan(va) or np.isnan(vb):
            continue
        # covariance term approximated (Hanley-McNeil)
        ra = np.argsort(np.argsort(pA[:, c])).astype(float)
        rb = np.argsort(np.argsort(pB[:, c])).astype(float)
        pos_a = ra[yc == 1].mean()
        neg_a = ra[yc == 0].mean()
        pos_b = rb[yc == 1].mean()
        neg_b = rb[yc == 0].mean()
        cov = (pos_a * pos_b / (len(yc) ** 2) +
               neg_a * neg_b / (len(yc) ** 2)) / 4
        vars_sum += np.array([[va, cov], [cov, vb]]) / K
        diff += (aa - bb) / K
    v = vars_sum[0, 0] + vars_sum[1, 1] - 2 * vars_sum[0, 1]
    if v <= 0:
        return diff, np.nan
    z = diff / np.sqrt(v)
    return diff, 2 * (1 - norm.cdf(abs(z)))


def main(res_dir, tag):
    loo = pd.read_csv(res_dir / "loo_predictions.tsv", sep="\t")
    y = loo["y"].values
    # class order = top-3 frequent groups (same rule as run_generic)
    meta = pd.read_csv(ROOT / "metadata.tsv", sep="\t")
    from collections import Counter
    order = [c for c, _ in Counter(meta["Group"].astype(str).str.strip()
                                   .values).most_common(3)]
    model_cols = {}
    for c in loo.columns:
        if "_" in c and not c.startswith(("patient", "y")):
            m, cls = c.rsplit("_", 1)
            if cls in order:
                model_cols.setdefault(m, []).append(c)
    rows = []
    for m, cols in model_cols.items():
        cols = sorted(cols, key=lambda c: order.index(c.split("_")[-1]))
        P = loo[cols].values
        auc0 = macro_auc(y, P)
        rng = np.random.RandomState(42)
        aucs = []
        for _ in range(2000):
            idx = rng.choice(len(y), len(y), replace=True)
            aucs.append(macro_auc(y[idx], P[idx]))
        lo_, hi = np.percentile(aucs, [2.5, 97.5])
        rows.append({"model": m, "macro_auc": auc0, "ci_lo": lo_, "ci_hi": hi})

    df = pd.DataFrame(rows).sort_values("macro_auc", ascending=False)
    df.to_csv(OUT / f"robustness_ci_{tag}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))

    # pairwise comparison: paired bootstrap p-value of AUC difference
    top5 = df.head(5)["model"].tolist()
    pr = []
    rng = np.random.RandomState(42)
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            mA, mB = top5[i], top5[j]
            PA = loo[[c for c in model_cols[mA]]].values
            PB = loo[[c for c in model_cols[mB]]].values
            d_obs = macro_auc(y, PA) - macro_auc(y, PB)
            cnt = 0
            B = 1000
            for _ in range(B):
                idx = rng.choice(len(y), len(y), replace=True)
                if macro_auc(y[idx], PA[idx]) - macro_auc(y[idx], PB[idx]) > abs(d_obs):
                    cnt += 1
            pr.append({"modelA": mA, "modelB": mB,
                       "auc_diff": d_obs,
                       "p_bootstrap": (1 + cnt) / (1 + B)})
    pdf = pd.DataFrame(pr)
    pdf.to_csv(OUT / f"robustness_pairwise_{tag}.tsv", sep="\t", index=False)
    print(pdf.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    res = Path(sys.argv[1])
    for i, a in enumerate(sys.argv):
        if a == "--meta-root" and i + 1 < len(sys.argv):
            ROOT = Path(sys.argv[i + 1])
    main(res, res.name)
