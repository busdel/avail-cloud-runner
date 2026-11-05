# -*- coding: utf-8 -*-
"""Ordinal-aware fusion experiment: does embedding the HC<MCI<PDD ordering
in the meta-learner loss help the hardest class (MCI)?

Compares the MLP meta-learner with and without ordinal regularization
(expected |E[y_hat]-y| penalty) on the n=150 zero-loss cohort, LOO.

Outputs: experiments/results/ordinal_results.tsv
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
__ap.add_argument("--base-preds",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results/base_predictions_generic.tsv")
__ap.add_argument("--loo",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results/loo_predictions.tsv")
__ap.add_argument("--out",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results")
__a2, _ = __ap.parse_known_args()
OUT = Path(__a2.out)

from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import _MLP, train_nn, predict_nn  # noqa: E402
from sklearn.metrics import roc_auc_score


def main():
    bp = pd.read_csv(__a2.base_preds, sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    loo = pd.read_csv(__a2.loo, sep="\t")
    ids = loo["patient"].astype(str).tolist()
    y = loo["y"].values
    K = 3
    piv = {}
    for _, r in bp.iterrows():
        piv.setdefault(r["patient"], {})[r["modality"]] = \
            np.array([r[f"p_{c}"] for c in range(K)])
    mod_order = sorted({m for _, mm in piv.items() for m in mm})
    # build the meta-feature matrix manually (p + availability flag)
    per = K + 1
    X = np.zeros((len(ids), len(mod_order) * per))
    for r, p in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * per
            if p in piv and m in piv[p]:
                X[r, b:b + K] = piv[p][m]
                X[r, b + K] = 1.0

    rows = []
    for tag, lam in [("MLP_plain", 0.0), ("MLP_ordinal", 0.5),
                     ("MLP_ordinal_strong", 1.5)]:
        P = np.zeros((len(y), K))
        for k in range(len(y)):
            tr = [t for t in range(len(y)) if t != k]
            seeds = []
            for s in range(10):
                mdl = _MLP(X.shape[1], K)
                try:
                    train_nn(mdl, X[tr], y[tr], seed=42 + s, K=K,
                             ordinal_lambda=lam)
                    seeds.append(predict_nn(mdl, X[k:k + 1])[0])
                except Exception:
                    seeds.append(np.full(K, 1 / K))
            P[k] = np.mean(seeds, axis=0)
        rows.append({"model": tag, "subgroup": "all",
                     "macro_auc": macro_auc(y, P)})
        for c, cls in enumerate(["PDD", "MCI", "HC"]):
            yc = (y == c).astype(int)
            if len(np.unique(yc)) == 2:
                rows.append({"model": tag, "subgroup": f"class_{cls}",
                             "macro_auc": roc_auc_score(yc, P[:, c])})
        print(tag, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ordinal_results.tsv", sep="\t", index=False)
    print(df.pivot_table(index="model", columns="subgroup",
                         values="macro_auc").round(3).to_string())
    print("ordinal experiment done")


if __name__ == "__main__":
    main()
