# -*- coding: utf-8 -*-
"""Hybrid strategy comparison for user-uploaded generic data (leak-free).

Hybrid = fold-internal imputed base predictions + REAL raw availability
flags from the unimputed layers. Compares against the avail_flags baseline
on the same LOO / 5-fold-CV protocol as imputation_benchmark_v4.

Outputs: <out>/hybrid_comparison_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--data-root", required=True)
__ap.add_argument("--out", required=True)
__a2, _ = __ap.parse_known_args()

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "experiments"))

from imputation_benchmark_v4 import (  # noqa: E402
    load_layers, stage1_loo, stage1_cv, meta_features, meta_loo, meta_cv,
    macro_auc, MODELS)
from imputation_benchmark_v4 import ROOT as _ROOT, OUT as _OUT  # noqa: E402
import imputation_benchmark_v4 as v4  # noqa: E402

v4.ROOT = Path(__a2.data_root)
v4.OUT = Path(__a2.out)
ROOT = v4.ROOT
OUT = v4.OUT

from sklearn.impute import SimpleImputer, IterativeImputer  # noqa: E402


def main():
    ids, y, layers = load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}", flush=True)
    results = []

    # baseline: avail_flags (LOO, no imputation)
    piv = stage1_loo(layers, ids, y, K)
    X = meta_features(piv, ids, mod_order, K, with_flags=True)
    preds = meta_loo(X, y, K, MODELS, flags=True)
    for m in MODELS:
        results.append({"strategy": "avail_flags", "model": m, "n": len(ids),
                        "seed": "", "macro_auc": macro_auc(y, preds[m])})
    print("avail_flags done", flush=True)

    # raw availability flags from unimputed layers
    raw_avail = np.zeros((len(ids), len(mod_order)))
    for c, m in enumerate(mod_order):
        for r, p in enumerate(ids):
            if p in layers[m].dropna().index:
                raw_avail[r, c] = 1.0

    factories = {
        "hybrid_mean_imp": lambda: SimpleImputer(strategy="mean"),
        "hybrid_mice_imp": lambda: IterativeImputer(max_iter=2, random_state=42,
                                                    n_nearest_features=50),
    }
    for strat, fac in factories.items():
        for seed in range(3):
            piv_i = stage1_cv(layers, ids, y, K, imputer_factory=fac, seed=seed)
            Xi = meta_features(piv_i, ids, mod_order, K, with_flags=True)
            per = K + 1
            for c in range(len(mod_order)):
                Xi[:, c * per + K] = raw_avail[:, c]
            preds_i = meta_cv(Xi, y, K, MODELS, flags=True, seed=seed)
            for m in MODELS:
                results.append({"strategy": strat, "model": m, "n": len(ids),
                                "seed": seed,
                                "macro_auc": macro_auc(y, preds_i[m])})
        print(f"{strat} done (3 seeds)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "hybrid_comparison_results.tsv", sep="\t", index=False)
    print(df.pivot_table(index="strategy", columns="model",
                         values="macro_auc").round(3).to_string())
    print("hybrid comparison done")


if __name__ == "__main__":
    main()
