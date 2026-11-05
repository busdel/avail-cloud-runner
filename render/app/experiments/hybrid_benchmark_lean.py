# -*- coding: utf-8 -*-
"""Lean hybrid benchmark: ONLY the hybrid_mean strategy (3 seeds, 5-fold CV).

The other 16 strategies are already in imputation_benchmark_v4_results.tsv
from the full v4 run (identical protocol). The hybrid reuses the same
stage1_cv/meta_cv machinery with REAL raw availability flags.

Outputs: <out>/hybrid_benchmark_results.tsv
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

import imputation_benchmark_v4 as v4  # noqa: E402
v4.ROOT = Path(__a2.data_root)
v4.OUT = Path(__a2.out)
ROOT = v4.ROOT
OUT = v4.OUT

from sklearn.impute import SimpleImputer  # noqa: E402


def main():
    ids, y, layers = v4.load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}", flush=True)
    results = []

    raw_avail = np.zeros((len(ids), len(mod_order)))
    for c, m in enumerate(mod_order):
        for r, p in enumerate(ids):
            if p in layers[m].dropna().index:
                raw_avail[r, c] = 1.0

    fac = lambda: SimpleImputer(strategy="mean")
    for seed in range(3):
        piv_i = v4.stage1_cv(layers, ids, y, K, imputer_factory=fac,
                             seed=seed)
        Xi = v4.meta_features(piv_i, ids, mod_order, K, with_flags=True)
        per = K + 1
        for c in range(len(mod_order)):
            Xi[:, c * per + K] = raw_avail[:, c]
        preds_i = v4.meta_cv(Xi, y, K, v4.MODELS, flags=True, seed=seed)
        for m in v4.MODELS:
            results.append({"strategy": "hybrid_mean_imp", "model": m,
                            "n": len(ids), "seed": seed,
                            "macro_auc": v4.macro_auc(y, preds_i[m])})
        print(f"hybrid_mean seed={seed} done", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "hybrid_benchmark_results.tsv", sep="\t", index=False)
    print(df.pivot_table(index="strategy", columns="model",
                         values="macro_auc").round(3).to_string())
    print("lean hybrid benchmark done")


if __name__ == "__main__":
    main()
