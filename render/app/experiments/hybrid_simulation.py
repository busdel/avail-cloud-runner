# -*- coding: utf-8 -*-
"""Lean simulation for the hybrid strategy.

Same generative grid and seeds as simulation_study.py (MCAR/MAR/MNAR x
p={0.2,0.4,0.6} x 40 reps) but only three methods: avail_sm, mean_imp_sm,
hybrid_sm. The remaining methods are already available in
simulation_results.tsv from the full run on the identical grid/seeds.

Outputs: experiments/results/hybrid_simulation_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))
OUT = BASE / "experiments" / "results"

import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--mechanism", choices=["MCAR", "MAR", "MNAR", "all"],
                  default="all")
__a2, _ = __ap.parse_known_args()
MECHS = (["MCAR", "MAR", "MNAR"] if __a2.mechanism == "all"
         else [__a2.mechanism])

from simulation_study import (gen_data, apply_missingness, stage1_loo,  # noqa: E402
                              meta, macro_auc, K, M, N, REPS)

from sklearn.impute import SimpleImputer  # noqa: E402


def main():
    rows = []
    for mech in MECHS:
        for p in [0.2, 0.4, 0.6]:
            res = {"mechanism": mech, "p_missing": p}
            for method in ["avail_sm", "mean_imp_sm", "hybrid_sm"]:
                res[method] = []
            for rep in range(REPS):
                y, mods = gen_data(rep)
                masked = apply_missingness(mods, y, p, mech, rep)
                piv_a, av_a = stage1_loo(masked, y)
                P_sm, _ = meta(piv_a, av_a, y)
                res["avail_sm"].append(macro_auc(y, P_sm))
                piv_m, _ = stage1_loo(
                    masked, y, imputer=lambda s=0: SimpleImputer(
                        strategy="mean"))
                av_ones = {m: np.ones(N) for m in range(M)}
                P_ms, _ = meta(piv_m, av_ones, y)
                res["mean_imp_sm"].append(macro_auc(y, P_ms))
                P_hs, _ = meta(piv_m, av_a, y)
                res["hybrid_sm"].append(macro_auc(y, P_hs))
            for method in res:
                if method in ("mechanism", "p_missing"):
                    continue
                vals = res[method]
                rows.append({"mechanism": mech, "p_missing": p,
                             "method": method,
                             "auc_mean": float(np.mean(vals)),
                             "auc_se": float(np.std(vals) / np.sqrt(len(vals))),
                             "n_reps": len(vals)})
            print(f"{mech} p={p} done", flush=True)
    df = pd.DataFrame(rows)
    tag = __a2.mechanism if __a2.mechanism != "all" else "all"
    df.to_csv(OUT / f"hybrid_simulation_{tag}.tsv", sep="\t", index=False)
    piv = df.pivot_table(index=["mechanism", "p_missing"], columns="method",
                         values="auc_mean").round(3)
    print(piv.to_string())
    print("hybrid simulation done")


if __name__ == "__main__":
    main()
