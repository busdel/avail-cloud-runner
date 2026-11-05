#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 5 fix: covariate control with flags kept intact."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/roi_ladder/integrity"
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
SEED = 42
K = 3
PER = K + 1


def load_meta_features(arm):
    base_files = sorted((OUT.parent).glob("stage1_full_*.tsv.gz"))
    base = {}
    for f in base_files:
        parts = f.name.split(".")[0].split("_")
        name = "_".join(parts[2:-1])
        n = int(parts[-1])
        t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
        t.index = t.index.astype(str).str.strip()
        base[(name, n)] = {i: t.loc[i].values.astype(float) for i in t.index}
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    comp = set(meta["SampleID"])
    for m in MODULES:
        comp &= feat[m]
    comp = sorted(comp)
    ids = comp if arm == "complete" else sorted(meta["SampleID"])
    subset = {}
    for m in MODULES:
        n_ = len([i for i in ids if i in feat[m]])
        key = (m, n_)
        if key not in base:
            raise KeyError(f"missing base cache {key}")
        subset[m] = base[key]
    X = np.zeros((len(ids), len(MODULES) * PER))
    y = []
    for r, pid in enumerate(ids):
        row = meta[meta["SampleID"] == pid].iloc[0]
        y.append(["HC", "MCI", "PDD"].index(row.Group))
        for c, m in enumerate(MODULES):
            b = c * PER
            if pid in subset[m]:
                X[r, b:b + K] = subset[m][pid]
                X[r, b + K] = 1.0
    y = np.array(y)
    cov = np.column_stack([
        meta.set_index("SampleID").loc[ids, "Age"].values.astype(float),
        meta.set_index("SampleID").loc[ids, "Sex"].values.astype(float),
        meta.set_index("SampleID").loc[ids, "Education"].values.astype(float)])
    return X, y, cov, ids


def residualize_proba_only(X, cov):
    """Residualize probability columns on covariates within-row-wise via OLS
    fit on the full cohort; flags left untouched."""
    Xr = X.copy()
    for c in range(len(MODULES)):
        b = c * PER
        cols = [b, b + 1, b + 2]
        C = np.column_stack([np.ones(X.shape[0]), cov])
        B, *_ = np.linalg.lstsq(C, X[:, cols], rcond=None)
        Xr[:, cols] = X[:, cols] - C @ B
    return Xr


def main():
    rows = []
    for arm in ["complete", "avail"]:
        print(f"[faz5fix] arm={arm}", flush=True)
        X, y, cov, ids = load_meta_features(arm)
        Xr = residualize_proba_only(X, cov)
        P = run_meta_loo(Xr, y, len(MODULES), K, model_names=MODELS,
                         device=None, random_state=SEED, n_seeds=5,
                         augment_flag=True, aug_p=0.3, verbose=False)
        for m in MODELS:
            rows.append({"arm": arm, "protocol": "covresid_fixed",
                         "models": m,
                         "obs_macro_auc": macro_auc(y, P[m]),
                         "null_mean": np.nan, "null_sd": np.nan,
                         "p_perm": np.nan})
    df = pd.DataFrame(rows)
    old = pd.read_csv(OUT / "permutation_and_covariate.tsv", sep="\t")
    old = old[old["protocol"] != "covresid"]
    out = pd.concat([old, df], ignore_index=True)
    out.to_csv(OUT / "permutation_and_covariate.tsv", sep="\t", index=False)
    print(out.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
