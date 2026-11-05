#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis (b): biomarker stability selection.

L1-logistic stability selection (B = 100 half-sample subsamples) on the
binary HC vs PDD task per layer:
  - stool16S genera (n=112), stoolSG MGS (n=104), fMRI networkFC (n=83)
Selection frequency = fraction of subsamples with non-zero coefficient.
Reports the top features and cross-checks the three-evidence taxa.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import LogCenterTransformer, CLRTransformer  # noqa

DATA = ROOT / "experiments/results/ladder_generic"
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
THREE_EVIDENCE = ["Bifidobacterium", "Lachnospira", "Faecalibacterium",
                  "Fusicatenibacter", "Roseburia", "Oscillibacter",
                  "Agathobacter"]
B = 100
SEED = 42


def load_feat(name, sample_ids):
    t = pd.read_csv(DATA / "features" / f"{name}.tsv", sep="\t")
    if "SampleID" not in t.columns and "patient" in t.columns:
        t = t.rename(columns={"patient": "SampleID"})
    t["SampleID"] = t["SampleID"].astype(str).str.strip()
    t = t.set_index("SampleID")
    t = t.loc[[i for i in sample_ids if i in t.index]]
    return t


def clr_or_log(X):
    rs = X.sum(axis=1)
    if rs.min() >= 0 and rs.mean() <= 150:
        return LogCenterTransformer().transform(X)
    return CLRTransformer().transform(X)


def stability(features, y, tag):
    from sklearn.impute import SimpleImputer
    X = features.values.astype(float)
    cols = np.array(features.columns)
    allnan = np.isnan(X).all(axis=0)
    if allnan.any():
        X = X[:, ~allnan]
        cols = cols[~allnan]
        print(f"[stability] {tag}: dropped {int(allnan.sum())} all-NaN "
              f"column(s)", flush=True)
    X = clr_or_log(X) if tag != "fmri_networkfc" else X
    imp = SimpleImputer(strategy="median").fit(X)
    X = imp.transform(X)
    sc = StandardScaler().fit(X)
    X = sc.transform(X)
    rng = np.random.RandomState(SEED)
    n = len(y)
    freq = np.zeros(X.shape[1])
    for b in range(B):
        idx = rng.choice(n, size=n // 2, replace=False)
        if len(np.unique(y[idx])) < 2:
            continue
        m = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                               class_weight="balanced", max_iter=5000)
        m.fit(X[idx], y[idx])
        freq += (np.abs(m.coef_).ravel() > 1e-8).astype(float)
    freq /= B
    d = pd.DataFrame({"feature": cols, "selection_freq": freq})
    d = d.sort_values("selection_freq", ascending=False).reset_index(drop=True)
    d.to_csv(OUT / f"stability_{tag}.tsv", sep="\t", index=False)
    top = d.head(15)
    print(f"=== {tag}: top-15 by stability (n={n})", flush=True)
    print(top.to_string(index=False), flush=True)
    hits = [c for c in cols if any(te in c for te in THREE_EVIDENCE)]
    sub = d[d["feature"].isin(hits)]
    if len(sub):
        print(f"-- three-evidence taxa stability ({tag}):", flush=True)
        print(sub.to_string(index=False), flush=True)
    return d


def main():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    sub = meta[meta.Group.isin(["HC", "PDD"])]
    ids = sub["SampleID"].tolist()
    y = (sub.set_index("SampleID").loc[ids, "Group"] == "PDD").astype(int).values
    for name in ["stool16S", "stoolSG", "fmri_networkfc"]:
        T = load_feat(name, ids)
        if len(T) < 20:
            print(f"[stability] {name}: too few subjects ({len(T)})")
            continue
        yy = np.array([y[ids.index(i)] for i in T.index])
        stability(T, yy, name)
    print("[stability] DONE", flush=True)


if __name__ == "__main__":
    main()
