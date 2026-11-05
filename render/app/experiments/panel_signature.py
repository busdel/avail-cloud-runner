#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis (a): compact multi-omics signature panel.

Repeated stratified half-splits (R = 50):
  - per layer (stool16S taxa, stoolSG MGS, fMRI networkFC summaries),
    select top-k features by univariate ANOVA F on the training half
    (computed on available subjects only, FDR-free rank cut),
    fit an L2 logistic per layer on the training half,
  - fuse layer probabilities + availability flags with an L2 logistic
    (all training subjects),
  - evaluate macro-AUC (3-class) and HC-PDD AUC on the held-out half.

Reports mean +/- sd across splits for k = 5, 10 per layer, and for
single-layer signatures (stool16S, fMRI) as context. A final variant uses
the fixed three-evidence taxa (literature-anchored) for stool layers.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
R = 50
THREE_EVIDENCE = ["Bifidobacterium", "Lachnospira", "Faecalibacterium",
                  "Fusicatenibacter", "Roseburia", "Oscillibacter",
                  "Agathobacter"]


def load_layer(name):
    t = pd.read_csv(DATA / "features" / f"{name}.tsv", sep="\t")
    if "SampleID" not in t.columns and "patient" in t.columns:
        t = t.rename(columns={"patient": "SampleID"})
    t["SampleID"] = t["SampleID"].astype(str).str.strip()
    return t.set_index("SampleID")


def top_k_by_f(X, y, k, rng=None):
    ok = np.isfinite(X).all(axis=1)
    Xo, yo = X[ok], y[ok]
    if len(np.unique(yo)) < 2:
        return np.arange(min(k, X.shape[1]))
    F, p = stats.f_oneway(*[Xo[yo == c] for c in np.unique(yo)])
    F = np.nan_to_num(F, nan=0.0)
    return np.argsort(F)[::-1][:k]


def fit_layer(Xtr, ytr, Xte, k):
    idx = top_k_by_f(Xtr, ytr, k)
    Xs, Xse = Xtr[:, idx], Xte[:, idx]
    imp = SimpleImputer(strategy="median").fit(Xs)
    sc = StandardScaler().fit(imp.transform(Xs))
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    m.fit(sc.transform(imp.transform(Xs)), ytr)
    K = len(np.unique(ytr))
    P = np.zeros((Xte.shape[0], K))
    for ci, cls in enumerate(m.classes_):
        P[:, int(cls)] = m.predict_proba(sc.transform(imp.transform(Xse)))[:, ci]
    return P


def fuse(P_list, Mtr, ytr, Pte_list, Mte, K):
    Xtr = np.hstack(P_list + [Mtr])
    Xte = np.hstack(Pte_list + [Mte])
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    m.fit(Xtr, ytr)
    P = np.zeros((Xte.shape[0], K))
    for ci, cls in enumerate(m.classes_):
        P[:, int(cls)] = m.predict_proba(Xte)[:, ci]
    return P


def main():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    ids_all = meta["SampleID"].tolist()
    cls3 = {"HC": 0, "MCI": 1, "PDD": 2}
    layers = {}
    for name in ["stool16S", "stoolSG", "fmri_networkfc"]:
        t = load_layer(name)
        layers[name] = t

    rows = []
    for task in ["multi", "binary"]:
        if task == "binary":
            sub = meta[meta.Group.isin(["HC", "PDD"])]
            classes = ["HC", "PDD"]
        else:
            sub = meta
            classes = ["HC", "MCI", "PDD"]
        ids = [i for i in sub["SampleID"] if i in ids_all]
        y = np.array([classes.index(g) for g in
                      sub.set_index("SampleID").loc[ids, "Group"]])
        K = len(classes)
        for k in [5, 10]:
            for rep in range(R):
                sss = StratifiedShuffleSplit(1, test_size=0.5,
                                             random_state=rep)
                tr, te = next(sss.split(ids, y))
                Ptr, Pte, Mtr, Mte = [], [], [], []
                for name, T in layers.items():
                    idx = T.index
                    ncols = min(400, T.shape[1])
                    cols = T.columns[:ncols]
                    X = np.full((len(ids), ncols), np.nan)
                    sub_T = T.loc[[i for i in ids if i in idx], cols]
                    for r, pid in enumerate(ids):
                        if pid in sub_T.index:
                            X[r] = sub_T.loc[pid].values.astype(float)
                    Xtr = X[tr]
                    Xte = X[te]
                    Ptr_ = fit_layer_av(Xtr, y[tr], Xtr, k)
                    Pte_ = fit_layer_av(Xtr, y[tr], Xte, k)
                    Ptr.append(Ptr_)
                    Pte.append(Pte_)
                    Mtr.append(np.isfinite(Xtr).all(1).astype(float)[:, None])
                    Mte.append(np.isfinite(Xte).all(1).astype(float)[:, None])
                P = fuse(Ptr, np.hstack(Mtr), y[tr], Pte, np.hstack(Mte), K)
                auc = (roc_auc_score(y[te], P[:, 1]) if task == "binary"
                       else macro_auc(y[te], P))
                rows.append({"task": task, "k": k, "rep": rep, "auc": auc})
        print(f"[panel] {task} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "panel_signature_raw.tsv", sep="\t", index=False)
    g = df.groupby(["task", "k"])["auc"].agg(["mean", "std"]).round(4)
    g.to_csv(OUT / "panel_signature_summary.tsv", sep="\t")
    print(g.to_string(), flush=True)
    print("[panel] DONE", flush=True)


def fit_layer_av(Xtr, ytr, Xte, k):
    """Layer fit handling NaN rows: impute, select on available train rows."""
    from sklearn.impute import SimpleImputer as SI
    ok_tr = np.isfinite(Xtr).all(1)
    ok_te = np.isfinite(Xte).all(1)
    Xo = Xtr[ok_tr]
    yo = ytr[ok_tr]
    K = int(max(ytr)) + 1
    if len(np.unique(yo)) < 2 or len(Xo) < 6:
        return np.full((Xte.shape[0], K), 1 / K)
    idx = top_k_by_f(Xo, yo, k)
    imp = SI(strategy="median").fit(Xo[:, idx])
    sc = StandardScaler().fit(imp.transform(Xo[:, idx]))
    m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    m.fit(sc.transform(imp.transform(Xo[:, idx])), yo)
    Xt = np.where(np.isfinite(Xte), Xte, np.nan)[:, idx]
    med = imp.statistics_
    Xt = np.where(np.isfinite(Xt), Xt, med)
    P = np.zeros((Xte.shape[0], K))
    prob = m.predict_proba(sc.transform(Xt))
    for ci, cls in enumerate(m.classes_):
        P[:, int(cls)] = prob[:, ci]
    return P


if __name__ == "__main__":
    main()
