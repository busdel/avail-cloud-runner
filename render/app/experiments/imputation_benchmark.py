# -*- coding: utf-8 -*-
"""Robustness benchmark: AVAIL (p+flag) vs missing-data handling baselines.

Strategies on the same zero-loss n=150 dataset (7 layers):
  avail        : availability flags (ours, every subject stays)
  complete     : complete-case analysis (reports sample loss)
  mean_imp     : mean imputation + missing indicator per modality
  knn_imp      : KNN imputation (k=3) per modality
  mice_imp     : iterative (MICE) imputation per modality

Same stage-1 LOO RF base learners and same meta-learners
(SimpleMean, WSimpleMean, ElasticNetLR, XGBoost) for all strategies.

Outputs: experiments/results/imputation_benchmark_results.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]


def load_layers():
    meta = pd.read_csv(ROOT / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    classes = ["HC", "MCI", "PDD"]
    ids = meta[meta.Group.isin(classes)]["SampleID"].tolist()
    y = np.array([classes.index(g) for g in meta[meta.Group.isin(classes)]["Group"]])
    layers = {}
    for f in sorted((ROOT / "features").iterdir()):
        d = pd.read_csv(f, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d[d.SampleID.isin(ids)].set_index("SampleID").reindex(ids)
        layers[f.stem] = d
    return ids, y, layers


def stage1_loo(mods, ids, y, K, n_est=200):
    """LOO base predictions per modality -> {pid: {modality: pvec}}."""
    piv = {}
    for mname, Xdf in mods.items():
        sub = Xdf.dropna()
        sids = list(sub.index)
        if len(sids) < 10:
            continue
        X = sub.values.astype(float)
        yy = np.array([y[ids.index(p)] for p in sids])
        P = np.zeros((len(sids), K))
        for k in range(len(sids)):
            tr = [t for t in range(len(sids)) if t != k]
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                        n_jobs=6)
            rf.fit(X[tr], yy[tr])
            P[k] = rf.predict_proba(X[k:k + 1])[0]
        for p, pv in zip(sids, P):
            piv.setdefault(p, {})[mname] = pv
    return piv


def meta_features(piv, ids, mod_order, K):
    per = K + 1
    X = np.zeros((len(ids), len(mod_order) * per))
    for r, p in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * per
            if p in piv and m in piv[p]:
                X[r, b:b + K] = piv[p][m]
                X[r, b + K] = 1.0
    return X


def meta_loo(X, y, K, models):
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in models}
    per = K + 1
    nmod = X.shape[1] // per
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr = X[tr], y[tr]
        Xte = X[fold:fold + 1]
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
        preds["SimpleMean"][fold] = acc / wsum if wsum > 0 else np.full(K, 1 / K)

        w = []
        for c in range(nmod):
            b = c * per
            obs = Xtr[:, b + K] == 1
            if obs.sum() >= 10:
                w.append(1.0)
            else:
                w.append(0.5)
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if Xte[0, b + K] == 1:
                acc += w[c] * Xte[0, b:b + K]
                wsum += w[c]
        preds["WSimpleMean"][fold] = acc / wsum if wsum > 0 else np.full(K, 1 / K)

        try:
            lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                    l1_ratio=0.5, C=0.3, max_iter=3000,
                                    class_weight="balanced", random_state=0)
            lr.fit(Xtr, ytr)
            preds["ElasticNetLR"][fold] = lr.predict_proba(Xte)[0]
        except Exception:
            preds["ElasticNetLR"][fold] = preds["SimpleMean"][fold]

        try:
            xgb = XGBClassifier(n_estimators=200, max_depth=2,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, random_state=0, n_jobs=6,
                                objective="multi:softprob", num_class=K,
                                eval_metric="mlogloss")
            xgb.fit(Xtr, ytr)
            preds["XGBoost"][fold] = xgb.predict_proba(Xte)[0]
        except Exception:
            preds["XGBoost"][fold] = preds["SimpleMean"][fold]
        if (fold + 1) % 50 == 0:
            print(f"    fold {fold + 1}/{n}")
    return preds


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def main():
    ids, y, layers = load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}")

    results = []

    # ---- strategy 1: AVAIL flags (ours)
    piv = stage1_loo(layers, ids, y, K)
    X = meta_features(piv, ids, mod_order, K)
    preds = meta_loo(X, y, K, MODELS)
    for m in MODELS:
        results.append({"strategy": "avail_flags", "model": m,
                        "n": len(ids), "macro_auc": macro_auc(y, preds[m])})
    print("avail done")

    # ---- strategy 2: complete-case
    cc = [p for p in ids if all(p in layers[m].dropna().index for m in mod_order)]
    ccidx = [ids.index(p) for p in cc]
    if len(ccidx) >= 15:
        lay_cc = {m: layers[m].loc[cc] for m in mod_order}
        piv = stage1_loo(lay_cc, cc, y[ccidx], K)
        X = meta_features(piv, cc, mod_order, K)
        preds = meta_loo(X, y[ccidx], K, MODELS)
        for m in MODELS:
            results.append({"strategy": "complete_case", "model": m,
                            "n": len(cc), "macro_auc": macro_auc(y[ccidx], preds[m])})
        print(f"complete-case done (n={len(cc)})")

    # ---- strategies 3-5: imputation per modality
    for strat, imp in [
        ("mean_imp", SimpleImputer(strategy="mean")),
        ("knn_imp", KNNImputer(n_neighbors=3)),
        ("mice_imp", IterativeImputer(max_iter=10, random_state=42)),
    ]:
        imputed = {}
        for m in mod_order:
            d = layers[m]
            Ximp = imp.fit_transform(d.values.astype(float))
            imputed[m] = pd.DataFrame(Ximp, index=d.index, columns=d.columns)
        piv = stage1_loo(imputed, ids, y, K)
        X = meta_features(piv, ids, mod_order, K)
        preds = meta_loo(X, y, K, MODELS)
        for mm in MODELS:
            results.append({"strategy": strat, "model": mm,
                            "n": len(ids), "macro_auc": macro_auc(y, preds[mm])})
        print(f"{strat} done")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "imputation_benchmark_results.tsv", sep="\t", index=False)
    print(df.pivot(index="strategy", columns="model", values="macro_auc")
          .round(3).to_string())


if __name__ == "__main__":
    main()
