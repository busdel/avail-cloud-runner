# -*- coding: utf-8 -*-
"""Methodological extension: uncertainty-aware meta-features.

Adds per-modality predictive entropy H(p) of the base predictions as an
extra meta-feature next to [p_1..p_K, flag]:

    X = [p_1..p_K, flag, H]  per modality

This gives the meta-learner an explicit uncertainty signal (grounded
alternative to plain averaging) and addresses the "probabilities throw away
information" critique. Compared against p+flag with the same 4 fast
meta-learners on the zero-loss n=150 data.

Outputs: experiments/results/entropy_meta_results.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RES = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def main():
    bp = pd.read_csv(RES / "base_predictions_generic.tsv", sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    loo = pd.read_csv(RES / "loo_predictions.tsv", sep="\t")
    loo["patient"] = loo["patient"].astype(str).str.strip()
    ids = loo["patient"].tolist()
    y = loo["y"].values
    K = 3
    mod_order = sorted(bp["modality"].unique())

    piv = {}
    for _, r in bp.iterrows():
        piv.setdefault(r["patient"], {})[r["modality"]] = \
            np.array([r[f"p_{c}"] for c in range(K)])
    nmod = len(mod_order)

    def build(with_entropy):
        per = (K + 1) + (1 if with_entropy else 0)
        X = np.zeros((len(ids), nmod * per))
        for i, p in enumerate(ids):
            for c, m in enumerate(mod_order):
                b = c * per
                if p in piv and m in piv[p]:
                    pv = piv[p][m]
                    X[i, b:b + K] = pv
                    X[i, b + K] = 1.0
                    if with_entropy:
                        H = -np.sum(np.where(pv > 0, pv * np.log(pv + 1e-12), 0))
                        X[i, b + K + 1] = H
        return X

    def meta_loo(X, y, per):
        n = len(y)
        preds = {m: np.zeros((n, K)) for m in
                 ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]}
        for fold in range(n):
            tr = [t for t in range(n) if t != fold]
            Xtr, ytr, Xte = X[tr], y[tr], X[fold:fold + 1]
            acc = np.zeros(K)
            wsum = 0.0
            for c in range(nmod):
                b = c * per
                if Xte[0, b + K] == 1:
                    acc += Xte[0, b:b + K]
                    wsum += 1.0
            preds["SimpleMean"][fold] = acc / wsum if wsum > 0 else np.full(K, 1 / K)
            preds["WSimpleMean"][fold] = preds["SimpleMean"][fold]
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
                                    learning_rate=0.05, random_state=0,
                                    n_jobs=6, objective="multi:softprob",
                                    num_class=K, eval_metric="mlogloss")
                xgb.fit(Xtr, ytr)
                preds["XGBoost"][fold] = xgb.predict_proba(Xte)[0]
            except Exception:
                preds["XGBoost"][fold] = preds["SimpleMean"][fold]
        return preds

    rows = []
    for tag, we in [("p+flag", False), ("p+flag+entropy", True)]:
        per = (K + 1) + (1 if we else 0)
        X = build(we)
        preds = meta_loo(X, y, per)
        for m, P in preds.items():
            rows.append({"meta_features": tag, "model": m,
                         "macro_auc": macro_auc(y, P)})
        print(tag, "done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "entropy_meta_results.tsv", sep="\t", index=False)
    print(df.pivot(index="meta_features", columns="model",
                   values="macro_auc").round(3).to_string())


if __name__ == "__main__":
    main()
