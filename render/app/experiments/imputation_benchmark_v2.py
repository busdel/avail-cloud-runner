# -*- coding: utf-8 -*-
"""Robustness benchmark v2: AVAIL vs missing-data baselines (extended).

Strategies (same n=150 data, same LOO):
  avail_flags        ours (p + availability flag)
  complete_case      drop incomplete subjects
  mean_imp           mean imputation + indicator
  knn_imp            KNN (k=3) imputation
  mice_imp           iterative imputation
  log_mean_imp       log(x+1) then mean imputation (compositional-friendly)
  early_fusion_xgb   raw concatenated (mean-imputed + indicators) -> XGBoost
  flag_only_xgb      availability flags only -> XGBoost
  p_only_xgb         base probabilities without flags -> XGBoost

Plus a per-modality solo-AUC table (marginal contribution of each layer).

Outputs: experiments/results/imputation_benchmark_v2_results.tsv,
         modality_solo_auc.tsv
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
    m = meta[meta.Group.isin(classes)]
    ids = m["SampleID"].tolist()
    y = np.array([classes.index(g) for g in m["Group"]])
    layers = {}
    for f in sorted((ROOT / "features").iterdir()):
        d = pd.read_csv(f, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d[d.SampleID.isin(ids)].set_index("SampleID").reindex(ids)
        layers[f.stem] = d
    return ids, y, layers


def stage1_loo(mods, ids, y, K, n_est=200):
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


def meta_features(piv, ids, mod_order, K, with_flags=True):
    per = K + 1
    X = np.zeros((len(ids), len(mod_order) * per))
    for r, p in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * per
            if p in piv and m in piv[p]:
                X[r, b:b + K] = piv[p][m]
                X[r, b + K] = 1.0
    if not with_flags:
        keep = [i for i in range(X.shape[1]) if (i % per) < K]
        X = X[:, keep]
    return X


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def meta_loo(X, y, K, models, per=4, flags=True):
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in models}
    nmod = X.shape[1] // per if flags else (X.shape[1] // K)
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr = X[tr], y[tr]
        Xte = X[fold:fold + 1]
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if flags and Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
            elif not flags:
                acc += Xte[0, b * K:(b + 1) * K]
                wsum += 1.0
        preds["SimpleMean"][fold] = acc / wsum if wsum > 0 else np.full(K, 1 / K)
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if flags and Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
            elif not flags:
                acc += Xte[0, b * K:(b + 1) * K]
                wsum += 1.0
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


def main():
    ids, y, layers = load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}")
    results = []

    # ---- ours
    piv = stage1_loo(layers, ids, y, K)
    solo = {}
    for m in mod_order:
        sub = [p for p in ids if p in piv and m in piv[p]]
        if len(sub) >= 10:
            P = np.vstack([piv[p][m] for p in sub])
            solo[m] = (len(sub), macro_auc(
                np.array([y[ids.index(p)] for p in sub]), P))
    solo_df = pd.DataFrame(
        [{"modality": m, "n": v[0], "solo_macro_auc": v[1]}
         for m, v in solo.items()]).sort_values(
        "solo_macro_auc", ascending=False)
    solo_df.to_csv(OUT / "modality_solo_auc.tsv", sep="\t", index=False)
    print(solo_df.round(3).to_string(index=False))

    X = meta_features(piv, ids, mod_order, K, with_flags=True)
    preds = meta_loo(X, y, K, MODELS, flags=True)
    for m in MODELS:
        results.append({"strategy": "avail_flags", "model": m, "n": len(ids),
                        "macro_auc": macro_auc(y, preds[m])})
    print("avail_flags done")

    # ---- complete case
    cc = [p for p in ids if all(p in layers[m].dropna().index for m in mod_order)]
    ccidx = [ids.index(p) for p in cc]
    if len(ccidx) >= 15:
        lay_cc = {m: layers[m].loc[cc] for m in mod_order}
        piv_cc = stage1_loo(lay_cc, cc, y[ccidx], K)
        Xcc = meta_features(piv_cc, cc, mod_order, K, with_flags=True)
        preds_cc = meta_loo(Xcc, y[ccidx], K, MODELS, flags=True)
        for m in MODELS:
            results.append({"strategy": "complete_case", "model": m,
                            "n": len(cc),
                            "macro_auc": macro_auc(y[ccidx], preds_cc[m])})
        print(f"complete_case done (n={len(cc)})")

    # ---- imputation strategies
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
        piv_i = stage1_loo(imputed, ids, y, K)
        Xi = meta_features(piv_i, ids, mod_order, K, with_flags=True)
        preds_i = meta_loo(Xi, y, K, MODELS, flags=True)
        for mm in MODELS:
            results.append({"strategy": strat, "model": mm, "n": len(ids),
                            "macro_auc": macro_auc(y, preds_i[mm])})
        print(f"{strat} done")

    # ---- log-then-mean imputation
    imputed = {}
    for m in mod_order:
        d = layers[m]
        Xl = np.log(d.values.astype(float) + 1.0)
        imp = SimpleImputer(strategy="mean").fit(Xl)
        Ximp = imp.transform(Xl)
        imputed[m] = pd.DataFrame(Ximp, index=d.index, columns=d.columns)
    piv_l = stage1_loo(imputed, ids, y, K)
    Xl = meta_features(piv_l, ids, mod_order, K, with_flags=True)
    preds_l = meta_loo(Xl, y, K, MODELS, flags=True)
    for mm in MODELS:
        results.append({"strategy": "log_mean_imp", "model": mm, "n": len(ids),
                        "macro_auc": macro_auc(y, preds_l[mm])})
    print("log_mean_imp done")

    # ---- early fusion: raw mean-imputed concat + indicator -> XGB
    parts, inds = [], []
    for m in mod_order:
        d = layers[m]
        Ximp = SimpleImputer(strategy="mean").fit_transform(
            d.values.astype(float))
        parts.append(Ximp)
        inds.append((~d.notna().values).astype(float))
    Xraw = np.hstack(parts)
    Iraw = np.hstack(inds)
    Xef = np.hstack([Xraw, Iraw])
    Pef = np.zeros((len(y), K))
    for k in range(len(y)):
        tr = [t for t in range(len(y)) if t != k]
        xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.5, random_state=0,
                            n_jobs=6, objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
        xgb.fit(Xef[tr], y[tr])
        Pef[k] = xgb.predict_proba(Xef[k:k + 1])[0]
    results.append({"strategy": "early_fusion_xgb", "model": "XGBoost",
                    "n": len(ids), "macro_auc": macro_auc(y, Pef)})
    print("early_fusion done")

    # ---- flag-only + p-only ablations
    flags_only = X[:, np.arange(X.shape[1]) % (K + 1) == K]
    Pfo = np.zeros((len(y), K))
    for k in range(len(y)):
        tr = [t for t in range(len(y)) if t != k]
        xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                            random_state=0, n_jobs=6, objective="multi:softprob",
                            num_class=K, eval_metric="mlogloss")
        xgb.fit(flags_only[tr], y[tr])
        Pfo[k] = xgb.predict_proba(flags_only[k:k + 1])[0]
    results.append({"strategy": "flag_only_xgb", "model": "XGBoost",
                    "n": len(ids), "macro_auc": macro_auc(y, Pfo)})
    print("flag_only done")

    Xp = meta_features(piv, ids, mod_order, K, with_flags=False)
    preds_p = meta_loo(Xp, y, K, MODELS, flags=False)
    for mm in MODELS:
        results.append({"strategy": "p_only", "model": mm, "n": len(ids),
                        "macro_auc": macro_auc(y, preds_p[mm])})
    print("p_only done")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "imputation_benchmark_v2_results.tsv", sep="\t", index=False)
    print(df.pivot_table(index="strategy", columns="model", values="macro_auc")
          .round(3).to_string())


if __name__ == "__main__":
    main()
