# -*- coding: utf-8 -*-
"""Robustness benchmark v3: AVAIL vs EVERY missing-data strategy we can run.

Strategies (same n=150 zero-loss data, same LOO protocol):
  avail_flags        ours: p + availability flag
  complete_case      drop incomplete subjects
  zero_imp           fill 0 (compositional-aware)
  mean_imp           mean imputation
  mean_imp_ind       mean + missing-indicator columns
  median_imp         median imputation
  knn_imp            KNN (k=3)
  log_mean_imp       log(x+1) then mean
  clr_mean_imp       pseudo-CLR (log(x+1) centered) then mean
  log_knn_imp        log(x+1) then KNN
  mice_imp           iterative (linear) imputation
  missforest_imp     iterative imputation with RandomForest estimator
                     (MissForest approximation)
  softimpute_imp     SVD soft-thresholding (SoftImpute)
  class_mean_imp     ORACLE reference: class-conditional mean
  early_fusion_xgb   raw concatenated (mean-imputed + indicators) -> XGBoost
  flag_only_xgb      availability flags only -> XGBoost (ablation)
  p_only             probabilities without flags (ablation)

Plus per-modality solo-AUC table.

Outputs: experiments/results/imputation_benchmark_v3_results.tsv,
         modality_solo_auc.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import argparse as _ap
_ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
_OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
_argv = _ap.ArgumentParser()
_argv.add_argument("--data-root", default=str(_ROOT))
_argv.add_argument("--out", default=str(_OUT))
_a, _ = _argv.parse_known_args()
ROOT = Path(_a.data_root)
OUT = Path(_a.out)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]


class SoftImpute:
    """SVD soft-thresholding imputation (SoftImpute)."""

    def __init__(self, max_rank=5, n_iter=10):
        self.max_rank = max_rank
        self.n_iter = n_iter

    def fit_transform(self, X):
        from sklearn.decomposition import TruncatedSVD
        X = np.asarray(X, dtype=float)
        mask = ~np.isnan(X)
        X0 = np.nan_to_num(X, nan=0.0)
        for _ in range(self.n_iter):
            svd = TruncatedSVD(n_components=self.max_rank, random_state=0)
            Xr = svd.fit_transform(X0).dot(np.diag(svd.singular_values_)).dot(
                svd.components_)
            X0 = np.where(mask, X, Xr)
        return X0


class MissForest:
    """Iterative RF imputation (MissForest approximation)."""

    def fit_transform(self, X):
        imp = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=20, n_jobs=6,
                                            random_state=0),
            max_iter=3, random_state=0)
        return imp.fit_transform(X)


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
    nmod = min(X.shape[1] // per if flags else (X.shape[1] // K),
              X.shape[1] // (per if flags else K))
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr, Xte = X[tr], y[tr], X[fold:fold + 1]
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if flags and b * per + K < Xte.shape[1] and Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
            elif not flags:
                if (b + 1) * K <= Xte.shape[1]:
                    acc += Xte[0, b * K:(b + 1) * K]
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
                                learning_rate=0.05, random_state=0, n_jobs=6,
                                objective="multi:softprob", num_class=K,
                                eval_metric="mlogloss")
            xgb.fit(Xtr, ytr)
            preds["XGBoost"][fold] = xgb.predict_proba(Xte)[0]
        except Exception:
            preds["XGBoost"][fold] = preds["SimpleMean"][fold]
        if (fold + 1) % 50 == 0:
            print(f"    fold {fold + 1}/{n}")
    return preds


def run_strategy(strat, mods, ids, y, K, results, done, with_flags=True):
    if strat in done:
        print(f"{strat} skipped (resume)", flush=True)
        return
    piv = stage1_loo(mods, ids, y, K)
    X = meta_features(piv, ids, sorted(mods), K, with_flags=with_flags)
    preds = meta_loo(X, y, K, MODELS, flags=with_flags)
    for m in MODELS:
        results.append({"strategy": strat, "model": m, "n": len(ids),
                        "macro_auc": macro_auc(y, preds[m])})
    print(f"{strat} done", flush=True)


def main():
    ids, y, layers = load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}", flush=True)
    done_strats = set()
    resfile = OUT / "imputation_benchmark_v3_results.tsv"
    if resfile.exists():
        done_strats = set(pd.read_csv(resfile, sep="\t")["strategy"])
        print("resuming, skipping:", sorted(done_strats), flush=True)
    results = []

    # ours + ablations
    piv = stage1_loo(layers, ids, y, K)
    solo = {}
    for m in mod_order:
        sub = [p for p in ids if p in piv and m in piv[p]]
        if len(sub) >= 10:
            P = np.vstack([piv[p][m] for p in sub])
            solo[m] = (len(sub), macro_auc(
                np.array([y[ids.index(p)] for p in sub]), P))
    pd.DataFrame([{"modality": m, "n": v[0], "solo_macro_auc": v[1]}
                  for m, v in solo.items()]).sort_values(
        "solo_macro_auc", ascending=False).to_csv(
        OUT / "modality_solo_auc.tsv", sep="\t", index=False)

    X = meta_features(piv, ids, mod_order, K, with_flags=True)
    preds = meta_loo(X, y, K, MODELS, flags=True)
    for m in MODELS:
        results.append({"strategy": "avail_flags", "model": m, "n": len(ids),
                        "macro_auc": macro_auc(y, preds[m])})
    print("avail_flags done", flush=True)

    cc = [p for p in ids if all(p in layers[m].dropna().index for m in mod_order)]
    ccidx = [ids.index(p) for p in cc]
    if len(ccidx) >= 15:
        lay_cc = {m: layers[m].loc[cc] for m in mod_order}
        run_strategy("complete_case", lay_cc, cc, y[ccidx], K, results, done_strats)
        # rewrite n for complete-case (run_strategy wrote len(ids))
        for r in results:
            if r["strategy"] == "complete_case":
                r["n"] = len(cc)

    Xp = meta_features(piv, ids, mod_order, K, with_flags=False)
    preds_p = meta_loo(Xp, y, K, MODELS, flags=False)
    for m in MODELS:
        results.append({"strategy": "p_only", "model": m, "n": len(ids),
                        "macro_auc": macro_auc(y, preds_p[m])})
    print("p_only done", flush=True)

    flags_only = X[:, np.arange(X.shape[1]) % (K + 1) == K]
    Pfo = np.zeros((len(y), K))
    for k in range(len(y)):
        tr = [t for t in range(len(y)) if t != k]
        xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                            random_state=0, n_jobs=6,
                            objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
        xgb.fit(flags_only[tr], y[tr])
        Pfo[k] = xgb.predict_proba(flags_only[k:k + 1])[0]
    results.append({"strategy": "flag_only_xgb", "model": "XGBoost",
                    "n": len(ids), "macro_auc": macro_auc(y, Pfo)})
    print("flag_only done", flush=True)

    # simple imputers
    for strat, imp in [
        ("zero_imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("mean_imp", SimpleImputer(strategy="mean")),
        ("median_imp", SimpleImputer(strategy="median")),
        ("knn_imp", KNNImputer(n_neighbors=3)),
        ("mice_imp", IterativeImputer(max_iter=10, random_state=42)),
    ]:
        imputed = {}
        for m in mod_order:
            Ximp = imp.fit_transform(layers[m].values.astype(float))
            imputed[m] = pd.DataFrame(Ximp, index=layers[m].index,
                                      columns=layers[m].columns)
        run_strategy(strat, imputed, ids, y, K, results, done_strats)

    # mean + missing indicators
    imputed = {}
    for m in mod_order:
        d = layers[m]
        Ximp = SimpleImputer(strategy="mean").fit_transform(
            d.values.astype(float))
        ind = (~d.notna().values).astype(float)
        Xc = np.hstack([Ximp, ind])
        imputed[m] = pd.DataFrame(Xc, index=d.index,
                                  columns=list(d.columns) +
                                  [f"miss_{c}" for c in d.columns])
    run_strategy("mean_imp_ind", imputed, ids, y, K, results, done_strats)

    # log-based
    for strat, use_knn in [("log_mean_imp", False), ("log_knn_imp", True)]:
        imputed = {}
        for m in mod_order:
            Xl = np.log(layers[m].values.astype(float) + 1.0)
            imp = (KNNImputer(n_neighbors=3) if use_knn
                   else SimpleImputer(strategy="mean")).fit(Xl)
            imputed[m] = pd.DataFrame(imp.transform(Xl),
                                      index=layers[m].index,
                                      columns=layers[m].columns)
        run_strategy(strat, imputed, ids, y, K, results, done_strats)

    # pseudo-CLR then mean
    imputed = {}
    for m in mod_order:
        Xl = np.log(layers[m].values.astype(float) + 1.0)
        Xl = Xl - Xl.mean(axis=1, keepdims=True)
        imp = SimpleImputer(strategy="mean").fit(Xl)
        imputed[m] = pd.DataFrame(imp.transform(Xl),
                                  index=layers[m].index,
                                  columns=layers[m].columns)
    run_strategy("clr_mean_imp", imputed, ids, y, K, results, done_strats)

    # missforest approximation
    imputed = {}
    for m in mod_order:
        d = layers[m]
        imp = MissForest().fit_transform(d.values.astype(float))
        imputed[m] = pd.DataFrame(imp, index=d.index, columns=d.columns)
    run_strategy("missforest_imp", imputed, ids, y, K, results, done_strats)

    # softimpute
    imputed = {}
    for m in mod_order:
        d = layers[m]
        imp = SoftImpute().fit_transform(d.values.astype(float))
        imputed[m] = pd.DataFrame(imp, index=d.index, columns=d.columns)
    run_strategy("softimpute_imp", imputed, ids, y, K, results, done_strats)

    # ORACLE: class-conditional mean
    imputed = {}
    for m in mod_order:
        d = layers[m]
        Ximp = np.zeros_like(d.values, dtype=float)
        for cls in range(K):
            cmask = np.array([y[ids.index(p)] == cls for p in d.index])
            col_mean = np.nanmean(d.values[cmask], axis=0)
            Ximp[cmask] = np.where(np.isnan(d.values[cmask]),
                                   col_mean, d.values[cmask])
        imputed[m] = pd.DataFrame(Ximp, index=d.index, columns=d.columns)
    run_strategy("class_mean_oracle", imputed, ids, y, K, results, done_strats)

    # early fusion
    parts, inds = [], []
    for m in mod_order:
        d = layers[m]
        Ximp = SimpleImputer(strategy="mean").fit_transform(
            d.values.astype(float))
        parts.append(Ximp)
        inds.append((~d.notna().values).astype(float))
    Xef = np.hstack(parts + inds)
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
    print("early_fusion done", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT / "imputation_benchmark_v3_results.tsv", sep="\t",
              index=False)
    print(df.pivot_table(index="strategy", columns="model",
                         values="macro_auc").round(3).to_string())


if __name__ == "__main__":
    main()
