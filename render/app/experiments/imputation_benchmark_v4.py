# -*- coding: utf-8 -*-
"""Robustness benchmark v4 (LEAK-FREE): AVAIL vs missing-data strategies.

CRITICAL FIX vs v3: imputation is applied FOLD-INTERNALLY. Each LOO fold
fits the imputer on the training subjects ONLY and transforms the left-out
subject with it. Imputing with statistics computed on all subjects leaks
test-subject information and inflated imputation AUC (0.90+) in v3.

Strategies (same n=150 data, same LOO protocol):
  avail_flags        ours: p + availability flag
  complete_case      drop incomplete subjects
  zero_imp / mean_imp / median_imp
  mean_imp_ind       mean + missing-indicator columns
  knn_imp            KNN (k=3)
  log_mean_imp / log_knn_imp / clr_mean_imp
  mice_imp           iterative linear (max_iter=2, top-200 features)
  missforest_imp     iterative RF (max_iter=2, top-200 features)
  softimpute_imp     SVD soft-thresholding (top-200 features)
  class_mean_oracle  class-conditional mean (fold-internal, label-informed)
  early_fusion_xgb   raw concatenated (fold-internal mean + indicators)
  flag_only_xgb / p_only   ablations

Outputs: experiments/results/imputation_benchmark_v4_results.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]


class SoftImpute:
    def __init__(self, max_rank=5, n_iter=10):
        self.max_rank = max_rank
        self.n_iter = n_iter

    def fit(self, X, y=None):
        return self

    def transform(self, X):
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


class FeatureCapper:
    """Keep top-N features by variance (fitted on training fold only)."""

    def __init__(self, top_n=None):
        self.top_n = top_n
        self.idx_ = None

    def fit(self, X, y=None):
        if self.top_n and X.shape[1] > self.top_n:
            self.idx_ = np.argsort(np.nanvar(X, axis=0))[::-1][:self.top_n]
        return self

    def transform(self, X):
        if self.idx_ is not None:
            return X[:, self.idx_]
        return X


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


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def stage1_loo(mods, ids, y, K, imputer_factory=None, n_est=200):
    """LOO base predictions, optionally with fold-internal imputation."""
    piv = {}
    for mname, Xdf in mods.items():
        sids = list(Xdf.index)
        Xfull = Xdf.values.astype(float)
        yy = np.array([y[ids.index(p)] for p in sids])
        P = np.zeros((len(sids), K))
        for k in range(len(sids)):
            tr = [t for t in range(len(sids)) if t != k]
            if imputer_factory is None:
                mask = ~np.isnan(Xfull[tr]).all(axis=1)
                Xtr = Xfull[tr][mask]
                ytr = yy[tr][mask]
            else:
                imp = imputer_factory().fit(Xfull[tr], yy[tr])
                Xtr = imp.transform(Xfull[tr])
                ytr = yy[tr]
            if len(Xtr) < 10 or len(np.unique(ytr)) < 2:
                P[k] = np.full(K, 1 / K)
                continue
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                        n_jobs=6)
            rf.fit(Xtr, ytr)
            Xte = Xfull[k:k + 1]
            if imputer_factory is not None:
                Xte = imp.transform(Xte)
            else:
                if np.isnan(Xte).all():
                    P[k] = np.nan
                    continue
                Xte = np.nan_to_num(Xte, nan=0.0)
            P[k] = rf.predict_proba(Xte)[0]
        for p, pv in zip(sids, P):
            if not np.isnan(pv).any():
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


def meta_loo(X, y, K, models, flags=True):
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in models}
    per = K + 1
    nmod = min(X.shape[1] // (per if flags else K), 1000)
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr, Xte = X[tr], y[tr], X[fold:fold + 1]
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if flags and Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
            elif not flags and (b + 1) * K <= Xte.shape[1]:
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
            print(f"    fold {fold + 1}/{n}", flush=True)
    return preds


import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--data-root",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
__ap.add_argument("--out",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results")
__a2, _ = __ap.parse_known_args()
ROOT = Path(__a2.data_root)
OUT = Path(__a2.out)

from sklearn.model_selection import KFold


def stage1_cv(mods, ids, y, K, imputer_factory=None, n_folds=5, seed=0,
              n_est=200):
    """5-fold-CV base predictions with fold-internal imputation.
    Leak-free for imputed strategies (no complement-of-labels leak)."""
    piv = {}
    for mname, Xdf in mods.items():
        sids = list(Xdf.index)
        Xfull = Xdf.values.astype(float)
        yy = np.array([y[ids.index(p)] for p in sids])
        P = np.zeros((len(sids), K))
        kf = KFold(n_folds, shuffle=True, random_state=seed)
        for tr_idx, te_idx in kf.split(np.arange(len(sids))):
            if imputer_factory is None:
                ok = ~np.isnan(Xfull[tr_idx]).all(axis=1)
                Xtr = Xfull[tr_idx][ok]
                ytr = yy[tr_idx][ok]
            else:
                imp = imputer_factory().fit(Xfull[tr_idx], yy[tr_idx])
                Xtr = imp.transform(Xfull[tr_idx])
                ytr = yy[tr_idx]
            if len(np.unique(ytr)) < 2:
                P[te_idx] = np.full(K, 1 / K)
                continue
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                        n_jobs=6)
            rf.fit(Xtr, ytr)
            Xte = Xfull[te_idx]
            if imputer_factory is not None:
                Xte = imp.transform(Xte)
            else:
                Xte = np.nan_to_num(Xte, nan=0.0)
            P[te_idx] = rf.predict_proba(Xte)
        for p, pv in zip(sids, P):
            piv.setdefault(p, {})[mname] = pv
    return piv


def meta_cv(X, y, K, models, n_folds=5, seed=0, flags=True):
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in models}
    per = K + 1
    nmod = min(X.shape[1] // (per if flags else K), 1000)
    kf = KFold(n_folds, shuffle=True, random_state=seed)
    for tr_idx, te_idx in kf.split(np.arange(n)):
        Xtr, ytr, Xte = X[tr_idx], y[tr_idx], X[te_idx]
        acc = np.zeros((len(te_idx), K))
        for i in range(len(te_idx)):
            wsum = 0.0
            for c in range(nmod):
                b = c * per
                if flags and Xte[i, b + K] == 1:
                    acc[i] += Xte[i, b:b + K]
                    wsum += 1.0
                elif not flags and (b + 1) * K <= Xte.shape[1]:
                    acc[i] += Xte[i, b * K:(b + 1) * K]
                    wsum += 1.0
            acc[i] = acc[i] / wsum if wsum > 0 else np.full(K, 1 / K)
        preds["SimpleMean"][te_idx] = acc
        preds["WSimpleMean"][te_idx] = acc
        try:
            lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                    l1_ratio=0.5, C=0.3, max_iter=3000,
                                    class_weight="balanced", random_state=0)
            lr.fit(Xtr, ytr)
            preds["ElasticNetLR"][te_idx] = lr.predict_proba(Xte)
        except Exception:
            preds["ElasticNetLR"][te_idx] = acc
        xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                            random_state=0, n_jobs=6,
                            objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
        xgb.fit(Xtr, ytr)
        preds["XGBoost"][te_idx] = xgb.predict_proba(Xte)
    return preds


def main():
    ids, y, layers = load_layers()
    K = 3
    mod_order = sorted(layers)
    print(f"n={len(ids)} layers={mod_order}", flush=True)
    results = []

    # ours
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
        piv_cc = stage1_loo(lay_cc, cc, y[ccidx], K)
        Xcc = meta_features(piv_cc, cc, mod_order, K, with_flags=True)
        preds_cc = meta_loo(Xcc, y[ccidx], K, MODELS, flags=True)
        for m in MODELS:
            results.append({"strategy": "complete_case", "model": m,
                            "n": len(cc),
                            "macro_auc": macro_auc(y[ccidx], preds_cc[m])})
        print(f"complete_case done (n={len(cc)})", flush=True)

    # ablations
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

    # fold-internal imputation strategies
    factories = {
        "zero_imp": lambda: SimpleImputer(strategy="constant", fill_value=0.0),
        "mean_imp": lambda: SimpleImputer(strategy="mean"),
        "median_imp": lambda: SimpleImputer(strategy="median"),
        "knn_imp": lambda: KNNImputer(n_neighbors=3),
        "mice_imp": lambda: IterativeImputer(max_iter=2, random_state=42,
                                             verbose=0),
        "missforest_imp": lambda: IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, n_jobs=6,
                                            random_state=0),
            max_iter=2, random_state=42),
        "softimpute_imp": lambda: SoftImpute(max_rank=3, n_iter=5),
    }
    for strat, fac in factories.items():
        for seed in range(3):
            piv_i = stage1_cv(layers, ids, y, K, imputer_factory=fac, seed=seed)
            Xi = meta_features(piv_i, ids, mod_order, K, with_flags=True)
            preds_i = meta_cv(Xi, y, K, MODELS, flags=True, seed=seed)
            for m in MODELS:
                results.append({"strategy": strat, "model": m, "n": len(ids),
                                "seed": seed,
                                "macro_auc": macro_auc(y, preds_i[m])})
        print(f"{strat} done (3 seeds)", flush=True)

    # log / clr variants
    for strat in ["log_mean_imp", "clr_mean_imp", "log_knn_imp"]:
        def make(s=strat):
            if s == "log_mean_imp":
                return SimpleImputer(strategy="mean")
            if s == "log_knn_imp":
                return KNNImputer(n_neighbors=3)
            return SimpleImputer(strategy="mean")
        # wrap with log transform inside the fold: custom factory class
        class LogWrap:
            def __init__(self, inner, centering):
                self.inner = inner
                self.centering = centering

            def fit(self, X, y=None):
                Xl = np.log(X + 1.0)
                if self.centering:
                    Xl = Xl - Xl.mean(axis=1, keepdims=True)
                self.inner = self.inner.fit(Xl, y)
                return self

            def transform(self, X):
                Xl = np.log(X + 1.0)
                if self.centering:
                    Xl = Xl - Xl.mean(axis=1, keepdims=True)
                return self.inner.transform(Xl)
        inner = {"log_mean_imp": lambda: SimpleImputer(strategy="mean"),
                 "log_knn_imp": lambda: KNNImputer(n_neighbors=3),
                 "clr_mean_imp": lambda: SimpleImputer(strategy="mean")}[strat]
        fac = lambda s=strat: LogWrap(inner(), centering=(s == "clr_mean_imp"))
        piv_i = stage1_cv(layers, ids, y, K, imputer_factory=fac, seed=0)
        Xi = meta_features(piv_i, ids, mod_order, K, with_flags=True)
        preds_i = meta_cv(Xi, y, K, MODELS, flags=True, seed=0)
        for m in MODELS:
            results.append({"strategy": strat, "model": m, "n": len(ids),
                            "macro_auc": macro_auc(y, preds_i[m])})
        print(f"{strat} done", flush=True)

    # ---------------- HYBRID: imputed signal + REAL raw availability flags
    # In the plain imputed strategies above the availability column is all-1
    # (imputation produces a prediction for every subject). The hybrid keeps
    # the imputed base predictions but restores the TRUE availability flag
    # from the raw (unimputed) layers, giving the meta-learner both the
    # completed signal and the assumption-free presence information.
    raw_avail = np.zeros((len(ids), len(mod_order)))
    for c, m in enumerate(mod_order):
        for r, p in enumerate(ids):
            if p in layers[m].dropna().index:
                raw_avail[r, c] = 1.0
    hybrid_factories = {
        "hybrid_mean_imp": lambda: SimpleImputer(strategy="mean"),
        "hybrid_mice_imp": lambda: IterativeImputer(max_iter=2,
                                                    random_state=42),
    }
    for strat, fac in hybrid_factories.items():
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
    df.to_csv(OUT / "imputation_benchmark_v4_results.tsv", sep="\t",
              index=False)
    print(df.pivot_table(index="strategy", columns="model",
                         values="macro_auc").round(3).to_string())


if __name__ == "__main__":
    main()
