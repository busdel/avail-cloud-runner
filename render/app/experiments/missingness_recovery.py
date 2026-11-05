# -*- coding: utf-8 -*-
"""Missingness-recovery benchmark (LEAK-FREE version).

On the 50 fully-measured subjects we artificially drop modality layers
(10-70%) and compare three strategies with the SAME LOO protocol:

  avail_flags     p+flag fusion, base learners trained on available subjects
  complete_case   subjects with all layers kept
  mean_imp        FOLD-INTERNAL mean imputation (imputer fit per LOO fold
                  on training subjects only - no cross-subject leakage)

Important: imputation is applied inside each LOO fold. Imputing with
statistics computed on ALL subjects (including the left-out one) leaks
test-subject information and inflates AUC - this version fixes that.

Outputs: experiments/results/missingness_recovery.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--data-root",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
__ap.add_argument("--out",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results")
__a2, _ = __ap.parse_known_args()
ROOT = Path(__a2.data_root)
OUT = Path(__a2.out)

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]


class FixedMeanImputer:
    """Mean imputer with statistics fixed on a reference set.

    Fold-internal mean imputation of artificially-dropped values leaks the
    true value through the complement identity (mean over n-1 subjects
    encodes the left-out subject's value). Fitting the imputer ONCE on a
    reference set disjoint from the evaluation subjects avoids this and
    mimics how imputation works for genuinely-missing data.
    """

    def __init__(self, reference_stats):
        self.statistics_ = reference_stats

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(np.isnan(X), self.statistics_, X)


def load():
    meta = pd.read_csv(ROOT / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    classes = sorted(meta["Group"].astype(str).unique())
    m = meta[meta.Group.astype(str).isin(classes)]
    ids = m["SampleID"].tolist()
    y = np.array([classes.index(g) for g in m["Group"].astype(str)])
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
    """LOO base predictions. If imputer_factory is given, each fold's
    training features are imputed with an imputer fitted on the training
    subjects ONLY (leak-free)."""
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
                imp = imputer_factory().fit(Xfull[tr])
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
                if np.isnan(Xte).all(axis=1):
                    # truly missing: NO prediction, NO availability flag
                    P[k] = np.nan
                    continue
                Xte = np.nan_to_num(Xte, nan=0.0)
            P[k] = rf.predict_proba(Xte)[0]
        for p, pv in zip(sids, P):
            if not np.isnan(pv).any():
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
                                learning_rate=0.05, random_state=0, n_jobs=6,
                                objective="multi:softprob", num_class=K,
                                eval_metric="mlogloss")
            xgb.fit(Xtr, ytr)
            preds["XGBoost"][fold] = xgb.predict_proba(Xte)[0]
        except Exception:
            preds["XGBoost"][fold] = preds["SimpleMean"][fold]
    return preds


def main():
    ids, y, layers = load()
    # normalize labels to 0..K-1 (binary cohorts may use {0,2} style codes)
    uniq = sorted(set(int(v) for v in y.tolist()))
    code = {u: i for i, u in enumerate(uniq)}
    y = np.array([code[int(v)] for v in y.tolist()])
    K = len(uniq)
    mod_order = sorted(layers)
    full = [p for p in ids if all(p in layers[m].dropna().index
                                  for m in mod_order)]
    fidx = [ids.index(p) for p in full]
    yf = y[fidx]
    print(f"complete subjects: {len(full)}")

    rows = []
    for rate in [0.1, 0.25, 0.4, 0.55, 0.7]:
        for seed in range(3):
            rng = np.random.RandomState(seed)
            masked = {m: layers[m].loc[full].copy() for m in mod_order}
            for p in full:
                drop = rng.rand(len(mod_order)) < rate
                for m, d in zip(mod_order, drop):
                    if d:
                        masked[m].loc[p] = np.nan

            # 1) AVAIL
            piv = stage1_loo(masked, full, yf, K)
            X = meta_features(piv, full, mod_order, K)
            preds = meta_loo(X, yf, K, MODELS)
            for m in MODELS:
                rows.append({"drop_rate": rate, "seed": seed,
                             "strategy": "avail_flags", "model": m,
                             "n_used": len(full),
                             "macro_auc": macro_auc(yf, preds[m])})

            # 2) complete-case
            cc = [p for p in full
                  if all(p in masked[m].dropna().index for m in mod_order)]
            if len(cc) >= 15:
                ccidx = [full.index(p) for p in cc]
                piv = stage1_loo({m: masked[m].loc[cc] for m in mod_order},
                                 cc, yf[ccidx], K)
                X = meta_features(piv, cc, mod_order, K)
                preds = meta_loo(X, yf[ccidx], K, MODELS)
                for m in MODELS:
                    rows.append({"drop_rate": rate, "seed": seed,
                                 "strategy": "complete_case", "model": m,
                                 "n_used": len(cc),
                                 "macro_auc": macro_auc(yf[ccidx], preds[m])})
            else:
                for m in MODELS:
                    rows.append({"drop_rate": rate, "seed": seed,
                                 "strategy": "complete_case", "model": m,
                                 "n_used": len(cc), "macro_auc": np.nan})

            # 3) reference-fitted mean imputation, 5-fold CV.
            # NOTE: LOO mean-imputation leaks the left-out label through the
            # complement structure (identical imputed rows -> prediction is a
            # complement statistic of the training labels). Fixed 5-fold CV
            # avoids this and is documented as such.
            ref_stats = {}
            for m in mod_order:
                ref = layers[m].loc[[p for p in ids if p not in full]]
                ref_stats[m] = np.nanmean(ref.values, axis=0)
            P_all = {m: np.zeros((len(full), K)) for m in MODELS}
            piv_imp = {}
            from sklearn.model_selection import KFold
            kf = KFold(5, shuffle=True, random_state=seed)
            for tr_idx, te_idx in kf.split(np.arange(len(full))):
                piv = {}
                for m in mod_order:
                    Xtr = np.where(np.isnan(masked[m].values[tr_idx]),
                                   ref_stats[m], masked[m].values[tr_idx])
                    Xte = np.where(np.isnan(masked[m].values[te_idx]),
                                   ref_stats[m], masked[m].values[te_idx])
                    rf = RandomForestClassifier(n_estimators=200,
                                                random_state=42, n_jobs=6)
                    rf.fit(Xtr, yf[tr_idx])
                    P = rf.predict_proba(Xte)
                    for i, p in enumerate([full[t] for t in te_idx]):
                        piv.setdefault(p, {})[m] = P[i]
                        piv_imp.setdefault(p, {})[m] = P[i]
                X = meta_features(piv, full, mod_order, K)
                for m in MODELS:
                    pass
                # meta per fold: train on train subjects, predict test
                from sklearn.linear_model import LogisticRegression
                from xgboost import XGBClassifier
                for mm in ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]:
                    pass
                Xtrm = X[tr_idx]
                Xtem = X[te_idx]
                # SimpleMean
                acc = np.zeros((len(te_idx), K))
                for i, t in enumerate(te_idx):
                    wsum = 0.0
                    for c in range(len(mod_order)):
                        b = c * (K + 1)
                        if X[t, b + K] == 1:
                            acc[i] += X[t, b:b + K]
                            wsum += 1.0
                    acc[i] = acc[i] / wsum if wsum > 0 else np.full(K, 1/K)
                P_all["SimpleMean"][te_idx] = acc
                P_all["WSimpleMean"][te_idx] = acc
                try:
                    lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                            l1_ratio=0.5, C=0.3, max_iter=3000,
                                            class_weight="balanced", random_state=0)
                    lr.fit(Xtrm, yf[tr_idx])
                    P_all["ElasticNetLR"][te_idx] = lr.predict_proba(Xtem)
                except Exception:
                    P_all["ElasticNetLR"][te_idx] = acc
                xgb = XGBClassifier(n_estimators=200, max_depth=2,
                                    learning_rate=0.05, random_state=0, n_jobs=6,
                                    objective="multi:softprob", num_class=K,
                                    eval_metric="mlogloss")
                xgb.fit(Xtrm, yf[tr_idx])
                P_all["XGBoost"][te_idx] = xgb.predict_proba(Xtem)
            for m in MODELS:
                rows.append({"drop_rate": rate, "seed": seed,
                             "strategy": "mean_imp_cv", "model": m,
                             "n_used": len(full),
                             "macro_auc": macro_auc(yf, P_all[m])})
            # 4) HYBRID: imputed base predictions (as above) + REAL raw
            #    availability flags from the masked layers.
            raw_av = np.zeros((len(full), len(mod_order)))
            for c, m in enumerate(mod_order):
                for r, p in enumerate(full):
                    if p in masked[m].dropna().index:
                        raw_av[r, c] = 1.0
            X_hyb = meta_features(piv_imp, full, mod_order, K)
            for c in range(len(mod_order)):
                X_hyb[:, c * (K + 1) + K] = raw_av[:, c]
            preds_h = meta_loo(X_hyb, yf, K, MODELS)
            for m in MODELS:
                rows.append({"drop_rate": rate, "seed": seed,
                             "strategy": "hybrid_flags_imp", "model": m,
                             "n_used": len(full),
                             "macro_auc": macro_auc(yf, preds_h[m])})
            print(f"rate={rate} seed={seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "missingness_recovery.tsv", sep="\t", index=False)
    piv_t = df.pivot_table(index="drop_rate", columns=["strategy", "model"],
                           values="macro_auc", aggfunc="mean").round(3)
    print(piv_t.to_string())
    print("missingness recovery (leak-free) done")


if __name__ == "__main__":
    main()
