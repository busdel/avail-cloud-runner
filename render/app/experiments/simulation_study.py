# -*- coding: utf-8 -*-
"""Controlled simulation study: when does the availability-aware fusion win?

Generative model: K classes, M modalities, each with class-informative +
noise features. Missingness mechanisms:
  MCAR  : cells dropped with probability p
  MAR   : drop probability depends on OTHER modalities (logistic)
  MNAR  : drop probability depends on the modality's OWN value (logistic)

Methods compared (same LOO protocol everywhere):
  avail_flags    p+flag fusion (SimpleMean + XGBoost meta)
  complete_case  listwise deletion
  mean_imp       mean imputation (reference-set fitted - leak-free)
  mice_imp       iterative imputation (fold-internal)
  early_fusion   concatenated imputed features -> XGBoost
  oracle         NO missingness (upper bound)

Grid: missingness {0.2, 0.4, 0.6} x mechanism {MCAR, MAR, MNAR} x 40 reps.

Outputs: experiments/results/simulation_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))
OUT = BASE / "experiments/results"

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

K = 2
M = 4
N = 200
N_FEAT = 40
REPS = 40


def gen_data(seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, K, N)
    mods = []
    inform = [0.9, 0.5, 0.3, 0.05]   # per-modality informativeness
    for m in range(M):
        X = np.zeros((N, N_FEAT))
        n_info = 10
        X[:, :n_info] = y[:, None] * inform[m] + rng.normal(0, 1, (N, n_info))
        X[:, n_info:] = rng.normal(0, 1, (N, N_FEAT - n_info))
        mods.append(X)
    return y, mods


def apply_missingness(mods, y, p, mech, seed):
    rng = np.random.RandomState(seed)
    masked = []
    for m in range(M):
        X = mods[m].copy()
        mask = np.ones((N, N_FEAT), dtype=bool)
        if mech == "MCAR":
            drop = rng.rand(N, N_FEAT) < p
        elif mech == "MAR":
            other = (m + 1) % M
            z = mods[other][:, 0] + mods[other][:, 1]
            pr = 1 / (1 + np.exp(-(z * 1.2)))
            pr = np.clip(pr * (p / pr.mean()), 0, 0.95)
            drop = rng.rand(N, N_FEAT) < pr[:, None]
        else:  # MNAR
            z = X[:, 0] + X[:, 1]
            pr = 1 / (1 + np.exp(-(z * 1.2)))
            pr = np.clip(pr * (p / pr.mean()), 0, 0.95)
            drop = rng.rand(N, N_FEAT) < pr[:, None]
        mask[drop] = False
        Xm = X.copy()
        Xm[~mask] = np.nan
        masked.append(Xm)
    return masked


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def stage1_loo(mods, y, imputer=None):
    piv = {}
    avail = {}
    for m, Xm in enumerate(mods):
        P = np.zeros((N, K))
        A = np.zeros(N)
        for k in range(N):
            tr = [t for t in range(N) if t != k]
            if imputer is None:
                ok = ~np.isnan(Xm[tr]).all(axis=1)
                Xtr = Xm[tr][ok]
                ytr = y[tr][ok]
                if np.isnan(Xm[k]).all():
                    P[k] = np.nan
                    A[k] = 0.0
                    continue
                Xte = np.nan_to_num(Xm[k:k + 1], nan=0.0)
                A[k] = 1.0
            else:
                imp = imputer().fit(Xm[tr])
                Xtr = imp.transform(Xm[tr])
                Xte = imp.transform(Xm[k:k + 1])
                ytr = y[tr]
                A[k] = 1.0
            if len(np.unique(ytr)) < 2:
                P[k] = np.full(K, 1 / K)
                continue
            rf = RandomForestClassifier(n_estimators=100, random_state=0,
                                        n_jobs=4)
            rf.fit(Xtr, ytr)
            P[k] = rf.predict_proba(Xte)[0]
        piv[m] = np.nan_to_num(P, nan=0.0)
        avail[m] = A
    return piv, avail


def meta(piv, avail, y):
    per = K + 1
    X = np.zeros((N, M * per))
    for m in range(M):
        X[:, m * per:m * per + K] = piv[m]
        X[:, m * per + K] = avail[m]
    # SimpleMean
    P_sm = np.zeros((N, K))
    for r in range(N):
        acc = np.zeros(K)
        w = 0.0
        for m in range(M):
            b = m * per
            if X[r, b + K] == 1:
                acc += X[r, b:b + K]
                w += 1.0
        P_sm[r] = acc / w if w > 0 else np.full(K, 1 / K)
    # XGBoost meta LOO
    P_xg = np.zeros((N, K))
    for k in range(N):
        tr = [t for t in range(N) if t != k]
        clf = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1,
                            random_state=0, n_jobs=4, eval_metric="logloss")
        clf.fit(X[tr], y[tr])
        P_xg[k] = clf.predict_proba(X[k:k + 1])[0]
    return P_sm, P_xg


def early_fusion(masked, y, ref_stats):
    parts, inds = [], []
    for m in range(M):
        X = np.where(np.isnan(masked[m]), ref_stats[m], masked[m])
        parts.append(X)
        inds.append(np.isnan(masked[m]).astype(float))
    Xef = np.hstack(parts + inds)
    P = np.zeros((N, K))
    for k in range(N):
        tr = [t for t in range(N) if t != k]
        clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                            random_state=0, n_jobs=4, eval_metric="logloss")
        clf.fit(Xef[tr], y[tr])
        P[k] = clf.predict_proba(Xef[k:k + 1])[0]
    return P


def main():
    rows = []
    for mech in ["MCAR", "MAR", "MNAR"]:
        for p in [0.2, 0.4, 0.6]:
            res = {"mechanism": mech, "p_missing": p}
            for method in ["avail_sm", "avail_xgb", "complete_case_sm",
                           "mean_imp_sm", "mice_imp_sm", "hybrid_sm",
                           "early_fusion", "oracle_sm"]:
                res[method] = []
            for rep in range(REPS):
                y, mods = gen_data(rep)
                masked = apply_missingness(mods, y, p, mech, rep)
                ref_stats = [np.nanmean(mods[m], axis=0) for m in range(M)]
                # oracle (no missingness)
                piv_o, av_o = stage1_loo(mods, y)
                P_o, _ = meta(piv_o, av_o, y)
                res["oracle_sm"].append(macro_auc(y, P_o))
                # avail
                piv_a, av_a = stage1_loo(masked, y)
                P_sm, P_xg = meta(piv_a, av_a, y)
                res["avail_sm"].append(macro_auc(y, P_sm))
                res["avail_xgb"].append(macro_auc(y, P_xg))
                # complete-case
                keep = np.array([not np.isnan(masked[m]).all(axis=1).any()
                                 for m in range(M)]).all(axis=0)
                if keep.sum() >= 15:
                    piv_c, av_c = stage1_loo([masked[m][keep] for m in range(M)],
                                             y[keep])
                    P_c, _ = meta(piv_c, av_c, y[keep])
                    res["complete_case_sm"].append(macro_auc(y[keep], P_c))
                # mean imputation (reference stats -> fixed imputer)
                piv_m, av_m = stage1_loo(masked, y,
                                         imputer=lambda s=0: SimpleImputer(
                                             strategy="mean"))
                P_ms, _ = meta(piv_m, av_m, y)
                res["mean_imp_sm"].append(macro_auc(y, P_ms))
                # mice
                piv_i, av_i = stage1_loo(masked, y,
                                         imputer=lambda: IterativeImputer(
                                             max_iter=3, random_state=0))
                P_is, _ = meta(piv_i, av_i, y)
                res["mice_imp_sm"].append(macro_auc(y, P_is))
                # hybrid: imputed base predictions + REAL raw availability
                # flags (the plain imputed variants carry all-1 flags, so
                # they cannot benefit from presence information).
                P_hs, _ = meta(piv_m, av_a, y)
                res["hybrid_sm"].append(macro_auc(y, P_hs))
                # early fusion
                res["early_fusion"].append(
                    macro_auc(y, early_fusion(masked, y, ref_stats)))
            for method in res:
                if method in ("mechanism", "p_missing"):
                    continue
                vals = res[method]
                if vals:
                    rows.append({"mechanism": mech, "p_missing": p,
                                 "method": method,
                                 "auc_mean": float(np.mean(vals)),
                                 "auc_se": float(np.std(vals) / np.sqrt(len(vals))),
                                 "n_reps": len(vals)})
            print(f"{mech} p={p} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "simulation_results.tsv", sep="\t", index=False)
    piv = df.pivot_table(index=["mechanism", "p_missing"], columns="method",
                         values="auc_mean").round(3)
    print(piv.to_string())
    print("simulation study done")


if __name__ == "__main__":
    main()
