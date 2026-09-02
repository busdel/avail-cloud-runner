# -*- coding: utf-8 -*-
"""AVAIL-AI learners: leak-free base learners + availability-aware meta-learners."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .preprocess import preproc, apply_preproc


def macro_auc(y, P):
    aucs = []
    K = P.shape[1]
    for c in range(K):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs)) if aucs else 0.5


def base_predictions_loo(mods, ids, label, conf, K, micro, residualize=False,
                         n_estimators=300, random_state=42):
    """Stage 1: leak-free per-modality base predictions via leave-one-out."""
    piv = {}
    for mname, feat_map in mods.items():
        sub_ids = sorted({i for i in ids if i in feat_map})
        X = np.vstack([feat_map[i] for i in sub_ids])
        y = np.array([label[i] for i in sub_ids])
        C = np.vstack([conf[i] for i in sub_ids])
        is_micro = mname in micro
        proba = np.zeros((len(sub_ids), K))
        for k in range(len(sub_ids)):
            tr = [t for t in range(len(sub_ids)) if t != k]
            sel, sel2, res, sc, Xtr = preproc(X[tr], y[tr], C[tr], is_micro, residualize)
            Xte = apply_preproc(sel, sel2, res, sc, X[k:k + 1], C[k:k + 1])
            m = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
            m.fit(Xtr, y[tr])
            for ci, cls in enumerate(m.classes_):
                proba[k, int(cls)] = m.predict_proba(Xte)[0, ci]
        for pid, pv in zip(sub_ids, proba):
            piv.setdefault(pid, {})[mname] = pv
    return piv


def build_meta_features(piv, ids, label, n_mods, mod_order, K, masked_value=0.0):
    D = n_mods * (K + 1)
    X = np.zeros((len(ids), D))
    y = np.array([label[i] for i in ids])
    for r, pid in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * (K + 1)
            if pid in piv and m in piv[pid]:
                X[r, b:b + K] = piv[pid][m]
                X[r, b + K] = 1.0
            else:
                X[r, b:b + K] = masked_value
                X[r, b + K] = 0.0
    return X, y


def simple_mean_probs(X, n_mods, K):
    per = K + 1
    n = X.shape[0]
    P = np.zeros((n, K))
    for s in range(n):
        acc = np.zeros(K)
        w = 0.0
        for m in range(n_mods):
            b = m * per
            if X[s, b + K] == 1:
                acc += X[s, b:b + K]
                w += 1.0
        P[s] = acc / w if w > 0 else np.full(K, 1.0 / K)
    return P


def modality_weights(Xtr, ytr, n_mods, K):
    """Fold-internal AUC-based weights (weak modalities get lower weight)."""
    per = K + 1
    w = []
    for m in range(n_mods):
        b = m * per
        obs = Xtr[:, b + K] == 1
        if obs.sum() >= 10:
            P = Xtr[obs, b:b + K]
            w.append(max(macro_auc(ytr[obs], P), 0.5))
        else:
            w.append(0.5)
    return np.array(w)


def weighted_simple_mean(X, y, n_mods, K):
    """WeightedSimpleMean: LOO with fold-internal AUC weighting."""
    n = len(y)
    P = np.zeros((n, K))
    per = K + 1
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        w = modality_weights(X[tr], y[tr], n_mods, K)
        acc = np.zeros(K)
        wsum = 0.0
        for m in range(n_mods):
            b = m * per
            if X[fold, b + K] == 1:
                acc += w[m] * X[fold, b:b + K]
                wsum += w[m]
        P[fold] = acc / wsum if wsum > 0 else np.full(K, 1.0 / K)
    return P


def elasticnet_loo(X, y, K, C=0.3):
    n = len(y)
    P = np.zeros((n, K))
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=C,
                                max_iter=5000, class_weight='balanced',
                                random_state=0)
        try:
            lr.fit(X[tr], y[tr])
            P[fold] = lr.predict_proba(X[fold:fold + 1])[0]
        except Exception:
            P[fold] = np.full(K, 1.0 / K)
    return P


def run_meta_benchmark(X, y, n_mods, K, model_names=('SimpleMean', 'WSimpleMean', 'ElasticNetLR')):
    """Returns {model_name: (n, K) probability matrix} for the requested meta-learners."""
    preds = {}
    if 'SimpleMean' in model_names:
        preds['SimpleMean'] = simple_mean_probs(X, n_mods, K)
    if 'WSimpleMean' in model_names:
        preds['WSimpleMean'] = weighted_simple_mean(X, y, n_mods, K)
    if 'ElasticNetLR' in model_names:
        preds['ElasticNetLR'] = elasticnet_loo(X, y, K)
    return preds
