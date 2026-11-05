# -*- coding: utf-8 -*-
"""Virtual Modality Reconstruction (VMR).

Meta-level imputation: for every subject MISSING a modality, we reconstruct
that modality's calibrated probability vector from the subject's AVAILABLE
modalities using a cross-modal regressor trained leak-free (LOO) on subjects
who have the target modality. The reconstruction enters the meta-feature
space as an extra block [virtual_p, virtual_flag] - the meta-learner can
decide how much to trust it.

This addresses "availability flags ignore cross-modal predictive structure"
without ever imputing raw high-dimensional features.
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_piv(bp_path, K):
    bp = pd.read_csv(bp_path, sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    piv = {}
    for _, r in bp.iterrows():
        piv.setdefault(r["patient"], {})[r["modality"]] = \
            np.array([r[f"p_{c}"] for c in range(K)], dtype=float)
    return piv


def cross_modal_features(piv, pid, mod_order, K, exclude):
    """Concatenated p+flag blocks of all modalities except `exclude`."""
    per = K + 1
    vec = []
    for m in mod_order:
        if m == exclude:
            continue
        b = len(vec)  # placeholder
        if pid in piv and m in piv[pid]:
            vec += list(piv[pid][m]) + [1.0]
        else:
            vec += [0.0] * K + [0.0]
    return np.array(vec, dtype=float)


def virtual_reconstruction(piv, ids, mod_order, K, n_est=100, seed=42):
    """Returns (virtual, recon_r2) where virtual[pid][m] = reconstructed p_m
    for subjects missing m, and recon_r2[m] = LOO reconstruction R2 among
    subjects who have m (quality check)."""
    virtual = {}
    recon_r2 = {}
    for m in mod_order:
        have = [p for p in ids if p in piv and m in piv[p]]
        missing = [p for p in ids if p not in piv or m not in piv[p]]
        if len(have) < 20:
            continue
        Xreg = np.vstack([cross_modal_features(piv, p, mod_order, K, m)
                          for p in have])
        Yreg = np.vstack([piv[p][m] for p in have])
        # LOO reconstruction quality (subjects WITH m)
        P = np.zeros_like(Yreg)
        for k in range(len(have)):
            tr = [t for t in range(len(have)) if t != k]
            rf = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=n_est, random_state=seed,
                                      n_jobs=-1))
            rf.fit(Xreg[tr], Yreg[tr])
            P[k] = rf.predict(Xreg[k:k + 1])[0]
        ss_res = ((Yreg - P) ** 2).sum(axis=0)
        ss_tot = ((Yreg - Yreg.mean(axis=0)) ** 2).sum(axis=0)
        recon_r2[m] = float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))
        # full model for the actually-missing subjects
        rf = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_est, random_state=seed,
                                  n_jobs=-1))
        rf.fit(Xreg, Yreg)
        if missing:
            Xm = np.vstack([cross_modal_features(piv, p, mod_order, K, m)
                            for p in missing])
            Pm = rf.predict(Xm)
            for p, pv in zip(missing, Pm):
                pv = np.clip(pv, 0, 1)
                pv = pv / np.maximum(pv.sum(), 1e-9)
                virtual.setdefault(p, {})[m] = pv
    return virtual, recon_r2


def build_meta_features(piv, ids, mod_order, K, virtual=None):
    """Standard [p, flag] or VMR-extended [p, flag, vp, vflag] per modality."""
    has_vmr = virtual is not None
    per = (K + 1) * 2 if has_vmr else K + 1
    X = np.zeros((len(ids), len(mod_order) * per))
    for r, p in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * per
            if p in piv and m in piv[p]:
                X[r, b:b + K] = piv[p][m]
                X[r, b + K] = 1.0
                if has_vmr:
                    X[r, b + K + 1:b + K + 1 + K] = 0.0
                    X[r, b + 2 * K + 1] = 0.0
            else:
                X[r, b:b + K] = 0.0
                X[r, b + K] = 0.0
                if has_vmr and virtual is not None and p in virtual \
                        and m in virtual[p]:
                    X[r, b + K + 1:b + K + 1 + K] = virtual[p][m]
                    X[r, b + 2 * K + 1] = 1.0
    return X


def virtual_reconstruction_knn(piv, ids, mod_order, K, k=5):
    """kNN variant: virtual p_m = average p_m of the k nearest subjects in
    the space of the AVAILABLE modalities' meta blocks."""
    from sklearn.neighbors import NearestNeighbors
    virtual = {}
    for m in mod_order:
        have = [p for p in ids if p in piv and m in piv[p]]
        missing = [p for p in ids if p not in piv or m not in piv[p]]
        if len(have) < 20:
            continue
        Xhave = np.vstack([cross_modal_features(piv, p, mod_order, K, m)
                           for p in have])
        Yhave = np.vstack([piv[p][m] for p in have])
        nn = NearestNeighbors(n_neighbors=min(k, len(have)))
        nn.fit(Xhave)
        if missing:
            Xm = np.vstack([cross_modal_features(piv, p, mod_order, K, m)
                            for p in missing])
            dist, idx = nn.kneighbors(Xm)
            w = 1.0 / np.maximum(dist, 1e-6)
            w = w / w.sum(axis=1, keepdims=True)
            Pm = np.einsum("ij,ijk->ik", w, Yhave[idx])
            for p, pv in zip(missing, Pm):
                pv = np.clip(pv, 0, 1)
                pv = pv / np.maximum(pv.sum(), 1e-9)
                virtual.setdefault(p, {})[m] = pv
    return virtual
