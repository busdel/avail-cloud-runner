# -*- coding: utf-8 -*-
"""Robustness toolkit: label-permutation test and modality sensitivity.

Both routines reuse the exact leak-free stage-1/stage-2 protocol of the
main pipeline (fold-internal preprocessing, LOO base learners, LOO
meta-learners) and work on any generic AVAIL layout
(``metadata.tsv`` + ``features/*.tsv``).
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .generic import detect_compositional, load_generic
from .learners import base_predictions_loo, build_meta_features, macro_auc
from .meta_learners import run_meta_loo, MODEL_NAMES


def _prepare(data_root, classes=None):
    group, conf_raw, mods, feat_names = load_generic(data_root)
    cnt = {}
    for v in group.values():
        cnt[v] = cnt.get(v, 0) + 1
    classes = list(classes) if classes else \
        [c for c, _ in sorted(cnt.items(), key=lambda kv: -kv[1])[:3]]
    K = len(classes)
    ids = sorted([i for i in group if group[i] in classes])
    label = {i: classes.index(group[i]) for i in ids}
    n_mods = len(mods)
    mod_order = sorted(mods.keys())
    conf_vec = {}
    for i in ids:
        c = conf_raw.get(i, [np.nan] * 3)[:3]
        conf_vec[i] = np.array(c + [np.nan] * (3 - len(c)), dtype=float)
    Cmat = np.vstack([conf_vec[i] for i in ids])
    for j in range(3):
        med = np.nanmedian(Cmat[:, j])
        Cmat[:, j] = np.where(np.isnan(Cmat[:, j]), med, Cmat[:, j])
    conf = {i: Cmat[k] for k, i in enumerate(ids)}
    micro = detect_compositional(data_root, mods, ids, verbose=False)
    return ids, label, conf, micro, mods, mod_order, K


def _run_one(label, mods, ids, conf, micro, mod_order, K, models,
             n_estimators=300, random_state=42):
    piv = base_predictions_loo(mods, ids, label, conf, K, micro,
                               n_estimators=n_estimators,
                               random_state=random_state)
    X, y = build_meta_features(piv, ids, label, len(mod_order), mod_order,
                               K, 0.0)
    preds = run_meta_loo(X, y, len(mod_order), K, model_names=models,
                         random_state=random_state, verbose=False)
    return {m: float(macro_auc(y, preds[m])) for m in models}


def label_permutation_test(data_root, out, n_perm=25, jobs=4,
                           models=("SimpleMean", "WSimpleMean",
                                   "ElasticNetLR", "MLP"),
                           seed=0, n_estimators=300):
    """Full-pipeline label-permutation test on a generic AVAIL layout.

    The anchor run (perm 0) uses the true labels; permutations re-run the
    complete stage-1/stage-2 protocol with scrambled labels. Returns a
    DataFrame with one row per permutation and prints per-model null
    distributions and p-values.
    """
    warnings.filterwarnings("ignore")
    data_root = Path(data_root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids, base_label, conf, micro, mods, mod_order, K = _prepare(data_root)
    models = [m for m in models if m in MODEL_NAMES]

    def shuffled(seed_i):
        rng = np.random.default_rng(seed_i)
        vals = [base_label[i] for i in ids]
        order = rng.permutation(len(ids))
        shuf = [vals[t] for t in order]
        return {i: shuf[t] for t, i in enumerate(ids)}

    def worker(job):
        perm_id, seed_i = job
        res = _run_one(shuffled(seed_i), mods, ids, conf, micro,
                       mod_order, K, models, n_estimators=n_estimators)
        print(f"[perm] {perm_id}: {res}", flush=True)
        return perm_id, res

    rows = []
    anchor = _run_one(base_label, mods, ids, conf, micro, mod_order, K,
                      models, n_estimators=n_estimators)
    print(f"[perm] anchor: {anchor}", flush=True)
    rows.append({"perm": 0, **anchor})
    jobs_list = [(p, seed * 100000 + p) for p in range(1, n_perm + 1)]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for perm_id, res in ex.map(worker, jobs_list):
                rows.append({"perm": perm_id, **res})
    else:
        for job in jobs_list:
            _, res = worker(job)
            rows.append({"perm": job[0], **res})
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    for m in models:
        v = df[df.perm > 0][m].values
        obs = df[df.perm == 0][m].values[0]
        pval = (np.sum(v >= obs) + 1) / (len(v) + 1)
        print(f"[perm] {m}: null mean={v.mean():.4f} sd={v.std():.4f} "
              f"max={v.max():.4f} observed={obs:.4f} p={pval:.4f}", flush=True)
    return df


def sensitivity_subsets(data_root, out, models=None, n_estimators=300,
                        random_state=42):
    """Modality sensitivity on cached stage-1 predictions: full, no_<m>,
    <m>_only and without_<m> subsets with the full meta-benchmark."""
    warnings.filterwarnings("ignore")
    data_root = Path(data_root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ids, label, conf, micro, mods, mod_order, K = _prepare(data_root)
    models = list(models) if models else list(MODEL_NAMES)
    piv = base_predictions_loo(mods, ids, label, conf, K, micro,
                               n_estimators=n_estimators,
                               random_state=random_state)

    def build(kept):
        D = len(kept) * (K + 1)
        X = np.zeros((len(ids), D))
        for r, pid in enumerate(ids):
            for c, m in enumerate(kept):
                b = c * (K + 1)
                if pid in piv and m in piv[pid]:
                    X[r, b:b + K] = piv[pid][m]
                    X[r, b + K] = 1.0
        return X

    y = np.array([label[i] for i in ids])
    subsets = [("full", mod_order)] + \
        [("no_" + m, [x for x in mod_order if x != m]) for m in mod_order] + \
        [("only_" + m, [m]) for m in mod_order]
    rows = []
    for name, kept in subsets:
        preds = run_meta_loo(build(kept), y, len(kept), K,
                             model_names=models, random_state=random_state,
                             verbose=False)
        row = {"subset": name, "n_mods": len(kept)}
        row.update({m: float(macro_auc(y, preds[m])) for m in models})
        rows.append(row)
        print(f"[sens] {name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    return df
