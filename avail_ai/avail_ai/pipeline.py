# -*- coding: utf-8 -*-
"""AVAIL-AI pipeline orchestrator: full classification + reporting flow."""
import sys
if sys.version_info < (3, 6):
    raise RuntimeError('AVAIL-AI requires Python 3.6 or newer. Run with `python3`.')
import numpy as np
import pandas as pd
from pathlib import Path
from .data import load_labels, load_confounders, load_modalities
from .learners import (macro_auc, base_predictions_loo, build_meta_features,
                       run_meta_benchmark)

MOD_ORDER = ['fMRI', 'stool16S', 'saliva16S', 'stoolSG', 'salivaSG']
MICRO = {'stool16S', 'saliva16S', 'stoolSG', 'salivaSG'}


def run_pipeline(data_root, out_dir, classes=('HC', 'MCI', 'PDD'),
                 residualize=False, min_prevalence=5, n_estimators=300,
                 meta_models=('SimpleMean', 'WSimpleMean', 'ElasticNetLR'),
                 cache_version='v3_kegg', use_cache=True, random_state=42):
    """Run the core AVAIL-AI pipeline and return a results summary dict.

    Parameters
    ----------
    data_root : path-like
        Root of the data package (contains busra_pd_multisite/, etc.).
    out_dir : path-like
        Where results (TSV) are written.
    classes : tuple
        Ordered class labels -> 0,1,2 (e.g. ('HC','MCI','PDD')).
    residualize : bool
        Whether to residualize confounders in base-learner preprocessing.
    meta_models : tuple
        Which meta-learners to evaluate (SimpleMean / WSimpleMean / ElasticNetLR).
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group = load_labels(data_root)
    conf_raw = load_confounders(data_root)
    mods = load_modalities(data_root, min_prevalence=min_prevalence)

    K = len(classes)
    ids = sorted([i for i in group if group[i] in classes])
    label = {i: classes.index(group[i]) for i in ids}

    # confounders with median imputation
    conf_vec = {}
    for i in ids:
        c = conf_raw.get(i, (np.nan, np.nan, np.nan))
        conf_vec[i] = np.array(c, dtype=float)
    Cmat = np.vstack([conf_vec[i] for i in ids])
    for j in range(3):
        med = np.nanmedian(Cmat[:, j])
        Cmat[:, j] = np.where(np.isnan(Cmat[:, j]), med, Cmat[:, j])
    conf = {i: Cmat[k] for k, i in enumerate(ids)}

    # Stage 1 (cache-aware)
    cache = out_dir / 'base_predictions_{}class_{}.tsv'.format(len(classes), cache_version)
    piv = None
    if use_cache and cache.exists():
        df = pd.read_csv(cache, sep='\t')
        piv = {}
        for _, r in df.iterrows():
            piv.setdefault(str(r['patient']), {})[r['modality']] = [r['p_{}'.format(c)] for c in range(K)]
    if piv is None:
        piv = base_predictions_loo(mods, ids, label, conf, K, MICRO,
                                   residualize=residualize, n_estimators=n_estimators,
                                   random_state=random_state)
        rows = []
        for pid, mm in piv.items():
            for m, pv in mm.items():
                row = {'patient': pid, 'modality': m}
                for c in range(K):
                    row['p_{}'.format(c)] = pv[c]
                rows.append(row)
        pd.DataFrame(rows).to_csv(cache, sep='\t', index=False)

    X, y = build_meta_features(piv, ids, label, len(mods), MOD_ORDER, K, 0.0)
    preds = run_meta_benchmark(X, y, len(mods), K, model_names=meta_models)

    # summary
    rows = []
    complete = np.array([all(i in mods[mo] for mo in MOD_ORDER) for i in ids])
    for m in meta_models:
        P = preds[m]
        for tag, mask in [('all', np.ones(len(y), bool)), ('complete5', complete)]:
            rows.append({'model': m, 'subgroup': tag, 'n': int(mask.sum()),
                         'macro_auc': macro_auc(y[mask], P[mask])})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / 'meta_benchmark_results.tsv', sep='\t', index=False)
    return {'results': res, 'preds': preds, 'X': X, 'y': y,
            'ids': ids, 'label': label, 'mods': mods, 'classes': classes}
