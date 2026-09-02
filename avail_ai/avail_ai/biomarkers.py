# -*- coding: utf-8 -*-
"""Biomarker discovery for the generic pipeline (works on ANY omics layers).

Per modality:
  - SHAP importance of an XGBoost classifier (top-K features)
  - Differential abundance: Mann-Whitney per class pair with BH-FDR

Writes (into out_dir):
  biomarkers_top20_per_modality.tsv
  diff_abundance_<modality>.tsv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


def _shap_top(mod_name, X, y, names, patients, K, top_k=20):
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    import shap

    if K == 2:
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            random_state=0, n_jobs=-1, eval_metric="logloss")
    else:
        clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                            random_state=0, n_jobs=-1,
                            objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
    try:
        skf = StratifiedKFold(n_splits=min(5, max(2, len(y) // 10)), shuffle=True,
                              random_state=0)
        shaps = np.zeros(X.shape[1])
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            ex = shap.TreeExplainer(clf)
            shaps += np.abs(ex.shap_values(X[te])).mean(axis=(0, 2)) \
                if K > 2 else np.abs(ex.shap_values(X[te])).mean(axis=0)
        imp = shaps / max(1, skf.get_n_splits())
    except Exception:
        clf.fit(X, y)
        try:
            ex = shap.TreeExplainer(clf)
            vals = ex.shap_values(X)
            imp = np.abs(vals).mean(axis=(0, 2)) if K > 2 else np.abs(vals).mean(axis=0)
        except Exception:
            imp = np.abs(np.asarray(clf.feature_importances_))
    idx = np.argsort(imp)[::-1][:top_k]
    rows = []
    for i in idx:
        rows.append({"modality": mod_name, "feature": names[i],
                     "importance": float(imp[i])})
    return pd.DataFrame(rows)


def _diff_abundance(mod_name, X, y, names, classes, max_features=300):
    X = np.asarray(X, dtype=float)
    rows = []
    for j, feat in enumerate(names[:max_features]):
        vals = X[:, j]
        for a in range(len(classes)):
            for b in range(a + 1, len(classes)):
                va = vals[y == a]
                vb = vals[y == b]
                if len(va) < 5 or len(vb) < 5:
                    continue
                try:
                    stat, p = mannwhitneyu(va, vb, alternative="two-sided")
                except Exception:
                    continue
                rows.append({"modality": mod_name, "feature": feat,
                             "comparison": f"{classes[a]}_vs_{classes[b]}",
                             "U": float(stat), "p_value": float(p),
                             "log2fc": float(np.log2(
                                 np.median(vb + 1e-12) / np.median(va + 1e-12)))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = multipletests(df["p_value"], method="fdr_bh")[1]
    return df.sort_values("p_value")


def run_biomarker_analysis(mods, names, ids, label, classes, out_dir,
                           top_k=20, max_features=300):
    """Biomarker discovery across modalities. Returns (top_df, diff_dfs)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)

    top_parts = []
    diff_parts = []
    for mname in sorted(mods):
        sub = sorted([i for i in ids if i in mods[mname]])
        if len(sub) < 10:
            continue
        X = np.vstack([mods[mname][i] for i in sub])
        y = np.array([label[i] for i in sub])
        feat_names = list(names.get(mname, [f"f{j}" for j in range(X.shape[1])]))
        feat_names = feat_names[:X.shape[1]]
        feat_names += [f"f{j}" for j in range(len(feat_names), X.shape[1])]

        try:
            t = _shap_top(mname, X, y, feat_names, sub, K, top_k=top_k)
            top_parts.append(t)
            t.to_csv(out_dir / f"biomarkers_{mname}.tsv", sep="\t", index=False)
        except Exception as e:
            print(f"[biomarkers] SHAP skipped for {mname}: {type(e).__name__}", flush=True)

        try:
            d = _diff_abundance(mname, X, y, feat_names, classes,
                                max_features=max_features)
            if not d.empty:
                diff_parts.append(d)
                d.to_csv(out_dir / f"diff_abundance_{mname}.tsv",
                         sep="\t", index=False)
        except Exception as e:
            print(f"[biomarkers] diff-abundance skipped for {mname}: "
                  f"{type(e).__name__}", flush=True)

    if top_parts:
        top = pd.concat(top_parts, ignore_index=True)
        top.to_csv(out_dir / "biomarkers_top20_per_modality.tsv",
                   sep="\t", index=False)
    else:
        top = pd.DataFrame()
    if diff_parts:
        diff = pd.concat(diff_parts, ignore_index=True)
        diff.to_csv(out_dir / "diff_abundance_all.tsv", sep="\t", index=False)
    else:
        diff = pd.DataFrame()
    return top, diff
