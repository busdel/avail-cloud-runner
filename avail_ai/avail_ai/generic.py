# -*- coding: utf-8 -*-
"""AVAIL-AI generic pipeline: run the full meta-benchmark on ANY tabular
multimodal dataset with missing modalities.

Input layout:

    mydata/
      metadata.tsv        # SampleID + Group (+ optional Age, Education, Sex)
      features/           # one .tsv/.csv per modality
        brain.tsv         # rows=samples, first column=SampleID, rest=features
        stool16S.tsv
        ...

Class labels: any names (e.g. HC/MCI/PDD or Case/Control). If more than
3 classes are present, only the 3 most frequent are used. Missing modalities
are handled natively by the availability flags.
"""
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .learners import base_predictions_loo, build_meta_features, macro_auc
from .meta_learners import run_meta_loo, MODEL_NAMES

warnings.filterwarnings("ignore")


def _read_table(path):
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_generic(data_root):
    """Returns (group, conf, mods, feature_names) from a generic layout."""
    data_root = Path(data_root)
    meta_path = data_root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = data_root / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.tsv not found under {data_root}. Required columns: SampleID, Group")
    meta = _read_table(meta_path)
    id_col = "SampleID" if "SampleID" in meta.columns else \
        ("sample" if "sample" in meta.columns else None)
    if id_col is None:
        raise ValueError("metadata must have a SampleID column")
    group_col = "Group" if "Group" in meta.columns else "group"
    if group_col not in meta.columns:
        raise ValueError("metadata must have a Group column")
    meta[id_col] = meta[id_col].astype(str).str.strip()
    meta[group_col] = meta[group_col].astype(str).str.strip()
    group = dict(zip(meta[id_col], meta[group_col]))

    conf = {}
    for col in ["Age", "Education", "Sex"]:
        if col in meta.columns:
            vals = meta[col].astype(str).str.strip()
            if col == "Sex":
                nums = [1.0 if v.upper() in ("M", "MALE", "1") else
                        (0.0 if v.upper() in ("F", "FEMALE", "0") else np.nan)
                        for v in vals]
            else:
                nums = [np.nan if v in ("", "nan", "NA", "Missing_Data") else float(v)
                        for v in vals]
            for sid, v in zip(meta[id_col], nums):
                conf.setdefault(sid, []).append(v)
    for sid in meta[id_col]:
        conf.setdefault(sid, [np.nan] * 3)

    feats_dir = data_root / "features"
    if not feats_dir.exists():
        raise FileNotFoundError("features/ directory not found. "
                                "Put one .tsv/.csv per modality (rows=samples).")
    mods = {}
    names = {}
    for f in sorted(feats_dir.iterdir()):
        if f.suffix.lower() not in (".tsv", ".csv"):
            continue
        df = _read_table(f)
        if df.shape[1] < 2:
            continue
        sample_col = df.columns[0]
        df[sample_col] = df[sample_col].astype(str).str.strip()
        feats = df.columns[1:]
        X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        mname = f.stem
        mods[mname] = {pid: X[i] for i, pid in enumerate(df[sample_col])}
        names[mname] = list(feats)
    if not mods:
        raise ValueError("features/ is empty: add at least one modality file")
    return group, conf, mods, names


def run_generic_pipeline(data_root, out_dir, classes=None, n_estimators=300,
                         random_state=42, verbose=True):
    """Full generic pipeline: LOO base learners + 11 meta-learners + reports.

    Returns a dict with results, predictions, ids, classes, mods.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group, conf_raw, mods, feat_names = load_generic(data_root)
    cnt = Counter(group.values())
    classes = list(classes) if classes else [c for c, _ in cnt.most_common(3)]
    K = len(classes)
    if K < 2:
        raise ValueError(f"need at least 2 classes, found: {dict(cnt)}")
    dropped = {g: c for g, c in cnt.items() if g not in classes}
    if dropped and verbose:
        print(f"[avail] classes={classes} (dropped {dropped})", flush=True)

    ids = sorted([i for i in group if group[i] in classes])
    label = {i: classes.index(group[i]) for i in ids}
    n_mods = len(mods)
    mod_order = sorted(mods.keys())
    if verbose:
        print(f"[avail] n={len(ids)}, K={K}, modalities={mod_order}", flush=True)
        print("[avail] availability: " +
              ", ".join(f"{m}={sum(1 for i in ids if i in mods[m])}"
                        for m in mod_order), flush=True)

    conf_vec = {}
    for i in ids:
        c = conf_raw.get(i, [np.nan] * 3)[:3]
        conf_vec[i] = np.array(c + [np.nan] * (3 - len(c)), dtype=float)
    Cmat = np.vstack([conf_vec[i] for i in ids])
    for j in range(3):
        med = np.nanmedian(Cmat[:, j])
        Cmat[:, j] = np.where(np.isnan(Cmat[:, j]), med, Cmat[:, j])
    conf = {i: Cmat[k] for k, i in enumerate(ids)}

    micro = {m for m in mods
             if all(np.all(np.isfinite(mods[m][i]) & (mods[m][i] >= 0))
                    for i in ids if i in mods[m])}
    if verbose:
        print(f"[avail] compositional (CLR) modalities: {sorted(micro) or '-'}",
              flush=True)

    cache = out_dir / "base_predictions_generic.tsv"
    piv = None
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
        piv = {}
        for _, r in df.iterrows():
            piv.setdefault(str(r["patient"]).strip(), {})[r["modality"]] = \
                [r[f"p_{c}"] for c in range(K)]
    if piv is None:
        if verbose:
            print("[avail] stage 1: leak-free LOO base predictions ...", flush=True)
        piv = base_predictions_loo(mods, ids, label, conf, K, micro,
                                   n_estimators=n_estimators,
                                   random_state=random_state)
        rows = []
        for pid, mm in piv.items():
            for m, pv in mm.items():
                row = {"patient": pid, "modality": m}
                for c in range(K):
                    row[f"p_{c}"] = pv[c]
                rows.append(row)
        pd.DataFrame(rows).to_csv(cache, sep="\t", index=False)

    X, y = build_meta_features(piv, ids, label, n_mods, mod_order, K, 0.0)
    if verbose:
        print(f"[avail] meta-feature matrix: {X.shape}", flush=True)
        print(f"[avail] stage 2: LOO meta-benchmark ({len(MODEL_NAMES)} "
              "meta-learners) ...", flush=True)
    preds = run_meta_loo(X, y, n_mods, K, model_names=list(MODEL_NAMES),
                         random_state=random_state, verbose=verbose)

    rows = []
    complete = np.array([all(i in mods[mo] for mo in mod_order) for i in ids])
    half = max(2, n_mods // 2 + 1)
    ge_half = np.array([sum(i in mods[mo] for mo in mod_order) >= half
                        for i in ids])
    for m in MODEL_NAMES:
        P = preds[m]
        for tag, mask in [("all", np.ones(len(y), bool)),
                          ("complete", complete),
                          (f"ge{half}mod", ge_half)]:
            if mask.sum() >= 5:
                rows.append({"model": m, "subgroup": tag, "n": int(mask.sum()),
                             "macro_auc": macro_auc(y[mask], P[mask])})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "meta_benchmark_results.tsv", sep="\t", index=False)
    res.to_csv(out_dir / "meta_benchmark_3class_results.tsv", sep="\t",
               index=False)
    if verbose:
        print(res.pivot(index="model", columns="subgroup", values="macro_auc")
              .round(3).to_string(), flush=True)

    class_rows = []
    for m in MODEL_NAMES:
        P = preds[m]
        vals = {}
        for c, cls in enumerate(classes):
            yc = (y == c).astype(int)
            vals[cls] = macro_auc(yc[:, None], P[:, c:c + 1]) \
                if len(np.unique(yc)) == 2 else np.nan
        class_rows.append({"model": m, **vals})
    pd.DataFrame(class_rows).to_csv(out_dir / "per_class_auc.tsv",
                                    sep="\t", index=False)

    pred_df = pd.DataFrame({"patient": ids, "y": y})
    for m in MODEL_NAMES:
        for c, cls in enumerate(classes):
            pred_df[f"{m}_{cls}"] = preds[m][:, c]
    pred_df.to_csv(out_dir / "loo_predictions.tsv", sep="\t", index=False)

    try:
        from .biomarkers import run_biomarker_analysis
        if verbose:
            print("[avail] biomarker discovery (SHAP + differential abundance) ...",
                  flush=True)
        run_biomarker_analysis(mods, feat_names, ids, label, classes, out_dir)
    except Exception as e:
        if verbose:
            print(f"[avail] biomarker analysis skipped: "
                  f"{type(e).__name__}: {str(e)[:150]}", flush=True)

    try:
        from .figures_publication import make_all_figures
        if verbose:
            print("[avail] publication figures ...", flush=True)
        make_all_figures(out_dir)
    except Exception as e:
        if verbose:
            print(f"[avail] figures skipped: {type(e).__name__}: {str(e)[:150]}",
                  flush=True)

    return {"results": res, "preds": preds, "X": X, "y": y,
            "ids": ids, "label": label, "mods": mods, "classes": classes}
