#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVAIL upgrade test A: reliability-aware meta features.

Standard representation per modality: [p0..pK-1, flag].
Extended representation appends, per modality, [entropy, max-confidence]
plus the number of available modalities at the end of the vector.
SimpleMean/WSimpleMean are unaffected (they read the standard layout);
the learning meta-models can exploit the extra reliability cues.

Compares rung-5 (full stack) AUX with and without the extra features,
both tasks (multi: 3-class macro-AUC; binary: HC-PDD AUC).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "TabPFN", "TabICL"]
LEARNING = ["ElasticNetLR", "XGBoost", "MLP", "TabPFN", "TabICL"]


def load_cache(tag, mod, n):
    f = RES / f"{tag}_{mod}_{n}.tsv.gz"
    t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
    t.index = t.index.astype(str).str.strip()
    return {i: t.loc[i].values.astype(float) for i in t.index}


def build(task):
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    if task == "binary":
        meta = meta[meta.Group.isin(["HC", "PDD"])]
        classes = ["HC", "PDD"]
    else:
        classes = ["HC", "MCI", "PDD"]
    ids = sorted(meta["SampleID"].tolist())
    label = {r.SampleID: classes.index(r.Group) for r in meta.itertuples()}
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    tag = "stage1bin" if task == "binary" else "stage1_full"
    piv = {}
    extra = {}
    K = len(classes)
    for p in ids:
        d, ex = {}, {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat[m]])
            src = load_cache(tag, m, n_)
            if p in src:
                pr = src[p]
                d[m] = pr
                ent = -np.sum(pr * np.log(pr + 1e-12))
                ex[m] = np.array([ent, pr.max()])
            else:
                ex[m] = np.zeros(2)
        piv[p] = d
        extra[p] = (ex, len(d))
    X, y = build_meta_features(piv, ids, label, len(MODULES), MODULES, K)
    Xe = [X]
    ext_cols = []
    for m in MODULES:
        ext_cols.append(np.array([extra[p][0][m] for p in ids]))
    ext_cols.append(np.array([[extra[p][1]] for p in ids]))
    Xext = np.hstack([X] + ext_cols)
    return X, Xext, y, ids, K


def evaluate(task):
    X, Xext, y, ids, K = build(task)
    out = {}
    for name, XX in [("standard", X), ("extended", Xext)]:
        P = run_meta_loo(XX, y, len(MODULES), K, model_names=MODELS,
                         device=None, random_state=42, n_seeds=5,
                         augment_flag=True, aug_p=0.3, verbose=False)
        out[name] = P
    rows = []
    for m in LEARNING:
        if task == "binary":
            a0 = roc_auc_score(y, out["standard"][m][:, 1])
            a1 = roc_auc_score(y, out["extended"][m][:, 1])
        else:
            a0 = macro_auc(y, out["standard"][m])
            a1 = macro_auc(y, out["extended"][m])
        rows.append({"task": task, "model": m, "standard": a0,
                     "extended": a1, "delta": a1 - a0})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"upgradeA_{task}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[upgA] {task}: mean standard "
          f"{df['standard'].mean():.3f} -> extended "
          f"{df['extended'].mean():.3f}", flush=True)


if __name__ == "__main__":
    evaluate("multi")
    evaluate("binary")
    print("[upgA] DONE", flush=True)
