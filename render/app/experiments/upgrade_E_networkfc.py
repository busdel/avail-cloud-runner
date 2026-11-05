#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improvement test E: add an fMRI networkFC block to the rung-5 stack.

Builds the 190 network-level FC summaries (19 within + 171 between) for the
83 imaging subjects, trains a fold-internal RF specialist (same protocol),
and reruns the meta stage with six blocks:
  fmri_roi, fmri_networkfc, stool16S, stoolSG, saliva16S, salivaSG.
Compares against the five-block rung 5 (0.690 multi / 0.893 binary).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import preproc, apply_preproc  # noqa: E402
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
BASE_MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
NEW = "fmri_networkfc"
ALL_MODULES = ["fmri_roi", NEW, "stool16S", "stoolSG", "saliva16S",
               "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "TabPFN", "TabICL"]


def build_networkfc_table():
    w = pd.read_csv(BENCH / "yeni_eklenenler/paper_pipeline/integration/"
                    "fc_within_network_summary.csv")
    b = pd.read_csv(BENCH / "yeni_eklenenler/paper_pipeline/integration/"
                    "fc_between_network_summary.csv")
    ref = w.merge(b, on="patient", validate="1:1")
    ref["patient"] = ref["patient"].astype(str).str.strip()
    out_f = DATA / "features" / "fmri_networkfc.tsv"
    ref.to_csv(out_f, sep="\t", index=False)
    print(f"[upgE] networkFC table: {ref.shape}", flush=True)
    return {r[0]: np.array(r[1:], dtype=float)
            for r in ref.itertuples(index=False)}


def stage1_rf(feats, ids, label, K, tag):
    sub = sorted([i for i in ids if i in feats])
    X = np.vstack([feats[i] for i in sub])
    y = np.array([label[i] for i in sub])
    C = np.zeros((len(sub), 3))
    proba = np.zeros((len(sub), K))
    cache_f = RES / f"{tag}_{NEW}_{len(sub)}.tsv.gz"
    if cache_f.exists():
        t = pd.read_csv(cache_f, sep="\t", index_col=0, compression="gzip")
        t.index = t.index.astype(str).str.strip()
        return {i: t.loc[i].values.astype(float) for i in t.index}
    for k, pid in enumerate(sub):
        tr = [t for t in range(len(sub)) if t != k]
        sel, sel2, res, sc, Xt = preproc(X[tr], y[tr], C[tr], False, False)
        Xv = apply_preproc(sel, sel2, res, sc, X[k:k + 1], C[k:k + 1])
        m = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1)
        m.fit(Xt, y[tr])
        pr = m.predict_proba(Xv)[0]
        for ci, cls in enumerate(m.classes_):
            proba[k, int(cls)] = pr[ci]
    pd.DataFrame(proba, index=sub,
                 columns=[f"p{i}" for i in range(K)]).to_csv(
        cache_f, sep="\t", compression="gzip")
    return {pid: proba[k] for k, pid in enumerate(sub)}


def load_cache(tag, m, n):
    f = RES / f"{tag}_{m}_{n}.tsv.gz"
    t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
    t.index = t.index.astype(str).str.strip()
    return {i: t.loc[i].values.astype(float) for i in t.index}


def run_task(task):
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    if task == "binary":
        meta = meta[meta.Group.isin(["HC", "PDD"])]
        classes = ["HC", "PDD"]
    else:
        classes = ["HC", "MCI", "PDD"]
    ids = sorted(meta["SampleID"].tolist())
    label = {r.SampleID: classes.index(r.Group) for r in meta.itertuples()}
    K = len(classes)
    nfc = build_networkfc_table()
    p_nfc = stage1_rf(nfc, ids, label, K,
                      "stage1bin" if task == "binary" else "stage1_full")
    tag = "stage1bin" if task == "binary" else "stage1_full"
    feat_sets = {}
    for m in BASE_MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat_sets[m] = set(t["SampleID"].astype(str))
    piv = {}
    for p in ids:
        d = {}
        for m in BASE_MODULES:
            n_ = len([i for i in ids if i in feat_sets[m]])
            src = load_cache(tag, m, n_)
            if p in src:
                d[m] = src[p]
        if p in p_nfc:
            d[NEW] = p_nfc[p]
        piv[p] = d
    X, y = build_meta_features(piv, ids, label, len(ALL_MODULES),
                               ALL_MODULES, K)
    P = run_meta_loo(X, y, len(ALL_MODULES), K, model_names=MODELS,
                     device=None, random_state=42, n_seeds=5,
                     augment_flag=True, aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        v = (roc_auc_score(y, P[m][:, 1]) if task == "binary"
             else macro_auc(y, P[m]))
        rows.append({"task": task, "model": m, "with_networkfc": v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"upgradeE_{task}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[upgE] {task}: mean {df['with_networkfc'].mean():.3f}", flush=True)


if __name__ == "__main__":
    run_task("multi")
    run_task("binary")
    print("[upgE] DONE", flush=True)
