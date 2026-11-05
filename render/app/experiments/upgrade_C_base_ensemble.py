#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improvement test C: base-learner ensemble (RF + XGB) per modality.

Averages the cached RF-base and XGB-base probabilities per modality before
the meta stage, compared against RF-only and XGB-only bases.
Both tasks.
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


def load(tag, m, n):
    f = RES / f"{tag}_{m}_{n}.tsv.gz"
    if not f.exists():
        return None
    t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
    t.index = t.index.astype(str).str.strip()
    return {i: t.loc[i].values.astype(float) for i in t.index}


def main(task):
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
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    tag_rf = "stage1bin" if task == "binary" else "stage1_full"
    tag_xgb = f"stage1xgb_{task}"
    piv = {}
    for p in ids:
        d = {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat[m]])
            prf = load(tag_rf, m, n_)
            pxg = load(tag_xgb, m, n_)
            vals = []
            if prf is not None and p in prf:
                vals.append(prf[p])
            if pxg is not None and p in pxg:
                vals.append(pxg[p])
            if vals:
                d[m] = np.mean(vals, axis=0)
        piv[p] = d
    X, y = build_meta_features(piv, ids, label, len(MODULES), MODULES, K)
    P = run_meta_loo(X, y, len(MODULES), K, model_names=MODELS, device=None,
                     random_state=42, n_seeds=5, augment_flag=True,
                     aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        v = (roc_auc_score(y, P[m][:, 1]) if task == "binary"
             else macro_auc(y, P[m]))
        rows.append({"task": task, "model": m, "base_ensemble": v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"upgradeC_{task}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[upgC] {task}: mean {df['base_ensemble'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main("multi")
    main("binary")
    print("[upgC] DONE", flush=True)
