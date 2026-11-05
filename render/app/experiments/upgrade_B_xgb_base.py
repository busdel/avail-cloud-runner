#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVAIL upgrade test B: base-learner sensitivity (RF -> XGBoost).

Re-runs the stage-1 specialists at rung 5 (full stack) with XGBoost base
learners instead of random forests, using the identical fold-internal
preprocessing, then reruns the meta stage and compares against the
established RF-base results (ladder_summary / binary_ladder_summary).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import preproc, apply_preproc  # noqa: E402
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MICRO = ["stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "TabPFN", "TabICL"]


def stage1_xgb(feats, ids, label, K, tag):
    piv = {}
    for m in MODULES:
        F = feats[m]
        sub = sorted([i for i in ids if i in F.index])
        X = F.loc[sub].values.astype(float)
        y = np.array([label[i] for i in sub])
        C = np.zeros((len(sub), 3))
        is_micro = m in MICRO
        from xgboost import XGBClassifier
        proba = np.zeros((len(sub), K))
        cache_f = RES / f"stage1xgb_{tag}_{m}_{len(sub)}.tsv.gz"
        if cache_f.exists():
            t = pd.read_csv(cache_f, sep="\t", index_col=0,
                            compression="gzip")
            t.index = t.index.astype(str).str.strip()
            for k, pid in enumerate(sub):
                if pid in t.index:
                    proba[k] = t.loc[pid].values.astype(float)
            for k, pid in enumerate(sub):
                piv.setdefault(pid, {})[m] = proba[k]
            print(f"   [{tag}] stage1 {m} (cached)", flush=True)
            continue
        for k, pid in enumerate(sub):
            tr = [t for t in range(len(sub)) if t != k]
            try:
                sel, sel2, res, sc, Xt = preproc(X[tr], y[tr], C[tr],
                                                 is_micro, False)
                Xv = apply_preproc(sel, sel2, res, sc, X[k:k + 1], C[k:k + 1])
            except Exception:
                Xt = X[tr]
                Xv = X[k:k + 1]
            clf = XGBClassifier(
                objective="multi:softprob" if K > 2 else "binary:logistic",
                **({"num_class": K} if K > 2 else {}),
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=0,
                n_jobs=-1, verbosity=0)
            clf.fit(Xt, y[tr])
            pr = clf.predict_proba(Xv)[0]
            for ci, cls in enumerate(clf.classes_):
                proba[k, int(cls)] = pr[ci]
        for k, pid in enumerate(sub):
            piv.setdefault(pid, {})[m] = proba[k]
        pd.DataFrame(proba, index=sub,
                     columns=[f"p{i}" for i in range(K)]).to_csv(
            cache_f, sep="\t", compression="gzip")
        if (MODULES.index(m) + 1) % 2 == 0:
            print(f"   [{tag}] stage1 {m} done", flush=True)
    return piv


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
    feats = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t")
        t["SampleID"] = t["SampleID"].astype(str).str.strip()
        feats[m] = t.set_index("SampleID")
    print(f"[upgB] {task}: stage1 XGB LOO, n={len(ids)}", flush=True)
    piv = stage1_xgb(feats, ids, label, K, task)
    X, y = build_meta_features(piv, ids, label, len(MODULES), MODULES, K)
    P = run_meta_loo(X, y, len(MODULES), K, model_names=MODELS,
                     device=None, random_state=42, n_seeds=5,
                     augment_flag=True, aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        if task == "binary":
            a = roc_auc_score(y, P[m][:, 1])
        else:
            a = macro_auc(y, P[m])
        rows.append({"task": task, "model": m, "macro_or_auc": a})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"upgradeB_{task}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[upgB] {task}: mean {df['macro_or_auc'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main("multi")
    main("binary")
    print("[upgB] DONE", flush=True)
