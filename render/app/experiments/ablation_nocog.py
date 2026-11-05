#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Non-cognitive-variable ablation (analog of UniMIX-AD 5.4.4).

Our training never uses cognitive scores (MMSE, CDRS, ...); the only
education variable (a cognitive-reserve proxy) enters solely through the
optional clinical block of rung 6. This script re-runs rung 6 with the
clinical block restricted to age + sex (education removed) and compares
against rung 5 (no clinical) and rung 6 (age + sex + education).
Also runs a leakage audit: verifies no cognitive variable name appears in
any training feature table or in the training metadata.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost", "MLP",
          "TabPFN", "TabICL"]
COGNITIVE = ["MMSE", "CDRS", "GDS", "Apathy", "NPI_Frequency.Severity",
             "UPDRS", "DS_Forward", "DS_Backward", "ST_False_Answer",
             "ST_Time_Difference", "Categorical_Fluency_", "Phonemic_Fluency",
             "Clock_Drawing_Test", "WMS.VR_Short.Term_Memory",
             "WMS.VR_Long.Term_Memory", "OVMPT_Coding", "OVMPT_Recalling",
             "OVMPT_Recognition", "Benton_Face_Recognition_Test",
             "Benton_Line_Orientation_Test", "HYE", "Disease_Duration"]


def leakage_audit():
    lines = []
    ok = True
    # training metadata
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    bad = [c for c in meta.columns if c in COGNITIVE]
    lines.append(f"metadata.tsv columns: {list(meta.columns)}")
    lines.append(f"cognitive columns present in metadata.tsv: {bad}")
    ok &= not bad
    # feature tables
    for f in sorted((DATA / "features").glob("*.tsv")):
        cols = pd.read_csv(f, sep="\t", nrows=0).columns
        hit = [c for c in cols if any(s.lower() in str(c).lower()
                                      for s in ["mmse", "cdrs", "gds",
                                                "updrs", "fluency",
                                                "benton", "ovmpt"])]
        lines.append(f"{f.name}: {len(cols)} cols; cognitive-like: {hit[:5]}")
        ok &= not hit
    lines.append(f"AUDIT RESULT: {'PASS' if ok else 'FAIL'}")
    (OUT / "cognitive_leakage_audit.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    return ok


def clinical_base(ids, label, features, K):
    X = np.vstack([features[i] for i in ids])
    y = np.array([label[i] for i in ids])
    piv = {}
    for k, pid in enumerate(ids):
        tr = [t for t in range(len(ids)) if t != k]
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1)
        m.fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[k:k + 1]))[0]
        out = np.zeros(K)
        for ci, cls in enumerate(m.classes_):
            out[int(cls)] = p[ci]
        piv[pid] = out
    return piv


def load_cache(tag, m, n):
    t = pd.read_csv(RES / f"{tag}_{m}_{n}.tsv.gz", sep="\t", index_col=0,
                    compression="gzip")
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
    tag = "stage1bin" if task == "binary" else "stage1_full"
    feat_sets = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat_sets[m] = set(t["SampleID"].astype(str))
    cl_age_sex = {r.SampleID: np.array([r.Age, r.Sex], dtype=float)
                  for r in meta.itertuples()}
    piv_clin = clinical_base(ids, label, cl_age_sex, K)
    piv = {}
    for p in ids:
        d = {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat_sets[m]])
            src = load_cache(tag, m, n_)
            if p in src:
                d[m] = src[p]
        d["clinical"] = piv_clin[p]
        piv[p] = d
    mod_order = MODULES + ["clinical"]
    X, y = build_meta_features(piv, ids, label, len(mod_order), mod_order, K)
    P = run_meta_loo(X, y, len(mod_order), K, model_names=MODELS,
                     device=None, random_state=42, n_seeds=5,
                     augment_flag=True, aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        v = (roc_auc_score(y, P[m][:, 1]) if task == "binary"
             else macro_auc(y, P[m]))
        rows.append({"task": task, "model": m,
                     "clinical_age_sex_only": v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"ablation_nocog_{task}.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[nocog] {task}: age+sex-only clinical mean "
          f"{df['clinical_age_sex_only'].mean():.3f}", flush=True)


if __name__ == "__main__":
    leakage_audit()
    run_task("multi")
    run_task("binary")
    print("[nocog] DONE", flush=True)
