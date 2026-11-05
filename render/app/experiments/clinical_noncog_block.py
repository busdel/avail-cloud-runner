#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clinical non-cognitive block experiment (user request).

Adds a 6-feature clinical block [UPDRS, RBD, GDS, Apathy, NPI severity,
Disease_Duration] as an extra modality in the AVAIL stack (no cognitive
scores; leakage-audit compliant). Fold-internal median imputation + RF
specialist (LOO), then the standard meta protocol.

Variants compared (multi and binary):
  rung5                : 5 molecular blocks            (reference)
  rung6_asex_edu       : + age/sex/education           (existing)
  +clin_noncog         : 5 molecular + clinical block
  +clin_noncog+asex_edu: all seven blocks
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
CLIN_COLS = ["UPDRS", "RBD", "GDS", "Apathy", "NPI_Frequency.Severity",
             "Disease_Duration"]


def load_clinical():
    m = pd.read_csv(BENCH / "busra_pd_multisite/metadata.txt", sep="\t")
    m["SampleID"] = m["SampleID"].astype(str).str.strip()
    m = m.set_index("SampleID")
    out = {}
    for sid, row in m.iterrows():
        out[sid] = np.array([pd.to_numeric(row.get(c), errors="coerce")
                            for c in CLIN_COLS], dtype=float)
    return out


def clin_base(ids, label, clin, K):
    X = np.vstack([clin[i] for i in ids])
    y = np.array([label[i] for i in ids])
    piv = {}
    for k, pid in enumerate(ids):
        tr = [t for t in range(len(ids)) if t != k]
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        m = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1)
        m.fit(sc.transform(imp.transform(X[tr])), y[tr])
        p = m.predict_proba(sc.transform(imp.transform(
            X[k:k + 1])))[0]
        o = np.zeros(K)
        for ci, cls in enumerate(m.classes_):
            o[int(cls)] = p[ci]
        piv[pid] = o
    return piv


def load_cache(tag, m, n):
    t = pd.read_csv(RES / f"{tag}_{m}_{n}.tsv.gz", sep="\t", index_col=0,
                    compression="gzip")
    t.index = t.index.astype(str).str.strip()
    return {i: t.loc[i].values.astype(float) for i in t.index}


def run_variant(task, extra_blocks):
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
    piv = {}
    for p in ids:
        d = {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat_sets[m]])
            src = load_cache(tag, m, n_)
            if p in src:
                d[m] = src[p]
        piv[p] = d
    mod_order = list(MODULES)
    if "clin" in extra_blocks:
        clin = load_clinical()
        pc = clin_base(ids, label, clin, K)
        for p in ids:
            piv[p]["clin_noncog"] = pc[p]
        mod_order.append("clin_noncog")
    if "asexedu" in extra_blocks:
        ax = {r.SampleID: np.array([r.Age, r.Sex, r.Education], dtype=float)
              for r in meta.itertuples()}
        pa = clin_base(ids, label, ax, K)
        for p in ids:
            piv[p]["clin_asexedu"] = pa[p]
        mod_order.append("clin_asexedu")
    X, y = build_meta_features(piv, ids, label, len(mod_order), mod_order, K)
    P = run_meta_loo(X, y, len(mod_order), K, model_names=MODELS,
                     device=None, random_state=42, n_seeds=5,
                     augment_flag=True, aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        v = (roc_auc_score(y, P[m][:, 1]) if task == "binary"
             else macro_auc(y, P[m]))
        rows.append({"task": task, "variant": "+".join(extra_blocks) or "none",
                     "model": m, "metric": v})
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for task in ["multi", "binary"]:
        for extra in [[], ["clin"], ["clin", "asexedu"]]:
            df = run_variant(task, extra)
            all_rows.append(df)
            mean = df["metric"].mean()
            print(f"[clin] {task} variant={'+'.join(extra) or 'none'}: "
                  f"mean {mean:.3f}", flush=True)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(OUT / "clinical_noncog_block.tsv", sep="\t", index=False)
    piv = out.pivot_table(index=["task", "variant"], values="metric",
                          aggfunc="mean").round(3)
    print(piv.to_string(), flush=True)
    print("[clin] DONE", flush=True)


if __name__ == "__main__":
    main()
