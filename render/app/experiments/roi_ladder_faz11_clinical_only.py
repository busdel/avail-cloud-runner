#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 11: clinical-only arm (3-class) for incremental-value decomposition.

Question: does adding clinical demographics to the ladder (a) help, and
(b) how much of the combined performance is demographic alone?

Reports clinical-only macro/per-class AUC (7 models, LOO, n=150), to be
compared with:
  rung5 molecular-only AVAIL (0.690 mean) and rung6 +clinical (0.730 mean).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/roi_ladder"
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
CLASSES = ["HC", "MCI", "PDD"]


def main():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    ids = meta["SampleID"].tolist()
    label = {r.SampleID: CLASSES.index(r.Group) for r in meta.itertuples()}
    X3 = {r.SampleID: np.array([r.Age, r.Sex, r.Education], dtype=float)
          for r in meta.itertuples()}
    piv = {}
    for pid in ids:
        tr = [i for i in ids if i != pid]
        Xtr = np.vstack([X3[i] for i in tr])
        ytr = np.array([label[i] for i in tr])
        sc = StandardScaler().fit(Xtr)
        m = RandomForestClassifier(n_estimators=300, random_state=42,
                                   n_jobs=-1)
        m.fit(sc.transform(Xtr), ytr)
        p = m.predict_proba(sc.transform(X3[pid][None, :]))[0]
        out = np.zeros(3)
        for ci, cls in enumerate(m.classes_):
            out[int(cls)] = p[ci]
        piv[pid] = {"clinical": out}
    X, y = build_meta_features(piv, ids, label, 1, ["clinical"], 3)
    P = run_meta_loo(X, y, 1, 3, model_names=MODELS, device=None,
                     random_state=42, n_seeds=5, augment_flag=True,
                     aug_p=0.3, verbose=False)
    rows = []
    for m in MODELS:
        Pm = np.nan_to_num(P[m], nan=1.0 / 3)
        row = {"model": m, "n": len(ids), "macro_auc": macro_auc(y, Pm)}
        for ci, cn in enumerate(CLASSES):
            yc = (y == ci).astype(int)
            row[f"auc_{cn}"] = roc_auc_score(yc, Pm[:, ci])
        hc, pdd = 0, 2
        sel = np.where(np.isin(y, [hc, pdd]))[0]
        yb = (y[sel] == pdd).astype(int)
        Pb = Pm[sel][:, pdd] / (Pm[sel][:, [hc, pdd]].sum(axis=1) + 1e-12)
        row["auc_HC_PDD"] = roc_auc_score(yb, Pb)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "clinical_only_arm.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nclinical-only model-mean macro: {df['macro_auc'].mean():.3f}, "
          f"HC-PDD: {df['auc_HC_PDD'].mean():.3f}")


if __name__ == "__main__":
    main()
