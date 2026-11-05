# -*- coding: utf-8 -*-
"""Preprocessing sensitivity: delta-CLR vs log(x+1) centering.

Documents the preprocessing choice for relative-abundance tables. Compares
the LOO base-learner AUC of the stool16S layer (counts) and the shotgun MGS
layer (relative abundances) under both transforms.

Outputs: experiments/results/preprocessing_sensitivity.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))
OUT = BASE / "experiments/results"

from avail_ai.preprocess import CLRTransformer, LogCenterTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def loo_auc(X, y, transform):
    P = np.zeros(len(y))
    for k in range(len(y)):
        tr = [t for t in range(len(y)) if t != k]
        Xtr = transform.transform(X[tr])
        Xte = transform.transform(X[k:k + 1])
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=6)
        rf.fit(Xtr, y[tr])
        P[k] = rf.predict_proba(Xte)[0][1]
    return roc_auc_score(y, P)


def main():
    meta = pd.read_csv(BASE / "AVAIL_benchmark/busra_pd_multisite/metadata.txt",
                       sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    lab = dict(zip(meta.SampleID, meta.Group))

    rows = []
    # counts layer (16S)
    g = pd.read_csv(BASE / "AVAIL_benchmark/busra_pd_multisite/stool_genus_p.txt",
                    sep="\t", index_col=0)
    g.columns = [str(c).strip() for c in g.columns]
    g = g.loc[(g > 0).sum(axis=1) >= 5].T
    ids = [p for p in g.index if lab.get(p) in ("HC", "PDD")]
    y = np.array([1 if lab[p] == "PDD" else 0 for p in ids])
    X = g.loc[ids].values.astype(float)
    rows.append({"layer": "stool16S_counts", "transform": "delta_CLR",
                 "auc": loo_auc(X, y, CLRTransformer())})
    rows.append({"layer": "stool16S_counts", "transform": "log_center",
                 "auc": loo_auc(X, y, LogCenterTransformer())})

    # relative-abundance layer (shotgun MGS)
    g2 = pd.read_csv(BASE / "AVAIL_benchmark/yeni_eklenenler/paper_pipeline/out/"
                     "gut_mgs_abundance_freq.tsv", sep="\t", index_col=0)
    cw = pd.read_csv(BASE / "AVAIL_benchmark/yeni_eklenenler/paper_pipeline/"
                     "DELIVERABLE/raw_outputs/err_patient_crosswalk.tsv", sep="\t")
    cw["patient"] = cw["patient"].astype(str).str.strip()
    cmap = dict(zip(cw["run_accession"].astype(str), cw["patient"]))
    g2.columns = [cmap.get(c, c) for c in g2.columns]
    g2 = g2.loc[(g2 > 0).sum(axis=1) >= 5].T
    ids2 = [p for p in g2.index if lab.get(p) in ("HC", "PDD")]
    y2 = np.array([1 if lab[p] == "PDD" else 0 for p in ids2])
    X2 = g2.loc[ids2].values.astype(float)
    rows.append({"layer": "shotgun_MGS_relab", "transform": "delta_CLR",
                 "auc": loo_auc(X2, y2, CLRTransformer())})
    rows.append({"layer": "shotgun_MGS_relab", "transform": "log_center",
                 "auc": loo_auc(X2, y2, LogCenterTransformer())})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "preprocessing_sensitivity.tsv", sep="\t", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
