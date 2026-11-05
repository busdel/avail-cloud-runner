# -*- coding: utf-8 -*-
"""Calibrated availability-aware fusion: does explicitly calibrating each
base learner (isotonic, inner-CV within the LOO fold) improve the fusion?

Compares meta-learner AUC with raw RF probabilities vs isotonic-calibrated
probabilities on the n=150 zero-loss cohort (fast meta models).

Outputs: experiments/results/calibration_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))
import argparse as _ap
__ap = _ap.ArgumentParser()
__ap.add_argument("--data-root",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
__ap.add_argument("--out",
                  default="/media/suleyman_disk/AVAIL_GPU/experiments/results")
__a2, _ = __ap.parse_known_args()
_DATA = Path(__a2.data_root)
OUT = Path(__a2.out)

from avail_ai.learners import macro_auc  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def load_layers():
    meta = pd.read_csv(_DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    classes = ["HC", "MCI", "PDD"]
    m = meta[meta.Group.isin(classes)]
    ids = m["SampleID"].tolist()
    y = np.array([classes.index(g) for g in m["Group"]])
    layers = {}
    for f in sorted((_DATA / "features").iterdir()):
        d = pd.read_csv(f, sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        d = d[d.SampleID.isin(ids)].set_index("SampleID").reindex(ids)
        layers[f.stem] = d
    return ids, y, layers


def stage1_loo_cal(mods, ids, y, K, calibrate, n_est=200):
    piv = {}
    for mname, Xdf in mods.items():
        sub = Xdf.dropna()
        sids = list(sub.index)
        if len(sids) < 10:
            continue
        X = sub.values.astype(float)
        yy = np.array([y[ids.index(p)] for p in sids])
        P = np.zeros((len(sids), K))
        for k in range(len(sids)):
            tr = [t for t in range(len(sids)) if t != k]
            rf = RandomForestClassifier(n_estimators=n_est, random_state=42,
                                        n_jobs=6)
            if calibrate:
                clf = CalibratedClassifierCV(
                    rf, method="isotonic", cv=5, n_jobs=1)
            else:
                clf = rf
            clf.fit(X[tr], yy[tr])
            P[k] = clf.predict_proba(X[k:k + 1])[0]
        for p, pv in zip(sids, P):
            piv.setdefault(p, {})[mname] = pv
    return piv


def meta_loo(X, y, K, per):
    n = len(y)
    preds = {"SimpleMean": np.zeros((n, K)),
             "WSimpleMean": np.zeros((n, K)),
             "ElasticNetLR": np.zeros((n, K)),
             "XGBoost": np.zeros((n, K))}
    nmod = X.shape[1] // per
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr, Xte = X[tr], y[tr], X[fold:fold + 1]
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(nmod):
            b = c * per
            if Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
        sm = acc / wsum if wsum > 0 else np.full(K, 1 / K)
        preds["SimpleMean"][fold] = sm
        preds["WSimpleMean"][fold] = sm
        try:
            lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                    l1_ratio=0.5, C=0.3, max_iter=3000,
                                    class_weight="balanced", random_state=0)
            lr.fit(Xtr, ytr)
            preds["ElasticNetLR"][fold] = lr.predict_proba(Xte)[0]
        except Exception:
            preds["ElasticNetLR"][fold] = sm
        xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                            random_state=0, n_jobs=6,
                            objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
        xgb.fit(Xtr, ytr)
        preds["XGBoost"][fold] = xgb.predict_proba(Xte)[0]
    return preds


def main():
    ids, y, layers = load_layers()
    mod_order = sorted(layers)
    K = 3
    per = K + 1
    rows = []
    for tag, calibrate in [("raw", False), ("isotonic", True)]:
        piv = stage1_loo_cal(layers, ids, y, K, calibrate)
        X = np.zeros((len(ids), len(mod_order) * per))
        for r, p in enumerate(ids):
            for c, m in enumerate(mod_order):
                b = c * per
                if p in piv and m in piv[p]:
                    X[r, b:b + K] = piv[p][m]
                    X[r, b + K] = 1.0
        preds = meta_loo(X, y, K, per)
        for m, P in preds.items():
            rows.append({"calibration": tag, "model": m,
                         "macro_auc": macro_auc(y, P)})
        print(tag, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_results.tsv", sep="\t", index=False)
    print(df.pivot_table(index="model", columns="calibration",
                         values="macro_auc").round(3).to_string())
    print("calibration experiment done")


if __name__ == "__main__":
    main()
