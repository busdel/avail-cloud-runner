# -*- coding: utf-8 -*-
"""Cohort-stratified evaluation: leave-one-Year-out at the meta level.

Uses the zero-loss run's subject-LOO base predictions (leak-free per
subject) and evaluates meta-learners with group (Year) folds to quantify
the generalization gap vs subject-LOO.

Outputs: experiments/results/cohort_stratified_results.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
RES = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_results")


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def main():
    bp = pd.read_csv(RES / "base_predictions_generic.tsv", sep="\t")
    bp["patient"] = bp["patient"].astype(str).str.strip()
    loo = pd.read_csv(RES / "loo_predictions.tsv", sep="\t")
    loo["patient"] = loo["patient"].astype(str).str.strip()

    meta = pd.read_csv(ROOT / "busra_pd_multisite/metadata.txt", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    meta = meta[meta.Group.isin(["HC", "MCI", "PDD"])]
    year = {r.SampleID: str(r.Year) for r in meta.itertuples()}
    ymap = {"HC": 0, "MCI": 1, "PDD": 2}

    ids = loo["patient"].tolist()
    y = np.array([ymap[g] for g in
                  [meta.set_index("SampleID").loc[p, "Group"] for p in ids]])
    K = 3
    mod_order = sorted(bp["modality"].unique())
    per = K + 1
    X = np.zeros((len(ids), len(mod_order) * per))
    piv = {}
    for _, r in bp.iterrows():
        piv.setdefault(r["patient"], {})[r["modality"]] = \
            [r[f"p_{c}"] for c in range(K)]
    for i, p in enumerate(ids):
        for c, m in enumerate(mod_order):
            b = c * per
            if p in piv and m in piv[p]:
                X[i, b:b + K] = piv[p][m]
                X[i, b + K] = 1.0

    years = np.array([year.get(p, "?") for p in ids])
    uniq = sorted(set(years))
    if len(uniq) <= 1:
        note = ("single-cohort dataset: all subjects share Year="
                f"{uniq[0] if uniq else '?'}; leave-one-cohort-out "
                "evaluation is not possible on this data")
        with open(OUT / "cohort_stratified_results.tsv", "w") as f:
            f.write("note\t" + note + "\n")
        print(note)
        return
    rows = []
    for fold_year in sorted(set(years)):
        if fold_year == "?":
            continue
        tr = np.where(years != fold_year)[0]
        te = np.where(years == fold_year)[0]
        if len(te) < 5 or len(tr) < 20:
            continue
        # SimpleMean
        P = np.zeros((len(te), K))
        for k, i in enumerate(te):
            acc = np.zeros(K)
            wsum = 0.0
            for c in range(len(mod_order)):
                b = c * per
                if X[i, b + K] == 1:
                    acc += X[i, b:b + K]
                    wsum += 1.0
            P[k] = acc / wsum if wsum > 0 else np.full(K, 1 / K)
        rows.append({"model": "SimpleMean", "left_out": fold_year,
                     "n_test": len(te), "macro_auc": macro_auc(y[te], P)})

        # XGBoost
        from xgboost import XGBClassifier
        P = np.zeros((len(te), K))
        xgb = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                            random_state=0, n_jobs=6,
                            objective="multi:softprob", num_class=K,
                            eval_metric="mlogloss")
        xgb.fit(X[tr], y[tr])
        P = xgb.predict_proba(X[te])
        rows.append({"model": "XGBoost", "left_out": fold_year,
                     "n_test": len(te), "macro_auc": macro_auc(y[te], P)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cohort_stratified_results.tsv", sep="\t", index=False)
    print(df.round(3).to_string(index=False))
    print("\nsubject-LOO reference (from zero-loss results):")
    ref = pd.read_csv(RES / "meta_benchmark_results.tsv", sep="\t")
    print(ref[ref.subgroup == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
