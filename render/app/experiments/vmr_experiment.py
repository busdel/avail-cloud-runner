# -*- coding: utf-8 -*-
"""VMR experiment: p+flag vs p+flag+VMR on the n=150 zero-loss cohort.

Same 11 meta-learners, same LOO protocol, two meta-feature layouts.
Reports overall macro-AUC and per-availability-subgroup (complete /
ge4mod / le2mod) plus the cross-modal reconstruction quality (R2).

Outputs: experiments/results/vmr_results.tsv, vmr_reconstruction.tsv
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

from avail_ai.vmr import (build_piv, virtual_reconstruction,  # noqa: E402
                          virtual_reconstruction_knn, build_meta_features)
from avail_ai.learners import macro_auc  # noqa: E402

K = 3


def meta_loo_ext(X, y, n_mods, K, per, models):
    """LOO meta-benchmark over a custom per-modality block size."""
    from avail_ai.meta_learners import _tabular_predict, run_meta_loo
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    n = len(y)
    preds = {m: np.zeros((n, K)) for m in models}
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr, Xte = X[tr], y[tr], X[fold:fold + 1]
        # SimpleMean / WSimpleMean on the p-flag part only
        acc = np.zeros(K)
        wsum = 0.0
        for c in range(n_mods):
            b = c * per
            if Xte[0, b + K] == 1:
                acc += Xte[0, b:b + K]
                wsum += 1.0
        preds["SimpleMean"][fold] = acc / wsum if wsum > 0 else np.full(K, 1 / K)
        preds["WSimpleMean"][fold] = preds["SimpleMean"][fold]
        try:
            lr = LogisticRegression(penalty="elasticnet", solver="saga",
                                    l1_ratio=0.5, C=0.3, max_iter=5000,
                                    class_weight="balanced", random_state=0)
            lr.fit(Xtr, ytr)
            preds["ElasticNetLR"][fold] = lr.predict_proba(Xte)[0]
        except Exception:
            preds["ElasticNetLR"][fold] = preds["SimpleMean"][fold]
        try:
            pr = _tabular_predict("XGBoost", Xtr, ytr, Xte, K)
            preds["XGBoost"][fold] = pr
        except Exception:
            preds["XGBoost"][fold] = preds["SimpleMean"][fold]
        if (fold + 1) % 50 == 0:
            print(f"    fold {fold + 1}/{n}", flush=True)
    return preds


def main():
    bp_path = BASE / "experiments/results/zero_loss_results/base_predictions_generic.tsv"
    loo = pd.read_csv(BASE / "experiments/results/zero_loss_results/loo_predictions.tsv",
                      sep="\t")
    ids = loo["patient"].astype(str).tolist()
    y = loo["y"].values
    piv = build_piv(bp_path, K)
    mod_order = sorted({m for _, mm in piv.items() for m in mm})
    n_mods = len(mod_order)
    print(f"n={len(ids)}, modalities={mod_order}")

    models = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]

    # --- baseline: p+flag
    X0 = build_meta_features(piv, ids, mod_order, K, virtual=None)
    print("baseline p+flag ...")
    preds0 = meta_loo_ext(X0, y, n_mods, K, K + 1, models)

    # --- VMR
    virtual, recon_r2 = virtual_reconstruction(piv, ids, mod_order, K)
    pd.DataFrame([{"modality": m, "loo_recon_r2": v}
                  for m, v in recon_r2.items()]).to_csv(
        OUT / "vmr_reconstruction.tsv", sep="\t", index=False)
    print("reconstruction R2:", {m: round(v, 3) for m, v in recon_r2.items()})
    n_virt = sum(1 for p, mm in virtual.items() for _ in mm)
    print(f"virtual reconstructions: {n_virt}")
    X1 = build_meta_features(piv, ids, mod_order, K, virtual=virtual)
    print("VMR-extended ...")
    preds1 = meta_loo_ext(X1, y, n_mods, K, 2 * (K + 1), models)

    # kNN variant
    virtual_k = virtual_reconstruction_knn(piv, ids, mod_order, K)
    X2 = build_meta_features(piv, ids, mod_order, K, virtual=virtual_k)
    print("VMR-kNN ...")
    preds2 = meta_loo_ext(X2, y, n_mods, K, 2 * (K + 1), models)

    # --- report
    counts = pd.Series([sum(1 for m in mod_order if p in piv and m in piv[p])
                        for p in ids])
    rows = []
    for tag, P in [("p+flag", preds0), ("p+flag+VMR", preds1),
                   ("p+flag+VMR-kNN", preds2)]:
        for m in models:
            Pv = P[m]
            rows.append({"layout": tag, "model": m, "subgroup": "all",
                         "n": len(ids), "macro_auc": macro_auc(y, Pv)})
            for sub, mask in [
                    ("complete", (counts == n_mods).values),
                    ("ge4mod", (counts >= 4).values),
                    ("le2mod", (counts <= 2).values)]:
                if mask.sum() >= 5:
                    rows.append({"layout": tag, "model": m, "subgroup": sub,
                                 "n": int(mask.sum()),
                                 "macro_auc": macro_auc(y[mask], Pv[mask])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "vmr_results.tsv", sep="\t", index=False)
    piv_t = df.pivot_table(index=["layout", "model"], columns="subgroup",
                           values="macro_auc").round(3)
    print(piv_t.to_string())
    print("VMR experiment done")


if __name__ == "__main__":
    main()
