#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demographic fairness sensitivity for the Paper-1 headline models.

Tasks : 3-class staging (loo_predictions_3class.tsv) · dedicated HC-PDD
        binary (rung5_binary_avail.npz).
Strata: Sex (M/F) · Age (median split) · Education (median split).
Stats : per-subgroup macro-AUC / balanced accuracy with bootstrap 95%
        CIs (1000 resamples), parity gaps (max-min), and demographically
        reweighted balanced accuracy over age x sex and edu x sex cells.
Output: experiments/results/fairness/{fairness_results.tsv, SUMMARY.md}
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path("/mnt/limbic/AVAIL_GPU")
OUT = ROOT / "experiments/results/fairness"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)

meta = pd.read_csv(ROOT / "experiments/results/ladder_generic/metadata.tsv",
                   sep="\t")
meta["SampleID"] = meta.SampleID.astype(int)
cov = meta.set_index("SampleID")[["Age", "Sex", "Education"]].astype(float)

def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs)) if aucs else np.nan

def bacc(y, P):
    pred = P.argmax(1)
    rec = [(pred[y == c] == c).mean() for c in np.unique(y) if (y == c).sum()]
    return float(np.mean(rec))

def boot_ci(y, P, fn, n=1000):
    vals = []
    m, K = len(y), P.shape[1]
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(fn(yy, P[idx]))
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def reweighted_bacc(y, P, cells):
    """Balanced accuracy with inverse-frequency weights over cells."""
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    w = np.zeros(len(y))
    for c in np.unique(cells):
        m = cells == c
        w[m] = 1.0 / m.sum()
    w /= w.sum()
    return float((w * correct).sum())

def eval_task(task, ids, y, models, cov):
    ids = np.asarray(ids, dtype=int)
    covs = cov.reindex(ids)
    age_hi = np.nanmedian(covs.Age.values)
    edu_hi = np.nanmedian(covs.Education.values)
    strata = {
        "sex": np.where(covs.Sex.values >= 0.5, "male", "female"),
        "age": np.where(covs.Age.values >= age_hi, f"age>={age_hi:.0f}",
                        f"age<{age_hi:.0f}"),
        "edu": np.where(covs.Education.values >= edu_hi,
                        f"edu>={edu_hi:.0f}", f"edu<{edu_hi:.0f}"),
    }
    rows = []
    for name, P in models.items():
        row = {"task": task, "model": name, "n": len(y)}
        for attr, grp in strata.items():
            aucs = {}
            for g in np.unique(grp):
                m = grp == g
                if len(np.unique(y[m])) < 2 or m.sum() < 10:
                    continue
                a = macro_auc(y[m], P[m])
                lo, hi = boot_ci(y[m], P[m], macro_auc)
                aucs[g] = (a, lo, hi, int(m.sum()))
                rows.append({"task": task, "model": name, "attribute": attr,
                             "subgroup": g, "n_sub": int(m.sum()),
                             "macro_auc": round(a, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)})
            if len(aucs) >= 2:
                vals = [v[0] for v in aucs.values()]
                row[f"gap_{attr}"] = round(max(vals) - min(vals), 4)
        # reweighted accuracy over cross-cells
        for a1, a2, tag in [("Age", "Sex", "age_x_sex"),
                            ("Education", "Sex", "edu_x_sex")]:
            s1 = np.where(covs[a1].values >=
                          np.nanmedian(covs[a1].values), "hi", "lo")
            s2 = np.where(covs.Sex.values >= 0.5, "M", "F")
            cells = np.char.add(np.char.add(s1, "_"), s2)
            row[f"rBACC_{tag}"] = round(reweighted_bacc(y, P, cells), 4)
        row["BACC"] = round(bacc(y, P), 4)
        rows.append(row)
    return rows

all_rows = []
# --- 3-class staging ---
loo = pd.read_csv(ROOT / "avail_results/loo_predictions_3class.tsv", sep="\t")
classes = ["HC", "MCI", "PDD"]
y3 = loo["y"].values.astype(int)
m3 = {}
for model in ["SimpleMean", "WSimpleMean", "ElasticNetLR", "MLP", "MoE",
              "Transformer", "XGBoost"]:
    cols = [f"{model}_{c}" for c in classes]
    if all(c in loo.columns for c in cols):
        m3[model] = loo[cols].values.astype(float)
all_rows += eval_task("multi_3class", loo.patient.values, y3, m3, cov)

# --- dedicated HC-PDD binary ---
d = np.load(ROOT / "experiments/results/roi_ladder/predictions/rung5_binary_avail.npz",
            allow_pickle=True)
yb = d["y"].astype(int)
mb = {k: d[k] for k in d.files if k not in ("y", "ids")}
all_rows += eval_task("binary_HC_PDD", d["ids"], yb, mb, cov)

long_df = pd.DataFrame([r for r in all_rows if "subgroup" in r])
wide_df = pd.DataFrame([r for r in all_rows if "subgroup" not in r])
long_df.to_csv(OUT / "fairness_results.tsv", sep="\t", index=False)
wide_df.to_csv(OUT / "fairness_parity.tsv", sep="\t", index=False)

print("=== SUBGROUP AUC (basılık modeller) ===")
head = long_df[long_df.model.isin(["SimpleMean", "WSimpleMean", "MLP",
                                   "TabICL", "XGBoost"])]
print(head.to_string(index=False))
print("\n=== PARITY / REWEIGHTED (üst modeller) ===")
print(wide_df[wide_df.model.isin(["SimpleMean", "WSimpleMean", "MLP",
                                  "TabICL", "XGBoost"])].to_string(index=False))
