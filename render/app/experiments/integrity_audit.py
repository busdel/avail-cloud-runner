# -*- coding: utf-8 -*-
"""Final integrity audit: recompute every reported AUC from the saved
leave-one-out predictions and verify the benchmark tables match.

For each results directory with loo_predictions.tsv:
  - recompute macro-AUC and balanced accuracy per model
  - compare against meta_benchmark_results.tsv (all subgroup)
  - report OK / MISMATCH

Outputs: experiments/results/integrity_audit.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "experiments/results"

_ROOTS = [BASE / "avail_results", BASE / "experiments/results",
          BASE / "experiments/external_wallen/results", BASE / "uploads"]
RES_DIRS = []
_seen = set()
for _r in _ROOTS:
    if _r.exists():
        for _p in sorted(_r.rglob("loo_predictions.tsv")):
            if str(_p.parent) not in _seen:
                _seen.add(str(_p.parent))
                RES_DIRS.append(_p.parent)


def macro_auc(y, P):
    aucs = []
    for c in range(P.shape[1]):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            aucs.append(roc_auc_score(yc, P[:, c]))
    return float(np.mean(aucs))


def balanced_acc(y, P):
    rec = []
    for c in range(P.shape[1]):
        yc = (y == c)
        if yc.sum() > 0:
            rec.append((P.argmax(axis=1)[yc] == c).mean())
    return float(np.mean(rec))


def main():
    rows = []
    for d in RES_DIRS:
        p = d / "loo_predictions.tsv"
        if not p.exists():
            continue
        loo = pd.read_csv(p, sep="\t")
        y = loo["y"].values
        cols = [c for c in loo.columns if c.startswith("SimpleMean_")]
        if not cols:
            continue
        classes = [c.rsplit("_", 1)[1] for c in cols]
        models = sorted({c.rsplit("_", 1)[0] for c in loo.columns
                         if "_" in c and c.rsplit("_", 1)[1] in classes})
        bench = None
        for bf in ["meta_benchmark_results.tsv", "meta_benchmark_3class_results.tsv"]:
            bp = d / bf
            if bp.exists():
                bench = pd.read_csv(bp, sep="\t")
                break
        for m in models:
            P = loo[[f"{m}_{c}" for c in classes]].values
            ma = macro_auc(y, P)
            ba = balanced_acc(y, P)
            reported = None
            if bench is not None and "all" in bench.subgroup.values:
                sub = bench[(bench.model == m) & (bench.subgroup == "all")]
                if len(sub):
                    reported = float(sub.macro_auc.iloc[0])
            ok = (reported is None) or (abs(ma - reported) < 1e-3)
            rows.append({"dir": d.name, "model": m,
                         "recomputed_macro_auc": round(ma, 6),
                         "reported_macro_auc": (round(reported, 6)
                                                if reported is not None else None),
                         "balanced_acc": round(ba, 6),
                         "match": ok})
        print(d.name, "audited:", len(models), "models", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "integrity_audit.tsv", sep="\t", index=False)
    bad = df[~df.match]
    print(f"total={len(df)}, mismatches={len(bad)}")
    if len(bad):
        print(bad.to_string(index=False))


if __name__ == "__main__":
    main()
