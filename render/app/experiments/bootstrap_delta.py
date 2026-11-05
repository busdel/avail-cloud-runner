#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired bootstrap for AVAIL vs competitor architectures.

For each task (multi: 3-class macro-AUC; binary: HC-PDD AUC), resample
subjects with replacement (B=2000), recompute AVAIL (model-averaged) and
competitor metrics on the same resampled subjects, and report the delta
distribution (95% CI, fraction <= 0).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
COMP = ROOT / "experiments/results/competitors"
OUT = ROOT / "revizyon/makaleler/supplement"
OUT.mkdir(parents=True, exist_ok=True)
B = 2000
SEED = 42


def metric_multi(y, P):
    return macro_auc(y, P)


def metric_binary(y, P):
    return roc_auc_score(y, P[:, 1])


def paired(avail_P, comp_P, y, metric, rng):
    n = len(y)
    base = metric(y, avail_P) - metric(y, comp_P)
    deltas = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        try:
            d = metric(yy, avail_P[idx]) - metric(yy, comp_P[idx])
            deltas.append(d)
        except Exception:
            continue
    arr = np.array(deltas)
    return base, np.percentile(arr, 2.5), np.percentile(arr, 97.5), \
        float((arr <= 0).mean()), len(arr)


def align(npz_a, npz_b):
    """Return (id order common, index arrays)."""
    ia = np.array([str(x).strip() for x in npz_a["ids"]])
    ib = np.array([str(x).strip() for x in npz_b["ids"]])
    common = [i for i in ia if i in set(ib)]
    idx_a = {i: k for k, i in enumerate(ia)}
    idx_b = {i: k for k, i in enumerate(ib)}
    return common, np.array([idx_a[i] for i in common]), \
        np.array([idx_b[i] for i in common])


def run_task(task, avail_file, comp_files, metric):
    if not avail_file.exists():
        print(f"[delta] {task}: missing {avail_file.name}; skipped")
        return
    za = np.load(avail_file)
    models = [k for k in za.files if k not in ("ids", "y")]
    Pa = np.mean([za[m] for m in models], axis=0)
    ya = za["y"]
    rows = []
    for cf in comp_files:
        if not cf.exists():
            print(f"[delta] {task}: missing {cf.name}; skipped")
            continue
        zb = np.load(cf)
        common, ia, ib = align(za, zb)
        if len(common) < 20:
            print(f"[delta] {task}: too few common ids for {cf.name}")
            continue
        P_a, P_b, y = Pa[ia], zb["probs"][ib] if "probs" in zb.files else None, ya[ia]
        if P_b is None:
            # competitor npz: each method is a separate key
            for m in [k for k in zb.files if k not in ("ids", "y")]:
                rng = np.random.default_rng(SEED)
                base, lo, hi, p, nb = paired(P_a, zb[m][ib], y, metric, rng)
                rows.append({"task": task, "competitor": m,
                             "avail_metric": metric(y, P_a),
                             "comp_metric": metric(y, zb[m][ib]),
                             "delta": base, "lo95": lo, "hi95": hi,
                             "p_le0": p, "n_boot": nb, "n": len(common)})
        else:
            rng = np.random.default_rng(SEED)
            base, lo, hi, p, nb = paired(P_a, P_b, y, metric, rng)
            rows.append({"task": task, "competitor": cf.stem,
                         "avail_metric": metric(y, P_a),
                         "comp_metric": metric(y, P_b),
                         "delta": base, "lo95": lo, "hi95": hi,
                         "p_le0": p, "n_boot": nb, "n": len(common)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / f"TableS5_delta_ci_{task}.tsv", sep="\t", index=False)
        print(df.round(3).to_string(index=False))


def main():
    run_task("multi", RES / "predictions/rung5_avail_11model.npz",
             [COMP / "competitors_loo_predictions.npz",
              COMP / "competitors2_loo_predictions_multi.npz"],
             metric_multi)
    run_task("binary", RES / "predictions/rung5_binary_avail.npz",
             [COMP / "competitors_loo_predictions_binary.npz",
              COMP / "competitors2_loo_predictions_binary.npz"],
             metric_binary)


if __name__ == "__main__":
    main()
