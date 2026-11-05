#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improvement test D: greedy meta-learner ensemble selection.

Caruana-style greedy forward selection (with replacement, 15 rounds) over
the cached rung-5 LOO predictions, maximizing the ensemble metric; compared
against the uniformly-averaged ensemble and the best single model.
Note: selection is performed on the same LOO predictions (mild optimistic
selection bias; used here as an upper-bound probe).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder/predictions"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)


def metric(task, y, P):
    return roc_auc_score(y, P[:, 1]) if task == "binary" else macro_auc(y, P)


def run(task, npz_file, rounds=15):
    z = np.load(npz_file)
    models = [k for k in z.files if k not in ("ids", "y")]
    y = z["y"]
    Ps = {m: z[m] for m in models}
    R = {m: metric(task, y, Ps[m]) for m in models}
    best_single = max(R.items(), key=lambda kv: kv[1])
    uniform = metric(task, y, np.mean([Ps[m] for m in models], axis=0))
    selected = []
    cur_sum = None
    best_auc = -1
    history = []
    for r in range(rounds):
        cand_best, cand_auc = None, -1
        for m in models:
            new_sum = Ps[m] if cur_sum is None else cur_sum + Ps[m]
            p = new_sum / (len(selected) + 1)
            a = metric(task, y, p)
            if a > cand_auc:
                cand_best, cand_auc = m, a
        selected.append(cand_best)
        cur_sum = Ps[cand_best] if cur_sum is None else cur_sum + Ps[cand_best]
        best_auc = cand_auc
        history.append({"round": r + 1, "added": cand_best,
                        "ensemble_auc": cand_auc})
    df = pd.DataFrame(history)
    df.to_csv(OUT / f"upgradeD_{task}_history.tsv", sep="\t", index=False)
    counts = pd.Series(selected).value_counts().to_dict()
    print(f"[upgD] {task}: best single {best_single[0]} "
          f"{best_single[1]:.3f}; uniform {uniform:.3f}; "
          f"greedy({rounds}) {best_auc:.3f}", flush=True)
    print(f"       composition: {counts}", flush=True)
    return best_single, uniform, best_auc


if __name__ == "__main__":
    run("multi", RES / "rung5_avail_11model.npz")
    run("binary", RES / "rung5_binary_avail.npz")
    print("[upgD] DONE", flush=True)
