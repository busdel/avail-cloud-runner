#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested validation of the ensemble policy (C x D).

Outer 5-fold repeated 10 times: greedy ensemble weights are selected using
only the inner-fold subjects' cached LOO meta-predictions, then evaluated
on the held-out outer subjects (pooled across folds per repeat).
Baselines under identical splitting: nested best-single-model selection and
the fixed uniform ensemble.

Feature sets compared:
  features_C  : base-ensemble (RF+XGB) meta features, 11-model zoo
  features_RF : standard RF-base features (11-model multi / 7-model binary)

Caveat documented: per-model predictions are LOO (each produced without its
own subject); this is the standard stacking setup and applies equally to
all compared methods. The check isolates the weight-selection bias.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

UP = ROOT / "experiments/results/upgrades"
PRED = ROOT / "experiments/results/roi_ladder/predictions"
OUT = UP
REPS = 10
FOLDS = 5
ROUNDS = 15


def metric(task, y, P):
    return roc_auc_score(y, P[:, 1]) if task == "binary" else macro_auc(y, P)


def greedy_weights(task, y, Ps, idx, rounds=ROUNDS):
    """Caruana greedy with replacement on inner subjects only."""
    models = list(Ps.keys())
    selected, cur_sum, best = [], None, -1
    for r in range(rounds):
        cand, cand_auc = None, -1
        for m in models:
            new_sum = Ps[m][idx] if cur_sum is None else cur_sum + Ps[m][idx]
            a = metric(task, y[idx], new_sum / (len(selected) + 1))
            if a > cand_auc:
                cand, cand_auc = m, a
        selected.append(cand)
        cur_sum = Ps[cand][idx] if cur_sum is None else cur_sum + Ps[cand][idx]
        best = cand_auc
    from collections import Counter
    c = Counter(selected)
    return {m: c.get(m, 0) / rounds for m in models}


def nested(task, z, name):
    y = z["y"]
    models = [k for k in z.files if k not in ("ids", "y")]
    Ps = {m: z[m] for m in models}
    n = len(y)
    rows = []
    for rep in range(REPS):
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=rep)
        oof_greedy = np.zeros((n, Ps[models[0]].shape[1]))
        oof_best = np.zeros_like(oof_greedy)
        oof_unif = np.zeros_like(oof_greedy)
        for tr, te in skf.split(np.zeros(n), y):
            # weight selection on inner subjects
            w = greedy_weights(task, y, Ps, tr)
            Pg = sum(w[m] * Ps[m][te] for m in models)
            oof_greedy[te] = Pg
            # best single by inner metric
            inner_auc = {m: metric(task, y[tr], Ps[m][tr]) for m in models}
            bs = max(inner_auc, key=inner_auc.get)
            oof_best[te] = Ps[bs][te]
            # uniform
            oof_unif[te] = np.mean([Ps[m][te] for m in models], axis=0)
        rows.append({
            "task": task, "features": name, "rep": rep,
            "greedy": metric(task, y, oof_greedy),
            "best_single": metric(task, y, oof_best),
            "uniform": metric(task, y, oof_unif),
        })
    df = pd.DataFrame(rows)
    out_f = OUT / f"nested_ensemble_{task}_{name}.tsv"
    df.to_csv(out_f, sep="\t", index=False)
    print(f"[nested] {task}/{name}: greedy "
          f"{df['greedy'].mean():.3f}±{df['greedy'].std():.3f} | "
          f"best-single {df['best_single'].mean():.3f}±"
          f"{df['best_single'].std():.3f} | "
          f"uniform {df['uniform'].mean():.3f}±{df['uniform'].std():.3f}",
          flush=True)
    return df


def main():
    # C features (base-ensemble)
    for task in ["multi", "binary"]:
        f = UP / f"upgradeF_predictions_{task}.npz"
        if f.exists():
            nested(task, np.load(f), "features_C")
        else:
            print(f"[nested] missing {f.name}")
    # RF features (standard)
    z = np.load(ROOT / "experiments/results/roi_ladder/predictions/"
                "rung5_avail_11model.npz")
    nested("multi", z, "features_RF")
    z = np.load(ROOT / "experiments/results/roi_ladder/predictions/"
                "rung5_binary_avail.npz")
    nested("binary", z, "features_RF")
    print("[nested] DONE", flush=True)


if __name__ == "__main__":
    main()
