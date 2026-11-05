#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Improvement test F: C x D combination.

C: base-learner ensemble (RF + XGB averaged per modality) for the meta
   features.
D: greedy (Caruana-style) weighted meta-ensemble selection on top.

Runs the full 11-model meta zoo on the base-ensemble features (LOO),
saves predictions, then performs greedy ensemble selection and reports
per-model, uniform-ensemble, best-single and greedy-ensemble metrics,
plus a conditional bootstrap CI for the greedy ensemble (weights fixed).
Both tasks.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = None
for cand in [Path("/media/suleyman_disk/AVAIL_GPU"),
             Path("/mnt/limbic/AVAIL_GPU")]:
    if cand.exists():
        ROOT = cand
        break
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DATA = None
for cand in [ROOT / "experiments_vox/results/ladder_generic",
             ROOT / "experiments/results/ladder_generic"]:
    if cand.exists():
        DATA = cand
        break
RES = ROOT / "experiments/results/roi_ladder"
OUT = ROOT / "experiments/results/upgrades"
OUT.mkdir(parents=True, exist_ok=True)
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "MLP", "MoE", "GNN",
          "Transformer", "XGBoost", "TabNet", "TabPFN", "TabICL"]
B = 2000
SEED = 42


def load(tag, m, n):
    f = RES / f"{tag}_{m}_{n}.tsv.gz"
    if not f.exists():
        return None
    t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
    t.index = t.index.astype(str).str.strip()
    return {i: t.loc[i].values.astype(float) for i in t.index}


def metric(task, y, P):
    return roc_auc_score(y, P[:, 1]) if task == "binary" else macro_auc(y, P)


def greedy(task, y, Ps, rounds=15):
    R = {m: metric(task, y, Ps[m]) for m in Ps}
    best_single = max(R.items(), key=lambda kv: kv[1])
    uniform = metric(task, y, np.mean([Ps[m] for m in Ps], axis=0))
    selected, cur_sum, best_auc = [], None, -1
    for r in range(rounds):
        cand_best, cand_auc = None, -1
        for m in Ps:
            new_sum = Ps[m] if cur_sum is None else cur_sum + Ps[m]
            a = metric(task, y, new_sum / (len(selected) + 1))
            if a > cand_auc:
                cand_best, cand_auc = m, a
        selected.append(cand_best)
        cur_sum = Ps[cand_best] if cur_sum is None else cur_sum + Ps[cand_best]
        best_auc = cand_auc
    counts = pd.Series(selected).value_counts()
    weights = {m: c / rounds for m, c in counts.items()}
    rng = np.random.default_rng(SEED)
    boots = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        Pg = sum(w * Ps[m][idx] for m, w in weights.items())
        boots.append(metric(task, yy, Pg))
    arr = np.array(boots)
    return {"best_single": best_single, "uniform": uniform,
            "greedy_auc": best_auc, "weights": weights,
            "greedy_lo95": float(np.percentile(arr, 2.5)),
            "greedy_hi95": float(np.percentile(arr, 97.5))}


def main(task):
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    if task == "binary":
        meta = meta[meta.Group.isin(["HC", "PDD"])]
        classes = ["HC", "PDD"]
    else:
        classes = ["HC", "MCI", "PDD"]
    ids = sorted(meta["SampleID"].tolist())
    label = {r.SampleID: classes.index(r.Group) for r in meta.itertuples()}
    K = len(classes)
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    tag_rf = "stage1bin" if task == "binary" else "stage1_full"
    tag_xgb = f"stage1xgb_{task}"
    piv = {}
    for p in ids:
        d = {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat[m]])
            prf, pxg = load(tag_rf, m, n_), load(tag_xgb, m, n_)
            vals = []
            if prf is not None and p in prf:
                vals.append(prf[p])
            if pxg is not None and p in pxg:
                vals.append(pxg[p])
            if vals:
                d[m] = np.mean(vals, axis=0)
        piv[p] = d
    X, y = build_meta_features(piv, ids, label, len(MODULES), MODULES, K)
    print(f"[upgF] {task}: meta LOO with {len(MODELS)} models...", flush=True)
    P = run_meta_loo(X, y, len(MODULES), K, model_names=MODELS, device=None,
                     random_state=42, n_seeds=5, augment_flag=True,
                     aug_p=0.3, verbose=False)
    np.savez_compressed(OUT / f"upgradeF_predictions_{task}.npz",
                        ids=np.array(ids), y=y,
                        **{m: P[m] for m in MODELS})
    res = greedy(task, y, P)
    rows = [{"task": task, "model": m, "auc": metric(task, y, P[m])}
            for m in MODELS]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"upgradeF_{task}_models.tsv", sep="\t", index=False)
    with open(OUT / f"upgradeF_{task}_summary.txt", "w") as f:
        f.write(f"task={task} n={len(ids)}\n")
        f.write(f"best_single={res['best_single'][0]} "
                f"{res['best_single'][1]:.4f}\n")
        f.write(f"uniform_ensemble={res['uniform']:.4f}\n")
        f.write(f"greedy_ensemble={res['greedy_auc']:.4f} "
                f"[{res['greedy_lo95']:.4f}-{res['greedy_hi95']:.4f}]\n")
        f.write(f"weights={ {k: round(v,3) for k, v in res['weights'].items()} }\n")
    print(df.round(3).to_string(index=False), flush=True)
    print(f"[upgF] {task}: best single {res['best_single'][0]} "
          f"{res['best_single'][1]:.3f} | uniform {res['uniform']:.3f} | "
          f"GREEDY {res['greedy_auc']:.3f} "
          f"[{res['greedy_lo95']:.3f}-{res['greedy_hi95']:.3f}]", flush=True)
    print(f"[upgF] {task}: weights "
          f"{ {k: round(v, 3) for k, v in res['weights'].items()} }",
          flush=True)


if __name__ == "__main__":
    main("multi")
    main("binary")
    print("[upgF] DONE", flush=True)
