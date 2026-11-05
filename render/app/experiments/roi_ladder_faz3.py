#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 2b + FAZ 3: ROI expert + two-arm module ladder (complete-case vs AVAIL).

Ladder sequence: fmri_roi -> +stool16S -> +stoolSG -> +saliva16S -> +salivaSG
  Arm "complete": patients with ALL modules in the rung (83->64->55->53->50)
  Arm "avail":    all n=150 labeled patients (availability flags, zero sample
                  loss) -- the AVAIL approach.

Stage 1 (leak-free, per modality):
  - fmri_roi: ROI EXPERT = fold-internal staged selection
      S0 variance top-300 -> S1 ANOVA F + BH-FDR (cap 150, min 20)
      -> S2 LASSO stability (8 bootstrap reps, freq>=0.5, cap 100)
      -> S3 RandomForest(300) on the selected edges
  - microbiome: log(x+1) row-centering or CLR -> RF-importance top-100 -> RF(300)
Stage 2 (meta, LOO): SimpleMean, WSimpleMean, ElasticNetLR, XGBoost, MLP
  (5 seeds), TabPFN, TabICL -- meta features = [base probas | flags].

Outputs (experiments/results/roi_ladder/):
  stage1_*_base.tsv.gz      per-modality base probabilities
  {rung}_{arm}.json         per-model AUCs (macro, per-class, HC-PDD)
  predictions/*.npz         probas for downstream figures
  ladder_summary.tsv        combined table (for the ladder figure + delta table)

Usage:
  python roi_ladder_faz3.py --smoke        # tiny fast validation run
  python roi_ladder_faz3.py                # full run (checkpointed)
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import preproc, apply_preproc  # noqa: E402
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/roi_ladder"
(OUT / "predictions").mkdir(parents=True, exist_ok=True)

MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MICRO = ["stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
N_SEEDS = 5
SEED = 42
CLASSES = ["HC", "MCI", "PDD"]


# --------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------- #
def load_data():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    ids = meta["SampleID"].tolist()
    label = {r.SampleID: CLASSES.index(r.Group) for r in meta.itertuples()}
    conf = {}
    for r in meta.itertuples():
        age = r.Age if pd.notna(r.Age) else np.nan
        edu = r.Education if pd.notna(r.Education) else np.nan
        conf[r.SampleID] = np.array([age, r.Sex, edu], dtype=float)
    mods = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t")
        t["SampleID"] = t["SampleID"].astype(str).str.strip()
        feat = t.set_index("SampleID")
        mods[m] = {i: feat.loc[i].values.astype(float) for i in feat.index}
    return ids, label, conf, mods


# --------------------------------------------------------------------- #
# ROI expert (fold-internal staged selection)
# --------------------------------------------------------------------- #
def _roi_expert(Xtr, ytr, Xte):
    rng = np.random.RandomState(SEED)
    n, p = Xtr.shape
    var = Xtr.var(axis=0)
    top = np.argsort(var)[::-1][:300]
    Xv = Xtr[:, top]
    Xtv = Xte[:, top]
    if Xv.shape[0] >= 3 and len(np.unique(ytr)) >= 2:
        F, pval = f_classif(Xv, ytr)
        rej = multipletests(pval, method="fdr_bh")[0]
        keep = np.where(rej)[0]
        if len(keep) < 20:
            keep = np.argsort(pval)[:20]
        if len(keep) > 150:
            keep = keep[np.argsort(pval[keep])[:150]]
    else:
        keep = np.argsort(var[top])[::-1][:20]
    Xa, Xta = Xv[:, keep], Xtv[:, keep]
    d = Xa.shape[1]
    freq = np.zeros(d)
    for _ in range(8):
        idx = rng.choice(n, size=max(3, int(0.8 * n)), replace=False)
        Xs, ys = Xa[idx], ytr[idx]
        Xs = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-9)
        try:
            lr = LogisticRegression(penalty="l1", C=0.3, solver="saga",
                                    max_iter=3000, class_weight="balanced",
                                    random_state=0)
            lr.fit(Xs, ys)
            freq += (np.abs(lr.coef_).sum(axis=0) > 1e-8).astype(float)
        except Exception:
            freq += 0.0
    freq /= 8.0
    keep2 = np.where(freq >= 0.5)[0]
    if len(keep2) == 0:
        keep2 = np.argsort(freq)[::-1][:min(20, d)]
    if len(keep2) > 100:
        keep2 = keep2[np.argsort(freq[keep2])[::-1][:100]]
    Xs, Xts = Xa[:, keep2], Xta[:, keep2]
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xs, ytr)
    return m.predict_proba(Xts)[0], len(keep2)


def _micro_fold(Xtr, ytr, Ctr, Xte, Cte):
    sel, sel2, res, sc, Xt = preproc(Xtr, ytr, Ctr, True, residualize=False)
    Xtt = apply_preproc(sel, sel2, res, sc, Xte, Cte)
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xt, ytr)
    out = np.zeros(3)
    for ci, cls in enumerate(m.classes_):
        out[int(cls)] = m.predict_proba(Xtt)[0, ci]
    return out


def stage1_base(mods, label, conf, subset_ids, mod_name):
    """LOO base probabilities for one modality over subset_ids."""
    sub = sorted([i for i in subset_ids if i in mods[mod_name]])
    piv = {}
    X = np.vstack([mods[mod_name][i] for i in sub])
    y = np.array([label[i] for i in sub])
    C = np.vstack([conf[i] for i in sub])
    is_micro = mod_name in MICRO
    for k, pid in enumerate(sub):
        tr = [t for t in range(len(sub)) if t != k]
        if is_micro:
            pr = _micro_fold(X[tr], y[tr], C[tr], X[k:k + 1], C[k:k + 1])
        else:
            pr, _ = _roi_expert(X[tr], y[tr], X[k:k + 1])
        piv[pid] = pr
    return piv


# --------------------------------------------------------------------- #
# stage 2 meta
# --------------------------------------------------------------------- #
def stage2_meta(piv, arm_ids, label, n_mods, mod_order):
    X, y = build_meta_features(piv, arm_ids, label, n_mods, mod_order, 3)
    P = run_meta_loo(X, y, n_mods, 3, model_names=MODELS, device=None,
                     random_state=SEED, n_seeds=N_SEEDS,
                     augment_flag=True, aug_p=0.3, verbose=False)
    return X, y, P


def evaluate(y, P, arm_ids, label):
    """macro AUC, per-class AUC, HC-PDD binary AUC (renormalized)."""
    res = {"macro_auc": macro_auc(y, P)}
    for c, cn in enumerate(CLASSES):
        yc = (y == c).astype(int)
        if len(np.unique(yc)) == 2:
            res[f"auc_{cn}"] = float(roc_auc_score(yc, P[:, c]))
        else:
            res[f"auc_{cn}"] = np.nan
    hc = CLASSES.index("HC")
    pdd = CLASSES.index("PDD")
    sel = np.where(np.isin(y, [hc, pdd]))[0]
    if len(sel) > 0:
        yb = (y[sel] == pdd).astype(int)
        Pb = P[sel][:, pdd] / (P[sel][:, [hc, pdd]].sum(axis=1) + 1e-12)
        res["auc_HC_PDD"] = float(roc_auc_score(yb, Pb))
    else:
        res["auc_HC_PDD"] = np.nan
    return res


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
def run(smoke=False, arms=("complete", "avail")):
    ids, label, conf, mods = load_data()
    if smoke:
        rng = np.random.RandomState(0)
        ids = rng.choice(ids, size=24, replace=False).tolist()
        label = {i: label[i] for i in ids}
        conf = {i: conf[i] for i in ids}
    all_set = set(ids)
    avail = {m: sorted(all_set & set(mods[m].keys())) for m in MODULES}

    # needed base-run subsets: (modality, tuple-of-ids)
    needed = set()
    cur = all_set.copy()
    rungs = []
    for m in MODULES:
        cur = cur & set(mods[m].keys())
        rungs.append((m, sorted(cur)))
    for m in MODULES:
        needed.add((m, tuple(avail[m])))
    for k, (_, comp_k) in enumerate(rungs):
        for m in MODULES[:k + 1]:
            needed.add((m, tuple(comp_k)))

    base = {}
    tag = "smoke" if smoke else "full"
    for mod_name, sub in sorted(needed):
        key = (mod_name, sub)
        cache = OUT / f"stage1_{tag}_{mod_name}_{len(sub)}.tsv.gz"
        if cache.exists():
            try:
                t = pd.read_csv(cache, sep="\t", index_col=0, compression="gzip")
                t.index = t.index.astype(str).str.strip()
                piv = {i: t.loc[i].values.astype(float) for i in t.index}
                base[key] = piv
                print(f"[stage1] {mod_name} n={len(sub)} (cached)", flush=True)
                continue
            except Exception:
                pass
        print(f"[stage1] {mod_name} n={len(sub)} starting...", flush=True)
        piv = stage1_base(mods, label, conf, list(sub), mod_name)
        base[key] = piv
        t = pd.DataFrame.from_dict(piv, orient="index",
                                   columns=[f"p_{c}" for c in CLASSES])
        t.to_csv(cache, sep="\t", compression="gzip")
        print(f"[stage1] {mod_name} n={len(sub)} done", flush=True)

    # stage 2 per rung x arm
    summary_rows = []
    for k, (m_new, comp) in enumerate(rungs):
        mod_order = MODULES[:k + 1]
        print(f"[stage2] rung{k+1} modules={mod_order}", flush=True)
        if "complete" in arms:
            arm_ids = comp
            piv_comp = {p: {m: base[(m, tuple(comp))][p] for m in mod_order
                            if p in base[(m, tuple(comp))]}
                        for p in comp}
            X, y, P = stage2_meta(piv_comp, arm_ids, label, len(mod_order),
                                  mod_order)
            for model in MODELS:
                row = evaluate(y, P[model], arm_ids, label)
                row.update({"rung": k + 1, "arm": "complete", "model": model,
                            "n": len(arm_ids),
                            "modules": "+".join(mod_order)})
                summary_rows.append(row)
            np.savez_compressed(OUT / "predictions" /
                                f"rung{k+1}_complete.npz",
                                **{f"{m}": P[m] for m in MODELS},
                                y=y, ids=np.array(arm_ids))
            print(f"   complete n={len(arm_ids)} "
                  f"macro={summary_rows[-len(MODELS)]['macro_auc']:.3f}",
                  flush=True)
        if "avail" in arms:
            arm_ids = sorted(all_set)
            piv_av = {p: {m: base[(m, tuple(avail[m]))][p] for m in mod_order
                          if p in base[(m, tuple(avail[m]))]}
                      for p in arm_ids}
            X, y, P = stage2_meta(piv_av, arm_ids, label, len(mod_order),
                                  mod_order)
            for model in MODELS:
                row = evaluate(y, P[model], arm_ids, label)
                row.update({"rung": k + 1, "arm": "avail", "model": model,
                            "n": len(arm_ids),
                            "modules": "+".join(mod_order)})
                summary_rows.append(row)
            np.savez_compressed(OUT / "predictions" /
                                f"rung{k+1}_avail.npz",
                                **{f"{m}": P[m] for m in MODELS},
                                y=y, ids=np.array(arm_ids))
            print(f"   avail n={len(arm_ids)} "
                  f"macro={summary_rows[-len(MODELS)]['macro_auc']:.3f}",
                  flush=True)
        summ = pd.DataFrame(summary_rows)
        summ.to_csv(OUT / "ladder_summary.tsv", sep="\t", index=False)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(OUT / "ladder_summary.tsv", sep="\t", index=False)
    print("[faz3] DONE. ladder_summary.tsv written", flush=True)
    print(summ.pivot_table(index=["rung", "arm"], columns="model",
                           values="macro_auc").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", nargs="+", default=["complete", "avail"])
    a = ap.parse_args()
    run(smoke=a.smoke, arms=a.arms)
