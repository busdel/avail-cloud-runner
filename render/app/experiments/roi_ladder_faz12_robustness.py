#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 12: robustness suite for the new-method (ladder) paper.

A. Randomization stability: 10 seeds x rungs {3,5} x arms, 4 fast models.
B. Protocol robustness: repeated stratified 5x5 CV vs LOO at rung 5.
C. Availability-subgroup performance at rung 5 (all-5 / 3-4 / <=2).
D. Per-class bootstrap CIs at rung 5 (2000 subject resamples).
E. Calibration at rung 5: multiclass Brier + PDD reliability curve.
F. Mask-value ablation: missing probabilities as 0 vs 1/K.

Outputs: experiments/results/roi_ladder/robustness/
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
OUT = RES / "robustness"
OUT.mkdir(parents=True, exist_ok=True)
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
FAST = ["SimpleMean", "ElasticNetLR", "XGBoost", "MLP"]
SEED0 = 42
K = 3


def load_meta(arm):
    base = {}
    for f in sorted(RES.glob("stage1_full_*.tsv.gz")):
        parts = f.name.split(".")[0].split("_")
        name = "_".join(parts[2:-1])
        n = int(parts[-1])
        t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
        t.index = t.index.astype(str).str.strip()
        base[(name, n)] = {i: t.loc[i].values.astype(float) for i in t.index}
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    comp = set(meta["SampleID"])
    for m in MODULES:
        comp &= feat[m]
    comp = sorted(comp)
    ids = comp if arm == "complete" else sorted(meta["SampleID"])
    label = {r.SampleID: ["HC", "MCI", "PDD"].index(r.Group)
             for r in meta.itertuples()}
    piv = {}
    flags = {}
    for p in ids:
        d = {}
        fl = []
        for c, m in enumerate(MODULES):
            src = base[(m, len([i for i in ids if i in feat[m]]))]
            if p in src:
                d[m] = src[p]
                fl.append(1)
            else:
                fl.append(0)
        piv[p] = d
        flags[p] = fl
    return piv, label, ids, flags


def meta_x(piv, ids, label, mod_order, mask=0.0):
    X, y = build_meta_features(piv, ids, label, len(mod_order), mod_order, K,
                               masked_value=mask)
    return X, y


def loo_fast(X, y, seed, n_seeds=2, models=FAST, mods=None):
    all_models = list(dict.fromkeys(["SimpleMean", "WSimpleMean"]
                                    + list(models)))
    P = run_meta_loo(X, y, mods or (X.shape[1] // (K + 1)), K,
                     model_names=all_models, device=None, random_state=seed,
                     n_seeds=n_seeds, augment_flag=True, aug_p=0.3,
                     verbose=False)
    return {m: macro_auc(y, P[m]) for m in models}


def robust_xgb_tr(Xtr, ytr, Xte):
    return None  # placeholder (unused)


def _fit_predict(model, Xtr, ytr, Xte, seed):
    from sklearn.linear_model import LogisticRegression
    if model == "SimpleMean":
        per = K + 1
        acc = np.zeros((Xte.shape[0], K))
        wsum = np.zeros(Xte.shape[0])
        for c in range(Xtr.shape[1] // per):
            b = c * per
            acc += Xte[:, b:b + K] * (Xte[:, b + K] == 1)[:, None]
            wsum += (Xte[:, b + K] == 1)
        return np.where(wsum[:, None] > 0, acc / np.maximum(wsum, 1e-9)[:, None],
                        np.full_like(acc, 1.0 / K))
    if model == "ElasticNetLR":
        lr = LogisticRegression(penalty='elasticnet', solver='saga',
                                l1_ratio=0.5, C=0.3, max_iter=5000,
                                class_weight='balanced', random_state=0)
        lr.fit(Xtr, ytr)
        return lr.predict_proba(Xte)
    if model == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(objective='multi:softprob', num_class=K,
                            eval_metric='mlogloss', n_estimators=200,
                            max_depth=2, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)
    if model == "MLP":
        import torch
        from avail_ai.meta_learners import _MLP, train_nn, predict_nn
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        ps = []
        for s in range(2):
            mdl = _MLP(Xtr.shape[1], K)
            train_nn(mdl, Xtr, ytr, seed=seed + s, K=K, device=dev)
            ps.append(predict_nn(mdl, Xte, device=dev))
        return np.mean(ps, axis=0)
    raise ValueError(model)


def part_a():
    print("[A] seed stability...", flush=True)
    rows = []
    data = {}
    for arm in ["complete", "avail"]:
        piv, label, ids, flags = load_meta(arm)
        data[arm] = (piv, label, ids)
    for rung in [3, 5]:
        mod_order = MODULES[:rung]
        for arm in ["complete", "avail"]:
            piv, label, ids = data[arm]
            X, y = meta_x(piv, ids, label, mod_order)
            for seed in range(10):
                res = loo_fast(X, y, SEED0 + seed)
                for m, v in res.items():
                    rows.append({"rung": rung, "arm": arm, "seed": seed,
                                 "model": m, "macro_auc": v})
            print(f"   rung{rung} {arm} seeds done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "A_seed_stability_raw.tsv", sep="\t", index=False)
    agg = df.groupby(["rung", "arm", "model"])["macro_auc"].agg(
        ["mean", "std"]).reset_index()
    agg.to_csv(OUT / "A_seed_stability_summary.tsv", sep="\t", index=False)
    # delta per seed
    d = df.pivot_table(index=["rung", "seed", "model"], columns="arm",
                       values="macro_auc").reset_index()
    d["delta"] = d["avail"] - d["complete"]
    d.groupby(["rung", "model"])["delta"].agg(["mean", "std"]).to_csv(
        OUT / "A_seed_delta.tsv", sep="\t")
    print(agg.round(3).to_string(index=False), flush=True)
    print("[A] done", flush=True)


def part_b():
    print("[B] CV vs LOO at rung5...", flush=True)
    from sklearn.model_selection import StratifiedKFold
    lad = pd.read_csv(RES / "ladder_summary.tsv", sep="\t")
    lad5 = lad[lad["rung"] == 5]
    rows = []
    for arm in ["complete", "avail"]:
        piv, label, ids, flags = load_meta(arm)
        X, y = meta_x(piv, ids, label, MODULES)
        loo_vals = {}
        for m in FAST:
            loo_vals[m] = lad5[(lad5["arm"] == arm)
                               & (lad5["model"] == m)]["macro_auc"].iloc[0]
        for rep in range(5):
            skf = StratifiedKFold(5, shuffle=True, random_state=100 + rep)
            oof = {m: np.zeros((len(y), K)) for m in FAST}
            for tr, te in skf.split(X, y):
                for m in FAST:
                    oof[m][te] = _fit_predict(m, X[tr], y[tr], X[te],
                                              1000 + rep * 10)
            for m in FAST:
                rows.append({"arm": arm, "model": m, "rep": rep,
                             "cv_macro": macro_auc(y, oof[m]),
                             "loo_macro": loo_vals[m]})
        print(f"   {arm} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "B_cv_vs_loo_raw.tsv", sep="\t", index=False)
    df.groupby(["arm", "model"]).agg(
        cv_mean=("cv_macro", "mean"), cv_sd=("cv_macro", "std"),
        loo=("loo_macro", "first")).to_csv(
        OUT / "B_cv_vs_loo_summary.tsv", sep="\t")
    print(df.groupby(["arm", "model"])[["cv_macro", "loo_macro"]].mean()
          .round(3).to_string(), flush=True)
    print("[B] done", flush=True)


def part_cde():
    print("[C/D/E] subgroup + per-class CI + calibration...", flush=True)
    for arm in ["complete", "avail"]:
        z = np.load(RES / "predictions" / f"rung5_{arm}.npz")
        y = z["y"]
        ids = z["ids"]
        piv, label, ids2, flags = load_meta(arm)
        nmod = np.array([sum(flags[str(i)]) for i in ids])
        grp = np.where(nmod == 5, "all-5",
                       np.where(nmod >= 3, "3-4", "<=2"))
        rows = []
        for m in ["SimpleMean", "ElasticNetLR", "XGBoost", "MLP", "TabPFN",
                  "TabICL"]:
            P = np.nan_to_num(z[m], nan=1.0 / K)
            for g in ["all-5", "3-4", "<=2"]:
                sel = np.where(grp == g)[0]
                if len(sel) < 8:
                    continue
                rows.append({"arm": arm, "model": m, "subgroup": g,
                             "n": len(sel),
                             "macro_auc": macro_auc(y[sel], P[sel])})
        pd.DataFrame(rows).to_csv(OUT / f"C_subgroups_{arm}.tsv", sep="\t",
                                  index=False)
        # D. per-class bootstrap
        rng = np.random.default_rng(SEED0)
        from sklearn.metrics import roc_auc_score
        rows = []
        for m in ["ElasticNetLR", "XGBoost", "MLP", "TabPFN"]:
            P = np.nan_to_num(z[m], nan=1.0 / K)
            boots = {c: [] for c in range(K)}
            for _ in range(2000):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) < 3:
                    continue
                for c in range(K):
                    boots[c].append(roc_auc_score((y[idx] == c).astype(int),
                                                  P[idx][:, c]))
            for c, cn in enumerate(["HC", "MCI", "PDD"]):
                arr = np.array(boots[c])
                rows.append({"arm": arm, "model": m, "class": cn,
                             "auc": roc_auc_score((y == c).astype(int),
                                                  P[:, c]),
                             "lo95": np.percentile(arr, 2.5),
                             "hi95": np.percentile(arr, 97.5)})
        pd.DataFrame(rows).to_csv(OUT / f"D_perclass_bootstrap_{arm}.tsv",
                                  sep="\t", index=False)
        # E. calibration rung5
        rows = []
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for m, col in [("ElasticNetLR", "#1a73e8"), ("TabPFN", "#f29900"),
                       ("MLP", "#d93025")]:
            P = np.nan_to_num(z[m], nan=1.0 / K)
            brier = np.mean(np.sum((P - np.eye(K)[y]) ** 2, axis=1)) / K
            rows.append({"arm": arm, "model": m, "brier": brier})
            pdd = P[:, 2]
            bins = np.quantile(pdd, np.linspace(0, 1, 11))
            bins = np.unique(bins)
            xs, ys = [], []
            for i in range(len(bins) - 1):
                sel = (pdd >= bins[i]) & (pdd <= bins[i + 1])
                if sel.sum() == 0:
                    continue
                xs.append(pdd[sel].mean())
                ys.append((y[sel] == 2).mean())
            axes[0].plot(xs, ys, "-o", ms=3, color=col, label=m)
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_xlabel("predicted P(PDD)")
        axes[0].set_ylabel("observed PDD rate")
        axes[0].set_title(f"Reliability ({arm})")
        axes[0].legend(fontsize=8)
        # risk score histogram
        P = np.nan_to_num(z["ElasticNetLR"], nan=1.0 / K)
        r = P[:, 2] - P[:, 0]
        for cl, col in zip(range(3), ["#1a73e8", "#f29900", "#d93025"]):
            axes[1].hist(r[y == cl], bins=20, alpha=0.5, color=col,
                         label=["HC", "MCI", "PDD"][cl])
        axes[1].set_xlabel("risk score r = q(PDD) - q(HC)")
        axes[1].set_title(f"Risk score ({arm})")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"E_calibration_{arm}.png", dpi=300)
        plt.close(fig)
        pd.DataFrame(rows).to_csv(OUT / f"E_brier_{arm}.tsv", sep="\t",
                                  index=False)
        print(f"   {arm} C/D/E done", flush=True)
    print("[C/D/E] done", flush=True)


def part_f():
    print("[F] mask-value ablation...", flush=True)
    lad = pd.read_csv(RES / "ladder_summary.tsv", sep="\t")
    lad5 = lad[lad["rung"] == 5]
    rows = []
    for arm in ["complete", "avail"]:
        piv, label, ids, flags = load_meta(arm)
        for mask, mname in [(0.0, "zeros"), (1.0 / K, "uniform")]:
            X, y = meta_x(piv, ids, label, MODULES, mask=mask)
            res = loo_fast(X, y, SEED0)
            for m, v in res.items():
                rows.append({"arm": arm, "mask": mname, "model": m,
                             "macro_auc": v})
        print(f"   {arm} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "F_maskvalue.tsv", sep="\t", index=False)
    print(df.pivot_table(index="model", columns=["arm", "mask"],
                         values="macro_auc").round(3).to_string(), flush=True)
    print("[F] done", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["A", "B", "CDE", "F"])
    a = ap.parse_args()
    if a.only in (None, "A"):
        part_a()
    if a.only in (None, "B"):
        part_b()
    if a.only in (None, "CDE"):
        part_cde()
    if a.only in (None, "F"):
        part_f()
    print("[faz12] ALL DONE", flush=True)
