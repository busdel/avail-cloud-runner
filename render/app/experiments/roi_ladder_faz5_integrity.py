#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 5: integrity checks for the two-arm ladder (rung 5).

1. Label-permutation test (200 shuffles) of the meta-LOO macro-AUC for
   both arms -> null distribution and empirical p.
2. Covariate control: meta features residualized on age/sex/education
   (within-fold) -> macro-AUC drop.
3. Decision-curve analysis: net benefit for AVAIL vs complete-case probas.

Outputs: experiments/results/roi_ladder/integrity/
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo, augment  # noqa: E402
from avail_ai.meta_learners import (_tabular_predict, _MLP, train_nn,
                                     predict_nn)  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "experiments/results/roi_ladder/integrity"
OUT.mkdir(parents=True, exist_ok=True)
PRED = ROOT / "experiments/results/roi_ladder/predictions"

MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
PERM_MODELS = ["ElasticNetLR", "XGBoost", "MLP"]
SEED = 42
N_PERM = 200
N_FOLDS = 5
RUNG = 5


def meta_fit_predict(Xtr, ytr, Xte, model_name, K=3, seeds=2, seed=0):
    """Single split train->predict for one meta model (permutation null)."""
    if model_name == "SimpleMean":
        per = K + 1
        acc = np.zeros((Xte.shape[0], K))
        wsum = np.zeros(Xte.shape[0])
        for c in range(Xtr.shape[1] // per):
            base = c * per
            acc += Xte[:, base:base + K] * (Xte[:, base + K] == 1)[:, None]
            wsum += (Xte[:, base + K] == 1)
        return np.where(wsum[:, None] > 0, acc / wsum[:, None],
                        np.full_like(acc, 1.0 / K))
    if model_name == "WSimpleMean":
        per = K + 1
        w = _modality_weights_quick(Xtr, ytr, Xtr.shape[1] // per, K)
        acc = np.zeros((Xte.shape[0], K))
        wsum = np.zeros(Xte.shape[0])
        for c in range(Xtr.shape[1] // per):
            base = c * per
            acc += w[c] * Xte[:, base:base + K] * (Xte[:, base + K] == 1)[:, None]
            wsum += w[c] * (Xte[:, base + K] == 1)
        return np.where(wsum[:, None] > 0, acc / wsum[:, None],
                        np.full_like(acc, 1.0 / K))
    if model_name == "ElasticNetLR":
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(penalty='elasticnet', solver='saga',
                                l1_ratio=0.5, C=0.3, max_iter=5000,
                                class_weight='balanced', random_state=0)
        lr.fit(Xtr, ytr)
        return lr.predict_proba(Xte)
    if model_name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(objective='multi:softprob', num_class=K,
                            eval_metric='mlogloss', n_estimators=200,
                            max_depth=2, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=0, n_jobs=-1)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)
    if model_name == "MLP":
        dev = 'cuda' if _device() == 'cuda' else 'cpu'
        ps = []
        for s in range(seeds):
            mdl = _MLP(Xtr.shape[1], K)
            train_nn(mdl, Xtr, ytr, seed=seed + s, K=K, device=dev)
            ps.append(predict_nn(mdl, Xte, device=dev))
        return np.mean(ps, axis=0)
    raise ValueError(model_name)


def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _modality_weights_quick(Xtr, ytr, n_mods, K):
    per = K + 1
    w = []
    for c in range(n_mods):
        base = c * per
        sel = Xtr[:, base + K] == 1
        if sel.sum() >= 3 and len(np.unique(ytr[sel])) > 1:
            auc = macro_auc(ytr[sel], Xtr[sel][:, base:base + K])
        else:
            auc = 0.5
        w.append(max(auc, 0.5))
    w = np.array(w, dtype=float)
    return w / w.sum() if w.sum() > 0 else np.ones(n_mods) / n_mods


def meta_cv(X, y, n_mods, model_names, n_folds=5, seeds=2, seed=0):
    """Stratified k-fold meta benchmark -> mean macro-AUC across models."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    outs = []
    for tr, te in skf.split(X, y):
        Xtr, ytr = augment(X[tr], y[tr], n_mods, 3, aug_p=0.3,
                           random_state=SEED)
        for m in model_names:
            P = meta_fit_predict(Xtr, ytr, X[te], m, seeds=seeds, seed=seed)
            outs.append(macro_auc(y[te], P))
    return float(np.mean(outs))


def load_meta_features(arm):
    """Rebuild the rung-5 meta feature matrix for the given arm."""
    base_files = sorted(PRED.parent.glob("stage1_full_*.tsv.gz"))
    base = {}
    for f in base_files:
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
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t", usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    comp = set(meta["SampleID"])
    for m in MODULES:
        comp &= feat[m]
    comp = sorted(comp)
    ids = comp if arm == "complete" else sorted(meta["SampleID"])
    subset = {}
    for m in MODULES:
        n_ = len([i for i in ids if i in feat[m]])
        key = (m, n_)
        if key not in base:
            raise KeyError(f"missing base cache {key}")
        subset[m] = base[key]
    n_mods = len(MODULES)
    K = 3
    X = np.zeros((len(ids), n_mods * (K + 1)))
    y = []
    for r, pid in enumerate(ids):
        row = meta[meta["SampleID"] == pid].iloc[0]
        y.append(["HC", "MCI", "PDD"].index(row.Group))
        for c, m in enumerate(MODULES):
            b = c * (K + 1)
            if pid in subset[m]:
                X[r, b:b + K] = subset[m][pid]
                X[r, b + K] = 1.0
    y = np.array(y)
    cov = np.column_stack([
        meta.set_index("SampleID").loc[ids, "Age"].values.astype(float),
        meta.set_index("SampleID").loc[ids, "Sex"].values.astype(float),
        meta.set_index("SampleID").loc[ids, "Education"].values.astype(float)])
    return X, y, cov, ids


def meta_loo(X, y, n_mods, seeds=5):
    P = run_meta_loo(X, y, n_mods, 3, model_names=MODELS, device=None,
                     random_state=SEED, n_seeds=seeds,
                     augment_flag=True, aug_p=0.3, verbose=False)
    return {m: macro_auc(y, P[m]) for m in MODELS}


def residualize_within_fold(X, y, cov, n_mods, K=3):
    Xr = X.copy()
    for fold in range(len(y)):
        tr = [t for t in range(len(y)) if t != fold]
        Ctr = np.column_stack([np.ones(len(tr)), cov[tr]])
        B, *_ = np.linalg.lstsq(Ctr, X[tr], rcond=None)
        Cte = np.concatenate([[1.0], cov[fold]])
        Xr[fold] = X[fold] - Cte @ B
    return Xr


def permutation_test(arm):
    X, y, cov, ids = load_meta_features(arm)
    rng = np.random.default_rng(SEED)
    obs = meta_cv(X, y, len(MODULES), PERM_MODELS, n_folds=N_FOLDS,
                  seeds=5, seed=SEED)
    null = []
    for it in range(N_PERM):
        yp = rng.permutation(y)
        null.append(meta_cv(X, yp, len(MODULES), PERM_MODELS,
                            n_folds=N_FOLDS, seeds=2, seed=it))
    arr = np.array(null)
    p = float((arr >= obs).mean())
    return pd.DataFrame([{"arm": arm, "protocol": "5fold-CV",
                          "models": "+".join(PERM_MODELS),
                          "obs_macro_auc": obs,
                          "null_mean": float(arr.mean()),
                          "null_sd": float(arr.std()), "p_perm": p}])


def main():
    rows = []
    for arm in ["complete", "avail"]:
        print(f"[faz5] permutation: arm={arm}", flush=True)
        rows.append(permutation_test(arm))
        X, y, cov, ids = load_meta_features(arm)
        print(f"[faz5] covariate control: arm={arm}", flush=True)
        Xr = residualize_within_fold(X, y, cov, len(MODULES))
        res_res = meta_loo(Xr, y, len(MODULES))
        for m in MODELS:
            rows.append(pd.DataFrame([{"arm": arm, "protocol": "covresid",
                                       "models": m,
                                       "obs_macro_auc": res_res[m],
                                       "null_mean": np.nan,
                                       "null_sd": np.nan,
                                       "p_perm": np.nan}]))
    perm = pd.concat(rows, ignore_index=True)
    perm.to_csv(OUT / "permutation_and_covariate.tsv", sep="\t", index=False)
    print(perm.to_string(), flush=True)

    # decision curve (HC-PDD binary, per arm)
    print("[faz5] decision curve...", flush=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm, color in [("complete", "#b00020"), ("avail", "#1a73e8")]:
        z = np.load(PRED / f"rung{RUNG}_{arm}.npz")
        y = z["y"]
        hc, pdd = 0, 2
        sel = np.where(np.isin(y, [hc, pdd]))[0]
        yb = (y[sel] == pdd).astype(int)
        prob = z["ElasticNetLR"][sel][:, pdd] / (
            z["ElasticNetLR"][sel][:, [hc, pdd]].sum(axis=1) + 1e-12)
        thr = np.linspace(0.01, 0.99, 99)
        nb = []
        for t in thr:
            tp = (prob >= t) & (yb == 1)
            fp = (prob >= t) & (yb == 0)
            nb.append((tp.sum() / len(yb)) - (fp.sum() / len(yb)) * t / (1 - t))
        ax.plot(thr, nb, color=color, lw=2,
                label=f"{arm} (n={len(sel)})")
    prev = yb.mean()
    ax.plot(thr, prev - (1 - prev) * thr / (1 - thr), "k--", lw=1.2,
            label="treat all")
    ax.plot(thr, np.zeros_like(thr), "k:", lw=1.2, label="treat none")
    ax.set_xlabel("threshold probability")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve: HC vs PDD (rung 5, ElasticNetLR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "decision_curve.png", dpi=300)
    fig.savefig(OUT / "decision_curve.pdf")
    print("[faz5] DONE.", flush=True)


if __name__ == "__main__":
    main()
