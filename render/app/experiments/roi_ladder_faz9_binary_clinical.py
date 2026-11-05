#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 9: (A) dedicated binary HC-vs-PDD ladder at rung 5/6 and
(B) clinical-module extension of the 3-class ladder (rung 6).

Motivation: the established AVAIL benchmark evaluated with the clinical
layer (age/sex/edu) included and reported a dedicated binary HC-PDD task
(AUC ~0.96). The 5-module 3-class ladder is therefore not directly
comparable; this script runs the matched evaluations.

Outputs: experiments/results/roi_ladder/
  binary_ladder_summary.tsv
  clinical_extension_summary.tsv
  binary_ladder_figure.png/pdf
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.preprocess import preproc, apply_preproc  # noqa: E402
from avail_ai.learners import build_meta_features, macro_auc  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

DATA = ROOT / "experiments/results/ladder_generic"
RES = ROOT / "experiments/results/roi_ladder"
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MICRO = ["stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
SEED = 42
N_SEEDS = 5


def load_all():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    mods = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t")
        t["SampleID"] = t["SampleID"].astype(str).str.strip()
        feat = t.set_index("SampleID")
        mods[m] = {i: feat.loc[i].values.astype(float) for i in feat.index}
    clin = {r.SampleID: np.array([r.Age, r.Sex, r.Education], dtype=float)
            for r in meta.itertuples()}
    return meta, mods, clin


def roi_expert_bin(Xtr, ytr, Xte, rng):
    var = Xtr.var(axis=0)
    top = np.argsort(var)[::-1][:300]
    Xv, Xtv = Xtr[:, top], Xte[:, top]
    if Xv.shape[0] >= 3 and len(np.unique(ytr)) >= 2:
        F, pval = f_classif(Xv, ytr)
        rej = multipletests(pval, method="fdr_bh")[0]
        keep = np.where(rej)[0]
        if len(keep) < 20:
            keep = np.argsort(pval)[:20]
        if len(keep) > 150:
            keep = keep[np.argsort(pval[keep])[:150]]
    else:
        keep = np.arange(Xv.shape[1])[:20]
    Xa, Xta = Xv[:, keep], Xtv[:, keep]
    n, d = Xa.shape
    freq = np.zeros(d)
    for _ in range(8):
        idx = rng.choice(n, size=max(3, int(0.8 * n)), replace=False)
        Xs, ys = Xa[idx], ytr[idx]
        Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
        try:
            lr = LogisticRegression(penalty="l1", C=0.3, solver="saga",
                                    max_iter=3000, class_weight="balanced",
                                    random_state=0)
            lr.fit(Xs, ys)
            freq += (np.abs(lr.coef_).sum(axis=0) > 1e-8).astype(float)
        except Exception:
            pass
    freq /= 8.0
    k2 = np.where(freq >= 0.5)[0]
    if len(k2) == 0:
        k2 = np.argsort(freq)[::-1][:min(20, d)]
    if len(k2) > 100:
        k2 = k2[np.argsort(freq[k2])[::-1][:100]]
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xa[:, k2], ytr)
    out = np.zeros(2)
    for ci, cls in enumerate(m.classes_):
        out[int(cls)] = m.predict_proba(Xta[:, k2])[0, ci]
    return out


def stage1_bin(mods, label, conf, subset, mod_name):
    sub = sorted([i for i in subset if i in mods[mod_name]])
    X = np.vstack([mods[mod_name][i] for i in sub])
    y = np.array([label[i] for i in sub])
    C = np.vstack([conf[i] for i in sub])
    rng = np.random.RandomState(SEED)
    piv = {}
    for k, pid in enumerate(sub):
        tr = [t for t in range(len(sub)) if t != k]
        if mod_name in MICRO:
            sel, sel2, res, sc, Xt = preproc(X[tr], y[tr], C[tr], True,
                                             residualize=False)
            Xtt = apply_preproc(sel, sel2, res, sc, X[k:k + 1], C[k:k + 1])
            m = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                       n_jobs=-1)
            m.fit(Xt, y[tr])
            out = np.zeros(2)
            for ci, cls in enumerate(m.classes_):
                out[int(cls)] = m.predict_proba(Xtt)[0, ci]
        else:
            out = roi_expert_bin(X[tr], y[tr], X[k:k + 1], rng)
        piv[pid] = out
    return piv


def stage1_clin(clin, label, subset, K):
    ids = sorted([i for i in subset if i in clin])
    X = np.vstack([clin[i] for i in ids])
    y = np.array([label[i] for i in ids])
    piv = {}
    for k, pid in enumerate(ids):
        tr = [t for t in range(len(ids)) if t != k]
        sc = StandardScaler().fit(X[tr])
        m = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                   n_jobs=-1)
        m.fit(sc.transform(X[tr]), y[tr])
        out = np.zeros(K)
        p = m.predict_proba(sc.transform(X[k:k + 1]))[0]
        for ci, cls in enumerate(m.classes_):
            out[int(cls)] = p[ci]
        piv[pid] = out
    return piv


def main():
    meta, mods, clin = load_all()
    ids_all = meta["SampleID"].tolist()
    conf = {r.SampleID: np.array([r.Age, r.Sex,
                                  np.nan_to_num(r.Education)], dtype=float)
            for r in meta.itertuples()}
    feat_sets = {m: set(mods[m].keys()) for m in MODULES}

    # complete-case rung5 subset
    comp5 = set(ids_all)
    for m in MODULES:
        comp5 &= feat_sets[m]
    comp5 = sorted(comp5)

    # ---------- A. binary HC vs PDD ----------
    hcpdd = meta[meta["Group"].isin(["HC", "PDD"])]["SampleID"].tolist()
    lab_bin = {r.SampleID: (0 if r.Group == "HC" else 1)
               for r in meta.itertuples()}
    avail_bin = sorted(hcpdd)
    comp_bin = sorted(set(comp5) & set(hcpdd))

    # needed base runs: for each modality avail_bin & comp_bin; clinical too
    needed = set()
    for m in MODULES:
        needed.add((m, tuple(sorted([i for i in avail_bin if i in feat_sets[m]]))))
        needed.add((m, tuple(sorted([i for i in comp_bin if i in feat_sets[m]]))))
    cache = {}
    for m, sub in sorted(needed):
        f = RES / f"stage1bin_{m}_{len(sub)}.tsv.gz"
        if f.exists():
            t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
            t.index = t.index.astype(str).str.strip()
            cache[(m, sub)] = {i: t.loc[i].values.astype(float)
                               for i in t.index}
            print(f"[faz9] {m} n={len(sub)} cached", flush=True)
            continue
        print(f"[faz9] stage1bin {m} n={len(sub)}...", flush=True)
        piv = stage1_bin(mods, lab_bin, conf, list(sub), m)
        cache[(m, sub)] = piv
        t = pd.DataFrame.from_dict(piv, orient="index", columns=["p0", "p1"])
        t.to_csv(f, sep="\t", compression="gzip")

    cl_av = stage1_clin(clin, lab_bin, set(avail_bin), 2)
    cl_cp = stage1_clin(clin, lab_bin, set(comp_bin), 2)

    rows = []
    for arm, arm_ids, getter in [
            ("avail", avail_bin, lambda m: cache[(
                m, tuple(sorted([i for i in avail_bin
                                 if i in feat_sets[m]])))]),
            ("complete", comp_bin, lambda m: cache[(
                m, tuple(sorted([i for i in comp_bin
                                 if i in feat_sets[m]])))])]:
        for n_mods in [5, 6]:
            mod_order = MODULES + (["clinical"] if n_mods == 6 else [])
            piv = {}
            for p in arm_ids:
                d = {}
                for m in MODULES:
                    src = getter(m)
                    if p in src:
                        d[m] = src[p]
                if n_mods == 6:
                    src = cl_av if arm == "avail" else cl_cp
                    if p in src:
                        d["clinical"] = src[p]
                piv[p] = d
            X, y = build_meta_features(piv, arm_ids, lab_bin, n_mods,
                                       mod_order, 2)
            P = run_meta_loo(X, y, n_mods, 2, model_names=MODELS,
                             device=None, random_state=SEED, n_seeds=N_SEEDS,
                             augment_flag=True, aug_p=0.3, verbose=False)
            for mdl in MODELS:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(y, P[mdl][:, 1])
                rows.append({"arm": arm, "n_modules": n_mods,
                             "modules": "+".join(mod_order), "model": mdl,
                             "n": len(arm_ids), "auc": auc})
            print(f"[faz9-BIN] {arm} n_mods={n_mods} n={len(arm_ids)} "
                  f"mean={np.mean([r['auc'] for r in rows[-len(MODELS):]]):.3f}",
                  flush=True)
    bdf = pd.DataFrame(rows)
    bdf.to_csv(RES / "binary_ladder_summary.tsv", sep="\t", index=False)
    print(bdf.pivot_table(index="model", columns=["n_modules", "arm"],
                          values="auc").round(3).to_string(), flush=True)

    # ---------- B. clinical extension (3-class) ----------
    lab3 = {r.SampleID: ["HC", "MCI", "PDD"].index(r.Group)
            for r in meta.itertuples()}
    cl3_av = stage1_clin(clin, lab3, set(ids_all), 3)
    cl3_cp = stage1_clin(clin, lab3, set(comp5), 3)

    def load3(m, n):
        f = RES / f"stage1_full_{m}_{n}.tsv.gz"
        t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
        t.index = t.index.astype(str).str.strip()
        return {i: t.loc[i].values.astype(float) for i in t.index}

    rows3 = []
    for arm, arm_ids in [("avail", sorted(ids_all)), ("complete", comp5)]:
        piv = {}
        for p in arm_ids:
            d = {}
            for m in MODULES:
                src = load3(m, len([i for i in arm_ids if i in feat_sets[m]]))
                if p in src:
                    d[m] = src[p]
            src_cl = cl3_av if arm == "avail" else cl3_cp
            if p in src_cl:
                d["clinical"] = src_cl[p]
            piv[p] = d
        mod_order = MODULES + ["clinical"]
        X, y = build_meta_features(piv, arm_ids, lab3, 6, mod_order, 3)
        P = run_meta_loo(X, y, 6, 3, model_names=MODELS, device=None,
                         random_state=SEED, n_seeds=N_SEEDS, augment_flag=True,
                         aug_p=0.3, verbose=False)
        for mdl in MODELS:
            rows3.append({"arm": arm, "n_modules": 6,
                          "modules": "+".join(mod_order), "model": mdl,
                          "n": len(arm_ids), "macro_auc": macro_auc(y, P[mdl])})
        print(f"[faz9-CLIN] {arm} n={len(arm_ids)} "
              f"mean={np.mean([r['macro_auc'] for r in rows3[-len(MODELS):]]):.3f}",
              flush=True)
    cdf = pd.DataFrame(rows3)
    cdf.to_csv(RES / "clinical_extension_summary.tsv", sep="\t", index=False)
    print(cdf.pivot_table(index="model", columns="arm",
                          values="macro_auc").round(3).to_string(), flush=True)
    print("[faz9] DONE", flush=True)


if __name__ == "__main__":
    main()
