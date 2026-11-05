#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 7: Critical-thinking driven robustness checks for the AVAIL ladder.

A. Table 1 + standardized mean differences (complete-rung5 vs excluded vs avail)
B. Flags-only baseline: how much signal is in availability alone
C. Matched-n subsampling: AVAIL restricted to n=50 vs complete-case n=50
D. Bootstrap 95% CIs for every rung/arm (per model + model-averaged)
E. Age-(sex-edu)-residualized behavioral PLS sensitivity

Outputs: experiments/results/roi_ladder/critical/
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import macro_auc, build_meta_features  # noqa: E402
from avail_ai.meta_learners import (_MLP, train_nn, predict_nn,  # noqa: E402
                                    augment)

DATA = ROOT / "experiments/results/ladder_generic"
RES = ROOT / "experiments/results/roi_ladder"
OUT = RES / "critical"
OUT.mkdir(parents=True, exist_ok=True)
PRED = RES / "predictions"

MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
CLASSES = ["HC", "MCI", "PDD"]
K = 3
SEED = 42
FAST_MODELS = ["ElasticNetLR", "XGBoost", "MLP"]


# ---------------------------------------------------------------- helpers
def _device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def meta_fit_predict(Xtr, ytr, Xte, model_name, seeds=2, seed=0):
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
        dev = _device()
        ps = []
        for s in range(seeds):
            mdl = _MLP(Xtr.shape[1], K)
            train_nn(mdl, Xtr, ytr, seed=seed + s, K=K, device=dev)
            ps.append(predict_nn(mdl, Xte, device=dev))
        return np.mean(ps, axis=0)
    raise ValueError(model_name)


def meta_loo_fast(X, y, models=FAST_MODELS, seeds=2, seed=SEED):
    n = len(y)
    P = {m: np.zeros((n, K)) for m in models}
    for fold in range(n):
        tr = [t for t in range(n) if t != fold]
        Xtr, ytr = augment(X[tr], y[tr], X.shape[1] // (K + 1), K,
                           aug_p=0.3, random_state=seed)
        for m in models:
            P[m][fold] = meta_fit_predict(Xtr, ytr, X[fold:fold + 1], m,
                                          seeds=seeds, seed=seed + fold)
    return P


def hc_pdd_auc(y, P):
    hc, pdd = 0, 2
    sel = np.where(np.isin(y, [hc, pdd]))[0]
    if len(sel) == 0:
        return np.nan
    from sklearn.metrics import roc_auc_score
    yb = (y[sel] == pdd).astype(int)
    Pb = P[sel][:, pdd] / (P[sel][:, [hc, pdd]].sum(axis=1) + 1e-12)
    return float(roc_auc_score(yb, Pb))


def load_meta_and_flags(arm="avail"):
    base_files = sorted(RES.glob("stage1_full_*.tsv.gz"))
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
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    comp = set(meta["SampleID"])
    for m in MODULES:
        comp &= feat[m]
    comp = sorted(comp)
    ids = comp if arm == "complete" else sorted(meta["SampleID"])
    subset = {}
    for m in MODULES:
        n_ = len([i for i in ids if i in feat[m]])
        subset[m] = base[(m, n_)]
    X = np.zeros((len(ids), len(MODULES) * (K + 1)))
    F = np.zeros((len(ids), len(MODULES)))
    y = []
    for r, pid in enumerate(ids):
        row = meta[meta["SampleID"] == pid].iloc[0]
        y.append(CLASSES.index(row.Group))
        for c, m in enumerate(MODULES):
            b = c * (K + 1)
            if pid in subset[m]:
                X[r, b:b + K] = subset[m][pid]
                X[r, b + K] = 1.0
                F[r, c] = 1.0
    return X, np.array(y), F, ids, meta, feat


# ---------------------------------------------------------------- A. Table 1
def table1():
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
    meta["arm"] = np.where(meta["SampleID"].isin(comp), "complete50",
                           "excluded")
    meta.loc[meta["arm"] == "", "arm"] = "avail_only"
    rows = []
    for grp in ["complete50", "excluded"]:
        g = meta[meta["arm"] == grp]
        rows.append({"group": grp, "n": len(g),
                     "age_mean": g["Age"].mean(),
                     "age_sd": g["Age"].std(),
                     "sex_m": (g["Sex"] == 1).mean(),
                     "edu_mean": g["Education"].mean(),
                     "HC": (g["Group"] == "HC").sum(),
                     "MCI": (g["Group"] == "MCI").sum(),
                     "PDD": (g["Group"] == "PDD").sum()})
    df = pd.DataFrame(rows)

    def smd_cont(a, b):
        return ((a.mean() - b.mean())
                / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

    c = meta[meta["arm"] == "complete50"]
    e = meta[meta["arm"] == "excluded"]
    smds = {
        "age_smd": smd_cont(c["Age"], e["Age"]),
        "edu_smd": smd_cont(c["Education"], e["Education"]),
        "sex_smd": (c["Sex"].mean() - e["Sex"].mean())
        / np.sqrt((c["Sex"].var(ddof=1) + e["Sex"].var(ddof=1)) / 2),
    }
    from scipy import stats
    chi2 = stats.chi2_contingency(pd.crosstab(meta["arm"], meta["Group"]))
    smds["group_chi2_p"] = chi2[1]
    smds["age_ttest_p"] = stats.ttest_ind(c["Age"], e["Age"],
                                          equal_var=False).pvalue
    df.to_csv(OUT / "table1_arm_comparability.tsv", sep="\t", index=False)
    pd.Series(smds).to_csv(OUT / "table1_smd.tsv", sep="\t")
    print("[faz7-A] table1 done", flush=True)
    print(df.round(3).to_string(index=False))
    print({k: round(v, 4) for k, v in smds.items()})


# ---------------------------------------------------------------- B. flags
def flags_only():
    X, y, F, ids, meta, feat = load_meta_and_flags()
    res = {}
    for label, Fm in [("flags5", F), ("flags_fmri_only", F[:, [0]])]:
        P = meta_loo_fast(Fm, y, models=["ElasticNetLR", "XGBoost", "MLP"])
        for m in P:
            res[f"{label}_{m}_macro"] = macro_auc(y, P[m])
            res[f"{label}_{m}_HCPDD"] = hc_pdd_auc(y, P[m])
    pd.Series(res).to_csv(OUT / "flags_only_baseline.tsv", sep="\t")
    print("[faz7-B] flags-only done:", {k: round(v, 3)
                                        for k, v in res.items()},
          flush=True)


# ---------------------------------------------------------------- C. matched-n
def matched_n(K_sub=50, n_sub=50, seed=SEED):
    X, y, F, ids, meta, feat = load_meta_and_flags()
    rng = np.random.default_rng(seed)
    hc = np.where(y == 0)[0]
    mci = np.where(y == 1)[0]
    pdd = np.where(y == 2)[0]
    # class makeup mirrored on the complete-case rung-5 cohort (17/18/15)
    n_hc, n_mci, n_pdd = 17, 18, 15
    aucs = {m: [] for m in FAST_MODELS}
    hcpdd = {m: [] for m in FAST_MODELS}
    for it in range(K_sub):
        sel = np.concatenate([
            rng.choice(hc, n_hc, replace=False),
            rng.choice(mci, n_mci, replace=False),
            rng.choice(pdd, n_pdd, replace=False)])
        sel = np.sort(sel)
        P = meta_loo_fast(X[sel], y[sel])
        for m in FAST_MODELS:
            aucs[m].append(macro_auc(y[sel], P[m]))
            hcpdd[m].append(hc_pdd_auc(y[sel], P[m]))
        if (it + 1) % 10 == 0:
            print(f"[faz7-C] subsample {it+1}/{K_sub}", flush=True)
    rows = []
    main = pd.read_csv(RES / "ladder_summary.tsv", sep="\t")
    comp = main[(main["rung"] == 5) & (main["arm"] == "complete")]
    for m in FAST_MODELS:
        a = np.array(aucs[m])
        h = np.array(hcpdd[m])
        cm = comp[comp["model"] == m]["macro_auc"].iloc[0]
        ch = comp[comp["model"] == m]["auc_HC_PDD"].iloc[0]
        rows.append({"model": m,
                     "avail50_macro_mean": a.mean(),
                     "avail50_macro_lo": np.percentile(a, 2.5),
                     "avail50_macro_hi": np.percentile(a, 97.5),
                     "complete50_macro": cm,
                     "delta_macro": a.mean() - cm,
                     "avail50_hcpdd_mean": h.mean(),
                     "avail50_hcpdd_lo": np.percentile(h, 2.5),
                     "avail50_hcpdd_hi": np.percentile(h, 97.5),
                     "complete50_hcpdd": ch,
                     "delta_hcpdd": h.mean() - ch})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "matched_n_avail50_vs_complete50.tsv", sep="\t",
              index=False)
    print("[faz7-C] matched-n done", flush=True)
    print(df.round(3).to_string(index=False))


# ---------------------------------------------------------------- D. bootstrap CIs
def bootstrap_cis(n_boot=2000):
    rows = []
    for rung in range(1, 6):
        for arm in ["complete", "avail"]:
            z = np.load(PRED / f"rung{rung}_{arm}.npz")
            y = z["y"]
            models = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
                      "MLP", "TabPFN", "TabICL"]
            P = {m: z[m] for m in models}
            n = len(y)
            rng = np.random.default_rng(SEED)
            boot = {m: [] for m in models}
            boot_mean = []
            for _ in range(n_boot):
                idx = rng.integers(0, n, n)
                vals = []
                for m in models:
                    try:
                        v = macro_auc(y[idx], P[m][idx])
                    except Exception:
                        v = np.nan
                    boot[m].append(v)
                    vals.append(v)
                boot_mean.append(np.nanmean(vals))
            for m in models:
                arr = np.array(boot[m])
                rows.append({"rung": rung, "arm": arm, "model": m,
                             "macro_auc": macro_auc(y, P[m]),
                             "lo95": np.nanpercentile(arr, 2.5),
                             "hi95": np.nanpercentile(arr, 97.5)})
            arr = np.array(boot_mean)
            rows.append({"rung": rung, "arm": arm,
                         "model": "MODEL_AVERAGED",
                         "macro_auc": np.nanmean(
                             [macro_auc(y, P[m]) for m in models]),
                         "lo95": np.nanpercentile(arr, 2.5),
                         "hi95": np.nanpercentile(arr, 97.5)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_ci_ladder.tsv", sep="\t", index=False)
    piv = df[df["model"] == "MODEL_AVERAGED"].pivot_table(
        index="rung", columns="arm",
        values=["macro_auc", "lo95", "hi95"])
    print("[faz7-D] bootstrap done", flush=True)
    print(piv.round(3).to_string())


# ---------------------------------------------------------------- E. PLS sensitivity
def pls_sensitivity():
    from scipy import stats as st
    meta = pd.read_csv(ROOT / "AVAIL_benchmark/busra_pd_multisite/"
                       "metadata.txt", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    xl = pd.ExcelFile("/home/suleyman/Desktop/fMRI datasets/"
                      "partial_corr_reshaped_Spe21-23.xlsx")
    yx = pd.read_excel(ROOT / "AVAIL_benchmark/fMRI correlation and feature"
                       " selection/y_data.xlsx").sort_values(
        "Sort Number").reset_index(drop=True)
    patients = yx["PatientID"].astype(str).str.strip().tolist()
    tri = np.triu_indices(100, k=1)
    M = np.stack([xl.parse(s, header=None).values.astype(float)[tri]
                  for s in xl.sheet_names])
    COG = ["MMSE", "Categorical_Fluency_", "Phonemic_Fluency", "OVMPT_Coding",
           "OVMPT_Recalling", "DS_Forward", "DS_Backward",
           "Benton_Face_Recognition_Test", "Clock_Drawing_Test"]
    mm = meta.set_index("SampleID").loc[patients]
    Y = np.column_stack([pd.to_numeric(mm[c], errors="coerce")
                        for c in COG])
    ok = np.isfinite(Y).all(axis=1)
    age = mm["Age"].values.astype(float)
    sex = mm["Sex"].map({"M": 1.0, "F": 0.0}).values
    cov = np.column_stack([age, sex,
                           mm["Education"].values.astype(float)])
    X = M[ok]
    Y = Y[ok]
    C = cov[ok]
    C1 = np.column_stack([np.ones(len(X)), C])
    Bx, *_ = np.linalg.lstsq(C1, X, rcond=None)
    Xr = X - C1 @ Bx
    Bx1, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), age[ok]]),
                              X, rcond=None)
    Xr1 = X - np.column_stack([np.ones(len(X)), age[ok]]) @ Bx1

    def pls_pv(Xm, Ym, B=500, seed=SEED, return_scores=False):
        rng = np.random.default_rng(seed)
        Xz = st.zscore(Xm, axis=0)
        Yz = st.zscore(Ym, axis=0)
        Cm = Xz.T @ Yz / (len(Xz) - 1)
        U, s, Vt = np.linalg.svd(Cm, full_matrices=False)
        obs = (s ** 2).sum()
        null = []
        for _ in range(B):
            Yp = Yz[rng.permutation(len(Yz))]
            s2 = np.linalg.svd(Xz.T @ Yp / (len(Xz) - 1), compute_uv=False)
            null.append((s2 ** 2).sum())
        null = np.array(null)
        if return_scores:
            return (obs, float((null >= obs).mean()),
                    s[0] ** 2 / (s ** 2).sum(), Xz @ U[:, 0])
        return obs, float((null >= obs).mean()), s[0] ** 2 / (s ** 2).sum()

    def resid(A, z):
        Z = np.column_stack([np.ones(len(A)), z])
        B, *_ = np.linalg.lstsq(Z, A, rcond=None)
        return A - Z @ B

    # NOTE on the correct age-control test: residualize BOTH blocks under the
    # same transform before permutation. Residualizing X only while leaving Y
    # raw lets permuted-age covariance leak into the null (invalid test).
    o_raw, p_raw, lv_raw, sc_raw = pls_pv(X, Y, return_scores=True)
    r_age = st.spearmanr(sc_raw, age[ok])
    o_both, p_both, lv_both = pls_pv(resid(X, age[ok]), resid(Y, age[ok]))
    o_y, p_y, lv_y = pls_pv(X, resid(Y, age[ok]))
    with open(OUT / "pls_sensitivity.txt", "w") as f:
        f.write(f"n={X.shape[0]}\n")
        f.write(f"RAW: p={p_raw:.4f}, LV1={lv_raw:.3f}, "
                f"corr(LV1_scores, age) rho={r_age.statistic:+.3f} "
                f"p={r_age.pvalue:.2e}\n")
        f.write(f"Y|age (cognition residualized): p={p_y:.4f}, "
                f"LV1={lv_y:.3f}\n")
        f.write(f"X,Y|age (BOTH residualized, valid partial test): "
                f"p={p_both:.4f}, LV1={lv_both:.3f}\n")
        f.write("NOTE: an earlier X-only residualization variant is not a "
                "valid age-control (permuted-age covariance leaks into the "
                "null) and is not reported.\n")
    print(f"[faz7-E] PLS: raw p={p_raw:.4f} LV1={lv_raw:.3f} "
          f"(age rho={r_age.statistic:+.3f}); both-resid "
          f"p={p_both:.4f} LV1={lv_both:.3f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    choices=["A", "B", "C", "D", "E"])
    a = ap.parse_args()
    if a.only in (None, "A"):
        table1()
    if a.only in (None, "B"):
        flags_only()
    if a.only in (None, "D"):
        bootstrap_cis()
    if a.only in (None, "E"):
        pls_sensitivity()
    if a.only in (None, "C"):
        matched_n()
    print("[faz7] DONE", flush=True)
