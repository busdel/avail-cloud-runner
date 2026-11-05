#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""k^3 factorial design over imputation x availability representation x fusion.

Levels are defined in LEVELS. The availability factor is a 2^3 lattice of
three summaries (binary flags, coverage count, pattern one-hot): 4 levels use
the binary/count lattice; 8 levels add the pattern factor.

The expensive stage-1 base predictions depend only on (imputation, seed), so
they are cached; availability and fusion are applied at the meta-learner.

Usage:
  python experiments/factorial_k3.py --design 4 --seeds 3 --folds 5
  python experiments/factorial_k3.py --design 8 --seeds 2 --folds 5
"""
import argparse
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
LOCAL_IMPUTER = HERE / "imputation_benchmark_v4.py"
if LOCAL_IMPUTER.exists():
    # Portable layout: factorial_k3.py and imputation_benchmark_v4.py
    # sit next to data/ and results/.
    BASE = HERE
    DATA_DEFAULT = HERE / "data"
    OUT_DEFAULT = HERE / "results" / "factorial_k3_repro"
    sys.path.insert(0, str(HERE))
else:
    BASE = Path("/mnt/limbic/AVAIL_GPU")
    DATA_DEFAULT = BASE / "experiments/results/zero_loss_generic"
    OUT_DEFAULT = BASE / "experiments/results/factorial_k3"
    sys.path.insert(0, str(BASE / "experiments"))
import imputation_benchmark_v4 as v4  # noqa: E402
from imputation_benchmark_v4 import (  # noqa: E402
    load_layers, stage1_cv, macro_auc, SoftImpute)

K = 3

IMP4 = ["none", "mean", "knn", "mice"]
IMP8 = IMP4 + ["median", "missforest", "softimpute", "logmean"]
AVAIL4 = ["none", "binary", "count", "binary_count"]
AVAIL8 = ["none", "binary", "count", "binary_count",
          "pattern", "binary_pattern", "count_pattern", "full"]
FUS4 = ["SimpleMean", "ElasticNetLR", "XGBoost", "RandomForest"]
FUS8 = FUS4 + ["ExtraTrees", "HistGB", "MLP", "KNN"]


class LogWrap:
    def __init__(self, inner, centering=False):
        self.inner = inner
        self.centering = centering

    def fit(self, X, y=None):
        Xl = np.log(np.asarray(X, dtype=float) + 1.0)
        if self.centering:
            Xl = Xl - np.nanmean(Xl, axis=1, keepdims=True)
        self.inner = self.inner.fit(Xl, y)
        return self

    def transform(self, X):
        Xl = np.log(np.asarray(X, dtype=float) + 1.0)
        if self.centering:
            Xl = Xl - np.nanmean(Xl, axis=1, keepdims=True)
        return self.inner.transform(Xl)


def factories():
    return {
        "mean": lambda: SimpleImputer(strategy="mean"),
        "median": lambda: SimpleImputer(strategy="median"),
        "knn": lambda: KNNImputer(n_neighbors=3),
        "mice": lambda: IterativeImputer(max_iter=2, random_state=42,
                                         n_nearest_features=50),
        "zero": lambda: SimpleImputer(strategy="constant", fill_value=0.0),
        "missforest": lambda: IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, n_jobs=12,
                                            random_state=0),
            max_iter=2, random_state=42, n_nearest_features=50),
        "softimpute": lambda: SoftImpute(max_rank=3, n_iter=5),
        "logmean": lambda: LogWrap(SimpleImputer(strategy="mean")),
    }


def avail_parts(piv, ids, mods, avail, B_raw=None):
    n, M = len(ids), len(mods)
    P = np.zeros((n, M * K))
    B = np.zeros((n, M)) if B_raw is None else np.asarray(B_raw, dtype=float)
    for r, p in enumerate(ids):
        for c, m in enumerate(mods):
            if p in piv and m in piv[p]:
                P[r, c * K:(c + 1) * K] = piv[p][m]
    count = B.sum(axis=1, keepdims=True) / max(1, M)
    codes = B.dot((2 ** np.arange(M)).astype(float)).astype(int)
    pattern = np.eye(2 ** M)[codes]
    parts = [P]
    if avail == "full" or "binary" in avail:
        parts.append(B)
    if avail == "full" or "count" in avail:
        parts.append(count)
    if avail == "full" or "pattern" in avail:
        parts.append(pattern)
    return np.hstack(parts), P, B


def make_model(name):
    if name == "ElasticNetLR":
        return LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, C=0.3, max_iter=3000,
                                  class_weight="balanced", random_state=0)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05,
                             random_state=0, n_jobs=6,
                             objective="multi:softprob", num_class=K,
                             eval_metric="mlogloss")
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, n_jobs=6,
                                      class_weight="balanced", random_state=0)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=6,
                                    class_weight="balanced", random_state=0)
    if name == "HistGB":
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    if name == "MLP":
        return Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                                  early_stopping=True, n_iter_no_change=10,
                                  random_state=0)),
        ])
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    raise ValueError(name)


def simple_mean(P, B):
    n = P.shape[0]
    M = B.shape[1]
    out = np.zeros((n, K))
    for i in range(n):
        idx = np.where(B[i] > 0)[0]
        if len(idx) == 0:
            out[i] = np.full(K, 1 / K)
        else:
            out[i] = P[i, (idx * K)[:, None] + np.arange(K)].mean(axis=0)
    return out


def run_cell(X, P, B, y, fusion, folds, seed, mean_all=False):
    n = len(y)
    pred = np.zeros((n, K))
    kf = KFold(folds, shuffle=True, random_state=seed)
    M = B.shape[1]
    for tr, te in kf.split(np.arange(n)):
        if fusion == "SimpleMean":
            if mean_all:
                pred[te] = P[te].reshape(len(te), M, K).mean(axis=1)
            else:
                pred[te] = simple_mean(P[te], B[te])
        else:
            m = make_model(fusion)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])
    return macro_auc(y, pred)


def _dummies(df, col):
    return pd.get_dummies(df[col].astype(str), prefix=col, drop_first=True,
                          dtype=float)


def _term_matrix(df, term):
    factors = term.split(":")
    prod = _dummies(df, factors[0])
    for f in factors[1:]:
        d = _dummies(df, f)
        cols = []
        for c1 in prod.columns:
            for c2 in d.columns:
                cols.append((prod[c1] * d[c2]).rename(f"{c1}*{c2}"))
        prod = pd.concat(cols, axis=1)
    return prod


def _fit_rss(df, y, terms):
    X = pd.DataFrame({"intercept": np.ones(len(df))}, index=df.index)
    if terms:
        X = pd.concat([X, _term_matrix(df, terms[0])], axis=1)
        for term in terms[1:]:
            X = pd.concat([X, _term_matrix(df, term)], axis=1)
    Xv = X.values.astype(float)
    beta, *_ = np.linalg.lstsq(Xv, y, rcond=None)
    resid = y - Xv @ beta
    return float(resid @ resid), int(np.linalg.matrix_rank(Xv))


def _all_terms(factors):
    from itertools import combinations
    out = []
    for r in range(1, len(factors) + 1):
        for c in combinations(factors, r):
            out.append(":".join(c))
    return out


def _anova_manual(df, factors, out_path):
    y = df["macro_auc"].values.astype(float)
    terms = _all_terms(factors)
    rss_full, rank_full = _fit_rss(df, y, terms)
    df_resid = len(df) - rank_full
    rows = []
    for term in terms:
        reduced = [t for t in terms if t != term]
        rss_r, rank_r = _fit_rss(df, y, reduced)
        df_diff = rank_full - rank_r
        if df_resid <= 0 or df_diff <= 0:
            F = np.nan; p = np.nan
        else:
            F = ((rss_r - rss_full) / df_diff) / (rss_full / df_resid)
            p = float(stats.f.sf(F, df_diff, df_resid))
        eta = (rss_r - rss_full) / rss_r if rss_r > 0 else np.nan
        rows.append({"term": term, "df": df_diff, "F": F, "p": p,
                     "partial_eta_sq": eta})
    a = pd.DataFrame(rows).set_index("term")
    a.to_csv(out_path, sep="\t")
    return a


def analyze(df, design, out):
    df = df.copy().drop_duplicates(subset=["imp", "avail", "fusion", "seed"],
                                   keep="last")
    df["avail"] = df["avail"].astype(str)
    anova = _anova_manual(df, ["imp", "avail", "fusion"],
                          out / f"factorial_{design}_anova.tsv")

    d2 = df.copy()
    d2["bin"] = d2["avail"].str.contains("binary").astype(int).astype(str)
    d2["cnt"] = d2["avail"].str.contains("count").astype(int).astype(str)
    d2["pat"] = d2["avail"].str.contains("pattern").astype(int).astype(str)
    factors = ["imp", "bin", "cnt", "fusion"]
    if design == "8":
        factors = ["imp", "bin", "cnt", "pat", "fusion"]
    a2 = _anova_manual(d2, factors,
                       out / f"factorial_{design}_availability_anova.tsv")

    g = df.groupby(["imp", "avail", "fusion"], observed=True)["macro_auc"].agg(
        ["mean", "std", "count"]).reset_index()
    rows = []
    def m(imp=None, avail=None, fusion=None):
        q = g
        if imp is not None: q = q[q.imp == imp]
        if avail is not None: q = q[q.avail == avail]
        if fusion is not None: q = q[q.fusion == fusion]
        return float(q["mean"].mean())
    for fusion in sorted(df.fusion.unique()):
        for imp in sorted(df.imp.unique()):
            vals = {a: m(imp, a, fusion) for a in sorted(df.avail.unique())}
            base = vals.get("none", np.nan)
            for a, v in vals.items():
                if a != "none":
                    rows.append({"contrast": f"{a}_vs_none|{imp}|{fusion}",
                                 "estimate": v - base})
    for imp in sorted(df.imp.unique()):
        if imp != "none":
            vals = [m(imp, a, f) for a in sorted(df.avail.unique())
                    for f in sorted(df.fusion.unique())]
            base = [m("none", a, f) for a in sorted(df.avail.unique())
                    for f in sorted(df.fusion.unique())]
            rows.append({"contrast": f"{imp}_vs_none|main",
                         "estimate": float(np.mean(vals) - np.mean(base))})
    for fusion in sorted(df.fusion.unique()):
        if fusion != "SimpleMean":
            vals = [m(i, a, fusion) for i in sorted(df.imp.unique())
                    for a in sorted(df.avail.unique())]
            base = [m(i, a, "SimpleMean") for i in sorted(df.imp.unique())
                    for a in sorted(df.avail.unique())]
            rows.append({"contrast": f"{fusion}_vs_SimpleMean|main",
                         "estimate": float(np.mean(vals) - np.mean(base))})
    ct = pd.DataFrame(rows)
    ct.to_csv(out / f"factorial_{design}_contrasts.tsv", sep="\t", index=False)

    pivot = df.pivot_table(index=["imp", "avail"], columns="fusion",
                           values="macro_auc", aggfunc="mean")
    pivot.round(4).to_csv(out / f"factorial_{design}_pivot.tsv", sep="\t")
    return anova, a2, ct, pivot


def md_table(design, anova, a2, ct, pivot, out):
    lines = [f"# Table S12{'b' if design == '4' else 'c'}. "
             f"{design}x{design}x{design} factorial design", "",
             f"Factors: imputation, availability representation (binary/count"
             + ("/pattern" if design == "8" else "") +
             " lattice), fusion. Five-fold CV; seeds averaged.", "",
             "## Cell means (macro-AUC)", "", pivot.round(3).to_string(), "",
             "## Factorial ANOVA (type II)", "", anova.round(5).to_string(), "",
             "## Availability-summary ANOVA", "", a2.round(5).to_string(), "",
             "## Selected contrasts (descriptive)", ""]
    if len(ct):
        lines += [ct.head(40).round(4).to_string(index=False), ""]
    (out / f"TableS12{'b' if design == '4' else 'c'}_factorial_{design}.md").write_text(
        "\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["4", "8"], required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data-root", default=str(DATA_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "stage1").mkdir(exist_ok=True)
    v4.ROOT = Path(args.data_root)
    ids, y, layers = load_layers()
    mods = sorted(layers)
    B_raw = np.zeros((len(ids), len(mods)))
    for c, m in enumerate(mods):
        present = set(layers[m].dropna().index)
        for r, p in enumerate(ids):
            if p in present:
                B_raw[r, c] = 1.0
    imps = IMP4 if args.design == "4" else IMP8
    avails = AVAIL4 if args.design == "4" else AVAIL8
    fusions = FUS4 if args.design == "4" else FUS8
    facs = factories()
    cells_path = out / f"cells_{args.design}.tsv"
    if args.analyze_only:
        df = pd.read_csv(cells_path, sep="\t").drop_duplicates(
            subset=["imp", "avail", "fusion", "seed"], keep="last")
        df.to_csv(cells_path, sep="\t", index=False)
        anova, a2, ct, pivot = analyze(df, args.design, out)
        md_table(args.design, anova, a2, ct, pivot, out)
        print("analysis done:", out)
        return
    done = set()
    old_df = pd.DataFrame()
    if cells_path.exists():
        old_df = pd.read_csv(cells_path, sep="\t")
        done = set(zip(old_df.imp, old_df.avail, old_df.fusion, old_df.seed))
    rows = []
    for imp in imps:
        fac = None if imp == "none" else facs[imp]
        for seed in range(args.seeds):
            cpath = out / "stage1" / f"{imp}_{seed}.pkl"
            if cpath.exists():
                with open(cpath, "rb") as fh:
                    piv = pickle.load(fh)
            else:
                print(f"[stage1] {imp} seed={seed}", flush=True)
                piv = stage1_cv(layers, ids, y, K, imputer_factory=fac,
                                n_folds=args.folds, seed=seed)
                with open(cpath, "wb") as fh:
                    pickle.dump(piv, fh)
            if imp == "none":
                # absent modalities have no measured signal: zero them so that
                # the availability-flag contrast is meaningful.
                for c, m in enumerate(mods):
                    for r, p in enumerate(ids):
                        if B_raw[r, c] == 0 and p in piv and m in piv[p]:
                            piv[p][m] = np.zeros(K)
            for avail in avails:
                X, P, B = avail_parts(piv, ids, mods, avail, B_raw)
                for fusion in fusions:
                    key = (imp, avail, fusion, seed)
                    if key in done:
                        continue
                    auc = run_cell(X, P, B, y, fusion, args.folds, seed,
                                    mean_all=(avail == "none"))
                    rows.append({"imp": imp, "avail": avail, "fusion": fusion,
                                 "seed": seed, "macro_auc": auc})
                    print(f"{imp:10s} {avail:15s} {fusion:12s} seed={seed} "
                          f"auc={auc:.4f}", flush=True)
                    if len(rows) % 10 == 0:
                        cur = pd.concat([old_df, pd.DataFrame(rows)],
                                        ignore_index=True).drop_duplicates(
                            subset=["imp", "avail", "fusion", "seed"],
                            keep="last")
                        cur.to_csv(cells_path, sep="\t", index=False)
    new = pd.DataFrame(rows)
    df = pd.concat([old_df, new], ignore_index=True) if len(old_df) else new
    df = df.drop_duplicates(subset=["imp", "avail", "fusion", "seed"],
                            keep="last")
    df.to_csv(cells_path, sep="\t", index=False)
    anova, a2, ct, pivot = analyze(df, args.design, out)
    md_table(args.design, anova, a2, ct, pivot, out)
    print("done:", out)


if __name__ == "__main__":
    main()
