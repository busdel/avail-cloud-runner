#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Factorial design over the missing-data strategy space (Paper 2, workstream E).

Factors (2x2x2): imputation (none vs fold-internal imputed signal),
flags (off vs true availability flags), fusion (fixed averaging vs learned).
Cells: p_only (0,0) · avail_flags (0,1) · imputation (1,0) · HYBRID (1,1).
Two imputer replicates (mean, MICE) x four meta-learners, 3 seeds averaged.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats as _st

BASE = Path("/mnt/limbic/AVAIL_GPU")
V4 = BASE / "revizyon/makaleler/supplement/S8_sources/imputation_benchmark_v4_results.tsv"
HY = BASE / "experiments/results/hybrid_comparison_results.tsv"
OUT = BASE / "experiments/results/factorial"
OUT.mkdir(parents=True, exist_ok=True)

FIXED = {"SimpleMean", "WSimpleMean"}

v4 = pd.read_csv(V4, sep="\t")
v4m = (v4.groupby(["strategy", "model"])["macro_auc"].mean()
       .reset_index())

if not HY.exists():
    raise SystemExit(f"hybrid results not found at {HY} — run hybrid_comparison.py first")
hy = pd.read_csv(HY, sep="\t")
hym = (hy.groupby(["strategy", "model"])["macro_auc"].mean()
       .reset_index())

CELLS = {
    "p_only": (0, 0, "none"),
    "avail_flags": (0, 1, "none"),
    "mean_imp": (1, 0, "mean"),
    "hybrid_mean_imp": (1, 1, "mean"),
    "mice_imp": (1, 0, "mice"),
    "hybrid_mice_imp": (1, 1, "mice"),
}

rows = []
for strat, (imp, flg, impu) in CELLS.items():
    src = hym if strat.startswith("hybrid") else v4m
    d = src[src.strategy == strat]
    for _, r in d.iterrows():
        rows.append({"strategy": strat, "model": r["model"],
                     "imp": imp, "flags": flg, "imputer": impu,
                     "fusion": "fixed" if r["model"] in FIXED else "learned",
                     "macro_auc": r["macro_auc"]})
df = pd.DataFrame(rows)
df.to_csv(OUT / "factorial_design_cells.tsv", sep="\t", index=False)

def _design(d, terms):
    base = {"imp": d["imp"].values.astype(float),
            "flags": d["flags"].values.astype(float),
            "fusion": (d["fusion"] == "learned").astype(float).values}
    cols = [np.ones(len(d))]
    for t in terms:
        v = np.ones(len(d))
        for part in t.split(":"):
            v = v * base[part]
        cols.append(v)
    return np.column_stack(cols)

_full = ["imp", "flags", "fusion", "imp:flags", "imp:fusion",
         "flags:fusion", "imp:flags:fusion"]
y = df.macro_auc.values
X = _design(df, _full)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
rss_full = float(((y - X @ beta) ** 2).sum())
df_resid = len(y) - np.linalg.matrix_rank(X)
rows = []
for term in _full:
    keep = [t for t in _full if t != term]
    Xr = _design(df, keep)
    br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
    rss_r = float(((y - Xr @ br) ** 2).sum())
    df_diff = np.linalg.matrix_rank(X) - np.linalg.matrix_rank(Xr)
    F = ((rss_r - rss_full) / max(1, df_diff)) / (rss_full / df_resid)
    p = float(_st.f.sf(F, max(1, df_diff), df_resid))
    rows.append({"term": term, "df": int(df_diff), "F": F, "p": p,
                 "partial_eta_sq": (rss_r - rss_full) / (rss_r + rss_full) if (rss_r + rss_full) else np.nan})
anova = pd.DataFrame(rows).set_index("term")
anova.to_csv(OUT / "factorial_anova.tsv", sep="\t")
print("=== ANOVA (partial F-tests, 2x2x2; imputer replicates averaged) ===")
print(anova.round(5).to_string())

piv = df.pivot_table(index="model", columns="strategy",
                     values="macro_auc")
piv.to_csv(OUT / "factorial_pivot.tsv", sep="\t")
print("\n=== cell means ===")
print(piv.round(4).to_string())

contr = []
for model_name in sorted(df.model.unique()):
    def g(strat):
        return piv.loc[model_name, strat]
    contr.append({
        "model": model_name,
        "d_flags_vs_p_only": g("avail_flags") - g("p_only"),
        "d_imp_vs_p_only_mean": g("mean_imp") - g("p_only"),
        "d_hybrid_flags_gain_mean": g("hybrid_mean_imp") - g("mean_imp"),
        "d_hybrid_flags_gain_mice": g("hybrid_mice_imp") - g("mice_imp"),
        "d_hybrid_vs_avail_mean": g("hybrid_mean_imp") - g("avail_flags"),
        "d_hybrid_vs_avail_mice": g("hybrid_mice_imp") - g("avail_flags"),
    })
ct = pd.DataFrame(contr)
ct.to_csv(OUT / "factorial_contrasts.tsv", sep="\t", index=False)
print("\n=== contrasts ===")
print(ct.round(4).to_string(index=False))

tests = []
for col in ["d_hybrid_flags_gain_mean", "d_hybrid_flags_gain_mice",
            "d_hybrid_vs_avail_mean", "d_hybrid_vs_avail_mice"]:
    x = ct[col].values
    t, p = (np.nan, np.nan)
    if np.std(x) > 0:
        from scipy import stats
        t, p = stats.ttest_1samp(x, 0.0)
    tests.append({"contrast": col, "mean": np.mean(x), "sd": np.std(x, ddof=1),
                  "t": t, "p": p})
tt = pd.DataFrame(tests)
ok = tt.p.notna()
_p = tt.loc[ok, "p"].values.astype(float)
_n = len(_p)
if _n:
    _order = np.argsort(_p)
    _q = _p[_order] * _n / np.arange(1, _n + 1)
    _q = np.minimum.accumulate(_q[::-1])[::-1]
    _out = np.empty(_n)
    _out[_order] = np.minimum(_q, 1.0)
    tt.loc[ok, "p_bh"] = _out
tt.to_csv(OUT / "factorial_tests.tsv", sep="\t", index=False)
print("\n=== tests (BH-FDR across 4 contrast families) ===")
print(tt.round(4).to_string(index=False))

md = ["# Table S12. Factorial design over the strategy space (2x2x2, "
      "two imputer replicates)",
      "",
      "Factors: imputation (none vs fold-internal), availability flags "
      "(off vs true), fusion (fixed averaging vs learned). Four "
      "meta-learners; seeds averaged (3 for imputation/hybrid cells).",
      "",
      "## Cell means (macro-AUC)", "",
      piv.round(3).to_string(), "",
      "## Interaction ANOVA (type II)", "",
      anova.round(5).to_string(), "",
      "## Hybrid contrasts per model", "",
      ct.round(3).to_string(index=False), "",
      "```",
      tt.round(4).to_string(index=False),
      "```", ""]
(OUT / "TableS12_factorial_strategy_design.md").write_text("\n".join(md))
print("\nsaved to", OUT)
