#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless agent evaluation #1: recommendation quality across cohorts.

For every cohort with a meta-benchmark table, compares the advisor's
recommended model (headless Advisor via the configured LLM) with the
benchmark-optimal non-degenerate model; reports agreement and regret.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))

E = ROOT / "experiments/results"
OUT = E / "agent_eval"
OUT.mkdir(parents=True, exist_ok=True)

COHORTS = [
    ("wallen", E / "wallen_avail_v2/meta_benchmark_results.tsv"),
    ("ad", E / "ad_avail/meta_benchmark_results.tsv"),
    ("hmp2", E / "hmp2_avail/meta_benchmark_results.tsv"),
    ("metacardis", E / "metacardis_results/meta_benchmark_results.tsv"),
    ("metacardis_cc", E / "metacardis_cc/out/meta_benchmark_results.tsv"),
    ("main_3class", ROOT / "avail_results/meta_benchmark_3class_results.tsv"),
]

from avail_omics.advisor import Advisor                      # noqa: E402
from avail_omics.llm import MultiLLMClient, build_clients    # noqa: E402

adv = Advisor(MultiLLMClient(build_clients()))
rows = []
for name, path in COHORTS:
    if not path.exists():
        print(f"[skip] {name}: no results file")
        continue
    df = pd.read_csv(path, sep="\t")
    sub = df[df.subgroup == "all"].copy()
    elig = sub[sub.macro_auc >= 0.50]
    elig_conv = elig[~elig.model.str.contains("GNN")]  # paper convention
    if elig.empty:
        print(f"[skip] {name}: no non-degenerate models")
        continue
    opt = elig.sort_values("macro_auc", ascending=False).iloc[0]
    opt_c = elig_conv.sort_values("macro_auc", ascending=False).iloc[0] \
        if len(elig_conv) else opt
    try:
        rec = adv.recommend_model(sub, primary_metric="macro_auc")
        m = str(rec.get("recommended_model", "")).strip()
    except Exception as e:
        m = f"ERROR:{type(e).__name__}"
        rec = {}
    row_ = sub[sub.model == m]
    rec_auc = float(row_.macro_auc.iloc[0]) if len(row_) else float("nan")
    rows.append({"cohort": name, "n": int(opt.n),
                 "optimal_model": opt.model,
                 "optimal_auc": round(float(opt.macro_auc), 4),
                 "agent_model": m,
                 "agent_auc": None if pd.isna(rec_auc) else round(rec_auc, 4),
                 "agreement": m == opt.model,
                 "optimal_model_conv": opt_c.model,
                 "optimal_auc_conv": round(float(opt_c.macro_auc), 4),
                 "agreement_conv": m == opt_c.model,
                 "regret": None if pd.isna(rec_auc)
                 else round(float(opt.macro_auc) - rec_auc, 4),
                 "runner_up": rec.get("runner_up", ""),
                 "caveat": (rec.get("caveat") or "")[:160]})
    print(f"[eval] {name}: optimal={opt.model} agent={m} "
          f"agree={m == opt.model}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "recommendation_quality.tsv", sep="\t", index=False)
ok = res[res.agent_auc.notna()]
print("\n=== ÖZET ===")
print(f"kohort: {len(res)} | öneri hatası yok: {len(ok)}")
print(f"örtüşme (tüm modeller): {int(ok.agreement.sum())}/{len(ok)} "
      f"({ok.agreement.mean():.0%})" if len(ok) else "no data")
print(f"örtüşme (makale konvansiyonu, GNN hariç): "
      f"{int(ok.agreement_conv.sum())}/{len(ok)} "
      f"({ok.agreement_conv.mean():.0%})" if len(ok) else "")
print(f"ortalama pişmanlık (AUC): {ok.regret.mean():.4f}" if len(ok) else "")
