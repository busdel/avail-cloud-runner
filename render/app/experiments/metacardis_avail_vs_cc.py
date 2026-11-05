#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MetaCardis AVAIL vs complete-case comparison (Paper 2 deployment suite).

Run once metacardis_results/ (n=1087) and metacardis_cc/out/ (n=536) are
synced locally. Writes supplement/TableS14_metacardis_avail_vs_cc.md and a
TSV next to it.
"""
from pathlib import Path

import pandas as pd

B = Path("/mnt/limbic/AVAIL_GPU")
SUPP = B / "revizyon/makaleler/supplement"

FULL = B / "experiments/results/metacardis_results/meta_benchmark_results.tsv"
CC = B / "experiments/results/metacardis_cc/out/meta_benchmark_results.tsv"


def load(p):
    d = pd.read_csv(p, sep="\t")
    return d[d.subgroup == "all"].set_index("model")["macro_auc"]


full = load(FULL)
cc = load(CC)
common = sorted(set(full.index) & set(cc.index))

rows = []
for m in common:
    rows.append({"model": m, "avail_n1087": round(full[m], 4),
                 "complete_case_n536": round(cc[m], 4),
                 "delta": round(full[m] - cc[m], 4)})
df = pd.DataFrame(rows).sort_values("avail_n1087", ascending=False)
df.to_csv(SUPP / "TableS14_sources_metacardis.tsv", sep="\t", index=False)

summary_avail = full[common].mean()
summary_cc = cc[common].mean()

md = f"""# Table S14. MetaCardis: availability-aware vs complete-case arm

Same pipeline, same seeds; complete-case restricts to the 536 subjects
with all seven modalities (discards 51% of the cohort; LVEF measured in
84% of IHD vs 42% of controls, chi2 p = 9e-41).

| Arm | n | zoo-mean macro-AUC | best model |
|---|---|---|---|
| AVAIL (flags) | 1087 | {summary_avail:.3f} | {full[common].idxmax()} {full[common].max():.3f} |
| Complete-case | 536 | {summary_cc:.3f} | {cc[common].idxmax()} {cc[common].max():.3f} |

## Per-model

{df.to_markdown(index=False)}
"""
(SUPP / "TableS14_metacardis_avail_vs_cc.md").write_text(md)
print(df.to_string(index=False))
print(f"zoo mean: AVAIL {summary_avail:.4f} vs CC {summary_cc:.4f}")
