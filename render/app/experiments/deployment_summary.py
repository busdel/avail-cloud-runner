#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment summary for Paper 2, section 4.6 (run after overnight rsync).

Reads local results dirs for the three deployment cohorts and writes a
markdown block with the numbers, ready to paste into the draft.
"""
from pathlib import Path

import pandas as pd

B = Path("/mnt/limbic/AVAIL_GPU")
R = B / "experiments/results"
OUT = B / "revizyon/makaleler/supplement/TableS15_deployment_summary.md"

COHORTS = [
    ("MetaCardis", R / "metacardis_results/meta_benchmark_results.tsv"),
    ("MetaCardis (complete-case)", R / "metacardis_cc/out/meta_benchmark_results.tsv"),
    ("ds004392 (PD spectrum)", R / "ds004392_results/meta_benchmark_results.tsv"),
    ("ds005892 (HC/PD-MCI)", R / "ds005892_results/meta_benchmark_results.tsv"),
]

lines = ["# Table S15. Deployment suite (protocol-level)", "",
         "Identical pipeline, no code changes, on genuinely incomplete or "
         "independent cohorts.", "",
         "| Cohort | n | zoo-mean macro-AUC | best (non-GNN) |",
         "|---|---|---|---|"]
for name, p in COHORTS:
    if not p.exists():
        lines.append(f"| {name} | - | pending | pending |")
        continue
    d = pd.read_csv(p, sep="\t")
    sub = d[(d.subgroup == "all") & (~d.model.str.contains("GNN"))]
    zoo = sub.macro_auc.mean()
    r = sub.sort_values("macro_auc", ascending=False).iloc[0]
    n = int(sub.n.max())
    lines.append(f"| {name} | {n} | {zoo:.3f} | {r.model} {r.macro_auc:.3f} |")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
