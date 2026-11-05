#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate main-text Table 1 (cohort demographics + modality availability)."""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/mnt/limbic/AVAIL_GPU")
DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "revizyon/makaleler"
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
LABEL = {"fmri_roi": "Imaging (fMRI ROI connectivity)",
         "stool16S": "Stool 16S (genera)",
         "stoolSG": "Stool shotgun (MGS)",
         "saliva16S": "Saliva 16S (genera)",
         "salivaSG": "Saliva shotgun (MGS)"}

meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
meta["SampleID"] = meta["SampleID"].astype(str)
avail = {}
for m in MODULES:
    t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                    usecols=["SampleID"])
    avail[m] = set(t["SampleID"].astype(str))
for m in MODULES:
    meta[m] = meta["SampleID"].isin(avail[m])

groups = [("All", meta), ("HC", meta[meta.Group == "HC"]),
          ("MCI", meta[meta.Group == "MCI"]),
          ("PDD", meta[meta.Group == "PDD"])]

rows = []
rows.append(["N"] + [str(len(g)) for _, g in groups] + [""])

def fmt_ms(v):
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}"

age_p = stats.f_oneway(*[g["Age"] for _, g in groups[1:]]).pvalue
rows.append(["Age, years (mean ± SD)"]
            + [fmt_ms(g["Age"]) for _, g in groups] + [f"{age_p:.3f}"])

sex_tab = pd.crosstab(meta["Group"], meta["Sex"])
chi2, sex_p, _, _ = stats.chi2_contingency(sex_tab)
rows.append(["Male, n (%)"]
            + [f"{int(g['Sex'].sum())} ({100*g['Sex'].mean():.0f}%)"
               for _, g in groups] + [f"{sex_p:.3f}"])

edu_p = stats.f_oneway(*[g["Education"] for _, g in groups[1:]]).pvalue
rows.append(["Education, years (mean ± SD)"]
            + [fmt_ms(g["Education"]) for _, g in groups] + [f"{edu_p:.3f}"])

for m in MODULES:
    tab = pd.crosstab(meta["Group"], meta[m])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    rows.append([f"{LABEL[m]}, n (%)"]
                + [f"{int(g[m].sum())} ({100*g[m].mean():.0f}%)"
                   for _, g in groups] + [f"{p:.3f}"])

df = pd.DataFrame(rows, columns=["Characteristic", "All", "HC", "MCI",
                                 "PDD", "p-value"])
df.to_csv(OUT / "MAKALE1_Table1_cohort.tsv", sep="\t", index=False)

md = ["| Characteristic | All (n=150) | HC (n=29) | MCI (n=57) | "
      "PDD (n=64) | p |", "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append("| " + " | ".join(str(x) for x in r) + " |")
open(OUT / "MAKALE1_Table1_cohort.md", "w").write("\n".join(md))
print("\n".join(md))
print("\nwritten MAKALE1_Table1_cohort.tsv/.md")
