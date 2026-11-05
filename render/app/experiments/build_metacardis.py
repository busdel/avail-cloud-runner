#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the MetaCardis AVAIL generic dataset (external validation #2).

Modalities:
  taxa        : gut metagenome species (ST10)
  kegg        : gut metagenome GMM/KEGG pathways (ST11)
  serum       : serum metabolome, log-transformed (ST12)
  urine       : urine NMR metabolites (ST13)
  clin_basic  : age, BMI, waist, TG, adiponectin, CRP, HbA1c, SBP/DBP, sex,
                diabetic status, physical activity (sporadic missing)
  clin_lvef   : left ventricular ejection fraction (available for 613/1087;
                GENUINE informative missingness - measured mostly in IHD)
  clin_proanp : fasting plasma pro-ANP (944/1087)

Label: IHD (372) vs CTRL (715). Output: generic layout
  metacardis/avail_dataset/{metadata.tsv, features/*.tsv}
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/mnt/limbic/datasets/metacardis")
XLSX = BASE / "moesm3_springer.xlsx"
OUT = BASE / "avail_dataset"
(OUT / "features").mkdir(parents=True, exist_ok=True)

# ---- metadata + clinical ----
meta = pd.read_excel(XLSX, sheet_name="ST9", header=1).dropna(subset=["ID"])
u = meta.drop_duplicates("ID").set_index("ID")

IHD_SETS = {"IHD372", "CIHD158", "ACS112", "HF102"}
gmap = meta.groupby("ID")["Status"].apply(set)
lab = gmap.apply(lambda s: "IHD" if len(s & IHD_SETS) else "CTRL")
print("labels:", lab.value_counts().to_dict())

sex = u["Gender"].map({"Male": 1, "Female": 0}).astype(float)
md = pd.DataFrame({
    "SampleID": u.index,
    "Group": lab.loc[u.index].values,
    "Age": u["Age (years)"].astype(float).values,
    "Sex": sex.values,
})
md.to_csv(OUT / "metadata.tsv", sep="\t", index=False)
print("metadata:", md.shape, md["Group"].value_counts().to_dict())


def write_feat(df, name):
    df = df.copy()
    df.index.name = "SampleID"
    df.to_csv(OUT / "features" / f"{name}.tsv", sep="\t")
    print(f"{name}: {df.shape[0]} x {df.shape[1]}")


# ---- omics blocks (already deduped + ordered in saved files) ----
for src, name in [("meta_taxa.tsv.gz", "taxa"),
                  ("meta_kegg.tsv.gz", "kegg"),
                  ("serum_values.tsv.gz", "serum"),
                  ("meta_urine.tsv.gz", "urine")]:
    d = pd.read_csv(BASE / src, sep="\t", index_col=0)
    if "Status" in d.columns:
        d = d.drop(columns=["Status"])
    d = d.loc[u.index]
    write_feat(d.astype(float), name)

# ---- clinical blocks ----
cb = pd.DataFrame({
    "BMI": pd.to_numeric(u["BMI (kg/m²)"], errors="coerce"),
    "Waist": pd.to_numeric(u["Waist circumference (cm)"], errors="coerce"),
    "TG": pd.to_numeric(u["Fasting plasma triglycerides (mmol/L)"],
                        errors="coerce"),
    "Adiponectin": pd.to_numeric(u["Fasting plasma adiponectin (mg/L)"],
                                 errors="coerce"),
    "CRP": pd.to_numeric(u["Fasting plasma CRP (mg/L)"], errors="coerce"),
    "HbA1c": pd.to_numeric(u["HbA1c (%)"], errors="coerce"),
    "SBP": pd.to_numeric(u["Systolic blood pressure (mmHg)"], errors="coerce"),
    "DBP": pd.to_numeric(u["Diastolic blood pressure (mmHg)"], errors="coerce"),
    "Diabetic": u["Diabetic status"].map({"Yes": 1, "No": 0}).astype(float),
    "Activity": pd.to_numeric(u["Physical activity (h/week)"],
                              errors="coerce"),
})
cb.index.name = "SampleID"
cb.to_csv(OUT / "features" / "clin_basic.tsv", sep="\t")
print("clin_basic:", cb.shape,
      "missing:", int(cb.isna().sum().sum()))

lv = pd.to_numeric(u["Left ventricular ejection fraction (%)"],
                   errors="coerce")
lv = lv.dropna()
lvdf = pd.DataFrame({"LVEF": lv.values}, index=lv.index)
lvdf.index.name = "SampleID"
lvdf.to_csv(OUT / "features" / "clin_lvef.tsv", sep="\t")
print("clin_lvef available:", lvdf.shape[0], "of", len(u))

pa = pd.to_numeric(u["Fasting plasma pro-ANP (pmol/L)"], errors="coerce")
pa = pa.dropna()
padf = pd.DataFrame({"proANP": pa.values}, index=pa.index)
padf.index.name = "SampleID"
padf.to_csv(OUT / "features" / "clin_proanp.tsv", sep="\t")
print("clin_proanp available:", padf.shape[0], "of", len(u))

print("\nDONE ->", OUT)
