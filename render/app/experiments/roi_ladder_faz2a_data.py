#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 2a: Assemble the ladder generic dataset on the LOCAL machine.

Builds experiments/results/ladder_generic/ :
  metadata.tsv              150 patients: SampleID, Group, Age, Sex, Education
  features/fmri_roi.tsv     83 x 4950 edges (Fisher-z partial correlations)
  features/stool16S.tsv     112 x genera
  features/stoolSG.tsv      123 x MSPs (gut)
  features/saliva16S.tsv    125 x genera
  features/salivaSG.tsv     107 x MSPs (oral)

Labels from metadata.txt (HC/MCI/PDD). ROI sheets mapped to patients via
y_data.xlsx Sort Number order (validated in faz1). Covariates: Age/Sex/Edu
numeric, sex M=1/F=0.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/ladder_generic"
(OUT / "features").mkdir(parents=True, exist_ok=True)

ROI_XLSX = Path("/home/suleyman/Desktop/fMRI datasets/"
                "partial_corr_reshaped_Spe21-23.xlsx")
Y_F = BENCH / "fMRI correlation and feature selection/y_data.xlsx"
META_F = BENCH / "busra_pd_multisite/metadata.txt"
STOOL16 = BENCH / "busra_pd_multisite/stool_genus_p.txt"
SAL16 = BENCH / "busra_pd_multisite/saliva_genus_p.txt"
GUT_MGS = BENCH / "yeni_eklenenler/paper_pipeline/out/gut_mgs_abundance_freq.tsv"
ORAL_MGS = BENCH / "yeni_eklenenler/paper_pipeline/out/oral_mgs_abundance_freq.tsv"
CW = (BENCH / "yeni_eklenenler/paper_pipeline/DELIVERABLE/raw_outputs/"
      "err_patient_crosswalk.tsv")


def write_features(df, name, id_col="SampleID"):
    df = df.copy()
    df.insert(0, "SampleID", df.index.astype(str).str.strip())
    df.columns = [str(c) for c in df.columns]
    path = OUT / "features" / f"{name}.tsv"
    df.to_csv(path, sep="\t", index=False)
    print(f"[faz2a] {name}: {df.shape[0]} x {df.shape[1]-1} -> {path.name}")


def main():
    meta = pd.read_csv(META_F, sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    meta = meta[meta["Group"].isin(["HC", "MCI", "PDD"])].reset_index(drop=True)
    sex_num = meta["Sex"].map({"M": 1.0, "F": 0.0}).values
    out = pd.DataFrame({
        "SampleID": meta["SampleID"],
        "Group": meta["Group"],
        "Age": meta["Age"].astype(float),
        "Sex": sex_num,
        "Education": pd.to_numeric(meta["Education"], errors="coerce"),
    })
    out.to_csv(OUT / "metadata.tsv", sep="\t", index=False)
    print(f"[faz2a] metadata: {len(out)} patients "
          f"({out['Group'].value_counts().to_dict()})")

    # --- fMRI ROI: 83 x 4950 edges (upper triangle, Fisher z) ---
    y = pd.read_excel(Y_F).sort_values("Sort Number").reset_index(drop=True)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    xl = pd.ExcelFile(ROI_XLSX)
    tri = np.triu_indices(100, k=1)
    rows, cols = [], []
    for s in xl.sheet_names:
        m = xl.parse(s, header=None).values.astype(np.float64)
        rows.append(m[tri])
    roi = pd.DataFrame(np.vstack(rows),
                       index=y["PatientID"].values,
                       columns=[f"e{i}_{j}" for i, j in zip(*tri)])
    write_features(roi, "fmri_roi")

    # --- 16S genus tables (rows=genera, cols=patients) -> transpose ---
    for path, name in [(STOOL16, "stool16S"), (SAL16, "saliva16S")]:
        t = pd.read_csv(path, sep="\t", index_col=0)
        t.columns = [str(c).strip() for c in t.columns]
        t.index = [str(x).strip() for x in t.index]
        t = t.T
        t.index.name = "SampleID"
        write_features(t, name)

    # --- shotgun MGS: ERR cols -> patients via crosswalk ---
    cw = pd.read_csv(CW, sep="\t")
    cw["patient"] = cw["patient"].astype(str).str.strip()
    cw["run_accession"] = cw["run_accession"].astype(str).str.strip()
    for path, name, comp in [(GUT_MGS, "stoolSG", "G"),
                             (ORAL_MGS, "salivaSG", "S")]:
        t = pd.read_csv(path, sep="\t", index_col=0)
        t.columns = [str(c).strip() for c in t.columns]
        t.index = [str(x).strip() for x in t.index]
        sub = cw[(cw["compartment"] == comp)
                 & (cw["run_accession"].isin(t.columns))]
        sub = sub.drop_duplicates(subset="patient", keep="first")
        p = t[sub["run_accession"].tolist()]
        p.columns = sub["patient"].values
        p = p.T
        p.index.name = "SampleID"
        write_features(p, name)

    # --- coverage summary per rung (both arms) ---
    feat = {f.stem: pd.read_csv(f, sep="\t", usecols=["SampleID"])
            for f in (OUT / "features").glob("*.tsv")}
    ids = {k: set(v["SampleID"].astype(str)) for k, v in feat.items()}
    all_ids = set(out["SampleID"])
    order = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
    cur = set(all_ids)
    print("[faz2a] ladder coverage:")
    for m in order:
        have = ids[m] & all_ids
        cur = cur & have
        print(f"   +{m:10s} avail={len(have):3d}  complete-case so far={len(cur):3d}")


if __name__ == "__main__":
    main()
