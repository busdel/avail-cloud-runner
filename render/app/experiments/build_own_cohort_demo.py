#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""De-identified demo excerpt from our own multi-site PD cohort.

- stratified sample (default 29 per class: HC / MCI / PDD)
- SampleIDs remapped to S001..; other metadata dropped by design
- real feature names kept (network summaries, 16S genera)
- shotgun msp-ID layers excluded (need taxonomy mapping; not in excerpt)
Requires explicit PI approval before any public distribution.
"""
import argparse
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path("/mnt/limbic/AVAIL_GPU")
SRC = BASE / "experiments/results/ladder_generic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "data_demos/own_cohort_excerpt.zip"))
    ap.add_argument("--per-class", type=int, default=29)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    meta = pd.read_csv(SRC / "metadata.tsv", sep="\t")
    parts = []
    for g in ["HC", "MCI", "PDD"]:
        sub = meta[meta.Group == g]
        k = min(a.per_class, len(sub))
        parts.append(sub.sample(k, random_state=a.seed))
    sel = pd.concat(parts).reset_index(drop=True)
    idmap = {str(old): f"S{i:03d}" for i, old in enumerate(sel.SampleID, 1)}
    sel["SampleID"] = sel.SampleID.astype(str).map(idmap)
    # public-safe quasi-identifiers: age = 10-year midpoint, education bins
    age = pd.to_numeric(sel["Age"], errors="coerce")
    sel["Age"] = ((age // 10) * 10 + 5).astype("Int64")
    edu = pd.to_numeric(sel["Education"], errors="coerce")
    sel["Education"] = edu.map(lambda v: 8 if v <= 8 else (10 if v <= 12 else 13))
    out_meta = sel[["SampleID", "Group", "Age", "Sex", "Education"]]

    mods = {"fmri_networkfc": "fMRI network connectivity",
            "stool16S": "stool 16S rRNA genera",
            "saliva16S": "saliva 16S rRNA genera"}
    feats = {}
    for name, label in mods.items():
        df = pd.read_csv(SRC / "features" / f"{name}.tsv", sep="\t")
        idc = df.columns[0]
        df[idc] = df[idc].astype(str).str.strip()
        df = df[df[idc].isin(idmap)]
        df[idc] = df[idc].map(idmap)
        df = df.rename(columns={idc: "SampleID"})
        feats[name] = df

    readme = (
        "REAL-COHORT EXCERPT - de-identified\n"
        "===================================\n\n"
        "Source: the multi-site Parkinson's disease cohort of our companion\n"
        "paper (HC / MCI / PDD). Participant identifiers were remapped to\n"
        "S001..; no direct identifiers are included. Values are the real\n"
        "measured features (no synthetic values).\n\n"
        "De-identification for public use: participant IDs remapped; age\n"
        "stored as 10-year midpoints; education stored in three bins\n"
        "(<=8, 9-12, >12 coded 8/10/13); no names, sites, dates or other\n"
        "direct/quasi identifiers are included.\n\n"
        f"Excerpt: up to {a.per_class} subjects per class "
        "(HC / MCI / PDD), stratified.\n\n"
        "Modalities: fMRI network-connectivity summaries (real Schaefer\n"
        "network names), stool 16S genera, saliva 16S genera (real genus\n"
        "names). Shotgun msp-ID layers are not part of this excerpt.\n\n"
        "Missing modalities reflect the original cohort (availability is\n"
        "structural); no imputation was applied to these tables.\n\n"
        "USAGE: shared for the AVAIL demo with the cohort PIs' permission.\n"
        "Do not redistribute outside the platform without approval.\n"
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("own_cohort/metadata.tsv",
                    out_meta.to_csv(sep="\t", index=False))
        for name, df in feats.items():
            zf.writestr(f"own_cohort/features/{name}.tsv",
                        df.to_csv(sep="\t", index=False))
        zf.writestr("own_cohort/compositional.txt", "stool16S\nsaliva16S\n")
        zf.writestr("own_cohort/README.txt", readme)
    print(f"[own-demo] wrote {out} "
          f"({out.stat().st_size/1e3:.0f} kB; n={len(out_meta)})")
    print(out_meta.Group.value_counts().to_dict())


if __name__ == "__main__":
    main()
