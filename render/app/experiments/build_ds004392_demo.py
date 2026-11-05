#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the ds004392 real-cohort demo excerpt (OpenNeuro, CC0).

Reads AFNI/nilearn-derived Schaefer-100 summaries + phenotype files and
writes a ready-to-upload example_*/ zip for the AVAIL web demo.
Exits 0 only when the zip is (re)built; exits 3 when features are not yet
complete (used by the auto-watcher).
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-root", default="/media/suleyman_disk/datasets/ds004392")
    ap.add_argument("--features-dir", default="/media/suleyman_disk/ds_work/features")
    ap.add_argument("--out", default="/media/suleyman_disk/AVAIL_GPU/data_demos/ds004392_excerpt.zip")
    ap.add_argument("--min-subjects", type=int, default=58)
    args = ap.parse_args()

    ds_root = Path(args.ds_root)
    fdir = Path(args.features_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fs = sorted(fdir.glob("ds004392_*_networkfeatures.tsv"))
    subs = [f.name.split("_", 2)[1] for f in fs]
    if len(subs) < args.min_subjects:
        print(f"[demo-build] only {len(subs)} subjects ready, need {args.min_subjects}")
        return 3

    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    part["participant_id"] = part["participant_id"].astype(str)
    dx = pd.read_csv(ds_root / "phenotype" / "dx_deidentified.tsv", sep="\t")
    dx["participant_id"] = dx["participant_id"].astype(str)
    gmap = dict(zip(dx["participant_id"], dx["Dx"].astype(str)))
    hy = pd.read_csv(ds_root / "phenotype" / "hy_deidentified.tsv", sep="\t")
    hy["participant_id"] = hy["participant_id"].astype(str)
    cog = pd.read_csv(ds_root / "phenotype" / "cognitive_domains.tsv", sep="\t")
    cog["participant_id"] = cog["participant_id"].astype(str)
    tst = pd.read_csv(ds_root / "phenotype" / "cognitive_tests.tsv", sep="\t")
    tst["participant_id"] = tst["participant_id"].astype(str)

    meta = pd.DataFrame({"SampleID": part["participant_id"]})
    meta["Group"] = meta["SampleID"].map(gmap).replace(
        {"PD normal cognition": "PD-NC"})
    meta["Age"] = pd.to_numeric(part.get("age"), errors="coerce").round(1)
    sex = pd.to_numeric(part.get("sex"), errors="coerce")
    meta["Sex"] = sex.map({1.0: "M", 0.0: "F"})
    meta["Education"] = pd.to_numeric(part.get("years_of_education"),
                                      errors="coerce")
    meta = meta.dropna(subset=["Group"]).reset_index(drop=True)
    idx = set(meta["SampleID"])

    full, partf = {}, {}
    for f in fs:
        sub = f.name.split("_", 2)[1]
        if sub not in idx:
            continue
        d = pd.read_csv(f, sep="\t").iloc[0].to_dict()
        full[sub] = {k: v for k, v in d.items() if k.startswith("full_")}
        partf[sub] = {k: v for k, v in d.items() if k.startswith("part_")}

    def as_df(rows, order):
        d = pd.DataFrame.from_dict(rows, orient="index")
        d.index.name = "SampleID"
        d = d.reindex([s for s in order if s in d.index])
        return d.reset_index()

    order = sorted(full.keys())
    img_full = as_df(full, order)
    img_part = as_df(partf, order)

    def clin(df, order):
        d = df.rename(columns={"participant_id": "SampleID"})
        d = d[d["SampleID"].isin(order)].reset_index(drop=True)
        num = d.select_dtypes(include=[np.number]).columns
        d[num] = d[num].apply(lambda c: c.fillna(c.median()))
        return d

    clin_hy = clin(hy, order)
    clin_cog = clin(cog, order)
    clin_tst = clin(tst, order)
    meta = meta[meta["SampleID"].isin(order)].reset_index(drop=True)

    readme = """Real public cohort excerpt - OpenNeuro ds004392 (license CC0).
Source: https://openneuro.org/datasets/ds004392
Groups: PD-NC / PD-MCI / PDD (Parkinson's disease cognitive spectrum).
Features: Schaefer-100 connectivity summaries (full and partial
correlation, Fisher-z) computed with the AVAIL imaging adapter
(AFNI + nilearn); clinical block: Hoehn & Yahr stage, six cognitive
domain z-scores, and individual cognitive test scores.
Upload this folder (or its zip) to the AVAIL web platform, or run:
  avail-omics run-generic --data-root example_ds004392 --out results
This excerpt reproduces the ds004392 entry of the platform's external
validation section. Please cite the dataset:
https://doi.org/10.18112/openneuro.ds004392
"""

    root = "example_ds004392"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/metadata.tsv",
                    meta.to_csv(sep="\t", index=False))
        for name, d in [("img_full", img_full), ("img_partial", img_part),
                        ("clin_hy", clin_hy), ("clin_cog", clin_cog),
                        ("clin_tests", clin_tst)]:
            zf.writestr(f"{root}/features/{name}.tsv",
                        d.to_csv(sep="\t", index=False))
        zf.writestr(f"{root}/README.txt", readme)
    print(f"[demo-build] wrote {out} "
          f"({out.stat().st_size/1e6:.1f} MB, {len(order)} subjects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
