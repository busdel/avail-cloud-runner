#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract Schaefer-100 FC features for one subject (AFNI-preprocessed).
Generalized for ds005892 / ds004392 layouts.

Usage: rsfmri_extract_fc.py <dataset> <subject>
Env: IMG_WORK (work root), NILEARN_DATA (atlas cache)
"""
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.covariance import LedoitWolf

DS, SUBJ = sys.argv[1], sys.argv[2]
WORKROOT = Path(os.environ.get("IMG_WORK", "/mnt/limbic/ds_work"))
WORK = WORKROOT / DS / SUBJ
OUT = WORKROOT / "features"
OUT.mkdir(parents=True, exist_ok=True)

atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True,
                           detrend=False, resampling_target="data")
ts = masker.fit_transform(str(WORK / "bold_clean.nii.gz")).T  # (100, T)
ts = np.nan_to_num(np.asarray(ts, dtype=np.float64), nan=0.0,
                   posinf=0.0, neginf=0.0)
const = np.nanstd(ts, axis=1) < 1e-12
if const.any():
    ts[const] = 0.0

C = np.corrcoef(ts)
if not np.isfinite(C).all():
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 1.0)
Z = np.arctanh(np.clip(C, -0.999999, 0.999999))
np.fill_diagonal(Z, 0.0)

lw = LedoitWolf().fit(ts.T)
P = lw.precision_
d = np.sqrt(np.diag(P))
pc = -P / np.outer(d, d)
np.fill_diagonal(pc, 0.0)
Zp = np.arctanh(np.clip(pc, -0.999999, 0.999999))
np.fill_diagonal(Zp, 0.0)

import nilearn
order_txt = Path(atlas.maps).parent / "Schaefer2018_100Parcels_17Networks_order.txt"
raw = [l.split()[1] for l in order_txt.read_text().strip().split("\n")]


def canon_of(name):
    pairs = [("VisCent", "Visual Central"), ("VisPeri", "Visual Peripheral"),
             ("SomMotA", "Somatomotor A"),
             ("SomMotB_Aud", "Somatomotor B Auditory"),
             ("SomMotB", "Somatomotor B"),
             ("DorsAttnA", "Dorsal Attention A"),
             ("DorsAttnB", "Dorsal Attention B"),
             ("SalVentAttnA", "Salience Ventral Attention A"),
             ("SalVentAttnB", "Salience Ventral Attention B"),
             ("LimbicA", "Limbic Temporopolar"),
             ("LimbicB", "Limbic Orbitofrontal"),
             ("ContA", "Frontoparietal Control A"),
             ("ContB", "Frontoparietal Control B"),
             ("ContC", "Frontoparietal Control C"),
             ("DefaultA", "Default A"), ("DefaultB", "Default B"),
             ("DefaultC", "Default C"), ("TempPar", "Temporoparietal")]
    for k, v in pairs:
        if k in name:
            return v
    return "??"


labels = np.array([canon_of(n) for n in raw])
assert len(labels) == 100 and "??" not in labels
nets = sorted(set(labels))


def summaries(Zm):
    rows = {}
    for nn in nets:
        idx = np.where(labels == nn)[0]
        up = Zm[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
        rows[f"within_{nn}"] = np.mean(up) if len(up) else np.nan
    for i, a in enumerate(nets):
        for b in nets[i + 1:]:
            ia, ib = np.where(labels == a)[0], np.where(labels == b)[0]
            rows[f"between_{a}__{b}"] = Zm[np.ix_(ia, ib)].mean()
    return rows


df = pd.DataFrame([{"SampleID": SUBJ,
                    **{f"full_{k}": v for k, v in summaries(Z).items()},
                    **{f"part_{k}": v for k, v in summaries(Zp).items()}}])
df.to_csv(OUT / f"{DS}_{SUBJ}_networkfeatures.tsv", sep="\t", index=False)
np.savez_compressed(OUT / f"{DS}_{SUBJ}_matrices.npz", z_full=Z,
                    z_partial=Zp, labels=labels)
print(f"[{DS}/{SUBJ}] features saved: {df.shape}")
