#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract Schaefer-100 FC features for one ds005892 subject (after AFNI
preprocessing). Produces full and partial correlation matrices plus the
190 network summaries (canonical 19 sub-network mapping), matching the main
cohort's feature definition (Fisher-z partial correlations).
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018
from sklearn.covariance import LedoitWolf

SUBJ = sys.argv[1]
WORK = Path(f"/mnt/limbic/ds005892_work/{SUBJ}")
OUT = Path("/mnt/limbic/ds005892_work/features")
OUT.mkdir(parents=True, exist_ok=True)

atlas = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17)
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True,
                           detrend=False,
                           resampling_target="data")
ts = masker.fit_transform(str(WORK / "bold_clean.nii.gz"))  # (T, 100)
ts = ts.T  # (100, T)

# full correlation -> Fisher z
C = np.corrcoef(ts)
Z = np.arctanh(np.clip(C, -0.999999, 0.999999))
np.fill_diagonal(Z, 0.0)

# partial correlation via Ledoit-Wolf precision -> Fisher z
lw = LedoitWolf().fit(ts.T)
P = lw.precision_
d = np.sqrt(np.diag(P))
pc = -P / np.outer(d, d)
np.fill_diagonal(pc, 0.0)
Zp = np.arctanh(np.clip(pc, -0.999999, 0.999999))
np.fill_diagonal(Zp, 0.0)

# canonical 19-sub-network labels (same mapping as the main cohort)
dir_dat = Path("/home/suleyman/nilearn_data/schaefer_2018")
labels_file = dir_dat / "Schaefer2018_100Parcels_17Networks_order.txt"
raw = [l.split()[1] for l in open(labels_file).read().strip().split("\n")]


def canon_of(name):
    if "VisCent" in name:
        return "Visual Central"
    if "VisPeri" in name:
        return "Visual Peripheral"
    if "SomMotA" in name:
        return "Somatomotor A"
    if "SomMotB_Aud" in name:
        return "Somatomotor B Auditory"
    if "SomMotB" in name:
        return "Somatomotor B"
    if "DorsAttnA" in name:
        return "Dorsal Attention A"
    if "DorsAttnB" in name:
        return "Dorsal Attention B"
    if "SalVentAttnA" in name:
        return "Salience Ventral Attention A"
    if "SalVentAttnB" in name:
        return "Salience Ventral Attention B"
    if "LimbicA" in name:
        return "Limbic Temporopolar"
    if "LimbicB" in name:
        return "Limbic Orbitofrontal"
    if "ContA" in name:
        return "Frontoparietal Control A"
    if "ContB" in name:
        return "Frontoparietal Control B"
    if "ContC" in name:
        return "Frontoparietal Control C"
    if "DefaultA" in name:
        return "Default A"
    if "DefaultB" in name:
        return "Default B"
    if "DefaultC" in name:
        return "Default C"
    if "TempPar" in name:
        return "Temporoparietal"
    return "??"


labels = np.array([canon_of(n) for n in raw])
assert len(labels) == 100 and "??" not in labels
nets = sorted(set(labels))
tri = np.triu_indices(100, k=1)


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


s_full = summaries(Z)
s_part = summaries(Zp)
df = pd.DataFrame([{"SampleID": SUBJ, **{f"full_{k}": v for k, v in
                                         s_full.items()},
                    **{f"part_{k}": v for k, v in s_part.items()}}])
df.to_csv(OUT / f"{SUBJ}_networkfeatures.tsv", sep="\t", index=False)
np.savez_compressed(OUT / f"{SUBJ}_matrices.npz", z_full=Z,
                    z_partial=Zp, labels=labels)
print(f"[{SUBJ}] features saved: {df.shape}")
