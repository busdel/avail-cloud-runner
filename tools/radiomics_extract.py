#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional PyRadiomics block for the imaging adapter (supervisor request).

Extracts IBSI-compliant radiomics features from the MNI-normalised T1w
volume using the Schaefer-100 (17-network) atlas as the mask:

  wb_*        whole-brain (union of parcels; firstorder+texture+shape)
  net_<name>_* per-network aggregates (firstorder+texture), --mode both

Usage:
  radiomics_extract.py --t1 t1_mni.nii.gz --out ds004392_sub-XXXX_radiomics.tsv \
                       [--mode wb|both] [--binwidth 25]

Env: RADIOMICS_PYTHON if a dedicated venv is needed (this script is
imported by the generic subject runner only when radiomics is enabled).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

ATLAS_CANDIDATES = [
    Path.home() / "nilearn_data" / "schaefer_2018",
    Path("/media/suleyman_disk/AVAIL_GPU/atlas/schaefer_2018"),
    Path("/mnt/limbic/AVAIL_GPU/atlas/schaefer_2018"),
]
ATLAS_DIR = next((d for d in ATLAS_CANDIDATES
                  if (d / "Schaefer2018_100Parcels_17Networks_order.txt").exists()),
                 ATLAS_CANDIDATES[0])
ATLAS_IMG = ATLAS_DIR / "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_1mm.nii.gz"
ATLAS_ORDER = ATLAS_DIR / "Schaefer2018_100Parcels_17Networks_order.txt"

PAIRS = [("VisCent", "Visual Central"), ("VisPeri", "Visual Peripheral"),
         ("SomMotA", "Somatomotor A"), ("SomMotB_Aud", "Somatomotor B Auditory"),
         ("SomMotB", "Somatomotor B"), ("DorsAttnA", "Dorsal Attention A"),
         ("DorsAttnB", "Dorsal Attention B"),
         ("SalVentAttnA", "Salience Ventral Attention A"),
         ("SalVentAttnB", "Salience Ventral Attention B"),
         ("LimbicA", "Limbic Temporopolar"), ("LimbicB", "Limbic Orbitofrontal"),
         ("ContA", "Frontoparietal Control A"),
         ("ContB", "Frontoparietal Control B"),
         ("ContC", "Frontoparietal Control C"),
         ("DefaultA", "Default A"), ("DefaultB", "Default B"),
         ("DefaultC", "Default C"), ("TempPar", "Temporoparietal")]


def canon(name):
    for k, v in PAIRS:
        if k in name:
            return v
    return None


def load_network_labels():
    labels = {}
    for line in ATLAS_ORDER.read_text().strip().split("\n"):
        idx, name = line.split()[:2]
        net = canon(name)
        if net:
            labels.setdefault(net, []).append(int(idx))
    return labels


def _afni(args, afni_bin):
    env = dict(os.environ)
    env["PATH"] = afni_bin + ":" + env.get("PATH", "")
    r = subprocess.run([str(Path(afni_bin) / args[0])] + list(args[1:]),
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"AFNI {args[0]} failed: {r.stderr[-300:]}")


def make_map_maps(bold, outdir, afni_bin, blur=None):
    """ALFF / fALFF / ReHo maps from preprocessed rs-fMRI volumes.

    bold_clean is already band-passed (0.01-0.10 Hz) by the pipeline, so
    ALFF = stdev(bold_clean) and fALFF = ALFF / stdev(bold_blur) where
    bold_blur is the pre-band-pass (but smoothed) volume.
    """
    bold = str(bold)
    if blur is None:
        blur = bold.replace("bold_clean", "bold_blur")
    small = outdir / "bold_small.nii.gz"
    if not small.exists():
        _afni(["3dresample", "-dxyz", "3", "3", "3", "-input", bold,
               "-prefix", str(small), "-overwrite"], afni_bin)
    blur_small = None
    if Path(blur).exists():
        blur_small = outdir / "blur_small.nii.gz"
        if not blur_small.exists():
            _afni(["3dresample", "-dxyz", "3", "3", "3", "-input", blur,
                   "-prefix", str(blur_small), "-overwrite"], afni_bin)
    out = {}
    alff = outdir / "alff.nii.gz"
    _afni(["3dTstat", "-stdev", "-prefix", str(alff), str(small),
           "-overwrite"], afni_bin)
    out["alff"] = alff
    if blur_small is not None:
        tot = outdir / "tot.nii.gz"
        _afni(["3dTstat", "-stdev", "-prefix", str(tot), str(blur_small),
               "-overwrite"], afni_bin)
        falff = outdir / "falff.nii.gz"
        _afni(["3dcalc", "-a", str(alff), "-b", str(tot), "-expr", "a/b",
               "-prefix", str(falff), "-overwrite"], afni_bin)
        out["falff"] = falff
    reho = outdir / "reho.nii.gz"
    _afni(["3dReHo", "-inset", str(small), "-prefix", str(reho),
           "-nneigh", "27", "-overwrite"], afni_bin)
    out["reho"] = reho
    return out


def robust_unit(img):
    """Clip to 2-98th percentile and rescale to [0,1] (map comparability)."""
    a = sitk.GetArrayFromImage(img).astype(np.float64)
    m = a > 0
    if m.sum() == 0:
        return img
    lo, hi = np.percentile(a[m], [2, 98])
    a = np.clip(a, lo, hi)
    a = (a - lo) / max(hi - lo, 1e-12)
    o = sitk.GetImageFromArray(a.astype(np.float32))
    o.CopyInformation(img)
    return o


def extract_maps(maps_arg, bold, map_dir, afni_bin, atlas_path, arr, ref_img,
                 binwidth):
    wanted = [m.strip().lower() for m in maps_arg.split(",") if m.strip()]
    present = [p.name for p in map_dir.glob("*.nii.gz")]
    need = [m for m in wanted if f"{m}.nii.gz" not in present]
    if need:
        make_map_maps(bold, map_dir, afni_bin)
    rows = {}
    atlas_img = sitk.ReadImage(str(atlas_path))
    classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    for m in wanted:
        f = map_dir / f"{m}.nii.gz"
        if not f.exists():
            continue
        raw = sitk.ReadImage(str(f))
        a_m = sitk.Resample(atlas_img, raw, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0,
                            atlas_img.GetPixelID())
        arr_m = sitk.GetArrayFromImage(a_m)
        raw_arr = sitk.GetArrayFromImage(raw)
        mask_arr = ((arr_m > 0) & (raw_arr > 0)).astype(np.uint8)
        if mask_arr.sum() < 512:
            continue
        mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(raw)
        img = robust_unit(raw)
        settings = {"binWidth": binwidth, "label": 1,
                    "resampledPixelSpacing": None, "correctMask": True}
        ex = featureextractor.RadiomicsFeatureExtractor(**settings)
        ex.disableAllFeatures()
        for c in classes:
            ex.enableFeatureClassByName(c)
        try:
            res = ex.execute(img, mask)
        except Exception:
            continue
        for k, v in res.items():
            if k.startswith("original_"):
                rows[f"{m}_wb_" + k[len("original_"):]] = float(v)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=str(ATLAS_IMG))
    ap.add_argument("--mode", choices=["wb", "both"], default="wb")
    ap.add_argument("--binwidth", type=float, default=25.0)
    ap.add_argument("--maps", default="",
                    help="comma list: alff,falff,reho (needs --bold)")
    ap.add_argument("--bold", default="",
                    help="preprocessed 4D rs-fMRI (bold_clean.nii.gz)")
    ap.add_argument("--afni-bin",
                    default=os.environ.get(
                        "AFNI_BIN",
                        "/media/suleyman_disk/tools/linux_ubuntu_16_64"))
    ap.add_argument("--map-binwidth", type=float, default=0.05)
    args = ap.parse_args()

    img = sitk.ReadImage(args.t1)
    atlas = sitk.ReadImage(args.atlas)
    atlas = sitk.Resample(atlas, img, sitk.Transform(), sitk.sitkNearestNeighbor,
                          0, atlas.GetPixelID())
    arr = sitk.GetArrayFromImage(atlas)

    classes = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    settings = {"binWidth": args.binwidth, "label": 1,
                "resampledPixelSpacing": None, "correctMask": True}
    rows = {}

    wb_mask = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    wb_mask.CopyInformation(img)
    ex_wb = featureextractor.RadiomicsFeatureExtractor(**settings)
    ex_wb.disableAllFeatures()
    for c in classes + ["shape"]:
        ex_wb.enableFeatureClassByName(c)
    res = ex_wb.execute(img, wb_mask)
    for k, v in res.items():
        if k.startswith("original_"):
            rows["wb_" + k[len("original_"):]] = float(v)

    if args.maps and args.bold:
        map_dir = Path(args.out).parent / (Path(args.out).stem + "_maps")
        map_dir.mkdir(parents=True, exist_ok=True)
        rows.update(extract_maps(args.maps, args.bold, map_dir,
                                 args.afni_bin, args.atlas, arr, img,
                                 args.map_binwidth))

    if args.mode == "both":
        labels = load_network_labels()
        ex_net = featureextractor.RadiomicsFeatureExtractor(**settings)
        ex_net.disableAllFeatures()
        for c in classes:
            ex_net.enableFeatureClassByName(c)
        for net, labs in sorted(labels.items()):
            m = np.isin(arr, labs).astype(np.uint8)
            if m.sum() < 64:
                continue
            mask = sitk.GetImageFromArray(m)
            mask.CopyInformation(img)
            try:
                r = ex_net.execute(img, mask)
            except Exception:
                continue
            pref = "net_" + re.sub(r"[^A-Za-z0-9]+", "", net.capitalize()) + "_"
            for k, v in r.items():
                if k.startswith("original_"):
                    rows[pref + k[len("original_"):]] = float(v)

    df = pd.DataFrame([rows])
    df.insert(0, "SampleID", Path(args.t1).parent.name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"[radiomics] {args.t1} -> {args.out} ({df.shape[1]-1} features)")


if __name__ == "__main__":
    sys.exit(main())
