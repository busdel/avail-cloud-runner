# -*- coding: utf-8 -*-
"""AVAIL imaging module: DICOM/NIfTI -> brain ROI features.

Pipeline per subject:
  1. dcm2niix (DICOM->NIfTI) or direct NIfTI
  2. Schaefer-100 parcellation -> ROI BOLD time series
  3. classical layer: Pearson functional-connectivity matrix (flattened);
     NaN/constant-parcel guard applied before correlation
  4. TimesFM-3 layer: zero-shot forecast representations per ROI
  5. optional: radiomics_features() -> whole-brain PyRadiomics features
     (avail-omics[radiomics] tier; bundled Schaefer atlas fallback)

Outputs under <data_root>/features/:
  imaging_classical.tsv   subjects x FC features
  imaging_timesfm.tsv     subjects x (100 ROIs x 32 horizon)
  roi_series.npy          per-subject ROI series (for later use)
Requires <data_root>/metadata.tsv (SampleID + Group) to match subjects.
"""
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("AVAIL_WEB_ROOT", "/media/suleyman_disk/AVAIL_GPU")).resolve()
PY = os.environ.get("AVAIL_PYTHON") or "python3"
DCM2NIIX = (os.environ.get("DCM2NIIX")
           or shutil.which("dcm2niix")
           or "/root/venv/bin/dcm2niix")
N_ROIS = 100
HORIZON = 32
ATLAS_CANDIDATES = [
    BASE / "atlas" / "schaefer_2018",
    Path("/media/suleyman_disk/AVAIL_GPU/atlas/schaefer_2018"),
    Path.home() / "nilearn_data" / "schaefer_2018",
]
RADIOMICS_CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "gldm",
                     "ngtdm", "shape"]

_tfm = None
_masker_obj = None


def log(msg):
    print(f"[imaging] {msg}", flush=True)


def _atlas_path(resolution_mm=2):
    """Local bundled atlas first (offline reproducibility), then nilearn."""
    for d in ATLAS_CANDIDATES:
        p = d / (f"Schaefer2018_100Parcels_17Networks_order_"
                 f"FSLMNI152_{resolution_mm}mm.nii.gz")
        if p.exists():
            return p
    return None


def _get_masker():
    global _masker_obj
    if _masker_obj is None:
        from nilearn.maskers import NiftiLabelsMasker
        p = _atlas_path(2)
        if p is None:
            from nilearn.datasets import fetch_atlas_schaefer_2018
            p = fetch_atlas_schaefer_2018(n_rois=N_ROIS,
                                          resolution_mm=2).maps
        _masker_obj = NiftiLabelsMasker(labels_img=str(p),
                                        standardize=False)
    return _masker_obj


def sanitize_ts(ts):
    """NaN / constant-parcel guard (deployment hardening).

    A single empty or constant parcel can propagate NaN through the whole
    correlation matrix (and silently zero a modality downstream). Non-finite
    entries are set to zero and constant columns are zeroed; the number of
    zeroed parcels is returned for QA logging.
    """
    ts = np.nan_to_num(np.asarray(ts, dtype=np.float64), nan=0.0,
                       posinf=0.0, neginf=0.0)
    bad = np.nanstd(ts, axis=0) < 1e-12
    if bad.any():
        ts = ts.copy()
        ts[:, bad] = 0.0
    return ts, int(bad.sum())


def radiomics_available():
    try:
        import radiomics  # noqa: F401
        import SimpleITK  # noqa: F401
        return True
    except Exception:
        return False


def radiomics_features(image_path, binwidth=25.0, atlas_path=None):
    """Optional whole-brain PyRadiomics block (anatomical T1 or maps).

    Requires the radiomics tier: pip install "avail-omics[radiomics]"
    (SimpleITK + pyradiomics; pyradiomics builds on Python <= 3.12).
    Returns a dict of IBSI features computed inside the Schaefer-100 mask.
    """
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
    except Exception as e:
        raise RuntimeError(
            "radiomics extras not installed: pip install "
            "'avail-omics[radiomics]' (Python <=3.12 recommended)") from e
    img = sitk.ReadImage(str(image_path))
    ap = atlas_path or _atlas_path(1)
    if ap is None:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        ap = fetch_atlas_schaefer_2018(n_rois=N_ROIS, resolution_mm=1).maps
    atlas = sitk.ReadImage(str(ap))
    atlas = sitk.Resample(atlas, img, sitk.Transform(),
                          sitk.sitkNearestNeighbor, 0,
                          atlas.GetPixelID())
    arr = sitk.GetArrayFromImage(atlas)
    mask = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
    mask.CopyInformation(img)
    ex = featureextractor.RadiomicsFeatureExtractor(binWidth=binwidth,
                                                    label=1)
    ex.disableAllFeatures()
    for c in RADIOMICS_CLASSES:
        ex.enableFeatureClassByName(c)
    res = ex.execute(img, mask)
    return {k[len("original_"):]: float(v) for k, v in res.items()
            if k.startswith("original_")}


def timesfm_available():
    try:
        import timesfm  # noqa: F401
        return True
    except Exception:
        return False


def _timesfm():
    global _tfm
    if _tfm is None:
        import timesfm
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        log(f"loading TimesFM-3 (google/timesfm-3.0-pytorch) on {device} ...")
        _tfm = timesfm.TimesFM3Forecaster.from_pretrained(
            "google/timesfm-3.0-pytorch", device=device)
    return _tfm


def _run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd or BASE))


def convert_dicom(dicom_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([DCM2NIIX, "-z", "y", "-f", "img", "-o", str(out_dir), str(dicom_dir)])
    nii = sorted(out_dir.glob("*.nii.gz"))
    return nii[0] if nii else None


def roi_series(nii_path):
    import nibabel as nib
    img = nib.load(str(nii_path))
    ts = _get_masker().fit_transform(img)
    return ts  # (T, 100)


def classical_features(ts):
    fc = np.corrcoef(ts.T)
    iu = np.triu_indices(N_ROIS, k=1)
    return np.nan_to_num(fc[iu], nan=0.0, posinf=0.0, neginf=0.0)


def timesfm_features(ts):
    tfm = _timesfm()
    feats = np.zeros((N_ROIS, HORIZON), dtype=np.float32)
    for r in range(N_ROIS):
        ctx = ts[:, r].astype(np.float32)
        if not np.isfinite(ctx).all():
            ctx = np.nan_to_num(ctx, nan=0.0)
        if float(np.std(ctx)) < 1e-9:
            continue  # constant ROI -> keep zero features
        try:
            out = tfm.predict(ctx, horizon=HORIZON)
            feats[r] = out.forecast.astype(np.float32)
        except Exception:
            feats[r] = 0.0
    return np.nan_to_num(feats.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)


def detect_imaging(data_root):
    for ext in ("*.dcm", "*.nii", "*.nii.gz", "*.ima"):
        if list(Path(data_root).rglob(ext)):
            return True
    return False


def collect_subjects(data_root):
    """Map subject id -> list of paths (nii files or dicom dirs)."""
    root = Path(data_root)
    subjects = {}
    # nifti files named <id>.nii[.gz]
    for p in root.rglob("*.nii.gz"):
        subjects.setdefault(p.stem.split(".")[0], []).append(p)
    for p in root.rglob("*.nii"):
        subjects.setdefault(p.stem, []).append(p)
    # dicom directories: one folder per subject
    for p in root.rglob("*.dcm"):
        d = p.parent
        if len(subjects.get(d.name, [])) == 0 and not any(
                str(x).endswith((".nii", ".nii.gz")) for x in subjects.values()):
            subjects.setdefault(d.name, []).append(d)
    # unique: dicom dirs handled separately below
    return subjects


def process_imaging(data_root, out_features_dir):
    """Run the imaging pipeline and write feature tables. Returns summary."""
    root = Path(data_root)
    feats_dir = Path(out_features_dir)
    feats_dir.mkdir(parents=True, exist_ok=True)

    meta_path = root / "metadata.tsv"
    if not meta_path.exists():
        meta_path = root / "metadata.csv"
    import pandas as pd
    sep = "\t" if str(meta_path).endswith(".tsv") else ","
    meta = pd.read_csv(meta_path, sep=sep)
    id_col = "SampleID" if "SampleID" in meta.columns else meta.columns[0]
    meta[id_col] = meta[id_col].astype(str).str.strip()
    ids = meta[id_col].tolist()

    # collect subject data
    nii_by_subj = {}
    for p in root.rglob("*.nii.gz"):
        nii_by_subj.setdefault(p.stem.split(".")[0], p)
    for p in root.rglob("*.nii"):
        nii_by_subj.setdefault(p.stem, p)
    dicom_dirs = {}
    for p in root.rglob("*.dcm"):
        dicom_dirs.setdefault(p.parent.name, p.parent)

    tmp = root / "_imaging_tmp"
    tmp.mkdir(exist_ok=True)

    have_tfm = timesfm_available()
    if not have_tfm:
        log("TimesFM layer unavailable (pip install 'avail-omics[timesfm]'); "
            "skipping that representation layer")

    rows_cl, rows_tf, series = {}, {}, {}
    matched = 0
    for sid in ids:
        nii = nii_by_subj.get(sid)
        if nii is None and sid in dicom_dirs:
            nii = convert_dicom(dicom_dirs[sid], tmp / sid)
        if nii is None:
            log(f"no imaging data for subject {sid} (skipped)")
            continue
        try:
            ts = roi_series(nii)
        except Exception as e:
            log(f"parcellation failed for {sid}: {e}")
            continue
        ts, nbad = sanitize_ts(ts)
        if nbad:
            log(f"subject {sid}: {nbad} empty/constant parcels zeroed")
        rows_cl[sid] = classical_features(ts)
        if have_tfm:
            try:
                rows_tf[sid] = timesfm_features(ts)
            except Exception as e:
                log(f"{sid}: TimesFM layer failed ({e})")
        series[sid] = ts
        matched += 1
        log(f"subject {sid}: ROI series {ts.shape} done ({matched})")

    if matched == 0:
        raise RuntimeError("no subject could be processed from the imaging data")

    np.save(feats_dir / "roi_series.npy", series, allow_pickle=True)

    def write(fname, rows, nfeat):
        with open(feats_dir / fname, "w") as f:
            f.write("SampleID\t" + "\t".join(f"{fname.split('.')[0]}_{i}"
                                             for i in range(nfeat)) + "\n")
            for sid in ids:
                if sid not in rows:
                    continue
                f.write(sid + "\t" + "\t".join(
                    f"{v:.6f}" for v in rows[sid]) + "\n")

    write("imaging_classical.tsv", rows_cl, (N_ROIS * (N_ROIS - 1)) // 2)
    if rows_tf:
        write("imaging_timesfm.tsv", rows_tf, N_ROIS * HORIZON)
    shutil.rmtree(tmp, ignore_errors=True)
    return {"subjects": matched, "n_rois": N_ROIS,
            "classical_features": (N_ROIS * (N_ROIS - 1)) // 2,
            "timesfm_features": (N_ROIS * HORIZON) if rows_tf else 0}


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(data_root) / "features")
    s = process_imaging(data_root, out)
    log(f"DONE {s}")
