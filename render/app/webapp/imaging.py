# -*- coding: utf-8 -*-
"""AVAIL imaging module: DICOM/NIfTI -> brain ROI features.

Pipeline per subject:
  1. dcm2niix (DICOM->NIfTI) or direct NIfTI
  2. Schaefer-100 parcellation -> ROI BOLD time series
  3. classical layer: Pearson functional-connectivity matrix (flattened)
  4. TimesFM-3 layer: zero-shot forecast representations per ROI

Outputs under <data_root>/features/:
  imaging_classical.tsv   subjects x FC features
  imaging_timesfm.tsv     subjects x (100 ROIs x 32 horizon)
  roi_series.npy          per-subject ROI series (for later use)
Requires <data_root>/metadata.tsv (SampleID + Group) to match subjects.
"""
import os
import re
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

_tfm = None
_masker_obj = None


def log(msg):
    print(f"[imaging] {msg}", flush=True)


ATLAS_CANDIDATES = [
    BASE / "atlas" / "schaefer_2018",
    Path("/media/suleyman_disk/AVAIL_GPU/atlas/schaefer_2018"),
    Path.home() / "nilearn_data" / "schaefer_2018",
]


def _atlas_path():
    for d in ATLAS_CANDIDATES:
        q = d / "Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"
        if q.exists():
            return q
    return None


def _get_masker():
    global _masker_obj
    if _masker_obj is None:
        from nilearn.maskers import NiftiLabelsMasker
        q = _atlas_path()
        if q is not None:
            order = Path(q).parent / "Schaefer2018_100Parcels_17Networks_order.txt"
            labels = [l.split()[1] for l in order.read_text().strip().split("\n")] \
                if order.exists() else []
        else:
            from nilearn.datasets import fetch_atlas_schaefer_2018
            atlas = fetch_atlas_schaefer_2018(n_rois=N_ROIS, yeo_networks=17,
                                              resolution_mm=2)
            q = atlas.maps
            labels = list(atlas.labels)
        _masker_obj = NiftiLabelsMasker(labels_img=str(q),
                                        standardize=True,
                                        detrend=False,
                                        resampling_target="data")
        _ATLAS["labels"] = labels
    return _masker_obj


_ATLAS = {}


def _networks_from_labels():
    """Schaefer 17-network labels: '17Networks_LH_VisCent_ExStr_1'
    -> network 'VisCent'. (First label is Background - skipped.)"""
    nets = []
    for lab in _ATLAS.get("labels", []):
        s = str(lab)
        if not s or s == "Background":
            continue
        parts = s.split("_")
        nets.append(parts[2] if len(parts) > 3 else "NA")
    if len(nets) != N_ROIS:
        nets = [f"n{i}" for i in range(N_ROIS)]
    return nets


def network_summaries(fc):
    """Within-network means (17) + between-network-pair means (136)."""
    nets = _networks_from_labels()
    uniq = sorted(set(nets))
    idx = {n: [i for i, v in enumerate(nets) if v == n] for n in uniq}
    within = np.array([np.mean(fc[np.ix_(idx[n], idx[n])]) for n in uniq])
    between = []
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            between.append(np.mean(fc[np.ix_(idx[uniq[a]], idx[uniq[b]])]))
    return np.concatenate([within, np.array(between)])


def alff_features(ts, tr=2.0, lo=0.01, hi=0.08):
    """Amplitude of low-frequency fluctuations per ROI (FFT bandpass)."""
    T = ts.shape[0]
    if T < 16:
        return np.zeros(N_ROIS)
    freqs = np.fft.rfftfreq(T, d=tr)
    spec = np.fft.rfft(ts, axis=0)
    band = np.where((freqs >= lo) & (freqs <= hi), 1.0, 0.0)
    return np.mean(np.abs(spec) * band[:, None], axis=0)


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


def sanitize_ts(ts):
    """NaN / constant-parcel guard (deployment hardening)."""
    ts = np.nan_to_num(np.asarray(ts, dtype=np.float64), nan=0.0,
                       posinf=0.0, neginf=0.0)
    bad = np.nanstd(ts, axis=0) < 1e-12
    if bad.any():
        ts = ts.copy()
        ts[:, bad] = 0.0
    return ts, int(bad.sum())


def timesfm_available():
    try:
        import timesfm  # noqa: F401
        return True
    except Exception:
        return False



NET_PAIRS = [("VisCent", "Visual Central"), ("VisPeri", "Visual Peripheral"),
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


def _canon_net(name):
    for k, v in NET_PAIRS:
        if k in name:
            return v
    return None


def _net_labels(p):
    order = Path(p).parent / "Schaefer2018_100Parcels_17Networks_order.txt"
    labels = []
    for line in order.read_text().strip().split("\n"):
        labels.append(_canon_net(line.split()[1]))
    return labels


def _net_summaries(mat, labels, nets):
    rows = {}
    for nn in nets:
        idx = np.where(np.array(labels) == nn)[0]
        up = mat[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
        rows["within_" + nn] = float(np.mean(up)) if len(up) else np.nan
    for i, a in enumerate(nets):
        for b in nets[i + 1:]:
            ia = np.where(np.array(labels) == a)[0]
            ib = np.where(np.array(labels) == b)[0]
            rows["between_" + a + "__" + b] = float(mat[np.ix_(ia, ib)].mean())
    return rows


def fc_protocol_features(ts):
    """Paper-protocol connectivity: full Pearson (Fisher-z) + partial
    (Ledoit-Wolf), within/between network summaries -> 171 + 171 features."""
    from sklearn.covariance import LedoitWolf
    q = _atlas_path()
    if q is None:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        q = fetch_atlas_schaefer_2018(n_rois=N_ROIS, yeo_networks=17,
                                      resolution_mm=2).maps
    labels = _net_labels(q)
    nets = sorted(set(labels))
    C = np.corrcoef(ts.T)
    if not np.isfinite(C).all():
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(C, 1.0)
    Z = np.arctanh(np.clip(C, -0.999999, 0.999999))
    np.fill_diagonal(Z, 0.0)
    lw = LedoitWolf().fit(ts)
    P = lw.precision_
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 0.0)
    Zp = np.arctanh(np.clip(pc, -0.999999, 0.999999))
    np.fill_diagonal(Zp, 0.0)
    rows = {f"full_{k}": v for k, v in _net_summaries(Z, labels, nets).items()}
    rows.update({f"part_{k}": v
                 for k, v in _net_summaries(Zp, labels, nets).items()})
    return {k: float(v) for k, v in rows.items()}

def classical_features(ts):
    fc = np.corrcoef(ts.T)
    iu = np.triu_indices(N_ROIS, k=1)
    return np.nan_to_num(
        np.concatenate([fc[iu], network_summaries(fc),
                        alff_features(ts)]),
        nan=0.0, posinf=0.0, neginf=0.0)


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
    all_nii = []
    for p in list(root.rglob("*.nii.gz")) + list(root.rglob("*.nii")):
        all_nii.append(p)
        nii_by_subj.setdefault(p.stem.split(".")[0], p)

    def _resolve_nii(sid):
        p = nii_by_subj.get(sid)
        if p is not None:
            return p
        for q in all_nii:
            qt = re.split(r"[^A-Za-z0-9\-]+", q.name)
            if sid in qt or sid in q.name:
                return q
        return None
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
        nii = _resolve_nii(sid)
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
            log(f"{sid}: {nbad} empty/constant parcels zeroed")
        rows_cl[sid] = fc_protocol_features(ts)
        if have_tfm:
            try:
                rows_tf[sid] = timesfm_features(ts)
            except Exception as e:
                log(f"{sid}: TimesFM layer failed ({e})")
        series[sid] = ts
        fc = np.corrcoef(ts.T)
        mat_dir = feats_dir / "matrices"
        mat_dir.mkdir(exist_ok=True)
        np.save(mat_dir / f"{sid}_fc_{N_ROIS}x{N_ROIS}.npy",
                np.nan_to_num(fc, nan=0.0))
        matched += 1
        log(f"subject {sid}: ROI series {ts.shape} done ({matched})")

    if matched == 0:
        raise RuntimeError("no subject could be processed from the imaging data")

    np.save(feats_dir / "roi_series.npy", series, allow_pickle=True)

    def write(fname, rows):
        header = None
        for sid in ids:
            if sid in rows:
                header = (list(rows[sid].keys()) if isinstance(rows[sid], dict)
                          else [f"{fname.split('.')[0]}_{i}"
                                for i in range(len(rows[sid]))])
                break
        with open(feats_dir / fname, "w") as f:
            f.write("SampleID\t" + "\t".join(header) + "\n")
            for sid in ids:
                if sid not in rows:
                    continue
                v = rows[sid]
                vals = list(v.values()) if isinstance(v, dict) else v
                f.write(sid + "\t" + "\t".join(
                    f"{float(x):.6f}" for x in vals) + "\n")

    write("imaging_classical.tsv", rows_cl)
    if rows_tf:
        write("imaging_timesfm.tsv", rows_tf)
    shutil.rmtree(tmp, ignore_errors=True)
    n_cl = max((len(v) for v in rows_cl.values()), default=0)
    return {"subjects": matched, "n_rois": N_ROIS,
            "classical_features": n_cl,
            "timesfm_features": (N_ROIS * HORIZON) if rows_tf else 0,
            "note": "classical = FC edges + network summaries + ALFF"}


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(data_root) / "features")
    s = process_imaging(data_root, out)
    log(f"DONE {s}")
