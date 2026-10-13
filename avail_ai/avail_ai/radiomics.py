# -*- coding: utf-8 -*-
"""AVAIL radiomics module (optional engineering block).

Ports the validated pipeline:
  - anatomical T1 (ideally MNI-registered): whole-brain (wb_*) and
    17-network (net_*) IBSI features (firstorder + texture [+shape for
    wb]), bin width 25, Schaefer-100 mask;
  - functional maps via AFNI (optional): ALFF = stdev(bold_clean),
    fALFF = ALFF / stdev(bold_blur), ReHo = 3dReHo (nneigh 27); maps are
    robustly rescaled to [0,1] (bin width 0.05), masks are resampled to
    each map grid with `3dresample -master` (validated fix for AFNI grid
    conventions), and out-of-coverage voxels are excluded;
  - every AFNI statistic is written as float (short scaling can
    silently quantize small temporal standard deviations to zero).

Requires: pip install "avail-omics[radiomics]" (SimpleITK + pyradiomics;
pyradiomics builds on Python <= 3.12). Functional maps additionally need
AFNI binaries (--afni-bin).
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .imaging import _atlas_path

CLASSES = ["firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
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


def log(msg):
    print(f"[radiomics] {msg}", flush=True)


def radiomics_available():
    try:
        import radiomics  # noqa: F401
        import SimpleITK  # noqa: F401
        return True
    except Exception:
        return False


def match_sid(path, ids_set):
    """Exact-token subject id matching (no substring collisions S1/S10)."""
    path = Path(path)
    for part in path.parts:
        if part in ids_set:
            return part
    tokens = re.split(r"[^A-Za-z0-9\-]+", path.name)
    for sid in ids_set:
        if sid in tokens:
            return sid
    return None


def _require():
    try:
        import SimpleITK as sitk
        from radiomics import featureextractor
        return sitk, featureextractor
    except Exception as e:
        raise RuntimeError(
            "radiomics extras not installed: pip install "
            "'avail-omics[radiomics]' (Python <=3.12 recommended); for "
            "functional maps also provide AFNI via --afni-bin") from e


def canon(name):
    for k, v in PAIRS:
        if k in name:
            return v
    return None


def order_path():
    p = _atlas_path(1)
    if p is not None:
        c = Path(p).parent / "Schaefer2018_100Parcels_17Networks_order.txt"
        if c.exists():
            return c
    from ._net import offline
    if offline():
        raise RuntimeError("offline mode: Schaefer order file not found "
                           "locally (bundle atlas/schaefer_2018/)")
    from nilearn.datasets import fetch_atlas_schaefer_2018
    m = fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=17).maps
    return Path(m).parent / "Schaefer2018_100Parcels_17Networks_order.txt"


def load_network_labels(order_txt=None):
    """{network: [parcel labels]} from the Schaefer order file.

    The Yeo "17-network" naming yields 18 named groups (SomMotB splits
    into auditory and rest), 100 parcels in total.
    """
    order_txt = Path(order_txt) if order_txt else order_path()
    labels = {}
    for line in order_txt.read_text().strip().split("\n"):
        idx, name = line.split()[:2]
        net = canon(name)
        if net:
            labels.setdefault(net, []).append(int(idx))
    return labels


def _afni(args, afni_bin):
    env = dict(os.environ)
    env["PATH"] = str(afni_bin) + ":" + env.get("PATH", "")
    r = subprocess.run([str(Path(afni_bin) / args[0])] + list(args[1:]),
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"AFNI {args[0]} failed: {r.stderr[-300:]}")


def make_maps(bold, outdir, afni_bin, blur=None):
    """ALFF / fALFF / ReHo maps (direct resolution, float outputs)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    blur = blur or str(bold).replace("bold_clean", "bold_blur")
    out = {}
    alff = outdir / "alff.nii.gz"
    _afni(["3dTstat", "-stdev", "-datum", "float", "-prefix", str(alff),
           str(bold), "-overwrite"], afni_bin)
    out["alff"] = alff
    if Path(blur).exists():
        tot = outdir / "tot.nii.gz"
        _afni(["3dTstat", "-stdev", "-datum", "float", "-prefix", str(tot),
               blur, "-overwrite"], afni_bin)
        falff = outdir / "falff.nii.gz"
        _afni(["3dcalc", "-a", str(alff), "-b", str(tot), "-expr", "a/b",
               "-datum", "float", "-prefix", str(falff), "-overwrite"],
              afni_bin)
        out["falff"] = falff
    reho = outdir / "reho.nii.gz"
    _afni(["3dReHo", "-inset", str(bold), "-prefix", str(reho),
           "-nneigh", "27", "-overwrite"], afni_bin)
    out["reho"] = reho
    return out


def _robust_unit(sitk, img):
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


def _mask_in_grid(sitk, atlas_img, ref_img, coverage_arr=None):
    """Resample the atlas to the reference grid (nearest) and build the
    binary mask; optionally intersect with a coverage map (map > 0)."""
    a = sitk.Resample(atlas_img, ref_img, sitk.Transform(),
                      sitk.sitkNearestNeighbor, 0, atlas_img.GetPixelID())
    arr = sitk.GetArrayFromImage(a) > 0
    if coverage_arr is not None:
        arr &= coverage_arr > 0
    m = sitk.GetImageFromArray(arr.astype(np.uint8))
    m.CopyInformation(ref_img)
    return m, int(arr.sum())


def extract_subject(t1=None, bold=None, mode="wb", maps="",
                    maps_dir=None, afni_bin=None, atlas_path=None,
                    binwidth=25.0, map_binwidth=0.05, out_tsv=None):
    """Extract radiomics features for one subject -> dict of features."""
    sitk, featureextractor = _require()
    rows = {}

    ap = atlas_path or _atlas_path(1)
    if ap is None:
        from ._net import offline
        if offline():
            raise RuntimeError("offline mode: no local Schaefer atlas found; "
                               "pass atlas_path or bundle atlas/schaefer_2018/")
        from nilearn.datasets import fetch_atlas_schaefer_2018
        ap = fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1).maps
    atlas_img = sitk.ReadImage(str(ap))

    if t1 is not None:
        img = sitk.ReadImage(str(t1))
        mask, nvox = _mask_in_grid(sitk, atlas_img, img)
        if nvox < 512:
            raise RuntimeError(
                f"atlas/image overlap too small ({nvox} voxels): check that "
                "the T1 is registered to MNI space")
        ex = featureextractor.RadiomicsFeatureExtractor(binWidth=binwidth,
                                                        label=1)
        ex.disableAllFeatures()
        for c in CLASSES + ["shape"]:
            ex.enableFeatureClassByName(c)
        res = ex.execute(img, mask)
        rows.update({"wb_" + k[len("original_"):]: float(v)
                     for k, v in res.items()
                     if k.startswith("original_")})
        if mode == "both":
            labs = load_network_labels()
            ex2 = featureextractor.RadiomicsFeatureExtractor(binWidth=binwidth,
                                                             label=1)
            ex2.disableAllFeatures()
            for c in CLASSES:
                ex2.enableFeatureClassByName(c)
            arr = sitk.GetArrayFromImage(atlas_img)
            for net, ls in sorted(labs.items()):
                m = np.isin(arr, ls).astype(np.uint8)
                if m.sum() < 64:
                    continue
                mask2 = sitk.GetImageFromArray(m)
                mask2.CopyInformation(atlas_img)
                mm = sitk.Resample(mask2, img, sitk.Transform(),
                                   sitk.sitkNearestNeighbor, 0,
                                   sitk.sitkUInt8)
                try:
                    r = ex2.execute(img, mm)
                except Exception:
                    continue
                pref = "net_" + re.sub(r"[^A-Za-z0-9]+", "", net.capitalize()) + "_"
                rows.update({pref + k[len("original_"):]: float(v)
                             for k, v in r.items()
                             if k.startswith("original_")})

    if maps and bold is not None:
        if not afni_bin:
            raise RuntimeError("functional maps require AFNI: pass "
                               "--afni-bin (or set AFNI_BIN)")
        mdir = Path(maps_dir) if maps_dir else Path(bold).parent / "rad_maps"
        if out_tsv is not None:
            mdir = Path(str(out_tsv).replace(".tsv", "_maps"))
        mdir.mkdir(parents=True, exist_ok=True)
        make_maps(bold, mdir, afni_bin)
        for m in [x.strip().lower() for x in maps.split(",") if x.strip()]:
            f = mdir / f"{m}.nii.gz"
            if not f.exists():
                continue
            raw = sitk.ReadImage(str(f))
            cov = sitk.GetArrayFromImage(raw)
            a_m_path = mdir / "atlas_in_map.nii.gz"
            if not a_m_path.exists():
                _afni(["3dresample", "-master", str(f), "-input", str(ap),
                       "-prefix", str(a_m_path), "-overwrite"], afni_bin)
            arr_m = sitk.GetArrayFromImage(sitk.ReadImage(str(a_m_path)))
            mask_arr = ((arr_m > 0) & (cov > 0)).astype(np.uint8)
            nvox = int(mask_arr.sum())
            if nvox < 512:
                log(f"{m}: coverage overlap too small ({nvox}), skipped")
                continue
            mask = sitk.GetImageFromArray(mask_arr)
            mask.CopyInformation(raw)
            img = _robust_unit(sitk, raw)
            ex = featureextractor.RadiomicsFeatureExtractor(
                binWidth=map_binwidth, label=1)
            ex.disableAllFeatures()
            for c in CLASSES:
                ex.enableFeatureClassByName(c)
            try:
                res = ex.execute(img, mask)
            except Exception as e:
                log(f"{m}: extraction failed ({e})")
                continue
            rows.update({f"{m}_wb_" + k[len("original_"):]: float(v)
                         for k, v in res.items()
                         if k.startswith("original_")})
    return rows


def extract_dataset(data_root, out, mode="wb", maps="", afni_bin=None,
                    binwidth=25.0):
    """Extract radiomics for every subject in a data_root; writes
    <out>/radiomics.tsv in the standard modality layout."""
    import pandas as pd
    root = Path(data_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(root / "metadata.tsv", sep="\t")
    id_col = "SampleID" if "SampleID" in meta.columns else meta.columns[0]
    ids = meta[id_col].astype(str).tolist()

    ids_set = set(ids)

    t1_map, bold_map = {}, {}
    for p in root.rglob("*.nii.gz"):
        b = p.name.lower()
        sid = match_sid(p, ids_set)
        if sid is None:
            continue
        if "t1" in b or "t1w" in b:
            t1_map.setdefault(sid, p)
        if b.startswith("bold_clean") or "task-rest_bold" in b:
            bold_map.setdefault(sid, p)

    rows = {}
    for sid in ids:
        t1 = t1_map.get(sid)
        bold = bold_map.get(sid)
        if t1 is None and bold is None:
            continue
        try:
            r = extract_subject(t1=t1, bold=bold, mode=mode, maps=maps,
                                afni_bin=afni_bin, binwidth=binwidth)
        except Exception as e:
            log(f"{sid}: skipped ({e})")
            continue
        if r:
            rows[sid] = r
            log(f"{sid}: {len(r)} features")
    if not rows:
        raise RuntimeError("no subject produced radiomics features")
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "SampleID"
    df.reset_index().to_csv(out / "radiomics.tsv", sep="\t", index=False)
    log(f"wrote {out / 'radiomics.tsv'}: {df.shape[0]} x {df.shape[1]}")
    return {"subjects": df.shape[0], "features": df.shape[1]}
