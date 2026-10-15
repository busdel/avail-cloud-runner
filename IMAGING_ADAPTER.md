# Imaging adapter - from raw NIfTI to AVAIL modalities

The imaging path turns a BIDS-style cohort into the same tabular
metadata.tsv + features/*.tsv layout used everywhere else. Two feature
families are produced, in three steps:

## 1. Preprocessing (AFNI, per subject)
`tools/rsfmri_process_subject.sh` (voximul: linux_ubuntu_16_64):
3dUnifize -> 3dAllineate (T1 -> MNI152, cost LPC - robust two-pass; MI diverges on these cohorts) -> despike -> slice-timing ->
volume registration -> EPI -> T1 -> composite transform -> 6 mm smoothing
-> automask -> band-pass 0.01-0.10 Hz + motion regression.
Outputs: `t1_mni.nii.gz`, `bold_clean.nii.gz`, `bold_blur.nii.gz`
(plus intermediates). Restart-safe: every step is guarded by an
existence check and completed subjects are skipped (`skip if
features exist`).

## 2. Feature extraction (per subject, two blocks)

### a) Connectivity (nilearn; no GPU)
`rsfmri_extract_fc.py` -> Schaefer-100 (17 networks) atlas:
full and partial (Ledoit-Wolf) Fisher-z matrices -> 171 within/between
summaries each (`full_*`, `part_*`). Guards: NaN/constant-parcel
protection before correlation (zero-variance rows are set to zero, so a
single empty parcel can no longer NaN an entire matrix).

### b) Radiomics (PyRadiomics; optional, engineering extension)
`radiomics_extract.py`:
- **anatomical**: T1 (MNI) x Schaefer-100 -> whole-brain (107) and
  17-network (1,674) IBSI features, bin width 25;
- **functional maps** (`--maps alff,falff,reho --bold bold_clean.nii.gz`):
  ALFF = stdev(bold_clean) (already band-passed), fALFF =
  ALFF / stdev(bold_blur), ReHo = 3dReHo (27-neighbourhood); every AFNI
  statistic is written as float (short storage silently quantizes small
  temporal standard deviations to zero - a failure mode we hit and
  guard against). Maps are computed at the native resolution with low
  parallelism to stay memory-safe;
  maps are robustly rescaled to [0,1] per subject (bin width 0.05), the
  atlas mask is resampled per-map grid, and out-of-coverage voxels are
  excluded.
Output: one row per subject, `{wb|net|alff|falff|reho}_*` columns
(~1,800 features in the reference configuration).

### Environment (voximul reference)
Python 3.11 venv (uv-managed; system 3.13 cannot build pyradiomics 3.0.1):
```bash
uv python install 3.11
<python3.11> -m venv --copies /root/radiomics_venv311
/root/radiomics_venv311/bin/pip install "numpy<2" "pandas<2.3" SimpleITK \
    setuptools wheel Cython versioneer
/root/radiomics_venv311/bin/pip install --no-build-isolation pyradiomics
```
Atlas: bundled in `atlas/schaefer_2018/` (1 mm + 2 mm + order file,
CC-BY, Schaefer et al. 2018); extractor falls back to the bundle when
the nilearn cache is absent.

## 3. Dataset assembly
`build_imaging_dataset.py` -> metadata.tsv (SampleID, Group, Age, Sex,
Education) + features/: `img_full.tsv`, `img_partial.tsv`,
`clin_hy.tsv`, `clin_cog.tsv`, `clin_tests.tsv`, and `radiomics.tsv`
(when present; constant columns dropped). The result is an ordinary
AVAIL dataset: `run_generic.py --data-root <dir> --out <results>`.

## Batch orchestration
`tools/reextract_batch.sh` (FC integrity pass), `tools/radiomics_batch.sh`
(+`_run_one.sh` runner, P=1-2 for memory safety), and the overnight
orchestrator `tools/overnight.sh` (phases, guards, and a single log at
`/media/suleyman_disk/OVERNIGHT.log`).
