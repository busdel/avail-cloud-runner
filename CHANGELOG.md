# Changelog

## 1.0.1 — robustness & imaging release
- **imaging**: `sanitize_ts` guard — non-finite entries zeroed and
  empty/constant parcels are neutralised before correlation, keeping
  feature tables well-defined under incomplete acquisition coverage.
- **imaging**: Schaefer-2018 atlas now resolves from the locally bundled
  copy first (`atlas/schaefer_2018/`, CC-BY) with nilearn as fallback —
  reproducible offline.
- **imaging**: optional `radiomics_features()` block (first-order,
  texture, shape; whole-brain mask; IBSI defaults, bin width 25).
  New extras: `avail-omics[imaging]`, `avail-omics[radiomics]`
  (pyradiomics builds on Python <= 3.12).
- **llm**: automatic retry with doubled token budget when a provider
  returns an empty message (observed with flash-tier models).
- **generic**: `compositional.txt` override plus a row-sum-scale
  heuristic; every classification is logged (silent CLR mis-application
  fixed).
- **radiomics**: new `avail_omics.radiomics` module + `avail-omics
  radiomics` CLI command — anatomical T1 (whole-brain + 17-network) and
  functional maps (ALFF/fALFF/ReHo via AFNI; per-map-grid mask
  resampling; float outputs; coverage masking) with a standard
  `radiomics.tsv` dataset output.
- **tests**: imaging + radiomics utility tests added; suite runs 12 tests.

## 1.0.0 — initial release
- Availability-aware two-stage integration; 11 meta-learners; 16+
  missing-data strategies + hybrid; leak-free validation protocol
  (permutation, dropout, MCAR/MAR/MNAR, conformal, calibration); CLI,
  generic pipeline, advisor, publication figures.
