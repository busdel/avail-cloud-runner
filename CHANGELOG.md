# Changelog

## 1.0.2 — maintenance release
- Unified compositional-modality detection between the main pipeline and
  the robustness toolkit (single helper; `compositional.txt` honoured
  everywhere, so permutation/sensitivity runs use the identical protocol).
- Base-prediction cache validated against a cohort fingerprint, class
  count and schema; stale caches recomputed automatically.
- Confounder parsing tolerates non-numeric entries (treated as missing).
- Imaging: configurable `dcm2niix`, automatic CPU/GPU device selection,
  bundled-atlas fallback, graceful skip when the optional TimesFM layer
  is absent (new `[timesfm]` extra).
- Radiomics: exact-token subject matching (no S1/S10 prefix collisions).
- LLM client: HTTP 403 without quota/rate context no longer counts as a
  quota error.
- `__version__` now consistent with distribution metadata.

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
