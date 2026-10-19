# Changelog

## 1.0.15
- LLM robustness: the Groq fallback now uses openai/gpt-oss-120b (tool
  calling, higher token-per-minute limits); per-provider model overrides
  via AVAIL_LLM_MODEL_<PROVIDER>.
- Web assistant trims its context and returns a clear rate-limit message
  instead of a raw provider error when all providers are throttled.

## 1.0.14
- Biological figures: top differential features are now shown per group
  (raw subject values with medians/IQR; fig8) and as signed effect-size
  panels with BH-FDR (fig9). The pipeline exports
  top_features_long.tsv for these panels.

## 1.0.13
- Fix: per-class one-vs-rest AUC was computed with the macro helper on a
  single column, which inverted the score (reported 1-AUC). It now uses
  roc_auc_score directly; subject_classes.tsv is written for group-aware
  figures.
- Web figures now show groups explicitly: coverage by group, class
  stripe on the availability matrix, per-comparison volcano panels, and
  a per-class AUC heatmap.

## 1.0.12
- Three additional web figures: availability-stratified model comparison
  (slope chart), subject-level availability pattern (matrix), and a
  differential-abundance volcano with a ranked callout panel. Label
  placement avoids overlaps; provenance stays in the UI captions.

## 1.0.11
- Web figures redesigned for legibility: lollipop charts with value
  labels, chance line explicit, coverage bars, and direction-colored
  feature attribution; provenance note moved off the axes.

## 1.0.10
- Figures: the web pipeline now ships three deliberately simple PNG
  figures (model performance, modality coverage, top features per
  modality) instead of the dense multi-panel manuscript set; the latter
  remains available in figures_publication for papers.
- Meta-learner subset: AVAIL_META_MODELS (comma list) selects a subset
  for fast runs; the built-in web demo uses four fast learners while
  advanced comparisons still cover all of them.

## 1.0.9
- Faster provider failover: the last-resort retry wait in the LLM client
  chain is now configurable (AVAIL_LLM_RETRY_SLEEP, default 15 s instead
  of a fixed 70 s), cutting worst-case interactive latency after a
  provider hiccup.

## 1.0.8
- LLM provider fallback now also triggers on transient failures
  (HTTP 5xx after in-provider retries, connection errors and timeouts),
  not only on quota/rate limits. With a secondary key configured
  (~/.config/avail/llm_key_groq), assistant requests survive primary-
  provider outages instead of surfacing a 503.

## 1.0.7
- Metadata parsing now tolerates missing Sex/Age/Education cells across
  pandas versions (the pandas 3 string dtype yields float NaN, which
  crashed the Sex normalizer); missing values map to NaN and are
  imputed downstream exactly as before.
- Offline mode (AVAIL_OFFLINE=1) now also disables the multi-database
  literature lookup and skips the advisor evidence loop, so no outbound
  request is attempted anywhere.
- Local-runner bundle: fixed the import path when the tools are run from
  the extracted ZIP layout; README command corrected.
- Integrity-audit script path handling made portable; scans all result
  directories containing saved leave-one-out predictions.
## 1.0.6 — bring your own server
- New `avail-omics worker` (+ `avail-omics worker-token`): a secure job
  server for your own machine — bearer-token auth (constant-time), binds
  to 127.0.0.1 by default, per-job sandbox directories with ZIP-slip
  protection, optional memory cap and nice, one job at a time. Connect it
  from any AVAIL interface ("Run on my own server"); expose it with an
  outbound-only tunnel. Docs: WORKER.md.
- Worker test suite added (auth, queueing, lifecycle).

## 1.0.5 — guard follow-up
- Protocol connectivity: correlation matrices are sanitized for
  zero-variance parcels (finite fallback, unit diagonal) before the
  Fisher-z/partial steps; a constant parcel can no longer leak NaN into
  the network summaries. Regression test included.

## 1.0.4 — one protocol everywhere
- Imaging feature definition aligned with the published extraction
  protocol: the automatic adapter (package and web) now produces the same
  342 connectivity features (full Pearson Fisher-z + partial Ledoit-Wolf,
  within/between network summaries) as the paper pipeline and the
  downloadable ds004392 FC tables. Masker settings (standardize, detrend,
  resampling) now match the protocol script exactly; NaN/constant guard
  retained.
- The imaging writer now emits semantically named columns
  (``full_*`` / ``part_*``).

## 1.0.3 — data-security release
- `AVAIL_OFFLINE=1` hard-disables every outbound call (LLM providers,
  PubMed look-ups, atlas downloads); PubMed helpers return empty results
  and imaging/radiomics raise an explicit error when no local atlas is
  bundled. For fully offline imaging: ship `atlas/schaefer_2018/`.
- New `docs/DATA_SECURITY.md`: complete outbound-call inventory with
  what is sent, when, and how to disable each path; verification recipes.
- No telemetry of any kind has ever been included.

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
