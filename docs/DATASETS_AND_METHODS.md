# Datasets and methods (Paper 2 / AVAIL)

This document is the reviewer-facing data and method supplement. It describes
what each cohort contributes, how the pipeline is fit, and where leakage
controls sit. It is intentionally aggregate-level; no subject-level clinical
data are redistributed.

## 1. Role-separated cohorts

| Role | Cohort | n | Purpose |
|---|---|---:|---|
| Anchor | MetaCardis | 1087 (complete-case 536) | reproduction and natural informative missingness |
| Independent replication | HMP2 / IBD | 131 (complete-block 82) | different disease and block missingness |
| Imaging failure mode | OpenNeuro ds004392 | 53 with imaging | imaging-only transfer / failure audit |
| Companion application | AVAIL cohort | 150 (147 with >=1 modality) | end-to-end software use; no new biology |
| Coverage audit | OpenNeuro ds004856 | 295 labelled (complete-case 60) | complete-case bias only; no performance claim |

### MetaCardis
- Role: anchor cohort; six modalities.
- `n=1087`, complete-case `n=536`.
- Missingness is natural and informative and is retained as availability flags.
- The complete-case arm is a separately trained model on the complete subjects.
- Source/access: MetaCardis consortium data; access conditions apply. The
  repository does not redistribute raw subject data.

### HMP2 / IBD
- Role: independent disease replication and block-missingness test.
- `n=131`; taxa 130, pathways 104, proteomics 89; complete-block subset 82.
- Three blocks (taxa, pathways, proteomics), natural block missingness.
- The 11-model all-available run and a separately trained complete-block
  11-model arm are kept distinct.
- Source/access: HMP2 / IBD Multi'omics Database portal; public data access
  conditions apply. Derived aggregate tables only are included here.

### OpenNeuro ds004392
- Role: imaging-only failure audit; not a performance-validation cohort.
- OpenNeuro DOI: `10.18112/openneuro.ds004392.v1.0.2` (CC0).
- 58 subjects in the processed metadata; 53 have usable imaging features.
- Imaging-only class counts: 5 PDD, 36 PD-MCI, 12 PD-NC.
- Features: 342 Schaefer-100 (17-network) rs-fMRI connectivity summaries
  (within/between-network full and partial correlation summaries).
- The 3-class 11-model macro-AUC is 0.386, below chance; a permutation check is
  consistent with no detectable transfer. Exploratory binary PDD contrasts
  are reported separately and are not a validation claim.

### OpenNeuro ds004856
- Role: coverage audit only; no AUC is estimated as a biological performance
  claim.
- OpenNeuro DOI: `10.18112/openneuro.ds004856.v1.3.0` (CC0).
- 295 amyloid-labelled subjects: 115 positive, 180 negative.
- Modalities: MRI (222), tau (60), cognition (295).
- Complete-case intersection (MRI + tau + cognition): 60 subjects, **zero
  amyloid-positive**; all 73 MRI-absent subjects are positive. A single
  availability indicator reaches AUC 0.82, so complete-case and all-subject
  AUCs are structurally confounded. This cohort is retained only to demonstrate
  why complete-case analysis can be non-estimable.

### AVAIL cohort
- Role: companion application/regression check; biological interpretation
  belongs to the companion paper.
- `n=150`; 147 have at least one modality; 3 subjects have no modality and are
  retained through availability flags in the manuscript protocol.
- Five modules: fMRI ROI connectivity, stool 16S, stool shotgun, saliva 16S,
  saliva shotgun. Complete-case `n=50`.
- Source/access: de-identified in-house cohort; raw data are not redistributed.
  Only aggregate results and derived feature tables are included.

## 2. Method summary

### 2.1 Notation and availability
For subject `i` and modality `m`, let `a_im in {0,1}` indicate whether modality
`m` is available. The model never fabricates a missing measurement. A subject
with no modality is excluded from the public pipeline and reported separately;
the manuscript also reports the n=147 sensitivity.

### 2.2 Stage 1: per-modality base learners
- One Random Forest (300 trees) per modality, trained only on subjects with
  that modality.
- Leave-one-out (LOO) predictions are generated for every available subject.
- All preprocessing is fitted inside each training fold:
  - microbiome layers: log-center/CLR-style transform, then RF-importance
    top-100 features;
  - imaging/other layers: variance top-50, MinMax scaling, ANOVA F-test with
    Benjamini-Hochberg FDR, then RF-importance top-15;
  - optional confounder residualization for age/sex/education when requested;
  - standardization fitted on the training fold only.
- The held-out subject never contributes to feature selection, scaling,
  residualization, or model fitting.

### 2.3 Stage 2: availability-aware fusion
- For each subject and modality the meta-feature vector stores the `K`
  predicted class probabilities and one availability flag:
  `[p_i1..p_iK, a_i1, ..., p_iM1..p_iMK, a_iM]`.
- The 11 meta-learners are: SimpleMean, WSimpleMean, ElasticNetLR, MLP, MoE,
  GNN, Transformer, XGBoost, TabNet, TabPFN, TabICL.
- Meta-level evaluation is LOO; `WSimpleMean` learns modality weights only on
  the training fold.
- GNN is a small graph neural network over modality nodes; it is not a
  pretrained foundation model. TabPFN/TabICL are tabular foundation-style
  learners; Transformer is trained from scratch in this implementation.

### 2.4 Validation and leakage controls
- LOO base predictions and LOO meta predictions.
- Fold-internal feature selection, imputation, scaling and weighting.
- Full-pipeline label permutation tests; observed AUC is compared with the
  permutation null.
- Complete-case comparators are separately trained where reported. For HMP2,
  the original "complete" subgroup was not a separate training arm; the
  revised complete-arm results are in `hmp2_complete_11model/`.
- Binary imaging audit: repeated stratified nested CV (5-fold outer, 3-fold
  inner), age/sex residualization inside each outer training fold, and 100
  label permutations.
- No test-subject information is used to choose features, hyperparameters,
  calibration, or thresholds.

### 2.5 Biomarker layer
- Fold-internal feature screening supports the specialists (variance,
  ANOVA-FDR, RF importance).
- After out-of-fold predictions are fixed, a biomarker panel reports SHAP top-k
  (or random-forest importance when SHAP is unavailable) and class-pair
  differential abundance with Mann-Whitney tests and BH-FDR.
- Biomarker ranking is reported alongside the decision, not used to tune the
  decision rule or the reported test metrics.

### 2.6 Factorial designs
- 2x2x2: imputation {none, mean} x availability {none, binary} x fusion
  {SimpleMean, ElasticNetLR}; 3 seeds, 5-fold CV.
- 4x4x4: imputation {none, mean, KNN, MICE} x availability {none, binary,
  count, binary_count} x fusion {SimpleMean, ElasticNetLR, XGBoost,
  RandomForest}; 2-3 seeds, 5-fold CV.
- 8x8x8: 4x4x4 plus median/MissForest/SoftImpute/log-mean, pattern/full
  availability, and ExtraTrees/HistGB/MLP/KNN fusion levels; 2 seeds, 5-fold CV.
- Values are descriptive design-level macro-AUCs. A significant factor effect
  means the factor matters in that cohort, not that it generalises.
- Current completion status is tracked in `02_MAKALE2/supplement/TableS24_factorial_grid.md`.

### 2.7 Metrics and uncertainty
- Primary: 3-class macro-AUC (one-vs-rest mean).
- Secondary: per-class AUC, balanced accuracy, Brier score, calibration,
  Mondrian conformal coverage, and decision curves.
- Binary application contrasts: HC vs PDD and PDD vs rest.
- Confidence intervals use subject bootstrap resampling; paired comparisons
  resample common subjects. Factorial ANOVAs are descriptive and are not
  population-inference claims.

## 3. Reproducibility
- Package: `avail-omics` (PyPI 1.9.4).
- Commands: `avail-omics run-generic`, `avail-omics figures`,
  `avail-omics compare`, `avail-omics report`.
- Source result folders are named in the tables. Random seed defaults are 42;
  RF = 300 trees; LOO and fold counts are reported beside each analysis.
- All figures are generated from checked-in source tables and deterministic
  scripts under `02_MAKALE2/figur_scriptleri/`.

## 4. Limitations
- HMP2 is a difficult external replication; the 11-model mean is 0.633 and
  the paired all-vs-complete difference has a CI crossing zero.
- ds004392 has only five PDD cases; binary PDD signals are exploratory.
- ds004856 has zero positive complete cases; it cannot support a performance
  claim.
- AVAIL cohort results are an application/regression check, not external
  validation.
- Factor effects are cohort-specific and should not be read as universal
  architecture recommendations.
