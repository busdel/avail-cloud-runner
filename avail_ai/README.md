# AVAIL-omics

**Availability-aware multimodal integration for clinical cohorts with missing modalities.**

Real multimodal cohorts are incomplete: an fMRI is missing here, a shotgun
metagenome there. Complete-case analysis discards subjects; AVAIL-omics drops
*nobody*. Every modality trains its own base learner on all available
subjects, a fixed-length meta-feature vector (base probabilities +
availability flags) is assembled per subject, and a meta-learner makes the
joint decision.

## Install

```bash
pip install avail-omics           # core: CPU-only, no CUDA needed
pip install "avail-omics[deep]"   # + neural (MLP/MoE/GNN/Transformer) and
                                  #   deep-tabular (TabNet/TabPFN/TabICL) meta-learners
pip install "avail-omics[full]"   # + SHAP-based biomarker interpretation
```

## Quick start (any tabular multimodal dataset)

```python
from avail_omics.generic import run_generic_pipeline

res = run_generic_pipeline(
    data_root="mydata",      # metadata.tsv (SampleID, Group) + features/*.tsv
    out_dir="results",       # one .tsv per modality -> 11 meta-learners, LOO
)
```

Data layout (one file per modality, rows = samples):

```
mydata/
  metadata.tsv          # SampleID + Group (+ optional Age, Education, Sex)
  features/
    brain.tsv
    stool16S.tsv
    ...
```

Or use the CLI:

```bash
avail-omics run --data-root mydata --out results
avail-omics robustness --data-root mydata --out robustness \
    --permutations 25 --jobs 4          # full-pipeline label-permutation test
avail-omics sensitivity --data-root mydata --out sensitivity
```

## Soft-palette visualization toolkit

```bash
avail-omics plot --out results --data-root mydata \
    --type network,roc,availability,umap,volcano,forest
```

- `network` — top-biomarker Spearman network (needs `data_root` for feature values;
  otherwise a modality–feature bipartite graph).
- `network_modules` — module-aware correlation network: automatic community
  detection, per-module panels or `--combined` single view with cross-module
  edges, hubs, optional taxonomy colouring (`--taxonomy`), and a
  `cross_module_edges.tsv` export. Correlations: `--method sparcc` (CLR
  approximation), `spearman` or `pearson`.
- `roc` — one-vs-rest ROC and reliability/calibration curves from
  `loo_predictions.tsv`.
- `availability` — subject × modality availability heatmap, group coverage and
  top availability patterns from `metadata.tsv` + `features/*.tsv`.
- `umap` — 2-D UMAP of the concatenated feature space (PCA fallback;
  install `avail-omics[viz]` for UMAP), coloured by group.
- `volcano` — log2 fold change vs q-value from `diff_abundance_*.tsv`.
- `forest` — effect sizes with 95% CI from a table such as
  `TableS5_delta_ci_*.tsv` or `forest.tsv`.

Outputs are PDF + 400 dpi PNG in `results/viz/` by default, using a muted,
colour-vision-friendly palette. Missing inputs are reported explicitly.

## AI advisor report and results agent (bring your own key)

The package ships the same LLM layer used by the web interface. Keys are
never bundled; each user supplies their own:

```bash
avail-omics set-key                 # or export AVAIL_LLM_API_KEY=...
avail-omics report --out results    # llm_advisor.json + llm_advisor_report.md
avail-omics chat --out results      # results-grounded Dr. AVAIL agent
avail-omics agent --out results     # alias for chat
```

Without a key the core pipeline runs normally; the AI sections are skipped.
Only derived summaries/prompts are sent to the model provider; raw data stay
on the user's machine. Built-in providers: DeepSeek, OpenAI, Groq, Gemini,
OpenRouter (OpenAI-compatible endpoints).

## What is included

- **Leak-free protocol**: preprocessing, feature selection, imputation and
  hyperparameters are computed strictly inside each cross-validation fold.
- **14 missing-data strategies + hybrid**: complete-case, zero/mean/median/KNN/
  MICE/MissForest/SoftImpute imputation, availability-flag encoding, and the
  hybrid (fold-internal imputation + *true* availability flags).
- **11 meta-learners**: SimpleMean, WSimpleMean, ElasticNetLR, MLP, MoE, GNN,
  Transformer, XGBoost, TabNet, TabPFN, TabICL.
- **Robustness toolkit**: synthetic dropout, MCAR/MAR/MNAR simulation,
  label-permutation test, modality sensitivity/ablation, bootstrap CIs,
  conformal prediction sets, calibration (isotonic) with Brier reporting,
  integrity audit.

## Evidence summary (primary cohort, n=150 oral-gut-brain)

- HC vs PDD binary AUC **0.978** (MLP, leave-one-out)
- Severity continuum: Spearman ρ = 0.633 vs clinical stage; risk score tracks
  MMSE (ρ = -0.47) and CDRS (ρ = 0.63) — scores never used in training
- Label-permutation test of the full pipeline: permuted macro-AUC ≈ 0.50
  (p = 0.039), ruling out leakage as the source of the separation
- Sustains prediction where complete-case collapses (25-55% dropout)
- External validation on public cohorts: Wallen 2022 PD shotgun (AUC 0.642),
  HMP2 IBD multi-omics with genuine per-sample missingness (macro-AUC 0.771),
  preclinical AD (0.542), label-free reconstruction (Vascellari 2020)

## R interoperability

```bash
avail-omics export-r --out results --data-root mydata \
  --dest r_export --taxonomy taxonomy.tsv
```

Writes `otu_table.tsv` (features × samples), `sample_data.tsv`,
`tax_table.tsv`, `feature_table.tsv`, results tables, GraphML/edge tables and
an `README_R.md` with phyloseq, vegan, igraph, SpiecEasi, NetCoMi and
mixOmics/PMA snippets. No R packages are bundled.

## Web platform

A self-hostable web interface accompanies this package. The public reviewer
site is demo-only; own-data analysis runs locally/on the user's worker. See
the project repository.

## Citing

Manuscript in preparation. Until the paper is out, please cite the
repository and package: `avail-omics` v1.0.19.

## License

MIT
