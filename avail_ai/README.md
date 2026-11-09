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
avail-omics run-generic --data-root mydata --out results
avail-omics robustness --data-root mydata --out robustness \
    --permutations 25 --jobs 4          # full-pipeline label-permutation test
avail-omics sensitivity --data-root mydata --out sensitivity
avail-omics compare --entry "Cohort A=results/a" \
  --entry "Cohort B=results/b" --out comparison
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
- `module_panels` — selected-module networks: a small, fully labelled network
  per module plus a compact cross-module heatmap (recommended for figures).
- `module_summary` — clean three-panel summary for dense networks: module
  sizes, cross-module link heatmap, and the strongest cross-module pairs.
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
- Role-separated public evidence: MetaCardis anchor (n=1087; reproduction +
  coverage), HMP2/IBD independent replication (n=130; non-GNN mean 0.639,
  best XGBoost 0.771; complete-block subset), and OpenNeuro ds004392 imaging failure-mode (n=53;
  no detectable transfer). Wallen 2022, preclinical AD and the label-free
  reconstruction are reported in the companion application paper.

## Validation suite and coverage audit

Full notes: [VALIDATION.md](https://github.com/busdel/avail-cloud-runner/blob/main/avail_ai/VALIDATION.md).

The software paper uses role-separated evidence rather than a leaderboard:

| Role | Dataset | n | What it tests |
|---|---|---:|---|
| Anchor | MetaCardis | 1087 (CC 536) | natural, informative missingness |
| Independent replication | HMP2 / IBD | 130 | different disease; complete-block subset |
| Imaging failure mode | OpenNeuro ds004392 | 53 | imaging-only transfer; failure handling |
| Companion application | AVAIL cohort | 150 | end-to-end software use; no new biology |
| Coverage audit | OpenNeuro ds004856 | 295 labeled | complete-case bias only |

The ds004856 audit is intentionally **not** a performance result. Its
processed OpenNeuro tables contain 295 amyloid-labeled subjects, but the
complete-case intersection (MRI + tau + cognition) has **60 subjects and zero
amyloid-positive subjects**; all 73 MRI-absent subjects are positive and the
60 tau-available subjects are negative. A single availability indicator
reaches AUC 0.82, so complete-case and all-subject AUCs are not interpretable
as biological estimates. The cohort is retained only to show that
complete-case analysis can be structurally non-estimable. It is not used to
claim diagnostic accuracy.

Figure: [cross-cohort comparison](https://raw.githubusercontent.com/busdel/avail-cloud-runner/main/viz/cohort_comparison.png).

The companion AVAIL cohort (n=150) is an application/regression check:
biological interpretation belongs to the companion paper; this package uses
the same verified cohort to exercise the command line, figures, report and R
export end to end.

## Cross-cohort comparison

`avail-omics compare` reads one or more results directories and writes a
machine-readable cohort table plus a two-panel scientific figure (complete-case
coverage and macro-AUC with the range across meta-learners). It is generic:
labels and cohorts come from the command line.

```bash
avail-omics compare \
  --entry "Cohort A=results/a" \
  --entry "Cohort B=results/b" \
  --out comparison
```

If the complete-case arm was run as a separate directory:

```bash
avail-omics compare \
  --entry "Cohort A=results/a_all::results/a_complete" \
  --entry "Cohort B=results/b" \
  --out comparison
```

Outputs: `cohort_comparison.tsv`, `cohort_comparison.pdf/.png/.tiff`; example table:
[`cohort_comparison.tsv`](https://raw.githubusercontent.com/busdel/avail-cloud-runner/main/viz/cohort_comparison.tsv).
GNN learners are excluded by default; use `--include-gnn` to retain them.

Figure: [example cross-cohort comparison](https://raw.githubusercontent.com/busdel/avail-cloud-runner/main/viz/cohort_comparison.png).

## Research platform

- Public overview: https://avail-research-platform.pages.dev/
- Fixed demo: https://avail-research-platform.pages.dev/demo
- Password-protected workspace: https://avail-research-platform.pages.dev/platform
- No public uploads and no public compute; full runs stay local, in Docker,
  Colab or on a user-owned GPU worker.

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
repository and package: `avail-omics` v1.8.0.

## License

MIT
