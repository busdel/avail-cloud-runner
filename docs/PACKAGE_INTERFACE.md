# Package interface

`avail-omics` is both a command-line pipeline and a Python library. The
installed package version is shown on PyPI; the interface below is stable
across the 1.9.x line.

## Install

```bash
pip install -U avail-omics                 # core, CPU
pip install "avail-omics[deep]"            # neural/tabular foundation learners
pip install "avail-omics[radiomics]"       # imaging feature extraction
```

Python: **>=3.9**.

## Command line

```bash
avail-omics run-generic --data-root mydata --out results
avail-omics figures     --out results
avail-omics compare     --entry cohort=results [--entry ...] --out comparison
avail-omics plot        --out results --data-root mydata \
  --type network,roc,availability,umap,volcano,forest
avail-omics report      --out results       # optional LLM advisor report
avail-omics chat        --out results       # optional results-grounded agent
```

`run-generic` accepts any tabular modality layout:

```text
mydata/
  metadata.tsv          # SampleID, Group (+ optional Age, Sex, Education)
  features/
    brain.tsv           # rows = samples; first column = SampleID
    stool16S.tsv
    ...
  compositional.txt     # optional modality names for CLR/log transforms
```

## Python library

```python
from avail_omics.generic import run_generic_pipeline

summary = run_generic_pipeline(
    "mydata",
    "results",
    classes=["HC", "MCI", "PDD"],
)
```

Lower-level building blocks are also exported:

```python
from avail_omics import (
    preproc, apply_preproc, macro_auc,
    base_predictions_loo, build_meta_features, run_meta_benchmark,
)

from avail_omics import viz
viz.plot_network("results", data_root="mydata")
viz.plot_roc_calibration("results")
viz.plot_availability("mydata")
```

## Output contract

```text
results/
  meta_benchmark_results.tsv
  meta_benchmark_3class_results.tsv
  per_class_auc.tsv
  loo_predictions.tsv
  base_predictions_generic.tsv
  biomarkers_top20_per_modality.tsv
  diff_abundance_all.tsv
  fig1_model_performance.{pdf,png,tiff}
  fig2_modality_coverage.{pdf,png,tiff}
  fig3_top_biomarkers.{pdf,png,tiff}
  fig4_subgroup_comparison.{pdf,png,tiff}
  fig5_availability_pattern.{pdf,png,tiff}
  fig6_diff_abundance.{pdf,png,tiff}
  fig7_per_class_auc.{pdf,png,tiff}
  fig8_group_profiles.{pdf,png,tiff}
  fig9_effect_sizes.{pdf,png,tiff}
```

Figures use the project palette and are exported as vector PDF plus
600-dpi PNG/TIFF. Table outputs are always available even if an optional
figure cannot be generated from a small or low-signal dataset.

## Reproducibility

- Seeds and fold counts are explicit in the commands and result tables.
- `run-generic` uses leave-one-out base and meta evaluation.
- The `compare` command reads multiple result folders and writes
  `cohort_comparison.tsv` plus publication-style figures.
- No raw subject data are sent to external services. Optional LLM features
  require a user-supplied key and operate on aggregate result context.
