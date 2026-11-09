# Documentation

## What this demo does

This is a **quick, demo-only** web interface for `avail-omics`. It accepts a ZIP
with the generic AVAIL layout, validates it, runs three fast meta-learners, and
returns a results ZIP plus a summary table.

For the full 11-model zoo, missing-data strategy benchmark, sensitivity,
robustness and R export, install the package and run locally:

```bash
pip install "avail-omics[deep]"
avail-omics run-generic --data-root mydata --out results
```

## Input layout

```
mydata.zip
  metadata.tsv          # required: SampleID, Group
  features/
    modality_a.tsv      # rows = samples, first column = SampleID
    modality_b.tsv
    ...
  compositional.txt     # optional: one modality name per line
```

- `metadata.tsv` may also contain `Age`, `Education`, `Sex`.
- Each feature table must have a unique `SampleID` column and numeric features.
- Missing modalities are represented by **absent rows** in that modality's
  feature file; do not write all-NaN rows.
- At least two classes are required. The demo uses the first three most frequent
  classes; the full package accepts explicit `--classes`.
- The demo does not accept raw DICOM/NIfTI; use the package's imaging adapter
  locally.

## Output

- `meta_benchmark_results.tsv` — meta-learner performance by subgroup
- `per_class_auc.tsv`, `loo_predictions.tsv`, `subject_classes.tsv`
- availability / coverage figures
- a ZIP containing the run directory

## Privacy

The public demo is for **de-identified or synthetic data only**. Do not upload
patient-identifiable data. For real clinical data, use the local package,
local runner, or your own worker. The demo keeps uploads only for the duration
of the job and deletes them afterward.
