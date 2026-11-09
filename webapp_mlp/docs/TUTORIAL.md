# Tutorial

## 1. Prepare a dataset

Create a ZIP with this structure:

```
mydata/
  metadata.tsv
  features/
    brain.tsv
    stool16S.tsv
```

Example `metadata.tsv`:

```tsv
SampleID	Group	Age	Sex
S001	HC	62	F
S002	MCI	71	M
S003	PDD	68	F
```

Example `features/brain.tsv`:

```tsv
SampleID	FC_A	FC_B	FC_C
S001	0.41	0.22	0.11
S002	0.37	0.19	0.14
S003	0.29	0.31	0.08
```

Use one file per modality. A subject absent from `features/stool16S.tsv` is
treated as missing that modality.

## 2. Upload and run

1. Open the **Demo** tab.
2. Upload the ZIP.
3. Click **Run quick benchmark**.
4. Inspect the result table and download the results ZIP.

The quick demo uses three meta-learners. It is intentionally not the full
11-model run.

## 3. Move to the full pipeline

```bash
pip install "avail-omics[deep]"
avail-omics run-generic --data-root mydata --out results
avail-omics robustness --data-root mydata --out robustness
avail-omics sensitivity --data-root mydata --out sensitivity
avail-omics compare \
  --entry "Cohort A=results" \
  --out comparison
```

## 4. Synthetic demo

Use **Run synthetic demo** if you want to see the output format without
uploading anything. The synthetic cohort is generated locally and contains
three modalities with deliberate missingness.
