# Reviewer quick start

This repository accompanies two companion manuscripts on availability-aware
multimodal learning for incomplete clinical cohorts. The guide below gives a
short, reproducible path through the software, the live interface, and the
validation evidence.

## 1. Live interface

- Landing page: **https://busdel.github.io/avail-cloud-runner/**
  (redirects to the current live session; access credentials are provided in
  the cover letter and should not be posted publicly).
- The hosted instance runs only the **fixed, de-identified demo** on the
  server. User-uploaded jobs run on the free cloud runner, the local runner,
  or a user-owned worker; `AVAIL_ALLOW_SERVER_RUN=0` is the default.
- Expected demo output: de-identified cohort, 87 subjects, model benchmark,
  9 figures, an advisor report with PubMed citations, and a results ZIP.
  Runtime on a CPU-only host is typically a few minutes.

## 2. Python package

```bash
pip install avail-omics==1.0.17
avail-omics --help
```

The package provides the same pipeline, CLI, validation utilities and
comparison suite as the web interface. Core install is CPU-only; the deep
tier adds neural/foundation meta-learners.

## 3. What to inspect

1. **Demo run:** does it complete with 87 subjects and 9 figures?
2. **Group-aware readouts:** coverage, per-class one-vs-rest AUC, and group
   profiles (HC / MCI / PDD).
3. **Advisor report:** are the PubMed links resolvable and is the evidence
   context visible?
4. **Audit outputs:** `meta_benchmark_results.tsv`,
   `per_class_auc.tsv`, `loo_predictions.tsv`, and the integrity audit.
5. **Deployment:** the same pipeline is available as a local runner and as a
   user-owned worker; no raw patient data is sent to an LLM provider.

## 4. Reproduce a run

```bash
avail-omics run-generic --data-root my_data --out results
avail-omics robustness  --data-root my_data --out results
avail-omics sensitivity --data-root my_data --out results
avail-omics figures     --out results
```

The generic input contract is `metadata.tsv` plus `features/*.tsv`; missing
subjects per modality are expected and handled by availability flags.

## 5. Tests

```bash
pytest tests/ -q
```

The current suite (24 tests) covers preprocessing, mask/flag semantics,
leak-free leave-one-out smoke tests, the generic loader, offline mode, imaging
utilities, radiomics utilities and the worker.

## 6. Data, privacy and limitations

- Raw patient data cannot be shared. The live demo uses a fixed,
  de-identified cohort excerpt.
- The public pipeline excludes subjects with **no modality at all** and reports
  the count; the manuscript also reports the n=147 sensitivity for this
  exclusion (0.663 vs 0.690 at rung 5).
- External cohorts are independent public datasets; the ds004392 imaging-only
  result is reported as no detectable transfer, not as anti-learning.
- MCI remains the hardest class; the framework reports a graded risk score
  rather than claiming a hard MCI label.
- This is research software, not a clinical decision tool.

## 7. Documentation

- [DEPLOY.md](DEPLOY.md) — stable deployment options
- [BYO_COMPUTE.md](BYO_COMPUTE.md) — local runner, self-hosted, cloud, worker
- [DATA_SECURITY.md](DATA_SECURITY.md) — outbound-call inventory and offline mode
- [WORKER.md](WORKER.md) — attaching a user-owned worker
- [IMAGING_ADAPTER.md](IMAGING_ADAPTER.md) — DICOM/NIfTI to connectivity/radiomics

## 8. Contact

Please use the repository issue tracker for reproducibility questions or
open an issue with the exact command, package version and run manifest.
