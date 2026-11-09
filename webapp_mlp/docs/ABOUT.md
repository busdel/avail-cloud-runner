# About

**AVAIL-omics** is an availability-aware multimodal integration framework for
clinical cohorts with missing modalities. Each modality trains a base learner
on the subjects who have it; a meta-learner combines the base probabilities
with explicit availability flags, so subjects are not discarded or imputed
away.

This web interface is a **demo**, not a medical device and not a clinical
decision tool. It is intended for:

- exploring the input/output format,
- checking availability and sample loss,
- demonstrating the software workflow.

## What the demo does not do

- It does not run the full 11-model benchmark.
- It does not make claims about diagnostic accuracy.
- It does not accept identifiable patient data.
- It does not replace local analysis or the published validation protocol.

## Citation and links

- Package: `avail-omics` (PyPI)
- Repository: https://github.com/busdel/avail-cloud-runner
- Validation notes: https://github.com/busdel/avail-cloud-runner/blob/main/avail_ai/VALIDATION.md
