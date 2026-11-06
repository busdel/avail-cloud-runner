# Validation suite and coverage audit

This note records the role-separated evaluation suite used by the software
paper. It is not a leaderboard, and it is not a clinical validation claim.

| Role | Dataset | n | Blocks | Missingness | Claim |
|---|---|---:|---|---|---|
| Anchor | MetaCardis | 1087 (complete-case 536) | 6 | natural, informative | reproduction + coverage |
| Independent replication | HMP2 / IBD | 130 | multi-omics | complete-block subset | disease/generalisation, not missingness |
| Imaging failure mode | OpenNeuro ds004392 | 53 | imaging-only | none | no detectable transfer; failure handling |
| Companion application | AVAIL cohort | 150 | five modules | natural | end-to-end software use; no new biology |
| Coverage audit | OpenNeuro ds004856 | 295 labeled | MRI + tau + cognition | label-confounded | complete-case bias only |

## ds004856 coverage audit

Source: OpenNeuro ds004856 (Dallas Lifespan Brain Study), CC0,
`10.18112/openneuro.ds004856.v1.3.0`; paper
`10.1038/s41597-025-04847-7`. The derived input is built from the
`derivatives/brainsummary` tables only; raw imaging is not redistributed.

- Labeled subjects: 295 (115 amyloid-positive, 180 negative).
- Modality availability: MRI 222, tau 60, cognition 295.
- Complete-case intersection: 60 subjects, **0 positive / 60 negative**.
- MRI-absent stratum: 73 subjects, **73 positive / 0 negative**.
- Tau-available stratum: 60 subjects, **0 positive / 60 negative**.
- Single availability indicator (MRI absent): AUC 0.817.

Because the processed tables have availability that is perfectly associated
with one class, complete-case performance is structurally non-estimable and
all-subject AUCs are not biological estimates. This cohort is used only to
demonstrate why complete-case analysis can be impossible and why role
separation and availability reporting are needed. No diagnostic accuracy is
claimed.

Artifacts: `data_demos/ds004856_avail_dataset/` (derived tables and dataset
card), `experiments/results/ds004856_coverage_audit.{tsv,json}`.
