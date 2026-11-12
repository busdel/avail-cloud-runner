# Paper division and novelty boundary

## Paper 1 — Bioinformatics
**Question:** biological and clinical meaning of availability-aware multimodal
integration in the Parkinson's cognitive spectrum.
**Claim:** imaging + oral/gut microbiome layers carry complementary information
across HC, MCI and PDD; missingness is part of the clinical phenotype.
**Evidence:** biological cohort results, module-additive ladder, external
biological readings, feature-level interpretation.
**Not claimed:** software novelty, web platform novelty, exhaustive
missing-data benchmark, or a universal 3-class leaderboard.

## Paper 2 — GigaScience
**Question:** can an availability-aware framework and software ecosystem
support incomplete multimodal cohorts without complete-case discard or
fabricated measurements?
**Claim:** the contribution is the framework + reproducible software +
leak-free benchmark + binary clinical contrasts + honest failure auditing.
**Evidence:** architecture, cohort-role separation, binary endpoints across
MetaCardis/HMP2/AVAIL, factorial designs, imaging failure/coverage audit,
web platform, reproducibility package.
**Not claimed:** new biology, diagnostic utility, or a state-of-the-art
3-class result. The 3-class analyses are supplementary.

## Shared material
- Same de-identified cohorts and derived feature tables, with role separation.
- Same code base and package version; Paper 1 cites Paper 2 for software.
- Paper 2 cites Paper 1 for biological interpretation; neither paper duplicates
  the other's primary claim.
- 3-class results: Supplementary Figures S1/S2 and supplementary tables in
  Paper 2; not the main evidence.

## Figure ownership
| Figure | Paper 2 | Paper 1 |
|---|---|---|
| Architecture | Figure 1 | — |
| Cohort roles / coverage | Figure 2 | — |
| Binary endpoints across cohorts | Figure 3 | — |
| Imaging binary + coverage audit | Figure 4 | — |
| 3-class paired/failure | Supplementary S1/S2 | biological figures only |
| Biological/clinical figures | — | Paper 1 figures |

## Novelty statements to use
- Paper 2: “We introduce an availability-aware integration framework and a
  reproducible evaluation protocol that keeps every subject with at least one
  modality, reports failure explicitly, and separates model choice from
  missingness handling in a factorial design.”
- Paper 1: “We show that brain–gut–oral multimodal information stages the
  Parkinson cognitive spectrum and that availability patterns themselves carry
  clinical information.”
