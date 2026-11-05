# AVAIL-AI — Theoretical Notes (draft for the manuscript)

## Notation

- Subjects i = 1..n, classes k = 1..K, modalities m = 1..M
- R_m(i) = 1 if subject i has modality m (availability)
- Base learner m produces calibrated probability p_m(i) = (p_m1, ..., p_mK)
- Meta-feature for subject i, modality m:
  - available:   [p_m(i), 1]
  - missing:     [0, ..., 0, 0]
- Fusion: f(x_i) with x_i = concat of the M blocks

## Lemma 1 (pattern-mixture equivalence in the calibrated space)

If the base probabilities are (approximately) calibrated,
P(Y = k | p_m(i)) ≈ p_mk(i), then the zero+indicator encoding
[p_m(i) * R_m(i), R_m(i)] is informationally equivalent to a
pattern-mixture model: conditioning the fusion on the availability
pattern R(i) = (R_1(i), ..., R_M(i)) through the indicator part, while
the probability part supplies the pattern-conditional sufficient
statistics. A sufficiently flexible meta-learner can therefore recover
any pattern-mixture fusion rule.

Argument: for each pattern r, the set of subjects with R(i) = r has
meta-features that are a deterministic function of their calibrated
base probabilities; the indicators identify r exactly. Hence the
meta-learner implements M-specific fusion rules f_r — precisely the
definition of a pattern-mixture model.

## Lemma 2 (indicator dimension: decision-level vs raw-level)

The missingness pattern is encoded with M indicator dimensions at the
decision level, versus sum_m d_m indicator dimensions at the raw level
(d_m = feature count of modality m). With n subjects, a flexible
meta-learner fits f on a feature space of dimension M(K+1) << sum_m d_m,
so the indicator is estimable at small n. At the raw level the
impute-and-indicate estimator must estimate indicator coefficients in a
sum_m d_m-dimensional space — infeasible or high-variance at the n of
clinical cohorts.

## Theorem 1 (complement leak of LOO mean imputation)

Consider mean imputation of an artificially (or structurally) missing
feature with the fold-internal mean computed over the LOO training set
T_k = {1..n} \ {k}. For the left-out subject k, the imputed value is

    x*_k = (S - x_k) / (n - 1),   S = sum_i x_i.

The imputed value is therefore a linear (complement) function of the
subject's own unobserved value x_k, and jointly over subjects the
imputed matrix is invertible to recover {x_k}. A supervised model
trained on the imputed training rows and evaluated on the imputed test
row can, in the limit of a well-specified learner, recover information
about x_k — hence about y_k whenever x_k is predictive of y_k. This is a
leakage: LOO mean imputation is NOT leave-one-out honest for predictive
evaluation. (Discovered empirically in our benchmark: mean-imputation
AUC rose monotonically with the artificial missingness rate - the exact
opposite of what an honest estimator can do.)

Proof sketch: S is constant across folds; x*_k is an affine function of
x_k with slope -1/(n-1) != 0. Hence the map x_k -> x*_k is injective.

Corollary: the AVAIL encoding is leak-free under LOO because the missing
block contributes the CONSTANT (0, 0) — no complement statistics of the
left-out subject are ever computed.

## Proposition (calibration is sufficient for uniform fusion)

If all base learners are calibrated and their errors are conditionally
independent given the availability pattern, the availability-weighted
mean (WSimpleMean with fold-internal AUC weights) is the minimum-variance
linear fusion among weights that sum to one. (Standard inverse-variance
weighting argument; included for completeness.)

## Hybrid strategy (imputed signal + real availability flags)

Motivation: the flag encoding is assumption-free but discards the
information in partially-observed features; imputation recovers that
signal but is only unbiased when the missingness mechanism matches the
imputer's assumptions (MAR for mean/MICE). The hybrid feeds BOTH to the
meta-learner: base predictions computed from fold-internal imputed
features, plus the TRUE availability flag of the raw (unimputed) layer.

Claim: for any missingness mechanism in {MCAR, MAR, MNAR}, the hybrid's
effective input space strictly contains that of the flags-only encoding
(the flag block is preserved) AND that of imputation-only (the imputed
block is preserved). Under a well-specified meta-learner the hybrid is
therefore at least as discriminative as the better of the two
component strategies, up to finite-sample estimation error.

Leakage note: imputation must remain fold-internal (imputer fitted on
training subjects only); the availability flags are computed from the
raw layer, never from the imputed matrix. This keeps the hybrid
leave-one-out honest.


## Availability as data: four properties (engineering paper, Section 3)

1. **Any pattern, one representation.** Every availability pattern maps to
   the same fixed-length vector z = [probabilities, flags] per modality;
   missing modalities contribute zeros with a flag of 0. The meta-learner
   needs no pattern-specific parameters and accepts patterns never seen in
   training, including single-modality subjects.

2. **Why complete-case cannot repair informative missingness.** When
   availability depends on the outcome, the complete-case subpopulation is
   a biased sample; the risk gap between complete-case and the population
   contains a term no learner trained on complete cases alone can remove.
   Availability flags let the fused model condition on the missingness
   pattern, so the correction is representable.

3. **The flags-only ceiling.** A predictor that uses only availability
   indicators cannot exceed the AUC of the pattern itself: 0.527
   (three-class) and 0.658 (HC-PDD) on the application cohort. Anything
   beyond this ceiling comes from measured signal, not from the
   informative-missingness shortcut.

4. **Per-pattern coverage.** Mondrian conformal prediction keeps its
   finite-sample coverage guarantee within each availability pattern
   (measured 89-94% per pattern, mean set size about two labels).

## How to read net-benefit (decision) curves

A decision curve answers a practical question: at which risk threshold
does acting on the model beat "treat everyone" and "treat none"? Net
benefit = TP/n - FP/n x theta/(1-theta). On the HC-PDD task the flag
representation (teal, AVAIL) stays above the matched complete-case arm
(red) across all clinically relevant thresholds: delta-NB about
+0.28 to +0.35, i.e. roughly 280-350 additional net detections per
1,000 subjects at the same decision rule. On MCI detection the two arms
overlap in the central range - the framework's value in this cohort is
coverage and stability, not per-subject superiority.

## The strategy space is a factorial experiment

The sixteen-plus missing-data strategies are one formal 2x2x2 design:
imputation (none vs imputed) x availability flags (off vs true) x fusion
(fixed averaging vs learned). p-only is (0,0); the framework's flag arm is
(0,1); standard imputation is (1,0); the hybrid is (1,1). Interaction
terms estimate what each ingredient contributes, with FDR control across
the contrast family (Supplementary Table S12).
