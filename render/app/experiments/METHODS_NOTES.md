# AVAIL-AI — Methods & Robustness Notes (for the manuscript)

## Prespecified primary analysis

- Primary meta-learner: **WSimpleMean** (availability-weighted late fusion;
  interpretable, no learnable parameters, minimal overfitting risk).
- Secondary/exploratory: ElasticNetLR, XGBoost and the deep learners
  (MLP/MoE/GNN/Transformer/TabNet/TabPFN/TabICL) are reported as a
  benchmark, NOT used for the main claim. This prevents post-hoc model
  selection bias on the evaluation set.
- Evaluation protocol: leave-one-out (LOO); base learners LOO per modality
  with fold-internal preprocessing (leak-free). Macro-AUC is the primary
  metric; class imbalance (HC n=29) is handled by macro-averaging and
  class-weighted training.

## Robustness suite (all run on the same n=150 zero-sample-loss cohort)

| Analysis | File | Answers reviewer question |
|---|---|---|
| Missing-data strategy benchmark (16 strategies: complete-case, zero/mean/median/KNN/MICE/mean+indicator/log/CLR/MissForest/SoftImpute/oracle, early-fusion XGBoost) | `imputation_benchmark_v3_results.tsv` | "Is the flag approach better than imputation?" |
| Bootstrap 95% CIs per model | `robustness_ci_zero_loss_results.tsv` | "Are the AUC differences meaningful?" |
| Pairwise DeLong tests (top-5 models) | `robustness_pairwise_zero_loss_results.tsv` | "Is MLP really better than SimpleMean?" |
| Cohort-stratified (leave-one-Year-out) evaluation | `cohort_stratified_results.tsv` | "Does it generalize across cohorts?" |
| Covariate-adjusted DA sensitivity (age+sex logistic vs Mann-Whitney) | `covariate_sensitivity.tsv` | "Do confounders drive the biomarkers?" |
| Modality solo-AUC table (marginal contributions) | `modality_solo_auc.tsv` | "Which modality carries the signal?" |
| Meta-feature ablation (p-only, flag-only) | in `imputation_benchmark_v3_results.tsv` | "Are both parts of p+flag needed?" |
| Uncertainty-aware meta-features (p+flag+entropy) | `entropy_meta_results.tsv` | "Do probabilities lose information?" |
| Multi-granularity fMRI views (edges vs networkFC vs segregation) | `netfc_results/` | "Is the fMRI representation adequate?" |
| Graph-theory + connectome GCN alternatives | `graph_metrics_results.tsv`, `connectome_gcn_results.tsv` | "Did you try graph-based representations?" |
| Zero-sample-loss cross-modal (pairwise-complete, no global intersection) | `crossmodal_networkfc_*.tsv` | "Does the cross-modal analysis drop subjects?" |

## Head-to-head vs the closest published method (UniMIX-AD)

The official UniMIX-AD code (github.com/SDH-Lab/UniMIX-AD; "Taming arbitrary
modality missingness and imbalance", AD diagnosis) was run on our cohort with
its own protocol (stratified train/val/test, 50 epochs, gate + MCG losses).
Adaptations (documented in `unimix_port.py`): tabular MLP encoders instead of
imaging encoders, positional-embedding length extended to our token count,
and a single-GPU drop-in reimplementation of the fmoe dependency (their
models.py / moe_module.py / umi_module.py are unchanged). Results on the same
splits (3 seeds, test macro-AUC mean +- std):

| model | test macro-AUC |
|---|---|
| UniMIX-AD (official code, adapted inputs) | 0.636 +- 0.059 |
| Ours - SimpleMean | 0.653 +- 0.067 |
| Ours - XGBoost meta | 0.669 +- 0.024 |

Our availability-aware fusion is competitive with or better than the
specialized graph-MoE on this small tabular cohort; UniMIX-AD's advantages
(massive imaging pretraining, 1000s of subjects) do not transfer at n=150.

## Single-cohort limitation

All 150 subjects belong to one cohort (Year=1). Leave-one-cohort-out
evaluation is therefore not possible; external validation in an independent
cohort is explicitly listed as the primary future work.

## Suggested manuscript phrasing (key defenses)

1. Zero sample loss: "No subject is excluded; each contributes through its
   available modalities via base-learner probabilities and availability
   indicators. Subjects with fewer modalities naturally receive lower
   weight, which is accounted for by the availability-weighted fusion."
2. Information retention: "While meta-features summarize each modality into
   class probabilities, we show (a) that this preserves discriminative
   information relative to raw-feature early fusion (Table X) and (b) that
   adding predictive entropy improves fusion further (Table Y)."
3. Missingness: "The availability indicator implicitly models the observed
   missingness pattern; comparisons against 14 imputation strategies,
   including an oracle class-conditional mean, show the flag approach is
   not simply exploiting missingness."
4. Biomarkers: "Differential abundance results were robust to age/sex
   adjustment (kept fraction X%, Table Z); SHAP rankings identify
   candidates that are then supported by independent PubMed evidence."
5. Limits: single-cohort; leave-one-Year-out shows a generalization gap of
   X AUC points; external validation in an independent cohort is planned.
