<p align="center">
  <img src="logo.png" width="140" alt="AVAIL-AI logo">
</p>

<h1 align="center">AVAIL-AI</h1>
<p align="center"><b>Availability-Aware Multi-Omics Integration &amp; Learning</b><br>
Every subject with at least one modality contributes to the model &mdash; no complete-case discard; all-missing subjects are explicitly reported.</p>

<p align="center"><b>Authors:</b> Busranur Delice &middot; Ahmet Kaplan <i>(corresponding)</i> &middot; Suleyman Yildirim <i>(corresponding)</i></p>

<p align="center">
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/v/avail-omics?color=0e7490" alt="PyPI"></a>
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/pyversions/avail-omics" alt="Python"></a>
  <a href="https://github.com/busdel/avail-cloud-runner/blob/main/LICENSE"><img src="https://img.shields.io/github/license/busdel/avail-cloud-runner" alt="MIT license"></a>
  <a href="https://github.com/busdel/avail-cloud-runner/releases"><img src="https://img.shields.io/github/v/release/busdel/avail-cloud-runner" alt="Latest release"></a>
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/dm/avail-omics" alt="Downloads"></a>
</p>

<p align="center">
</p>

---

## At a glance

| | |
|---|---|
| Current package | **avail-omics 1.9.4** on PyPI |
| Python | >=3.9 |
| Public demo | fixed de-identified CPU demo; no login/upload |
| Public demo | fixed CPU demo + illustrative Analyze/Dr. AVAIL/Imageomics |
| Docs | [datasets/methods](docs/DATASETS_AND_METHODS.md), [package interface](docs/PACKAGE_INTERFACE.md), [web app](docs/WEBAPP.md) |
| Citation | [CITATION.cff](CITATION.cff) |

## What is AVAIL-AI?

A framework for multimodal clinical cohorts with **missing modalities**. Built for the
oral&ndash;gut&ndash;brain axis in Parkinson's cognitive trajectory (HC vs MCI vs PDD),
combining fMRI connectivity, 16S rRNA, shotgun metagenomics (MGS + KEGG functional
modules) &mdash; and generalizable to **any tabular omics layers**, with genuine
missingness handled as data instead of discarded or fabricated.

### Pipeline

1. **Base learners** &mdash; one Random Forest per modality, leak-free leave-one-out (LOO)
2. **Meta-features** &mdash; `[p_1..p_K, flag_1..flag_K]` per subject (probabilities + availability flags)
3. **11 meta-learners** &mdash; SimpleMean, WSimpleMean, ElasticNetLR, MLP, MoE, GNN,
   Transformer, XGBoost, TabNet, TabPFN, TabICL
4. **Strategy benchmark** &mdash; 14 missing-data strategies (including hybrid) + decision curves,
   bootstrap CIs, conformal prediction sets, calibration
5. **Biomarker discovery** &mdash; SHAP importance + differential abundance (Mann&ndash;Whitney, BH-FDR)
6. **AI advisor** &mdash; model decision, biomarker plausibility with **real PubMed citations**
   (free NCBI E-utilities, no API key), structured interpretation
7. **Publication figures** &mdash; PDF plus 600 dpi PNG/TIFF

### Why it is different

- **Missingness is data, not noise** &mdash; availability flags enter the
  representation; sparse subjects are kept, never imputed away or dropped.
  Subjects with no modality at all are excluded by the public pipeline and
  reported; the reference protocol's n=147 sensitivity is 0.663 vs 0.690 at rung 5.
- **Leak-free by construction** &mdash; fold-internal preprocessing, meta-level
  leave-one-out, permutation nulls at chance.
- **Benchmarked, not just built** &mdash; every design choice quantified against
  14 missing-data strategies and 11 meta-learners.
- **Role-separated public evidence** &mdash; **MetaCardis** (n=1087;
  reproduction + coverage), **AVAIL cohort** (n=150; application check),
  **OpenNeuro ds004392** (n=53; imaging-only failure audit), and
  **OpenNeuro ds004856** (n=295 labelled; coverage audit). ds004856 is not a
  performance claim.
- **Group-aware readouts** &mdash; every run reports HC / MCI / PDD breakdowns:
  coverage by group, per-class one-vs-rest AUC, and group profiles of the top
  differential features (with individual subjects shown).
- **Yours to run** &mdash; local runner, self-hosted, Docker/Colab, or your own
  server. Raw data is never sent to AI providers or used for model training;
  user jobs run on the compute you choose. The local-runner
  bundle can also act as a worker (`python run_me.py --serve`) so the web page
  dispatches jobs straight to your machine, no pip install needed.

## Current evidence snapshot (11-model, 2026-11)

These are the current reference results, not universal performance claims.
See [docs/DATASETS_AND_METHODS.md](docs/DATASETS_AND_METHODS.md) for definitions.

| Comparison | Result | Reading |
|---|---|---|
| MetaCardis paired ΔAUC | −0.003 [−0.009, 0.002] | no difference |
| AVAIL application paired ΔAUC | +0.014 [−0.107, 0.122] | inconclusive |
| AVAIL HC vs PDD | 0.850 all / 0.795 complete | dedicated binary contrast |
| ds004392 3-class | 0.386 | imaging-only failure audit |
| ds004392 binary PDD vs rest | 0.850–0.900, exploratory | 5 PDD cases; not validation |

The take-home message is **coverage, role-separated validation and honest
failure reporting**, not a state-of-the-art 3-class leaderboard. The software
and reproducible evaluation protocol are the contribution.

### Limitations

- ds004392 has only five PDD cases; binary PDD results are exploratory.
- ds004856 has zero positive complete cases; it is a coverage audit only.
- The AVAIL cohort is an application/regression check, not external validation.
- Factorial-grid evidence is reported for the active cohorts; archived
  cross-disease analyses are not part of the current scope.

## Validation suite and coverage audit

The reference evaluation uses role-separated evidence rather than a leaderboard:

| Cohort | n | What it tests |
|---|---:|---|
| **MetaCardis** | 1087 (CC 536) | natural, informative missingness |
| **AVAIL cohort** | 150 | end-to-end software use; no new biology |
| **OpenNeuro ds004392** | 53 | imaging-only transfer; failure handling |
| **OpenNeuro ds004856** | 295 labeled | complete-case bias only |

The **ds004856 coverage audit** is intentionally not a performance result.
Its processed OpenNeuro tables contain 295 amyloid-labeled subjects, but the
complete-case intersection (MRI + tau + cognition) has **60 subjects and zero
amyloid-positive subjects**; all 73 MRI-absent subjects are positive and the
60 tau-available subjects are negative. A single availability indicator
reaches AUC 0.82, so complete-case and all-subject AUCs are not interpretable
as biological estimates. The cohort is retained only to show that
complete-case analysis can be structurally non-estimable; it is not used to
claim diagnostic accuracy.



The companion AVAIL cohort (n=150) is an application/regression check:
biological interpretation belongs to the application-specific analysis; this repository
uses the same verified cohort to exercise the command line, figures, report
and R export end to end.

## Cloud runner demo

The original cloud-runner landing is preserved as a static demo at
[cloud-runner/](cloud-runner/); it links to the public demo, package and deployment documentation.

## Public demo

A fixed, de-identified CPU demo is available at [demo/](demo/). It accepts no
account and no uploads; user data stays in the package, local runner,
Docker/Colab or a user-owned worker. Screenshots and the privacy model are in
[docs/WEBAPP.md](docs/WEBAPP.md).

## Install the Python package

**PyPI: [avail-omics](https://pypi.org/project/avail-omics/)** (latest: **v1.9.4**, MIT, Python **>=3.9**) &mdash; `pip install avail-omics`

**Release status:** use **1.9.4**. Earlier `1.0.x`–`1.9.2` releases are
yanked; `1.9.3` remains available as the previous stable release.


```bash
pip install avail-omics            # core (CPU is fine)
pip install "avail-omics[deep]"    # + torch, TabNet, TabPFN, TabICL

avail-omics run-generic --data-root mydata --out results
avail-omics robustness  --data-root mydata --out results   # leak-free permutation
avail-omics sensitivity --data-root mydata --out results   # modality ablation
avail-omics figures     --out results
avail-omics report      --out results   # AI advisor report (own LLM key)
avail-omics chat        --out results   # results-grounded Dr. AVAIL agent
avail-omics plot --out results --data-root mydata \
    --type network,roc,availability,umap,volcano,forest   # soft-palette figures
```

Generic data layout (any omics layers):

```
mydata/
  metadata.tsv        # SampleID + Group (+ optional Age, Education, Sex)
  features/
    brain.tsv         # rows = samples, first column = SampleID, rest = features
    stool16S.tsv
  compositional.txt   # optional: one modality name per line (log/CLR layers)
```

Outputs: `meta_benchmark_results.tsv`, `per_class_auc.tsv`, `loo_predictions.tsv`,
`biomarkers_top20_per_modality.tsv`, `diff_abundance_*.tsv`, publication figures,
and the AI advisor report (with PubMed literature).

## Library interface

The same package is both a CLI and a Python library. Full interface reference:
[docs/PACKAGE_INTERFACE.md](docs/PACKAGE_INTERFACE.md).

```python
from avail_omics.generic import run_generic_pipeline
from avail_omics import viz

run_generic_pipeline("mydata", "results", classes=["HC", "MCI", "PDD"])
viz.plot_roc_calibration("results")
viz.plot_availability("mydata")
```

Standard output includes benchmark tables, per-class AUC, LOO predictions,
biomarkers/differential abundance, and `fig1`–`fig9` as PDF/PNG/TIFF.

## Visualization toolkit (v1.9.4)

Soft-palette, publication-ready figures from the package:

The package API exposes
`network`, `roc`, `availability`, `umap`, `volcano`, `forest`, and module-aware
network figures; the application-specific examples are kept outside the wheel.

```bash
avail-omics plot --out results --data-root mydata \
  --type network,roc,availability,umap,volcano,forest

# module-aware network, SparCC-like correlations, cross-module edges
avail-omics plot --out results --data-root mydata --modality taxa \
  --type network_modules --method sparcc --combined --taxonomy taxonomy.tsv
```

Figures use the CPAL palette and a publication-style preset (Arial,
7–8 pt, 600 dpi PDF/PNG/TIFF, panel letters). For dense networks, use the
clean `module_panels` (selected-module networks + cross-module heatmap) or
`module_summary` (module sizes, cross-module heatmap, strongest pairs)
instead of a hairball; `--method sparcc|spearman|pearson` selects the
correlation.

Also included: UMAP/PCA, volcano and forest plots. Correlations: `sparcc`
(CLR approximation), `spearman` or `pearson`. Full gallery:

**Module-aware network** — automatic community detection, per-module panels
or `--combined` single view with cross-module edges, hub highlighting,
taxonomy colouring and a `cross_module_edges.tsv` export.

## R interoperability

```bash
avail-omics export-r --out results --data-root mydata \
  --dest r_export --taxonomy taxonomy.tsv
```

Writes phyloseq-ready `otu_table.tsv`, `sample_data.tsv`, `tax_table.tsv`,
`feature_table.tsv`, result/network tables and an R guide covering
**phyloseq**, **vegan**, **igraph**, **SpiecEasi**, **NetCoMi** and
**mixOmics/PMA**. No R packages are bundled. Full guide:
[docs/R_INTEROP.md](docs/R_INTEROP.md).

## Research platform and own-data analysis

- **Public overview:** https://busdel.github.io/avail-cloud-runner/ &mdash; static introduction; **no uploads, no public compute**.
- **Canonical project page:** https://busdel.github.io/avail-cloud-runner/
- **Own data:** install the package, use the local runner, Docker, Colab or your
  own worker. See [BYO_COMPUTE.md](BYO_COMPUTE.md) (four deployment tiers),
  [DATA_SECURITY.md](DATA_SECURITY.md) and [WORKER.md](WORKER.md).

Imaging is first-class: raw rs-fMRI and T1 volumes are processed end-to-end
(Schaefer-100 connectivity + optional radiomics) on **your** machine &mdash; see
[IMAGING_ADAPTER.md](IMAGING_ADAPTER.md).

## LLM advisor

Optional AI automation (model selection, biomarker literature synthesis, results
interpretation). Any OpenAI-compatible endpoint works; Gemini is the default and
**Groq/OpenRouter free tiers** are automatic fallbacks on quota or provider
outages.

```bash
avail-omics set-key            # saves to ~/.config/avail/llm_key (0600)
avail-omics report --out results   # llm_advisor.json + llm_advisor_report.md
avail-omics chat --out results     # results-grounded Dr. AVAIL agent
avail-omics figures --out results
avail-omics advise  --out results
```

## Documentation

| File | What it covers |
|---|---|
| [DEPLOY.md](DEPLOY.md) | One-command deployment of the web platform |
| [BYO_COMPUTE.md](BYO_COMPUTE.md) | The four bring-your-own-compute tiers & security model |
| [WORKER.md](WORKER.md) | Attach your own server as a compute worker |
| [IMAGING_ADAPTER.md](IMAGING_ADAPTER.md) | Raw rs-fMRI / T1 processing protocol |
| [DATA_SECURITY.md](DATA_SECURITY.md) | Outbound-call inventory, offline mode |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
| [docs/DATASETS_AND_METHODS.md](docs/DATASETS_AND_METHODS.md) | Cohort cards, leakage controls, models, factorial designs, limitations |
| [docs/WEBAPP.md](docs/WEBAPP.md) | Web app tour, privacy model, screenshots |
| [docs/PACKAGE_INTERFACE.md](docs/PACKAGE_INTERFACE.md) | CLI, Python API, output contract, figure exports |

## Data & tools

External datasets and methods used by the platform and the demo:

- **OpenNeuro ds004392** — Wylie KP et al., *Parkinson's disease, functional
  connectivity, and cognition* (CC0) — [DOI 10.18112/openneuro.ds004392.v1.0.2](https://doi.org/10.18112/openneuro.ds004392.v1.0.2)
- **Schaefer atlas** — [DOI 10.1093/cercor/bhx179](https://doi.org/10.1093/cercor/bhx179)
- **Nilearn** — [DOI 10.3389/fninf.2014.00014](https://doi.org/10.3389/fninf.2014.00014) ·
  **AFNI** — [DOI 10.1006/cbmr.1996.0014](https://doi.org/10.1006/cbmr.1996.0014) ·
  **PyRadiomics** — [DOI 10.1158/0008-5472.CAN-17-0339](https://doi.org/10.1158/0008-5472.CAN-17-0339)
- **scikit-learn** — [JMLR 2011](https://jmlr.org/papers/v12/pedregosa11a.html) ·
  **SHAP** — [arXiv:1705.07874](https://arxiv.org/abs/1705.07874) ·
  **PubMed E-utilities** — [NCBI documentation](https://www.ncbi.nlm.nih.gov/books/NBK25499/)

## Repository layout

```
avail_ai/avail_ai/      # the avail-omics Python package
run_generic.py          # generic multi-omics pipeline entry point
BYO_COMPUTE.md          # bring-your-own-compute tiers & security guide
.github/workflows/run.yml   # free-cloud compute backend (GitHub Actions)
```

## Citation

If you use AVAIL-AI in academic work, please cite it &mdash; machine-readable
metadata is in [CITATION.cff](CITATION.cff):

```bibtex
@software{delice2026avail,
  title   = {AVAIL Research Platform: availability-aware multimodal analytics for incomplete clinical cohorts},
  author  = {Delice, Busranur and Kaplan, Ahmet and Yildirim, Suleyman},
  version = {1.9.4},
  year    = {2026},
  url     = {https://avail-research-platform.pages.dev/}
}
```

Citation metadata will be updated when a DOI is assigned.

## Authors

- **Busranur Delice**
- **Ahmet Kaplan** — corresponding author
- **Suleyman Yildirim** — corresponding author

## License

MIT &mdash; see [LICENSE](LICENSE).
