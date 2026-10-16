<p align="center">
  <img src="logo.png" width="140" alt="AVAIL-AI logo">
</p>

<h1 align="center">AVAIL-AI</h1>
<p align="center"><b>Availability-Aware Multi-Omics Integration &amp; Learning</b><br>
Every subject contributes to the model, regardless of which modalities they have &mdash; zero sample loss.</p>

<p align="center">
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/v/avail-omics?color=0e7490" alt="PyPI"></a>
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/pyversions/avail-omics" alt="Python"></a>
  <a href="https://github.com/busdel/avail-cloud-runner/blob/main/LICENSE"><img src="https://img.shields.io/github/license/busdel/avail-cloud-runner" alt="MIT license"></a>
  <a href="https://github.com/busdel/avail-cloud-runner/releases"><img src="https://img.shields.io/github/v/release/busdel/avail-cloud-runner" alt="Latest release"></a>
  <a href="https://pypi.org/project/avail-omics/"><img src="https://img.shields.io/pypi/dm/avail-omics" alt="Downloads"></a>
</p>

<p align="center">
  <img src="docs/assets/tour.gif" width="640" alt="AVAIL web interface &mdash; tour">
</p>

---

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
4. **Strategy benchmark** &mdash; 16 missing-data strategies + hybrid + decision curves,
   bootstrap CIs, conformal prediction sets, calibration
5. **Biomarker discovery** &mdash; SHAP importance + differential abundance (Mann&ndash;Whitney, BH-FDR)
6. **AI advisor** &mdash; model decision, biomarker plausibility with **real PubMed citations**
   (free NCBI E-utilities, no API key), manuscript-ready interpretation
7. **Publication figures** &mdash; 600 dpi PDF/PNG + 300 dpi TIFF

### Why it is different

- **Missingness is data, not noise** &mdash; availability flags enter the
  representation; sparse subjects are kept, never imputed away or dropped.
- **Leak-free by construction** &mdash; fold-internal preprocessing, meta-level
  leave-one-out, permutation nulls at chance.
- **Benchmarked, not just built** &mdash; every design choice quantified against
  16 missing-data strategies and 11 meta-learners.
- **Validated outside the lab** &mdash; Wallen 2022 (PD, n=724), preclinical AD
  (n=1457), HMP2 IBD, MetaCardis (n=1087), and independent rs-fMRI cohorts
  (OpenNeuro ds004392, ds005892) processed end-to-end with the same pipeline.
- **Yours to run** &mdash; free cloud, local runner, self-hosted, or your own
  server. Raw data never sits on maintainer infrastructure.

## Gallery

| Method schematic | Availability patterns |
|---|---|
| <img src="docs/assets/method_schematic.png" width="430"> | <img src="docs/assets/availability_patterns.png" width="430"> |

<p align="center"><img src="docs/assets/architecture.png" width="760" alt="System architecture: data layer, availability-aware core, strategy layer, interfaces"></p>

## Install the Python package

**PyPI: [avail-omics](https://pypi.org/project/avail-omics/)** (latest: **v1.0.8**, MIT, Python 3.10&ndash;3.13) &mdash; `pip install avail-omics`

```bash
pip install avail-omics            # core (CPU is fine)
pip install "avail-omics[deep]"    # + torch, TabNet, TabPFN, TabICL

avail-omics run-generic --data-root mydata --out results
avail-omics robustness  --data-root mydata --out results   # leak-free permutation
avail-omics sensitivity --data-root mydata --out results   # modality ablation
avail-omics figures     --out results
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

## Live web tool

**Stable link: https://busdel.github.io/avail-cloud-runner/** (redirects to the current
session; access password shared on request while the manuscript is under review).

Upload a ZIP &rarr; automatic validation + AI data-quality feedback &rarr; run on the
**free cloud (GitHub Actions)** or **on your own computer** &rarr; AI decision report +
publication figures. The hosted interface **never executes jobs on maintainer
infrastructure**: see [BYO_COMPUTE.md](BYO_COMPUTE.md) for the four deployment tiers
(local runner &middot; self-hosted &middot; your own cloud runner &middot; worker agent for
your own server), [DATA_SECURITY.md](DATA_SECURITY.md) for the outbound-call inventory
and offline mode, and [WORKER.md](WORKER.md) to attach your own server.

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
  title   = {AVAIL-AI: availability-aware multimodal learning for incomplete clinical cohorts},
  author  = {Delice, Busranur},
  version = {1.0.8},
  year    = {2026},
  url     = {https://github.com/busdel/avail-cloud-runner}
}
```

The companion manuscripts are in preparation; this entry will be updated with the DOIs.

## License

MIT &mdash; see [LICENSE](LICENSE).
