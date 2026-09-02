<p align="center">
  <img src="logo.png" width="140" alt="AVAIL-AI logo">
</p>

<h1 align="center">AVAIL-AI</h1>
<p align="center"><b>Availability-Aware Multi-Omics Integration &amp; Learning</b><br>
Every subject contributes to the model, regardless of which modalities they have &mdash; zero sample loss.</p>

---

## What is AVAIL-AI?

A framework for multimodal clinical cohorts with **missing modalities**. Built for the
oral&ndash;gut&ndash;brain axis in Parkinson's cognitive trajectory (HC vs MCI vs PDD),
combining fMRI connectivity, 16S rRNA, shotgun metagenomics (MGS + KEGG functional
modules) &mdash; and generalizable to **any tabular omics layers**.

### Pipeline

1. **Base learners** &mdash; one Random Forest per modality, leak-free leave-one-out (LOO)
2. **Meta-features** &mdash; `[p_1..p_K, flag_1..flag_K]` per subject (probabilities + availability flags)
3. **11 meta-learners** &mdash; SimpleMean, WSimpleMean, ElasticNetLR, MLP, MoE, GNN,
   Transformer, XGBoost, TabNet, TabPFN, TabICL
4. **Biomarker discovery** &mdash; SHAP importance + differential abundance (Mann&ndash;Whitney, BH-FDR)
5. **AI advisor** &mdash; model decision, biomarker plausibility with **real PubMed citations**
   (free NCBI E-utilities, no API key), manuscript-ready interpretation
6. **Publication figures** &mdash; 600 dpi PDF/PNG + 300 dpi TIFF in the manuscript style

## Web tool

A public web interface: upload a ZIP &rarr; automatic validation + AI data-quality
feedback &rarr; run on the free cloud (GitHub Actions), on the server GPU, or on your
own computer &rarr; AI decision report + publication figures.

> URL and access password are shared on request while the manuscript is under review.

## Try it locally

```bash
pip install avail-omics            # core (CPU is fine)
pip install "avail-omics[deep]"    # + torch, TabNet, TabPFN, TabICL

avail run-generic --data-root mydata --out results
```

Generic data layout (any omics layers):

```
mydata/
  metadata.tsv        # SampleID + Group (+ optional Age, Education, Sex)
  features/
    brain.tsv         # rows = samples, first column = SampleID, rest = features
    stool16S.tsv
    ...
```

Outputs: `meta_benchmark_results.tsv`, `per_class_auc.tsv`, `loo_predictions.tsv`,
`biomarkers_top20_per_modality.tsv`, `diff_abundance_*.tsv`, publication figures,
and the AI advisor report (with PubMed literature).

## LLM advisor

Optional AI automation (model selection, biomarker literature synthesis, results
interpretation). Any OpenAI-compatible endpoint works; Gemini is the default and
**Groq/OpenRouter free tiers** are used as automatic fallbacks when quotas run out.

```bash
avail set-key            # saves to ~/.config/avail/llm_key (0600)
avail figures --out results
avail advise --out results
```

## Repository layout

```
avail_ai/avail_ai/      # the avail-omics Python package
run_generic.py          # generic multi-omics pipeline entry point
.github/workflows/run.yml   # free-cloud compute backend (GitHub Actions)
```

## License

MIT. See the paper for citation (in preparation).
