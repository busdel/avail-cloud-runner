# -*- coding: utf-8 -*-
"""R interoperability export.

Writes a self-contained folder that can be loaded by phyloseq/vegan/
SpiecEasi/NetCoMi/igraph/mixOmics without re-implementing the AVAIL pipeline.
No R packages are bundled; the README lists the optional packages and snippets.
"""
from pathlib import Path
import shutil

import pandas as pd


def _read_features(data_root):
    feats = {}
    fdir = Path(data_root) / "features"
    if fdir.exists():
        for p in sorted(fdir.glob("*.tsv")):
            try:
                d = pd.read_csv(p, sep="\t")
                d["SampleID"] = d["SampleID"].astype(str).str.strip()
                feats[p.stem] = d.set_index("SampleID")
            except Exception:
                continue
    return feats


def export_r(out, data_root=None, dest=None, taxonomy=None):
    out = Path(out)
    dest = Path(dest) if dest else out / "r_export"
    dest.mkdir(parents=True, exist_ok=True)
    # Results tables
    for name in ("meta_benchmark_3class_results.tsv",
                 "meta_benchmark_results.tsv", "per_class_auc.tsv",
                 "loo_predictions.tsv", "diff_abundance_all.tsv",
                 "biomarkers_top20_per_modality.tsv",
                 "biomarkers_literature.tsv"):
        src = out / name
        if src.exists():
            shutil.copy2(src, dest / name)
    for name in ("network.graphml", "network_edges.tsv",
                 "cross_module_edges.tsv", "network_metrics.json"):
        src = out / name
        if not src.exists():
            src = out.parent / name
        if src.exists():
            shutil.copy2(src, dest / name)
    # Data layout
    meta_src = None
    if data_root:
        root = Path(data_root)
        for cand in (root / "metadata.tsv", root / "metadata.csv"):
            if cand.exists():
                meta_src = cand
                break
    if meta_src is not None:
        meta = pd.read_csv(meta_src, sep="\t" if meta_src.suffix == ".tsv" else ",",
                           dtype=str)
        meta.to_csv(dest / "sample_data.tsv", sep="\t", index=False)
        ids = meta.iloc[:, 0].astype(str).str.strip().tolist()
        feats = _read_features(data_root)
        if feats:
            frames = []
            for m, d in sorted(feats.items()):
                d = d.reindex(ids)
                d = d.apply(pd.to_numeric, errors="coerce")
                d = d.loc[:, d.notna().any(axis=0)]
                d.columns = [f"{m}::{c}" for c in d.columns]
                frames.append(d)
            if frames:
                wide = pd.concat(frames, axis=1)
                wide.insert(0, "SampleID", wide.index)
                wide.to_csv(dest / "feature_table.tsv", sep="\t", index=False)
                otu = wide.set_index("SampleID").T
                otu.index.name = "Feature"
                otu.to_csv(dest / "otu_table.tsv", sep="\t")
                tax_rows = []
                for c in otu.index:
                    mod, _, feat = str(c).partition("::")
                    tax_rows.append({"Feature": c, "Modality": mod,
                                     "FeatureName": feat})
                tax = pd.DataFrame(tax_rows).set_index("Feature")
                if taxonomy:
                    td = pd.read_csv(taxonomy, sep="\t")
                    fcol = "feature" if "feature" in td.columns else td.columns[0]
                    gcol = next((c for c in ("group", "taxon", "phylum",
                                             "class", "family", "genus")
                                 if c in td.columns), None)
                    if gcol:
                        mp = dict(zip(td[fcol].astype(str), td[gcol].astype(str)))
                        tax["Taxon"] = [mp.get(str(c).split("::")[-1], "unassigned")
                                        for c in tax.index]
                tax.to_csv(dest / "tax_table.tsv", sep="\t")
    (dest / "README_R.md").write_text('''# AVAIL → R export

This folder contains the AVAIL outputs and a phyloseq-ready data layout.
No R packages are bundled; install the optional packages you need.

## phyloseq

```r
library(phyloseq)
otu <- read.delim("otu_table.tsv", row.names = 1, check.names = FALSE)
sam <- read.delim("sample_data.tsv", row.names = 1, check.names = FALSE)
tax <- read.delim("tax_table.tsv", row.names = 1, check.names = FALSE)
ps  <- phyloseq(
  otu_table(as.matrix(otu), taxa_are_rows = TRUE),
  sample_data(sam),
  tax_table(as.matrix(tax))
)
```

## vegan

```r
library(vegan)
d <- vegdist(t(as.matrix(otu)), method = "bray")
adonis2(d ~ Group, data = sam)
mantel(d, dist(t(as.matrix(otu))))
```

## igraph / GraphML

```r
library(igraph)
g <- read_graph("network.graphml", format = "graphml")
```

## SpiecEasi / NetCoMi (optional)

```r
library(SpiecEasi)
se <- spiec.easi(t(as.matrix(otu)), method = "mb",
                 lambda.min.ratio = 1e-2, nlambda = 20)

library(NetCoMi)
net <- netConstruct(t(as.matrix(otu)), measure = "spieceasi",
                    sparsMethod = "none", verbose = 0)
```

## mixOmics / PMA (sGCCA, DIABLO)

```r
library(mixOmics)
# X: list of omic matrices (samples x features); Y: phenotype
# fit <- block.splsda(X, Y, ncomp = 2, keepX = list(...))
```

Notes: `otu_table.tsv` is features × samples; `feature_table.tsv` is
samples × features. `sample_data.tsv` is the de-identified metadata. Only
derived/de-identified data should be exported.
''')
    print(f"[export] wrote {dest}")
    return dest
