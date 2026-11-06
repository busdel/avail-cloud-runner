# AVAIL → R interoperability

`avail-omics export-r` writes a self-contained folder for R analyses. No R
packages are bundled; install the optional packages you need.

```bash
avail-omics export-r --out results --data-root mydata \
  --dest r_export --taxonomy taxonomy.tsv
```

Files:

| File | Shape | Purpose |
|---|---|---|
| `otu_table.tsv` | features × samples | phyloseq / SpiecEasi / NetCoMi |
| `feature_table.tsv` | samples × features | vegan / mixOmics / generic R |
| `sample_data.tsv` | samples × metadata | phyloseq `sample_data` |
| `tax_table.tsv` | features × taxonomy/modality | phyloseq `tax_table` |
| `meta_benchmark*.tsv`, `per_class_auc.tsv`, `loo_predictions.tsv` | results | model/per-class evidence |
| `diff_abundance_all.tsv`, `biomarkers_top20_per_modality.tsv` | results | feature-level evidence |
| `network.graphml`, `network_edges.tsv`, `cross_module_edges.tsv` | graph | igraph / Cytoscape / Gephi |
| `README_R.md` | guide | copy-paste snippets |

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

Only export de-identified/derived data. These snippets are documentation for
optional third-party packages; their licences and behaviour are governed by
those packages.
