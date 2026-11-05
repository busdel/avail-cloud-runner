# -*- coding: utf-8 -*-
"""Experiment: network-level fMRI FC  x  other omics layers.

ZERO-SAMPLE-LOSS version: every correlation is computed on the maximal
pairwise-complete subject set (no global intersection), exactly like the
AVAIL availability strategy. Each association reports its own n.

Outputs: experiments/results/crossmodal_networkfc_*.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")
GEN = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/netfc_generic/features")


def load_fc():
    net = pd.read_csv(GEN / "fMRI_networkFC.tsv", sep="\t")
    seg = pd.read_csv(GEN / "fMRI_segregation.tsv", sep="\t")
    edg = pd.read_csv(GEN / "fMRI_selected_edges.tsv", sep="\t")
    out = []
    for df in (net, seg, edg):
        df["SampleID"] = df["SampleID"].astype(str).str.strip()
        out.append(df.set_index("SampleID"))
    return tuple(out)


def load_16s(fname):
    df = pd.read_csv(ROOT / "busra_pd_multisite" / fname, sep="\t", index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[(df > 0).sum(axis=1) >= 5]
    return df.T  # samples x genera


def load_mgs():
    cw = pd.read_csv(ROOT / "yeni_eklenenler/paper_pipeline/DELIVERABLE/"
                     "raw_outputs/err_patient_crosswalk.tsv", sep="\t")
    cw["patient"] = cw["patient"].astype(str).str.strip()
    cmap = dict(zip(cw["run_accession"].astype(str), cw["patient"]))
    out = {}
    for name, fname in [("gut", "gut_mgs_abundance_freq.tsv"),
                        ("oral", "oral_mgs_abundance_freq.tsv")]:
        df = pd.read_csv(ROOT / "yeni_eklenenler/paper_pipeline/out" / fname,
                         sep="\t", index_col=0)
        df.columns = [cmap.get(c, c) for c in df.columns]
        df = df.loc[(df > 0).sum(axis=1) >= 5]
        out[name] = df.T
    return out


def pairwise_spearman(X, Y, Xnames, Ynames):
    """Pairwise-complete correlations: aligned subjects per layer pair."""
    common = X.index.intersection(Y.index)
    if len(common) < 10:
        return pd.DataFrame()
    Xa = X.loc[common]
    Ya = Y.loc[common]
    rows = []
    for i, xn in enumerate(Xnames):
        for j, yn in enumerate(Ynames):
            r, p = spearmanr(Xa.iloc[:, i], Ya.iloc[:, j])
            if not np.isnan(r):
                rows.append({"brain_feature": xn, "omic_feature": yn,
                             "rho": r, "p": p, "n": len(common)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q"] = multipletests(df["p"], method="fdr_bh")[1]
    return df.sort_values("p")


def main():
    net, seg, edg = load_fc()
    layers = [("networkFC", net), ("segregation", seg),
              ("selected_edges", edg)]

    all_parts = []
    for tag, fname in [("stool16S", "stool_genus_p.txt"),
                       ("saliva16S", "saliva_genus_p.txt")]:
        g = load_16s(fname)
        for label, feat in layers:
            d = pairwise_spearman(feat, g, list(feat.columns), list(g.columns))
            if d.empty:
                continue
            d.insert(0, "omic_layer", tag)
            d.insert(1, "brain_layer", label)
            all_parts.append(d)
            sig = d[d.q < 0.05]
            top = d.head(10)
            print(f"{tag} x {label}: {len(sig)} FDR-sig pairs "
                  f"(pairs tested={len(d)}, top rho={top.rho.abs().max():.3f}, "
                  f"n range {d.n.min()}-{d.n.max()})")
            d.to_csv(OUT / f"crossmodal_networkfc_{tag}_{label}.tsv",
                     sep="\t", index=False)

    for tag, g in load_mgs().items():
        for label, feat in layers:
            d = pairwise_spearman(feat, g, list(feat.columns), list(g.columns))
            if d.empty:
                continue
            d.insert(0, "omic_layer", f"shotgun_{tag}")
            d.insert(1, "brain_layer", label)
            all_parts.append(d)
            sig = d[d.q < 0.05]
            print(f"shotgun_{tag} x {label}: {len(sig)} FDR-sig pairs "
                  f"(pairs={len(d)}, top rho={d.rho.abs().max():.3f}, "
                  f"n range {d.n.min()}-{d.n.max()})")
            d.to_csv(OUT / f"crossmodal_networkfc_shotgun{tag}_{label}.tsv",
                     sep="\t", index=False)

    if all_parts:
        allr = pd.concat(all_parts, ignore_index=True)
        allr.to_csv(OUT / "crossmodal_networkfc_all.tsv", sep="\t", index=False)
    print("done (zero-sample-loss pairwise-complete)")


if __name__ == "__main__":
    main()
