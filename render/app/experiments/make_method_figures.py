#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fig. 2 (v2) method schematic + graphical abstract.

Fig 2 panels:
 (a) cohort availability matrix (real data; 150 x 5 blocks)
 (b) two-stage framework: preprocessing -> specialists -> flags ->
     representation z_i=[p|a] -> meta-learners -> outputs
 (c) evaluation ladder: two arms (complete-case vs AVAIL), leak-free LOO
 (d) validation and output layers (biomarkers, external cohorts, controls)
Graphical abstract: problem -> method -> results in one strip.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path("/mnt/limbic/AVAIL_GPU")
DATA = ROOT / "experiments/results/ladder_generic"
OUT = ROOT / "revizyon/figür_tablolari"
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
LABELS = ["fMRI ROI", "stool 16S", "stool SG", "saliva 16S", "saliva SG"]
COLORS = ["#1a73e8", "#188038", "#34a853", "#f29900", "#e37400"]
BOX = dict(boxstyle="round,pad=0.35", fc="#f5f7fa", ec="#9aa0a6", lw=1.1)


def load_avail():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    cls = {"HC": 0, "MCI": 1, "PDD": 2}
    order = meta.sort_values(["Group", "SampleID"],
                             key=lambda s: s.map(cls) if s.name == "Group"
                             else s)["SampleID"].tolist()
    A = np.zeros((len(order), len(MODULES)))
    for j, m in enumerate(MODULES):
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        ids = set(t["SampleID"].astype(str).str.strip())
        for i, pid in enumerate(order):
            A[i, j] = 1 if pid in ids else 0
    y = meta.set_index("SampleID").loc[order, "Group"].map(cls).values
    return A, y


def box(ax, x, y, w, h, text, fc="#f5f7fa", ec="#9aa0a6", fs=8.5, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#5f6368"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, color=color, lw=1.2))


def main():
    A, y = load_avail()
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.32,
                          wspace=0.18)

    # ---------------- (a) availability matrix ----------------
    axa = fig.add_subplot(gs[0, 0])
    img = np.zeros(A.shape + (3,))
    for j, c in enumerate(COLORS):
        rgb = matplotlib.colors.to_rgb(c)
        img[A[:, j] == 1, j] = rgb
        img[A[:, j] == 0, j] = (0.92, 0.92, 0.92)
    axa.imshow(img, aspect="auto", interpolation="nearest")
    axa.set_xticks(range(5), LABELS, rotation=30, ha="right", fontsize=8)
    bounds = np.cumsum([0] + [int((y == c).sum()) for c in range(3)])
    for b in bounds:
        axa.axhline(b - 0.5, color="k", lw=0.8)
    for c, name in enumerate(["HC", "MCI", "PDD"]):
        axa.text(4.75, (bounds[c] + bounds[c + 1]) / 2 - 0.5, name,
                 fontsize=8, va="center", fontweight="bold")
    axa.set_title("(a) Cohort: 150 subjects, 5 blocks; per-subject "
                  "availability\n(only 50 complete; imaging availability "
                  "79% HC vs 44% PDD, p=0.006)", fontsize=9)
    axa.set_yticks([])

    # ---------------- (b) framework ----------------
    axb = fig.add_subplot(gs[0, 1])
    axb.set_xlim(0, 10)
    axb.set_ylim(0, 10)
    axb.axis("off")
    axb.set_title("(b) Two-stage availability-aware framework", fontsize=9)
    box(axb, 0.1, 7.6, 3.0, 2.0,
        "Data blocks\nfMRI ROI (4,950 edges)\n16S genera; shotgun MGS\n"
        "(stool, saliva)", fs=7.6)
    box(axb, 3.5, 7.6, 2.9, 2.0,
        "Fold-internal\npreprocessing\nCLR / log-center;\nvariance-ANOVA-FDR;\n"
        "RF importance", fs=7.4)
    box(axb, 6.8, 7.6, 3.1, 2.0,
        "Specialists (stage 1)\nper-modality RF, LOO\n-> p_im = P(y|x_im)",
        fs=7.6)
    arrow(axb, 3.1, 8.6, 3.5, 8.6)
    arrow(axb, 6.4, 8.6, 6.8, 8.6)
    box(axb, 2.6, 4.7, 4.8, 1.9,
        "Representation\nz_i = [ p_i1, a_i1, ..., p_iM, a_iM ]\n"
        "missing modality -> p=0, flag=0", fs=7.8, ec="#1a73e8")
    arrow(axb, 8.35, 7.6, 6.5, 6.6)
    box(axb, 1.2, 1.6, 7.6, 2.1,
        "Meta-learners (stage 2, 11-model zoo; class-weighted, LOO)\n"
        "SimpleMean | WSimpleMean | ElasticNet | MLP | MoE | GNN |\n"
        "Transformer | XGBoost | TabNet | TabPFN | TabICL", fs=7.6)
    arrow(axb, 5.0, 4.7, 5.0, 3.7)
    box(axb, 0.1, 0.1, 9.8, 1.1,
        "Outputs: class probabilities q_i -> staging; risk score "
        "r_i = q_PDD - q_HC;  availability flags absorb informative "
        "missingness (flags-only baseline 0.527)", fs=7.8, ec="#188038")

    # ---------------- (c) ladder ----------------
    axc = fig.add_subplot(gs[1, 0])
    axc.set_xlim(0, 10)
    axc.set_ylim(0, 10)
    axc.axis("off")
    axc.set_title("(c) Module ladder: two arms under one leak-free protocol",
                  fontsize=9)
    xs = np.linspace(0.6, 8.8, 5)
    for k, (x, lab) in enumerate(zip(xs, LABELS)):
        box(axc, x - 0.55, 8.2, 1.3, 1.0, lab, fs=7.2, fc="#e8f0fe")
        if k:
            arrow(axc, xs[k - 1] + 0.75, 8.7, x - 0.55, 8.7)
    box(axc, 0.4, 5.4, 4.2, 2.0,
        "Complete-case arm\nsubjects with all rung modules\n"
        "n = 83 -> 64 -> 55 -> 53 -> 50", fs=7.8, ec="#b00020")
    box(axc, 5.2, 5.4, 4.4, 2.0,
        "AVAIL arm\nall n = 150, availability flags\n(zero sample loss)",
        fs=7.8, ec="#1a73e8")
    arrow(axc, 2.5, 8.2, 2.5, 7.4)
    arrow(axc, 7.4, 8.2, 7.4, 7.4)
    box(axc, 0.4, 2.6, 9.2, 2.0,
        "Evaluation: leave-one-out per rung and arm; identical models; "
        "matched-n control;\nfull-pipeline permutation p<0.005; bootstrap "
        "CIs; seed stability; nested ensemble validation", fs=7.6)
    arrow(axc, 5.0, 5.4, 5.0, 4.6)
    box(axc, 0.4, 0.2, 9.2, 1.8,
        "Result: complete-case peaks mid-ladder then degrades/collapses "
        "(0.619 at the full stack);\nAVAIL stable (0.679-0.702) at 3x "
        "coverage; +0.05 multi / +0.09-0.13 binary vs architectures",
        fs=7.8, ec="#188038")

    # ---------------- (d) validation layers ----------------
    axd = fig.add_subplot(gs[1, 1])
    axd.set_xlim(0, 10)
    axd.set_ylim(0, 10)
    axd.axis("off")
    axd.set_title("(d) Biological outputs and validation layers",
                  fontsize=9)
    box(axd, 0.2, 6.8, 4.6, 2.6,
        "Biomarker pipeline (3 criteria)\nunivariate FDR + multivariate "
        "importance +\n16S<->shotgun replication\n"
        "-> 7 stool taxa; DMN-FPN edges;\nSCFA/butyrate axis (Table S6)",
        fs=7.5)
    box(axd, 5.3, 6.8, 4.5, 2.6,
        "Clinical anchoring\nrisk score r_i vs MMSE (rho -0.47)\nand CDRS "
        "(+0.63) beyond age;\ntertile gradient (Kruskal p < 1e-7)",
        fs=7.5)
    box(axd, 0.2, 3.2, 4.6, 2.8,
        "External validation\nWallen PD (n=724); HMP2 IBD (n=130,\n"
        "genuine missingness); preclinical AD (n=164);\n"
        "cross-assay transfer 0.64/0.74;\ndropout profile replicated",
        fs=7.5)
    box(axd, 5.3, 3.2, 4.5, 2.8,
        "Controls and audits\nno cognitive variable in training (audit);\n"
        "education-removal ablation;\nflags-only baseline; matched-n;\n"
        "covariate residualization; verification ledger",
        fs=7.5)
    box(axd, 0.2, 0.5, 9.6, 1.7,
        "Full parameter registry: Supplementary Methods S7; complete "
        "model inventory: Table S8", fs=8, ec="#1a73e8")

    fig.suptitle("AVAIL: availability-aware multimodal integration for "
                 "staging the Parkinson cognitive spectrum", fontsize=11.5,
                 y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT / "fig2_method_schematic_v2.png", dpi=300)
    fig.savefig(OUT / "fig2_method_schematic_v2.pdf")
    plt.close(fig)
    print("[fig2v2] schematic saved")

    # ---------------- graphical abstract ----------------
    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    box(ax, 0.15, 0.3, 3.4, 3.0,
        "PROBLEM\nReal cohorts are incomplete:\nonly 50/150 complete;\n"
        "missingness is informative\n(imaging 79% HC vs 44% PDD,\np=0.006); "
        "complete-case\ndrops 2/3 non-randomly", fs=8, ec="#b00020")
    box(ax, 4.05, 0.3, 3.9, 3.0,
        "METHOD (two-stage, zero loss)\nper-modality specialists ->\n"
        "p_im, plus explicit availability flags a_im\n"
        "-> meta-learner fusion over ALL 150 subjects\n"
        "(11-model zoo; leak-free LOO; S7 registry)", fs=8, ec="#1a73e8")
    box(ax, 8.45, 0.3, 3.4, 3.0,
        "RESULTS\n3x coverage, stable across the ladder\n(0.68-0.70 vs "
        "complete-case collapse 0.62);\nHC-PDD 0.92-0.98; MCI risk score\n"
        "(MMSE -0.47, CDRS +0.63);\nSCFA/gut-brain biomarkers;\n"
        "4 external cohorts replicate", fs=8, ec="#188038")
    arrow(ax, 3.55, 1.8, 4.05, 1.8)
    arrow(ax, 7.95, 1.8, 8.45, 1.8)
    fig.savefig(OUT / "graphical_abstract.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT / "graphical_abstract.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[graphical-abstract] saved")


if __name__ == "__main__":
    main()
