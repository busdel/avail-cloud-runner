#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 6: Report generation (English DOCX + Turkish summary MD).

Reads all roi_ladder outputs and writes:
  experiments/results/roi_ladder/ROI_LADDER_REPORT.docx
  experiments/results/roi_ladder/ROI_LADDER_OZET.md   (Turkish summary)
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt

ROOT = Path("/mnt/limbic/AVAIL_GPU")
RES = ROOT / "experiments/results/roi_ladder"
AV = ROOT / "avail_results"


def add_table(doc, df, title=None, max_rows=25):
    if title:
        doc.add_heading(title, level=2)
    df = df.reset_index(drop=True)
    if len(df) > max_rows:
        df = df.head(max_rows)
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = "" if pd.isna(v) else f"{v:.4g}" \
                if isinstance(v, (float, np.floating)) else str(v)
    return t


def main():
    doc = Document()
    doc.add_heading("ROI x multi-omics ladder analysis - Report", level=0)
    doc.add_paragraph(
        "Two-arm module ladder (complete-case vs AVAIL zero-sample-loss) "
        "on the PD cognitive-spectrum cohort (n=150; HC/MCI/PDD). "
        "LOO cross-validation, leak-free staged feature selection, "
        "meta-learners: SimpleMean, WSimpleMean, ElasticNetLR, XGBoost, "
        "MLP, TabPFN, TabICL. Run on the local machine (GTX 1660 Ti).")

    # --- Faz 1 ---
    doc.add_heading("1. ROI data validation (Faz 1)", level=1)
    rep = pd.read_csv(RES / "roi_validation_report.tsv", sep="\t")
    doc.add_paragraph(
        f"83 sheets (100x100 Fisher-z partial correlation matrices) were "
        f"mapped to patients via the Sort-Number order of y_data.xlsx and "
        f"the 190 network summaries (19 within + 171 between) were "
        f"re-derived. Per-feature agreement with the reference "
        f"paper-pipeline CSVs: {int((rep['r'] >= 0.99).sum())}/190 features "
        f"with r >= 0.99 (29 exact). Discrepancies are confined to 4 "
        f"networks (Visual Central/Peripheral, Somatomotor A, Somatomotor B "
        f"Auditory) and reflect differences in the reference pipeline's "
        f"parcellation assignment; the sheet->patient mapping itself is "
        f"confirmed by the exact-match features.")
    doc.add_paragraph(
        "Downstream analyses use the canonical Schaefer-2018 100-parcel "
        "17-network labels (nilearn), not the legacy map CSV.")

    # --- Faz 3 ---
    doc.add_heading("2. Two-arm module ladder (Faz 2-3)", level=1)
    summ = pd.read_csv(RES / "ladder_summary.tsv", sep="\t")
    cov = pd.read_csv(RES / "ladder_coverage.tsv", sep="\t")
    add_table(doc, cov, "Coverage per rung")
    piv = summ.pivot_table(index=["rung", "arm"], columns="model",
                           values="macro_auc").round(3)
    add_table(doc, piv, "3-class macro-AUC (LOO)")
    delta = pd.read_csv(RES / "ladder_delta.tsv", sep="\t")
    dp = delta.pivot_table(index="model", columns="rung",
                           values="delta").round(3)
    add_table(doc, dp, "Delta = AVAIL - complete-case")
    doc.add_paragraph(
        "The ladder adds modules cumulatively: fMRI ROI -> +stool16S -> "
        "+stoolSG -> +saliva16S -> +salivaSG. The complete-case arm trains "
        "each stage on patients with all modules in the rung "
        "(83 -> 64 -> 55 -> 53 -> 50); the AVAIL arm keeps all n=150 "
        "patients and encodes availability flags.")

    # --- benchmark comparison ---
    bc = RES / "BENCHMARK_COMPARISON.tsv"
    if bc.exists():
        doc.add_heading("2b. Established benchmark comparison (Faz 10)",
                        level=2)
        bdf = pd.read_csv(bc, sep="\t")
        add_table(doc, bdf[["comparison", "protocol", "value"]],
                  "Established 8-layer benchmark vs ladder")
        doc.add_paragraph(
            "The 3-class macro-AUC of the +clinical rung (0.730) matches "
            "the established 8-layer benchmark (0.735). The MCI per-class "
            "AUC is the ceiling in both (0.549 vs 0.540). The dedicated "
            "binary HC-PDD task reaches 0.918 (rung6 +clinical, AVAIL) vs "
            "0.961 for the established dedicated 2-class setup.")

    # --- Faz 4 ---
    doc.add_heading("3. ROI x cognition association analyses (Faz 4)", level=1)
    assoc = RES / "assoc"
    roi = pd.read_csv(assoc / "roi_mass_univariate.tsv", sep="\t")
    roi_sig = roi[roi["sig"]]
    doc.add_paragraph(
        "ROI-level mass-univariate (mean edge strength x cognitive score, "
        "age-adjusted partial Spearman, BH-FDR over 100 ROIs). "
        "Significant ROI counts per score:")
    cnt = roi_sig.groupby("score")["roi"].nunique()
    add_table(doc, cnt.reset_index().rename(columns={"roi": "n_sig_ROIs"}))
    edge = pd.read_csv(assoc / "edge_mass_univariate.tsv.gz", sep="\t",
                       compression="gzip")
    esig = edge[edge["sig"]]
    doc.add_paragraph(
        "Edge-level (4950 edges, BH-FDR). Significant edge counts per "
        "score:")
    ecnt = esig.groupby("score")["edge"].nunique()
    add_table(doc, ecnt.reset_index().rename(columns={"edge": "n_sig_edges"}))
    top = pd.read_csv(assoc / "edge_top40.tsv", sep="\t")
    add_table(doc, top[["score", "roi_i", "roi_j", "net_i", "net_j",
                        "rho", "p_bh"]].head(15), "Top edges by |rho|")
    nbs = pd.read_csv(assoc / "nbs_components.tsv", sep="\t")
    add_table(doc, nbs, "NBS-style components (threshold |rho|>=0.35)")
    with open(assoc / "pls_summary.txt") as f:
        pls_txt = f.read()
    doc.add_heading("Behavioral PLS", level=2)
    doc.add_paragraph(pls_txt)

    # --- Faz 5 ---
    doc.add_heading("4. Integrity checks (Faz 5)", level=1)
    integ = RES / "integrity"
    pc = pd.read_csv(integ / "permutation_and_covariate.tsv", sep="\t")
    perm_rows = pc[pc["null_mean"].notna()]
    add_table(doc, perm_rows, "Label permutation (200 shuffles, rung 5)")
    cov_rows = pc[pc["null_mean"].isna()]
    add_table(doc, cov_rows, "Covariate-residualized macro-AUC (rung 5)")
    doc.add_paragraph(
        "Decision curve for HC vs PDD classification (rung 5, "
        "ElasticNetLR) is saved as decision_curve.png/pdf.")

    doc.save(RES / "ROI_LADDER_REPORT.docx")
    print("[faz6] DOCX written")

    # --- Turkish summary ---
    lines = []
    lines.append("# ROI × Multi-Omics Merdiven Analizi — Özet")
    lines.append("")
    lines.append("## Merdiven kapsama")
    for _, r in cov.iterrows():
        lines.append(f"- Rung {r['rung']} ({r['modules']}): complete-case "
                     f"n={r['n_complete']}, AVAIL n={r['n_avail']}")
    lines.append("")
    lines.append("## Macro-AUC (LOO)")
    lines.append(piv.round(3).to_markdown())
    lines.append("")
    lines.append("## Δ (AVAIL − complete-case)")
    lines.append(dp.round(3).to_markdown())
    lines.append("")
    lines.append("## ROI × biliş ilişkileri (BH-FDR)")
    lines.append("- ROI düzeyi anlamlı ROI sayıları: "
                 + ", ".join(f"{s}: {n}" for s, n in cnt.items()))
    lines.append("- Kenar düzeyi anlamlı kenar sayıları: "
                 + ", ".join(f"{s}: {n}" for s, n in ecnt.items()))
    lines.append("")
    lines.append("## Bütünlük")
    perm_rows2 = pc[pc["protocol"] == "5fold-CV"]
    for _, r in perm_rows2.iterrows():
        lines.append(f"- {r['arm']} ({r['models']}): permütasyon "
                     f"p={r['p_perm']:.3f}")
    cov_rows2 = pc[pc["protocol"] == "covresid_fixed"]
    for arm in ["complete", "avail"]:
        vals = cov_rows2[cov_rows2["arm"] == arm]["obs_macro_auc"].values
        if len(vals):
            lines.append(f"- {arm} kovaryat-düzeltmeli macro-AUC aralığı: "
                         f"{vals.min():.3f}-{vals.max():.3f}")
    with open(RES / "ROI_LADDER_OZET.md", "w") as f:
        f.write("\n".join(lines))
    print("[faz6] Turkish summary written")


if __name__ == "__main__":
    main()
