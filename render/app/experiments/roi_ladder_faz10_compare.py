#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 10: Consolidated benchmark comparison.

Established 8-layer AVAIL benchmark (avail_results) vs
the two-arm ladder (5 modules, +clinical extension) and the dedicated
binary HC-PDD task.

Outputs (experiments/results/roi_ladder/):
  BENCHMARK_COMPARISON.tsv / .md  (+ figure PNG/PDF)
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
RES = ROOT / "experiments/results/roi_ladder"
AV = ROOT / "avail_results"


def main():
    est = pd.read_csv(AV / "meta_benchmark_3class_results.tsv", sep="\t")
    est_all = est[est["subgroup"] == "all"]
    est_pc = pd.read_csv(AV / "per_class_auc.tsv", sep="\t")
    est_bin = pd.read_csv(AV / "pdd_vs_hc_auc_comparable.tsv", sep="\t")
    lad = pd.read_csv(RES / "ladder_summary.tsv", sep="\t")
    lad5 = lad[lad["rung"] == 5]
    clin = pd.read_csv(RES / "clinical_extension_summary.tsv", sep="\t")
    binr = pd.read_csv(RES / "binary_ladder_summary.tsv", sep="\t")

    def rng(s):
        return f"{s.mean():.3f} [{s.min():.3f}-{s.max():.3f}]"

    rows = []
    rows.append({"comparison": "3-class macro-AUC",
                 "protocol": "Established 8-layer benchmark (11 models, "
                             "n=150, clinical layer included)",
                 "value": rng(est_all["macro_auc"]),
                 "mean": est_all["macro_auc"].mean()})
    for arm, label in [("complete", "complete-case n=50"),
                       ("avail", "AVAIL n=150")]:
        s = lad5[lad5["arm"] == arm]["macro_auc"]
        rows.append({"comparison": "3-class macro-AUC",
                     "protocol": f"Ladder rung5 (5 modules, 7 models, {label})",
                     "value": rng(s), "mean": s.mean()})
    for arm, label in [("complete", "complete-case n=50"),
                       ("avail", "AVAIL n=150")]:
        s = clin[clin["arm"] == arm]["macro_auc"]
        rows.append({"comparison": "3-class macro-AUC",
                     "protocol": f"Ladder rung6 (+clinical, 6 modules, "
                                 f"7 models, {label})",
                     "value": rng(s), "mean": s.mean()})
    for cls in ["HC", "MCI", "PDD"]:
        s = est_pc[cls]
        rows.append({"comparison": f"per-class AUC ({cls})",
                     "protocol": "Established 8-layer benchmark",
                     "value": rng(s), "mean": s.mean()})
        s2 = lad5[lad5["arm"] == "avail"][f"auc_{cls}"]
        rows.append({"comparison": f"per-class AUC ({cls})",
                     "protocol": "Ladder rung5 AVAIL (7 models)",
                     "value": rng(s2), "mean": s2.mean()})
    rows.append({"comparison": "Binary HC-PDD AUC",
                 "protocol": "Established benchmark (dedicated 2-class)",
                 "value": rng(est_bin["PDD_vs_HC_AUC"]),
                 "mean": est_bin["PDD_vs_HC_AUC"].mean()})
    for nm, label in [(5, "rung5"), (6, "rung6 +clinical")]:
        s = binr[(binr["n_modules"] == nm) & (binr["arm"] == "avail")]["auc"]
        rows.append({"comparison": "Binary HC-PDD AUC",
                     "protocol": f"Dedicated binary ladder {label} "
                                 f"(AVAIL, n=93)",
                     "value": rng(s), "mean": s.mean()})
        s2 = binr[(binr["n_modules"] == nm) & (binr["arm"] == "complete")]["auc"]
        rows.append({"comparison": "Binary HC-PDD AUC",
                     "protocol": f"Dedicated binary ladder {label} "
                                 f"(complete-case, n=32)",
                     "value": rng(s2), "mean": s2.mean()})

    comp = pd.DataFrame(rows)
    comp.to_csv(RES / "BENCHMARK_COMPARISON.tsv", sep="\t", index=False)

    L = ["# Benchmark comparison — established framework vs ladder", ""]
    L.append("All values are model-mean [min-max] across the reported model "
             "zoo. The established benchmark uses 8 layers (clinical "
             "included) and 11 models on the full n=150 cohort; the ladder "
             "uses 5 molecular/imaging modules (7 models locally; an "
             "11-model replication is running on the GPU server).")
    L.append("")
    L.append("| Comparison | Protocol | Value |")
    L.append("|---|---|---|")
    for _, r in comp.iterrows():
        L.append(f"| {r['comparison']} | {r['protocol']} | {r['value']} |")
    L.append("")
    L.append("Key reading: the 3-class ceiling is set by the MCI class in "
             "BOTH the established benchmark and the ladder (MCI ~0.45-0.60 "
             "vs HC/PDD 0.75-0.93). The dedicated binary HC-PDD task "
             "reaches the established range once run as a 2-class problem, "
             "and the +clinical rung aligns the ladder with the established "
             "protocol's feature set.")
    (RES / "BENCHMARK_COMPARISON.md").write_text("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    labels = ["Established\n8-layer", "Ladder rung5\nAVAIL",
              "Ladder rung6\nAVAIL +clin", "Ladder rung5\ncomplete",
              "Ladder rung6\ncomplete +clin"]
    vals = [est_all["macro_auc"].mean(),
            lad5[lad5["arm"] == "avail"]["macro_auc"].mean(),
            clin[clin["arm"] == "avail"]["macro_auc"].mean(),
            lad5[lad5["arm"] == "complete"]["macro_auc"].mean(),
            clin[clin["arm"] == "complete"]["macro_auc"].mean()]
    colors = ["#444", "#1a73e8", "#5b9bd5", "#b00020", "#d98080"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("3-class macro-AUC (model mean)")
    ax.set_ylim(0.5, 0.85)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("3-class comparison")
    ax = axes[1]
    mci_e = est_pc["MCI"].mean()
    mci_l = lad5[lad5["arm"] == "avail"]["auc_MCI"].mean()
    hc_e, hc_l = est_pc["HC"].mean(), lad5[lad5["arm"] == "avail"]["auc_HC"].mean()
    pdd_e = est_pc["PDD"].mean()
    pdd_l = lad5[lad5["arm"] == "avail"]["auc_PDD"].mean()
    x = np.arange(3)
    ax.bar(x - 0.2, [hc_e, mci_e, pdd_e], 0.4, label="Established",
           color="#444")
    ax.bar(x + 0.2, [hc_l, mci_l, pdd_l], 0.4, label="Ladder rung5 AVAIL",
           color="#1a73e8")
    ax.set_xticks(x, ["HC", "MCI", "PDD"])
    ax.set_ylabel("per-class AUC")
    ax.legend(fontsize=8)
    ax.set_title("Per-class comparison (MCI bottleneck)")
    fig.tight_layout()
    fig.savefig(RES / "benchmark_comparison.png", dpi=300)
    fig.savefig(RES / "benchmark_comparison.pdf")
    print("[faz10] comparison written")
    print(comp[["comparison", "protocol", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
