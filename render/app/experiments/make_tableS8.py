#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble Table S8: complete inventory of evaluated models and methods.

Sections:
  A. Meta-learner zoo across the ladder (11 models; multi + binary)
  B. Missing-data strategies (14 strategies x 4 models, full detail)
  C. Competitor architectures (HeMIS/ModDrop/TMC/SMIL/mmFormer/early
     fusion + UniMIX-AD port) on both tasks
  D. Improvement probes A-G
  E. Strategy-level aggregates
Writes supplement/TableS8_all_models.md and copies source TSVs.
"""
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
RES = ROOT / "experiments/results"
SUP = ROOT / "revizyon/makaleler/supplement"
SRC = SUP / "S8_sources"
SRC.mkdir(parents=True, exist_ok=True)
L = ["# Table S8. Complete inventory of evaluated models and methods", ""]


def md_table(df, floatfmt="%.3f"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: floatfmt % v if pd.notna(v) else "")
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def copy_src(path, name=None):
    if path.exists():
        shutil.copy(path, SRC / (name or path.name))


# A. meta-learner zoo
z = pd.read_csv(RES / "roi_ladder/ladder_summary_full_zoo.tsv", sep="\t")
copy_src(RES / "roi_ladder/ladder_summary_full_zoo.tsv")
piv = z.pivot_table(index="model", columns=["rung", "arm"],
                    values="macro_auc").round(3)
L += ["## A. Meta-learner zoo, 3-class macro-AUC (server replication, "
      "11 models, 10 seeds)", "",
      md_table(piv.reset_index()), ""]
# A3. per-class AUC at the full stack (rung 5), both arms and tasks
r5 = z[z["rung"] == 5]
pc = r5.melt(id_vars=["model", "arm"],
             value_vars=["auc_HC", "auc_MCI", "auc_PDD"],
             var_name="class", value_name="auc")
pcp = pc.pivot_table(index="model", columns=["class", "arm"],
                     values="auc").round(3)
L += ["## A3. Per-class AUC at the full stack (rung 5; server "
      "replication)", "", md_table(pcp.reset_index()), ""]
b = pd.read_csv(RES / "roi_ladder/binary_ladder_summary.tsv", sep="\t")
copy_src(RES / "roi_ladder/binary_ladder_summary.tsv")
bp = b.pivot_table(index="model", columns=["n_modules", "arm"],
                   values="auc").round(3)
L += ["## A2. Dedicated binary HC-PDD ladder (7 models)", "",
      md_table(bp.reset_index()), ""]
bpc = b.melt(id_vars=["model", "arm", "n_modules"], value_vars=["auc"],
             var_name="metric", value_name="auc_value")
bpcp = bpc.pivot_table(index="model",
                       columns=["metric", "n_modules", "arm"],
                       values="auc_value").round(3)
L += ["## A4. Binary HC-PDD AUC (dedicated ladder, overall)", "",
      md_table(bpcp.reset_index()), ""]

# B. strategies full detail
imp = pd.read_csv(ROOT / "revizyon/figür_tablolari/fig4/"
                  "imputation_benchmark_v4_results.tsv", sep="\t")
copy_src(ROOT / "revizyon/figür_tablolari/fig4/"
         "imputation_benchmark_v4_results.tsv")
agg = imp.groupby("strategy").agg(n=("n", "first"),
                                  macro_mean=("macro_auc", "mean"),
                                  macro_sd=("macro_auc", "std"),
                                  macro_min=("macro_auc", "min"),
                                  macro_max=("macro_auc", "max")).round(3)
L += ["## B. Missing-data strategies (14 strategies x 4 meta-models; "
      "full per-model detail in the source TSV)", "",
      md_table(agg.reset_index()), ""]

# C. competitors
c1 = pd.read_csv(RES / "competitors/competitors_loo_results.tsv", sep="\t")
c1b = pd.read_csv(RES / "competitors/competitors_loo_results_binary.tsv",
                  sep="\t")
c2 = pd.read_csv(RES / "competitors/competitors2_loo_results_multi.tsv",
                 sep="\t")
c2b = pd.read_csv(RES / "competitors/competitors2_loo_results_binary.tsv",
                  sep="\t")
for f in ["competitors_loo_results.tsv", "competitors_loo_results_binary.tsv",
          "competitors2_loo_results_multi.tsv",
          "competitors2_loo_results_binary.tsv"]:
    copy_src(RES / "competitors" / f)
comp = pd.concat([
    c1.assign(task="multi"), c1b.assign(task="binary"),
    c2.assign(task="multi"), c2b.assign(task="binary")],
    ignore_index=True)[["task", "method", "macro_auc", "auc_HC", "auc_MCI",
                        "auc_PDD"]].round(3)
L += ["## C. Competitor missing-modality architectures (re-implementations; "
      "LOO, 2 seeds)", "", md_table(comp), ""]
u = pd.read_csv(RES / "unimix_ad_results.tsv", sep="\t")
copy_src(RES / "unimix_ad_results.tsv")
ua = u.groupby("model")["test_macro_auc"].agg(
    mean="mean", sd="std", min="min", max="max").round(3)
L += ["## C2. Official UniMIX-AD port (per-seed detail in source TSV; "
      "held-out split n=23)", "", md_table(ua.reset_index()), ""]

# D. probes
L += ["## D. Improvement probes A-G (see upgrade[A-G]* files)", ""]
for f in sorted((RES / "upgrades").glob("upgrade[A-E]_*.tsv")):
    copy_src(f)
    d = pd.read_csv(f, sep="\t")
    L += [f"### {f.name}", "", md_table(d.head(20)), ""]
for f in sorted((RES / "upgrades").glob("upgradeF_*summary.txt")):
    copy_src(f)
    L += [f"### {f.name}", "", "```", f.read_text().strip(), "```", ""]
for f in sorted((RES / "upgrades").glob("nested_ensemble_*.tsv")):
    copy_src(f)
    d = pd.read_csv(f, sep="\t")
    g = d.groupby(["task", "features"])[["greedy", "best_single",
                                         "uniform"]].agg(["mean", "std"])
    L += [f"### {f.name} (summary)", "", md_table(g.reset_index()), ""]

(SUP / "TableS8_all_models.md").write_text("\n".join(L))
print(f"[S8] written ({len(L)} lines); sources in {SRC}")
