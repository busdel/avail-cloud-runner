#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Follow-up: clinical non-cognitive block WITHOUT Disease_Duration
(the duration feature is a near-perfect HC-vs-PD proxy: 0 for HC by design).
Runs the two task variants on the reduced 5-feature block.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "experiments"))

import clinical_noncog_block as cnb  # noqa: E402

cnb.CLIN_COLS = ["UPDRS", "RBD", "GDS", "Apathy", "NPI_Frequency.Severity"]

rows = []
for task in ["multi", "binary"]:
    df = cnb.run_variant(task, ["clin"])
    df["variant"] = "clin_nodur"
    rows.append(df)
    print(f"[nodur] {task} done", flush=True)

out = pd.concat(rows, ignore_index=True)
out.to_csv(ROOT / "experiments/results/upgrades/clinical_nodur_results.tsv",
           sep="\t", index=False)
print(out.groupby(["task", "variant"])["metric"].mean().to_string())
print("[nodur] DONE")
