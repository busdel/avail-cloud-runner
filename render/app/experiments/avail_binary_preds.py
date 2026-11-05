#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate AVAIL binary (HC vs PDD) rung-5 predictions as npz for the
paired bootstrap comparison with competitor architectures (local run)."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
from avail_ai.learners import build_meta_features  # noqa: E402
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

RES = ROOT / "experiments/results/roi_ladder"
DATA = ROOT / "experiments/results/ladder_generic"
MODULES = ["fmri_roi", "stool16S", "stoolSG", "saliva16S", "salivaSG"]
MODELS = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost",
          "MLP", "TabPFN", "TabICL"]
K = 2


def main():
    meta = pd.read_csv(DATA / "metadata.tsv", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    ids = sorted(meta[meta.Group.isin(["HC", "PDD"])]["SampleID"].tolist())
    label = {r.SampleID: (0 if r.Group == "HC" else 1)
             for r in meta.itertuples()}
    feat = {}
    for m in MODULES:
        t = pd.read_csv(DATA / "features" / f"{m}.tsv", sep="\t",
                        usecols=["SampleID"])
        feat[m] = set(t["SampleID"].astype(str))
    base = {}
    for f in sorted(RES.glob("stage1bin_*.tsv.gz")):
        parts = f.name.split(".")[0].split("_")
        name = "_".join(parts[1:-1])
        n = int(parts[-1])
        t = pd.read_csv(f, sep="\t", index_col=0, compression="gzip")
        t.index = t.index.astype(str).str.strip()
        base[(name, n)] = {i: t.loc[i].values.astype(float) for i in t.index}
    piv = {}
    for p in ids:
        d = {}
        for m in MODULES:
            n_ = len([i for i in ids if i in feat[m]])
            src = base.get((m, n_))
            if src is not None and p in src:
                d[m] = src[p]
        piv[p] = d
    X, y = build_meta_features(piv, ids, label, len(MODULES), MODULES, K)
    P = run_meta_loo(X, y, len(MODULES), K, model_names=MODELS, device=None,
                     random_state=42, n_seeds=5, augment_flag=True,
                     aug_p=0.3, verbose=False)
    out = RES / "predictions"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "rung5_binary_avail.npz",
                        **{m: P[m] for m in MODELS},
                        y=y, ids=np.array(ids))
    print(f"[bin-preds] saved {len(ids)} subjects x {len(MODELS)} models")


if __name__ == "__main__":
    main()
