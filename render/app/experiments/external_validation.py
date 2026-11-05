# -*- coding: utf-8 -*-
"""External validation 1: method-level on Wallen et al. 2022 (Nat Commun).

Independent cohort: 724 subjects (490 PD, 234 HC), shotgun metagenomics
(MetaPhlAn species = ready MGS abundance + HUMAnN pathways). Our full
AVAIL pipeline (LOO base learners, p+flag meta-features, meta-learners)
runs on this external data with no retraining on our cohort.

Outputs: experiments/external_wallen/results/*
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))

from avail_ai.generic import load_generic  # noqa: E402
from avail_ai.learners import (base_predictions_loo, build_meta_features,  # noqa: E402
                               macro_auc)
from avail_ai.meta_learners import run_meta_loo  # noqa: E402

DATA = BASE / "experiments/external_wallen/generic"
OUT = BASE / "experiments/external_wallen/results"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    group, conf_raw, mods, names = load_generic(DATA)
    from collections import Counter
    classes = [c for c, _ in Counter(group.values()).most_common(3)]
    K = len(classes)
    ids = sorted([i for i in group if group[i] in classes])
    label = {i: classes.index(group[i]) for i in ids}
    mod_order = sorted(mods)
    print(f"external cohort: n={len(ids)}, K={K}, classes={classes}, "
          f"modalities={mod_order}")
    print("availability: " +
          ", ".join(f"{m}={sum(1 for i in ids if i in mods[m])}"
                    for m in mod_order))

    conf = {}
    for i in ids:
        c = conf_raw.get(i, [np.nan] * 3)[:3]
        conf[i] = np.array(c + [np.nan] * (3 - len(c)), dtype=float)
    Cmat = np.vstack([conf[i] for i in ids])
    for j in range(3):
        med = np.nanmedian(Cmat[:, j])
        Cmat[:, j] = np.where(np.isnan(Cmat[:, j]), med, Cmat[:, j])
    conf = {i: Cmat[k] for k, i in enumerate(ids)}

    micro = {m for m in mods
             if all(np.all(np.isfinite(mods[m][i]) & (mods[m][i] >= 0))
                    for i in ids if i in mods[m])}
    print(f"CLR modalities: {sorted(micro)}")

    cache = OUT / "base_predictions_generic.tsv"
    piv = None
    if cache.exists():
        df = pd.read_csv(cache, sep="\t")
        piv = {}
        for _, r in df.iterrows():
            piv.setdefault(str(r["patient"]).strip(), {})[r["modality"]] = \
                [r[f"p_{c}"] for c in range(K)]
    if piv is None:
        print("stage 1: leak-free LOO base predictions ...")
        piv = base_predictions_loo(mods, ids, label, conf, K, micro,
                                   n_estimators=300, random_state=42)
        rows = []
        for pid, mm in piv.items():
            for m, pv in mm.items():
                row = {"patient": pid, "modality": m}
                for c in range(K):
                    row[f"p_{c}"] = pv[c]
                rows.append(row)
        pd.DataFrame(rows).to_csv(cache, sep="\t", index=False)

    X, y = build_meta_features(piv, ids, label, len(mods), mod_order, K, 0.0)
    models = ["SimpleMean", "WSimpleMean", "ElasticNetLR", "XGBoost"]
    print(f"stage 2: LOO meta-benchmark ({models}) ...")
    preds = run_meta_loo(X, y, len(mods), K, model_names=models)

    res_rows = []
    for m in models:
        res_rows.append({"model": m, "subgroup": "all", "n": len(ids),
                         "macro_auc": macro_auc(y, preds[m])})
    res = pd.DataFrame(res_rows)
    res.to_csv(OUT / "meta_benchmark_results.tsv", sep="\t", index=False)
    res.to_csv(OUT / "meta_benchmark_3class_results.tsv", sep="\t", index=False)
    print(res.round(3).to_string(index=False))

    pc = []
    for m in models:
        vals = {}
        for c, cls in enumerate(classes):
            yc = (y == c).astype(int)
            vals[cls] = macro_auc(yc, preds[m][:, c:c + 1])
        pc.append({"model": m, **vals})
    pd.DataFrame(pc).to_csv(OUT / "per_class_auc.tsv", sep="\t", index=False)
    print(pd.DataFrame(pc).round(3).to_string(index=False))

    pred_df = pd.DataFrame({"patient": ids, "y": y})
    for m in models:
        for c, cls in enumerate(classes):
            pred_df[f"{m}_{cls}"] = preds[m][:, c]
    pred_df.to_csv(OUT / "loo_predictions.tsv", sep="\t", index=False)
    print("external validation (method-level) done")


if __name__ == "__main__":
    main()
