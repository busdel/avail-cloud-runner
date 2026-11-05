# -*- coding: utf-8 -*-
"""External validation 2: CROSS-COHORT MODEL TRANSFER.

Train a stool-genus PD-vs-HC classifier on OUR cohort (16S genera, CLR),
test it on the independent Wallen et al. 2022 cohort (MetaPhlAn species
aggregated to genus level, CLR) using only the genera shared by both
platforms (16S vs shotgun, genus-level - platform shift is expected).

This is the strongest external claim: a trained model, not just the method,
transfers to an independent cohort.

Outputs: experiments/results/transfer_wallen_results.tsv
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

BASE = Path("/media/suleyman_disk/AVAIL_GPU")
sys.path.insert(0, str(BASE / "avail_ai"))

ROOT = BASE / "AVAIL_benchmark"
OUT = BASE / "experiments/results"
WALLEN = BASE / "experiments/external_wallen"


def clr(df):
    x = np.log(df.values + 1.0)
    return pd.DataFrame(x - x.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def our_genera():
    g = pd.read_csv(ROOT / "busra_pd_multisite/stool_genus_p.txt", sep="\t",
                    index_col=0)
    g.columns = [str(c).strip() for c in g.columns]
    g = g.loc[(g > 0).sum(axis=1) >= 5].T
    meta = pd.read_csv(ROOT / "busra_pd_multisite/metadata.txt", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    lab = dict(zip(meta.SampleID, meta.Group))
    ids = [p for p in g.index if lab.get(p) in ("HC", "PDD")]
    y = np.array([1 if lab[p] == "PDD" else 0 for p in ids])
    return clr(g.loc[ids]), y


def wallen_genera():
    xl = pd.ExcelFile(WALLEN / "source_data.xlsx")
    meta = xl.parse("subject_metadata")
    lab = dict(zip(meta.sample_name, meta.Case_status))
    sp = xl.parse("metaphlan_rel_ab")
    sp = sp[sp.clade_name.str.contains("|s__", regex=False)].copy()
    sp["genus"] = sp.clade_name.str.extract(r"\|g__([^|]+)").fillna(
        sp.clade_name.str.split("|").str[-1])
    sp = sp.drop(columns=["clade_name"])
    g = sp.groupby("genus").sum()
    g = g.loc[(g > 0).sum(axis=1) >= 5]   # prevalence filter on rows (genera)
    g = g.T
    ids = [p for p in g.index if lab.get(p) in ("Control", "PD")]
    y = np.array([1 if lab[p] == "PD" else 0 for p in ids])
    return clr(g.loc[ids]), y


def main():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    Xo, yo = our_genera()
    Xw, yw = wallen_genera()
    shared = sorted(set(Xo.columns) & set(Xw.columns))
    print(f"our: {Xo.shape} ({len(np.unique(yo))} classes), "
          f"wallen: {Xw.shape}, shared genera: {len(shared)}")
    Xo = Xo[shared].values
    Xw = Xw[shared].values
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf.fit(Xo, yo)
    P = rf.predict_proba(Xw)[:, 1]
    auc = roc_auc_score(yw, P)
    rows = [{"transfer": "ours_16S_genus -> wallen_shotgun_genus",
             "n_train": len(yo), "n_test": len(yw),
             "shared_genera": len(shared), "test_auc": auc,
             "test_prev": float(yw.mean())}]
    # reversed direction
    rf2 = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    rf2.fit(Xw, yw)
    P2 = rf2.predict_proba(Xo)[:, 1]
    rows.append({"transfer": "wallen -> ours",
                 "n_train": len(yw), "n_test": len(yo),
                 "shared_genera": len(shared),
                 "test_auc": roc_auc_score(yo, P2),
                 "test_prev": float(yo.mean())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "transfer_wallen_results.tsv", sep="\t", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
