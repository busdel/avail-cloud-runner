# -*- coding: utf-8 -*-
"""Build the FULL zero-sample-loss generic dataset (n=150, 8 layers).

Layers: fMRI_selected_edges / fMRI_networkFC / fMRI_segregation (n=83 each),
stool16S (n=112), saliva16S (n=125), shotgun_stool (n=123), shotgun_saliva
(n=107), clinical (age/edu/sex, n~150). Labels merged from metadata.txt +
y_data.xlsx + ID_matching_master (HC/MCI/PDD). Every subject stays in the
cohort regardless of which layers they have - availability flags do the rest.

Outputs: experiments/results/zero_loss_generic/{metadata.tsv, features/*}
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/zero_loss_generic")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "features").mkdir(parents=True, exist_ok=True)
GEN = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/netfc_generic/features")


def labels_all():
    meta = pd.read_csv(ROOT / "busra_pd_multisite/metadata.txt", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    g_meta = dict(zip(meta["SampleID"], meta["Group"]))
    y = pd.read_excel(ROOT / "fMRI correlation and feature selection/y_data.xlsx")
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    gmap_f = {"control": "HC", "mci": "MCI", "dementia": "PDD"}
    g_f = dict(zip(y["PatientID"], y["Group"].map(gmap_f)))
    master = pd.read_csv(ROOT / "yeni_eklenenler/paper_pipeline/integration/"
                         "ID_matching_master.csv")
    master["patient"] = master["patient"].astype(str).str.strip()
    gmap_m = {"HC": "HC", "PD-NC": "PDNC", "PD-MCI": "MCI", "PDD": "PDD"}
    g_m = {p: gmap_m[g] for p, g in zip(master["patient"], master["group"])
           if g in gmap_m}
    group = {}
    for i in set(g_meta) | set(g_f) | set(g_m):
        g = g_meta.get(i) or g_f.get(i) or g_m.get(i)
        if g:
            group[i] = g
    return group


def main():
    group = labels_all()
    ids = sorted([i for i in group if group[i] in ("HC", "MCI", "PDD")])
    print(f"cohort: {len(ids)} subjects "
          f"({pd.Series([group[i] for i in ids]).value_counts().to_dict()})")

    rows_meta = ["SampleID\tGroup"]
    for p in ids:
        rows_meta.append(f"{p}\t{group[p]}")
    Path(OUT / "metadata.tsv").write_text("\n".join(rows_meta))

    def write_layer(name, df):
        df = df.copy()
        df.columns = [f"f{j}" for j in range(df.shape[1])]
        df.index = [str(i).strip() for i in df.index]
        df = df[df.index.isin(ids)]
        df.insert(0, "SampleID", df.index)
        df.to_csv(OUT / f"features/{name}.tsv", sep="\t", index=False)
        print(f"  {name}: n={len(df)} x {df.shape[1]-1}")

    # fMRI layers (83 subjects)
    for name in ["fMRI_selected_edges", "fMRI_networkFC", "fMRI_segregation"]:
        d = pd.read_csv(GEN / f"{name}.tsv", sep="\t")
        d["SampleID"] = d["SampleID"].astype(str).str.strip()
        write_layer(name, d.set_index("SampleID"))

    # 16S layers
    for name, fname in [("stool16S", "stool_genus_p.txt"),
                        ("saliva16S", "saliva_genus_p.txt")]:
        g = pd.read_csv(ROOT / "busra_pd_multisite" / fname, sep="\t",
                        index_col=0)
        g.columns = [str(c).strip() for c in g.columns]
        g = g.loc[(g > 0).sum(axis=1) >= 5].T
        write_layer(name, g)

    # shotgun layers
    cw = pd.read_csv(ROOT / "yeni_eklenenler/paper_pipeline/DELIVERABLE/"
                     "raw_outputs/err_patient_crosswalk.tsv", sep="\t")
    cw["patient"] = cw["patient"].astype(str).str.strip()
    cmap = dict(zip(cw["run_accession"].astype(str), cw["patient"]))
    for name, fname in [("shotgun_stool", "gut_mgs_abundance_freq.tsv"),
                        ("shotgun_saliva", "oral_mgs_abundance_freq.tsv")]:
        g = pd.read_csv(ROOT / "yeni_eklenenler/paper_pipeline/out" / fname,
                        sep="\t", index_col=0)
        g.columns = [cmap.get(c, c) for c in g.columns]
        g = g.loc[(g > 0).sum(axis=1) >= 5].T
        write_layer(name, g)

    print("zero-loss dataset written")


if __name__ == "__main__":
    main()
