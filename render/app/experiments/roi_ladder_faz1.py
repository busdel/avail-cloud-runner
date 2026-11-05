#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FAZ 1: ROI partial-correlation matrices -> 190 network summaries + cross-check.

- Reads the 83 patient sheets (100x100 partial corr) from the Desktop xlsx.
- Maps sheet order -> Sort Number -> PatientID (y_data.xlsx).
- Uses roi_network_map.csv (100 ROIs -> 19 sub-networks) to derive the same
  19 within-network + 171 between-network summary features as the existing
  paper_pipeline CSVs.
- Cross-checks the derived values against the reference CSVs
  (fc_within_network_summary.csv / fc_between_network_summary.csv):
  per-feature Pearson r, per-patient r, and max absolute deviation.
  This also validates the sheet->patient mapping.

Outputs (experiments/results/roi_ladder/):
  roi_derived_190.tsv         derived 83 x 190 summaries
  roi_validation_report.tsv   per-feature and per-patient agreement stats
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/mnt/limbic/AVAIL_GPU")
BENCH = ROOT / "AVAIL_benchmark"
OUT = ROOT / "experiments/results/roi_ladder"
OUT.mkdir(parents=True, exist_ok=True)

ROI_XLSX = Path("/home/suleyman/Desktop/fMRI datasets/"
                "partial_corr_reshaped_Spe21-23.xlsx")
Y_F = BENCH / "fMRI correlation and feature selection/y_data.xlsx"
W_F = BENCH / "yeni_eklenenler/paper_pipeline/integration/fc_within_network_summary.csv"
B_F = BENCH / "yeni_eklenenler/paper_pipeline/integration/fc_between_network_summary.csv"
MAP_F = BENCH / "yeni_eklenenler/paper_pipeline/metadata_fmr/roi_network_map.csv"


def load_roi_matrices():
    xl = pd.ExcelFile(ROI_XLSX)
    sheets = [s for s in xl.sheet_names]
    mats = []
    for s in sheets:
        m = xl.parse(s, header=None).values.astype(np.float64)
        assert m.shape == (100, 100), f"{s} shape {m.shape}"
        mats.append(m)
    return np.stack(mats), sheets


def load_patient_order():
    y = pd.read_excel(Y_F)
    y = y.sort_values("Sort Number").reset_index(drop=True)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    return y


def build_summaries(mats, net_labels):
    n = mats.shape[0]
    nets = sorted(set(net_labels))
    labels = np.array(net_labels)
    within_cols = [f"within_{nn}" for nn in nets]
    between_cols = []
    for i, a in enumerate(nets):
        for b in nets[i + 1:]:
            between_cols.append(f"between_{a}__{b}")
    within = np.zeros((n, len(nets)))
    between = np.zeros((n, len(between_cols)))
    for k in range(n):
        m = mats[k]
        for j, nn in enumerate(nets):
            idx = np.where(labels == nn)[0]
            sub = m[np.ix_(idx, idx)]
            within[k, j] = np.mean(sub[np.triu_indices(len(idx), k=1)])
        c = 0
        for i, a in enumerate(nets):
            for b in nets[i + 1:]:
                ia, ib = np.where(labels == a)[0], np.where(labels == b)[0]
                between[k, c] = np.mean(m[np.ix_(ia, ib)])
                c += 1
    W = pd.DataFrame(within, columns=within_cols)
    B = pd.DataFrame(between, columns=between_cols)
    return W, B


def main():
    mats, sheets = load_roi_matrices()
    y = load_patient_order()
    print(f"[faz1] loaded {mats.shape[0]} matrices; y rows {len(y)}")
    assert mats.shape[0] == len(y) == 83

    rmap = pd.read_csv(MAP_F)
    net_labels = [str(x).strip() for x in rmap["17 Network"]]
    assert len(net_labels) == 100

    W, B = build_summaries(mats, net_labels)
    derived = pd.concat([W, B], axis=1)
    derived.insert(0, "patient", y["PatientID"].values)
    derived.columns = [str(c).strip() for c in derived.columns]
    derived.to_csv(OUT / "roi_derived_190.tsv", sep="\t", index=False)
    print(f"[faz1] derived 83 x {derived.shape[1]-1} written")

    wref = pd.read_csv(W_F)
    bref = pd.read_csv(B_F)
    wref["patient"] = wref["patient"].astype(str).str.strip()
    bref["patient"] = bref["patient"].astype(str).str.strip()
    ref = wref.merge(bref, on="patient", validate="1:1")
    ref.columns = [str(c).strip() for c in ref.columns]
    ref = ref.set_index("patient").loc[y["PatientID"].values].reset_index()

    report_rows = []
    for col in ref.columns[1:]:
        canon = col
        if "__" in col:
            a, b = col.split("__", 1)
            canon = f"{a.strip()}__{b.strip()}"
        if canon not in derived.columns:
            print(f"[faz1] WARN: no derived match for {col!r}")
            continue
        r = np.corrcoef(derived[canon], ref[col])[0, 1]
        mad = np.abs(derived[canon] - ref[col]).max()
        mae = np.abs(derived[canon] - ref[col]).mean()
        report_rows.append({"feature": canon, "r": r, "max_abs_diff": mad,
                            "mean_abs_diff": mae})
    rep = pd.DataFrame(report_rows)
    rep.to_csv(OUT / "roi_validation_report.tsv", sep="\t", index=False)

    canon_cols = [r["feature"] for r in report_rows]
    ref_map = {}
    for col in ref.columns[1:]:
        if "__" in col:
            a, b = col.split("__", 1)
            ref_map[col] = f"{a.strip()}__{b.strip()}"
        else:
            ref_map[col] = col.strip()
    ref_canon = ref.rename(columns=ref_map)
    pr = []
    for pid in y["PatientID"].values:
        d = derived.loc[derived["patient"] == pid, canon_cols].values.ravel()
        rv = ref_canon.loc[ref_canon["patient"] == pid, canon_cols].values.ravel()
        pr.append(np.corrcoef(d, rv)[0, 1])
    pr = np.array(pr)
    print(f"[faz1] per-feature r: min {rep['r'].min():.4f} "
          f"median {rep['r'].median():.4f} max {rep['r'].max():.4f}")
    print(f"[faz1] per-patient r: min {pr.min():.4f} median {np.median(pr):.4f} "
          f"max {pr.max():.4f}")
    print(f"[faz1] max abs diff overall: {rep['max_abs_diff'].max():.6f}")
    bad = (rep["r"] < 0.999).sum()
    print(f"[faz1] features with r < 0.999: {bad} / {len(rep)}")
    print(f"[faz1] patients with r < 0.999: {(pr < 0.999).sum()} / 83")
    if bad or (pr < 0.999).sum():
        print("[faz1] WARNING: mismatch detected — see report")
    else:
        print("[faz1] OK: sheet->patient mapping + summary derivation verified")


if __name__ == "__main__":
    main()
