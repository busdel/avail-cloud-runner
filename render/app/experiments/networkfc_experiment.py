# -*- coding: utf-8 -*-
"""Experiment: network-level FC modality from within/between summaries.

Builds per subject a 17-network x 17-network FC matrix (from the 19 within +
172 between values), derives:
  - networkFC features (lower triangle, 136)
  - segregation / ratio features (within/(within+between) per network, etc.)
Writes a GENERIC AVAIL layout (metadata.tsv + features/*.tsv) so the full
11-meta-learner pipeline can treat each granularity as a separate modality.

Outputs: experiments/results/netfc_generic/{metadata.tsv, features/*}
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results/netfc_generic")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "features").mkdir(parents=True, exist_ok=True)

W_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_within_network_summary.csv"
B_F = ROOT / "yeni_eklenenler/paper_pipeline/integration/fc_between_network_summary.csv"
Y_F = ROOT / "fMRI correlation and feature selection/y_data.xlsx"
S_F = ROOT / "fMRI correlation and feature selection/selected_feature_data.xlsx"
META_F = ROOT / "busra_pd_multisite/metadata.txt"

COARSE = {
    "Default": "DMN", "Dorsal Attention": "DAN", "Frontoparietal Control": "FPN",
    "Limbic": "LIM", "Salience Ventral Attention": "SAL", "Somatomotor": "SMN",
    "Temporoparietal": "TPN", "Visual": "VIS",
}


def coarse(name):
    for k, v in COARSE.items():
        if name.startswith(k):
            return v
    return name.strip()


def main():
    w = pd.read_csv(W_F)
    b = pd.read_csv(B_F)
    y = pd.read_excel(Y_F)
    y["PatientID"] = y["PatientID"].astype(str).str.strip()
    gmap = {"control": "HC", "mci": "MCI", "dementia": "PDD"}
    y["cls"] = y["Group"].map(gmap)
    labels = dict(zip(y["PatientID"], y["cls"]))
    w["patient"] = w["patient"].astype(str).str.strip()
    b["patient"] = b["patient"].astype(str).str.strip()
    common = sorted(set(w.patient) & set(b.patient) & set(labels))
    common = [p for p in common if labels[p] in ("HC", "MCI", "PDD")]
    print(f"subjects: {len(common)}")

    nets = sorted({coarse(n) for n in
                   [c.replace("within_", "") for c in w.columns[1:]
                    if not w[c].isna().all()]})
    print(f"networks: {nets}")

    # age/sex for metadata
    try:
        meta = pd.read_csv(META_F, sep="\t")
        meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
        age_sex = {r.SampleID: (r.Age, r.Sex) for r in meta.itertuples()}
    except Exception:
        age_sex = {}

    rows_meta = ["SampleID\tGroup"]
    rows_edges, rows_net, rows_seg = [], [], []
    net_feats = []
    for p in common:
        rows_meta.append(f"{p}\t{labels[p]}")
        # network matrix
        M = np.zeros((len(nets), len(nets)))
        rw = w[w.patient == p].iloc[0]
        for c in w.columns[1:]:
            v = rw[c]
            if pd.isna(v):
                continue
            node = c.replace("within_", "")
            M[nets.index(coarse(node)), nets.index(coarse(node))] += float(v)
        rb = b[b.patient == p].iloc[0]
        cnt = np.zeros_like(M)
        for c in b.columns[1:]:
            v = rb[c]
            if pd.isna(v):
                continue
            cc = c.replace("between_", "")
            a, bb = cc.split("__")
            i, j = nets.index(coarse(a)), nets.index(coarse(bb))
            M[i, j] += float(v)
            M[j, i] += float(v)
            cnt[i, j] += 1
            cnt[j, i] += 1
        M = np.where(cnt > 0, M / np.maximum(cnt, 1), M)
        tri = M[np.tril_indices(len(nets), k=-1)]
        net_feats.append(tri)
        rows_net.append([p] + [f"{v:.6f}" for v in tri])
        # segregation per network + ratios
        seg = []
        for i in range(len(nets)):
            wsum = M[i, i]
            bsum = M[i, :].sum() - wsum
            seg.append(wsum / (wsum + bsum + 1e-9))
        rows_seg.append([p] + [f"{v:.6f}" for v in seg])

    tri_names = [f"{nets[i]}-{nets[j]}" for i in range(len(nets))
                 for j in range(i) if j < i]
    # write features tables (rows=samples)
    edges = pd.read_excel(S_F)
    edges["PatientID"] = edges["PatientID"].astype(str).str.strip()
    edges = edges[edges.PatientID.isin(common)].set_index("PatientID").loc[common]
    ec = [f"edge_{i}" for i in range(edges.shape[1])]
    edges.columns = ec
    edges.insert(0, "SampleID", common)
    edges.to_csv(OUT / "features/fMRI_selected_edges.tsv", sep="\t", index=False)

    netdf = pd.DataFrame(rows_net, columns=["SampleID"] + tri_names)
    netdf.to_csv(OUT / "features/fMRI_networkFC.tsv", sep="\t", index=False)
    segdf = pd.DataFrame(rows_seg, columns=["SampleID"] + nets)
    segdf.to_csv(OUT / "features/fMRI_segregation.tsv", sep="\t", index=False)

    Path(OUT / "metadata.tsv").write_text("\n".join(rows_meta))
    print(f"wrote generic layout: edges {edges.shape}, networkFC "
          f"{netdf.shape}, segregation {segdf.shape}")


if __name__ == "__main__":
    main()
