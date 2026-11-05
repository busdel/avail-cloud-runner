import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DS_ROOT = Path("/mnt/limbic/datasets")


def sex_norm(v):
    s = str(v).strip().upper()
    if s in ("1", "1.0", "M", "MALE"):
        return "M"
    if s in ("0", "0.0", "2", "2.0", "F", "FEMALE"):
        return "F"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ds004392", "ds005892"])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--ds-root", default=str(DS_ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-edges", action="store_true")
    ap.add_argument("--radiomics-dir", default="/media/suleyman_disk/ds_work/radiomics")
    args = ap.parse_args()

    ds = args.dataset
    ds_root = Path(args.ds_root) / ds
    froot = Path(args.features_dir)
    out = Path(args.out)
    (out / "features").mkdir(parents=True, exist_ok=True)

    part = pd.read_csv(ds_root / "participants.tsv", sep="\t")
    meta = pd.DataFrame({"SampleID": part["participant_id"].astype(str)})
    meta["Age"] = pd.to_numeric(part.get("age"), errors="coerce")
    meta["Sex"] = part["sex"].map(sex_norm) if "sex" in part else ""
    edu_col = "years_of_education"
    if edu_col in part:
        meta["Education"] = pd.to_numeric(part[edu_col], errors="coerce")

    if ds == "ds004392":
        dx = pd.read_csv(ds_root / "phenotype" / "dx_deidentified.tsv", sep="\t")
        g = dict(zip(dx["participant_id"].astype(str), dx["Dx"].astype(str)))
        meta["Group"] = (meta["SampleID"].map(g)
                         .replace({"PD normal cognition": "PD-NC"}))
    else:
        gcol = "group" if "group" in part else "Group"
        meta["Group"] = part[gcol].astype(str)

    meta = meta.dropna(subset=["Group"]).reset_index(drop=True)
    ids = set(meta["SampleID"])

    files = sorted(froot.glob(f"{ds}_*_networkfeatures.tsv"))
    rows_full, rows_part, ids_feat = {}, {}, set()
    for f in files:
        sub = f.name.split("_", 2)[1]
        if sub in ids:
            d = pd.read_csv(f, sep="\t")
            d = d.iloc[0].to_dict()
            rows_full[sub] = {k: v for k, v in d.items() if k.startswith("full_")}
            rows_part[sub] = {k: v for k, v in d.items() if k.startswith("part_")}
            ids_feat.add(sub)

    def emit(df_rows, name):
        if not df_rows:
            print(f"[warn] no {name} rows")
            return
        d = pd.DataFrame.from_dict(df_rows, orient="index")
        d.index.name = "SampleID"
        d = d.apply(pd.to_numeric, errors="coerce")
        d = d.loc[:, d.notna().mean() > 0.5]
        d = d.apply(lambda c: c.fillna(c.median()), axis=0)
        d.reset_index().to_csv(out / "features" / f"{name}.tsv", sep="\t",
                               index=False)
        print(f"[ok] {name}: {d.shape[0]} subj x {d.shape[1]} feats")

    emit(rows_full, "img_full")
    emit(rows_part, "img_partial")

    if ds == "ds004392":
        hy = pd.read_csv(ds_root / "phenotype" / "hy_deidentified.tsv", sep="\t")
        hy = hy.rename(columns={"participant_id": "SampleID"})
        hy["SampleID"] = hy["SampleID"].astype(str)
        hy = hy[hy["SampleID"].isin(ids)]
        hy.to_csv(out / "features" / "clin_hy.tsv", sep="\t", index=False)

        cog = pd.read_csv(ds_root / "phenotype" / "cognitive_domains.tsv", sep="\t")
        cog = cog.rename(columns={"participant_id": "SampleID"})
        cog["SampleID"] = cog["SampleID"].astype(str)
        cog = cog[cog["SampleID"].isin(ids)]
        num = cog.select_dtypes(include=[np.number]).columns
        cog[num] = cog[num].apply(lambda c: c.fillna(c.median()))
        cog.to_csv(out / "features" / "clin_cog.tsv", sep="\t", index=False)

        tests = pd.read_csv(ds_root / "phenotype" / "cognitive_tests.tsv", sep="\t")
        tests = tests.rename(columns={"participant_id": "SampleID"})
        tests["SampleID"] = tests["SampleID"].astype(str)
        tests = tests[tests["SampleID"].isin(ids)]
        num = tests.select_dtypes(include=[np.number]).columns
        tests[num] = tests[num].apply(lambda c: c.fillna(c.median()))
        tests.to_csv(out / "features" / "clin_tests.tsv", sep="\t", index=False)
        print(f"[ok] clinical: hy {len(hy)}, cog {len(cog)}, tests {len(tests)}")

    rd = Path(args.radiomics_dir)
    rfiles = sorted(rd.glob(f"{ds}_*_radiomics.tsv")) if rd.exists() else []
    if rfiles:
        frames = [pd.read_csv(f, sep="\t") for f in rfiles]
        R = pd.concat(frames, ignore_index=True)
        R["SampleID"] = R["SampleID"].astype(str)
        R = R[R["SampleID"].isin(ids)]
        num = R.select_dtypes(include=[np.number]).columns
        keep = num[R[num].std() > 0]
        R = R[["SampleID"] + list(keep)]
        R.to_csv(out / "features" / "radiomics.tsv", sep="\t", index=False)
        print(f"[ok] radiomics: {R.shape[0]} subj x {R.shape[1]-1} feats")

    if args.with_edges:
        e_full, e_part = {}, {}
        for sub in sorted(ids_feat):
            npz = froot / f"{ds}_{sub}_matrices.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            iu = np.triu_indices(z["z_full"].shape[0], k=1)
            e_full[sub] = z["z_full"][iu]
            e_part[sub] = z["z_partial"][iu]
        for rows, name in ((e_full, "fc_full"), (e_part, "fc_partial")):
            if rows:
                d = pd.DataFrame.from_dict(rows, orient="index")
                d.columns = [f"e{i}" for i in range(d.shape[1])]
                d.index.name = "SampleID"
                d.reset_index().to_csv(out / "features" / f"{name}.tsv",
                                       sep="\t", index=False)
                print(f"[ok] {name}: {d.shape}")

    meta.to_csv(out / "metadata.tsv", sep="\t", index=False)
    print(f"[ok] metadata: {meta.shape}, groups: "
          f"{meta['Group'].value_counts().to_dict()}")
    print(f"[ok] feature subjects: {len(ids_feat)} / {len(ids)}")


if __name__ == "__main__":
    main()
