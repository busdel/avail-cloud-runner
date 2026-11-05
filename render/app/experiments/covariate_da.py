# -*- coding: utf-8 -*-
"""Sensitivity: does covariate adjustment (age+sex) change DA conclusions?

Per genus (stool/saliva 16S, CLR or log-transformed):
  A) unadjusted Mann-Whitney (as in the main analysis)
  B) age+sex adjusted logistic regression (LR p-value)
FDR-corrected overlap: how many unadjusted hits survive adjustment.

Outputs: experiments/results/covariate_sensitivity.tsv
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression

ROOT = Path("/media/suleyman_disk/AVAIL_GPU/AVAIL_benchmark")
OUT = Path("/media/suleyman_disk/AVAIL_GPU/experiments/results")


def clr(df):
    x = np.log(df.values + 1.0)
    return pd.DataFrame(x - x.mean(axis=1, keepdims=True),
                        index=df.index, columns=df.columns)


def main():
    meta = pd.read_csv(ROOT / "busra_pd_multisite/metadata.txt", sep="\t")
    meta["SampleID"] = meta["SampleID"].astype(str).str.strip()
    meta = meta[meta.Group.isin(["HC", "MCI", "PDD"])]
    conf = {}
    for r in meta.itertuples():
        age = np.nan if str(r.Age) in ("Missing_Data", "") else float(r.Age)
        sex = 1.0 if str(r.Sex).strip() == "M" else 0.0
        conf[r.SampleID] = (age, sex)
    ymap = {"HC": 0, "MCI": 1, "PDD": 2}
    y = {r.SampleID: ymap[r.Group] for r in meta.itertuples()}

    rows = []
    for tag, fname in [("stool16S", "stool_genus_p.txt"),
                       ("saliva16S", "saliva_genus_p.txt")]:
        g = pd.read_csv(ROOT / "busra_pd_multisite" / fname, sep="\t",
                        index_col=0)
        g.columns = [str(c).strip() for c in g.columns]
        g = g.loc[(g > 0).sum(axis=1) >= 5].T
        common = sorted(set(g.index) & set(conf))
        g = clr(g.loc[common])
        yy = np.array([y[p] for p in common])
        for pi in range(len(common)):  # pairwise class comparisons
            for ci in range(3):
                for cj in range(ci + 1, 3):
                    mask = (yy == ci) | (yy == cj)
                    if mask.sum() < 10:
                        continue
                    ybin = (yy[mask] == cj).astype(int)
                    pvals_u, pvals_lr = [], []
                    for col in g.columns:
                        x = g.loc[np.array(common)[mask], col].values
                        try:
                            _, p = mannwhitneyu(x[ybin == 1], x[ybin == 0])
                            pvals_u.append(p)
                        except Exception:
                            pvals_u.append(np.nan)
                        C = np.column_stack([
                            [conf[p][0] for p in np.array(common)[mask]],
                            [conf[p][1] for p in np.array(common)[mask]]])
                        C = np.nan_to_num(C, nan=np.nanmedian(
                            np.nan_to_num(C, nan=0)) or 0.0)
                        try:
                            lr = LogisticRegression(max_iter=2000)
                            lr.fit(C, ybin)
                            from scipy.stats import chi2
                            base = -2 * np.mean([ybin * 0])
                            ll0 = -len(ybin) * np.log(2) * 1.0
                            # likelihood ratio: model with cov only vs +feature
                            lr0 = LogisticRegression(max_iter=2000).fit(C, ybin)
                            p0 = lr0.predict_proba(C)[:, 1]
                            eps = 1e-9
                            ll_cov = np.mean(ybin * np.log(p0 + eps)
                                             + (1 - ybin) * np.log(1 - p0 + eps))
                            Xf = np.column_stack([C, x])
                            lr1 = LogisticRegression(max_iter=2000).fit(Xf, ybin)
                            p1 = lr1.predict_proba(Xf)[:, 1]
                            ll_feat = np.mean(ybin * np.log(p1 + eps)
                                              + (1 - ybin) * np.log(1 - p1 + eps))
                            stat = 2 * len(ybin) * (ll_feat - ll_cov)
                            pvals_lr.append(1 - chi2.cdf(stat, 1))
                        except Exception:
                            pvals_lr.append(np.nan)
                    qu = multipletests(pvals_u, method="fdr_bh")[1]
                    ql = multipletests(pvals_lr, method="fdr_bh")[1]
                    sig_u = set(np.where(qu < 0.05)[0])
                    sig_l = set(np.where(ql < 0.05)[0])
                    rows.append({
                        "layer": tag,
                        "comparison": f"cls{ci}_vs_cls{cj}",
                        "n": int(mask.sum()),
                        "n_sig_unadj": len(sig_u),
                        "n_sig_adj": len(sig_l),
                        "overlap": len(sig_u & sig_l),
                        "kept_fraction": (len(sig_u & sig_l) / len(sig_u)
                                          if sig_u else np.nan),
                    })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "covariate_sensitivity.tsv", sep="\t", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
