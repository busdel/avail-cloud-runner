# -*- coding: utf-8 -*-
"""AVAIL-AI data loading: labels, confounders, modalities (with feature names)."""
import numpy as np
import pandas as pd
from pathlib import Path

from .networks import abbrev_network


def load_labels(data_root):
    """Unified group labels keyed by anonymized patient ID.

    Merges three sources: metadata.txt (SampleID->Group), y_data.xlsx
    (fMRI PatientID->Group), ID_matching_master.csv (patient->Group).
    Returns {patient_id: 'HC'|'MCI'|'PDD'|'PDNC'}.
    """
    meta = pd.read_csv(data_root / 'busra_pd_multisite' / 'metadata.txt', sep='\t')
    meta['SampleID'] = meta['SampleID'].astype(str).str.strip()
    g_meta = dict(zip(meta['SampleID'], meta['Group']))

    y = pd.read_excel(data_root / 'fMRI correlation and feature selection' / 'y_data.xlsx')
    y['PatientID'] = y['PatientID'].astype(str).str.strip()
    gmap_f = {'control': 'HC', 'mci': 'MCI', 'dementia': 'PDD'}
    g_f = dict(zip(y['PatientID'], y['Group'].map(gmap_f)))

    master = pd.read_csv(data_root / 'yeni_eklenenler' / 'paper_pipeline' /
                         'integration' / 'ID_matching_master.csv')
    master['patient'] = master['patient'].astype(str).str.strip()
    gmap_m = {'HC': 'HC', 'PD-NC': 'PDNC', 'PD-MCI': 'MCI', 'PDD': 'PDD'}
    g_m = {p: gmap_m[g] for p, g in zip(master['patient'], master['group']) if g in gmap_m}

    group = {}
    for i in set(g_meta) | set(g_f) | set(g_m):
        g = g_meta.get(i) or g_f.get(i) or g_m.get(i)
        if g:
            group[i] = g
    return group


def load_confounders(data_root):
    """(age, education, sex) per patient; NaN where missing."""
    meta = pd.read_csv(data_root / 'busra_pd_multisite' / 'metadata.txt', sep='\t')
    meta['SampleID'] = meta['SampleID'].astype(str).str.strip()
    conf = {}
    for _, r in meta.iterrows():
        sid = r['SampleID']

        def _f(v):
            return np.nan if str(v) in ('Missing_Data', '', 'nan') else float(v)
        sex = 1.0 if str(r['Sex']).strip() == 'M' else (0.0 if str(r['Sex']).strip() == 'F' else np.nan)
        conf[sid] = (_f(r['Age']), _f(r['Education']), sex)
    return conf


def load_modalities(data_root, min_prevalence=5):
    """Per-modality feature dict {patient_id: feature_vector}.

    Modalities: fMRI, stool16S, saliva16S, stoolSG (MGS+KEGG), salivaSG (MGS+KEGG).
    """
    mods = {}
    sf = pd.read_excel(data_root / 'fMRI correlation and feature selection' / 'selected_feature_data.xlsx')
    sf['PatientID'] = sf['PatientID'].astype(str).str.strip()
    X_f = sf.drop(columns=['PatientID']).values.astype(np.float64)
    mods['fMRI'] = {pid: X_f[i] for i, pid in enumerate(sf['PatientID'])}

    for name, fname in [('stool16S', 'stool_genus_p.txt'), ('saliva16S', 'saliva_genus_p.txt')]:
        df = pd.read_csv(data_root / 'busra_pd_multisite' / fname, sep='\t', index_col=0)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[(df > 0).sum(axis=1) >= min_prevalence]
        X = df.T.values.astype(np.float64)
        mods[name] = {pid: X[i] for i, pid in enumerate(df.columns)}

    cw = pd.read_csv(data_root / 'yeni_eklenenler' / 'paper_pipeline' / 'DELIVERABLE' /
                     'raw_outputs' / 'err_patient_crosswalk.tsv', sep='\t')
    cw['patient'] = cw['patient'].astype(str).str.strip()
    cmap = dict(zip(cw['run_accession'].astype(str), cw['patient']))

    outdir = data_root / 'yeni_eklenenler' / 'paper_pipeline' / 'out'
    gut = pd.read_csv(outdir / 'gut_mgs_abundance_freq.tsv', sep='\t', index_col=0)
    oral = pd.read_csv(outdir / 'oral_mgs_abundance_freq.tsv', sep='\t', index_col=0)
    gut_keg = pd.read_csv(outdir / 'stool_modules.tsv', sep='\t', index_col=0)
    oral_keg = pd.read_csv(outdir / 'saliva_modules.tsv', sep='\t', index_col=0)
    gcom = sorted(set(gut.columns) & set(gut_keg.columns))
    ocom = sorted(set(oral.columns) & set(oral_keg.columns))
    gut = pd.concat([gut[gcom], gut_keg[gcom]], axis=0)
    oral = pd.concat([oral[ocom], oral_keg[ocom]], axis=0)
    gut.columns = [cmap.get(c, c) for c in gut.columns]
    oral.columns = [cmap.get(c, c) for c in oral.columns]
    gut = gut.loc[(gut > 0).sum(axis=1) >= min_prevalence]
    oral = oral.loc[(oral > 0).sum(axis=1) >= min_prevalence]
    mods['stoolSG'] = {pid: gut.T.values.astype(np.float64)[i] for i, pid in enumerate(gut.columns)}
    mods['salivaSG'] = {pid: oral.T.values.astype(np.float64)[i] for i, pid in enumerate(oral.columns)}
    return mods


def load_named_features(data_root, min_prevalence=5):
    """Per-modality feature matrices WITH feature names (for biomarker analysis)."""
    from .preprocess import CLRTransformer  # noqa: F401  (kept for symmetry)
    out = {}
    roi = pd.read_csv(data_root / 'yeni_eklenenler' / 'paper_pipeline' / 'metadata_fmr' /
                      'roi_network_map.csv')
    node2net = dict(zip(roi['Node Number'], roi['17 Network']))
    nodes = list(range(1, 100))
    tri = [(a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:]]

    def fmri_name(f):
        try:
            r, c = tri[int(f) - 1]
            a, b = abbrev_network(node2net.get(r + 1, r + 1)), abbrev_network(node2net.get(c + 1, c + 1))
            return f'within {a}' if a == b else '-'.join(sorted([a, b]))
        except Exception:
            return f'ROI_{f}'

    sf = pd.read_excel(data_root / 'fMRI correlation and feature selection' / 'selected_feature_data.xlsx')
    sf['PatientID'] = sf['PatientID'].astype(str).str.strip()
    fdf = sf.drop(columns=['PatientID'])
    out['fMRI'] = dict(X=fdf.values.astype(float),
                       names=[fmri_name(c) for c in fdf.columns],
                       patients=sf['PatientID'].tolist())

    for name, fname in [('stool16S', 'stool_genus_p.txt'), ('saliva16S', 'saliva_genus_p.txt')]:
        df = pd.read_csv(data_root / 'busra_pd_multisite' / fname, sep='\t', index_col=0)
        df.columns = [str(c).strip() for c in df.columns]
        df = df.loc[(df > 0).sum(axis=1) >= min_prevalence]
        out[name] = dict(X=df.T.values.astype(float), names=list(df.index), patients=list(df.columns))

    cw = pd.read_csv(data_root / 'yeni_eklenenler' / 'paper_pipeline' / 'DELIVERABLE' /
                     'raw_outputs' / 'err_patient_crosswalk.tsv', sep='\t')
    cmap = dict(zip(cw['run_accession'].astype(str), cw['patient'].astype(str)))
    outdir = data_root / 'yeni_eklenenler' / 'paper_pipeline' / 'out'
    for name, fname in [('stoolSG', 'gut_mgs_abundance_freq.tsv'),
                        ('salivaSG', 'oral_mgs_abundance_freq.tsv')]:
        df = pd.read_csv(outdir / fname, sep='\t', index_col=0)
        df.columns = [cmap.get(c, c) for c in df.columns]
        df = df.loc[(df > 0).sum(axis=1) >= min_prevalence]
        out[name] = dict(X=df.T.values.astype(float), names=list(df.index), patients=list(df.columns))

    st_tax = pd.read_csv(outdir / 'stool_msp_taxonomy.tsv', sep='\t', index_col=0)
    sa_tax = pd.read_csv(outdir / 'saliva_msp_taxonomy.tsv', sep='\t', index_col=0)
    return out, st_tax, sa_tax
