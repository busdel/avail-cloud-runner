# -*- coding: utf-8 -*-
"""Publication figures in the manuscript style (Nature/Science-like, clean).

Style extracted from the paper's unified_manuscript_figures.py:
  - DejaVu Sans, white background, thin spines, no top/right spines
  - bold panel letters (a/b/c), bold figure titles, "Key message" text panels
  - consistent palette, 600 dpi PDF/PNG + 300 dpi TIFF (LZW)
"""
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------- style
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'figure.dpi': 200, 'savefig.dpi': 600, 'axes.grid': False,
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
})

MCTRL = '#a6dba0'; MPD = '#1b9e77'; FCTRL = '#c2a5cf'; FPD = '#7570b3'
MALE_C = '#2166ac'; FEMALE_C = '#b2182b'; SHARED_C = '#762a83'
DARK = '#2d2d2d'; MED = '#666666'; LIGHT = '#e0e0e0'; ACCENT = '#e6ab02'
BLUE = '#4393c3'; RED = '#d6604d'; GREEN = '#4dac26'; ORANGE = '#f4a582'
MOD_PAL = ['#2166ac', '#1b9e77', '#7570b3', '#e6ab02', '#b2182b', '#762a83',
           '#4393c3', '#4dac26', '#d6604d', '#8c6d31', '#666666']
CLASS_PAL = ['#4393c3', '#e6ab02', '#b2182b']


def panel(ax, letter, x=-0.08, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=13,
            fontweight='bold', color=DARK, va='bottom')


def clean(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)


def save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [('pdf', 600), ('png', 600), ('tiff', 300)]:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(out_dir / f'{name}.{ext}', dpi=dpi, facecolor='white',
                    bbox_inches='tight', pad_inches=0.2, **kw)
    plt.close(fig)
    print(f'  ✓ {name} (pdf/png/tiff)', flush=True)


def keypanel(ax, title, text):
    ax.axis('off')
    ax.text(0.5, 0.52, text, transform=ax.transAxes, fontsize=9.5, ha='center',
            va='center', color=DARK, linespacing=1.5)
    ax.text(0.5, 0.95, title, transform=ax.transAxes, fontsize=11,
            fontweight='bold', color=DARK, ha='center')


def _read(out_dir, *names):
    for n in names:
        p = out_dir / n
        if p.exists():
            return pd.read_csv(p, sep='\t')
    return None


# ---------------------------------------------------------------- figures
def fig_benchmark(out_dir):
    res = _read(out_dir, 'meta_benchmark_3class_results.tsv',
                'meta_benchmark_results.tsv')
    if res is None:
        return None
    per_class = _read(out_dir, 'per_class_auc.tsv')
    subs = [s for s in ['all', 'complete5', 'complete', 'ge3mod', 'le2mod']
            if (res['subgroup'] == s).any()]
    piv = res.pivot(index='model', columns='subgroup', values='macro_auc')[subs]
    piv = piv.sort_values('all', ascending=False)
    models = list(piv.index)[:8]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white',
                             gridspec_kw={'width_ratios': [1.5, 1.2, 0.9]})

    ax = axes[0]
    x = np.arange(len(models))
    w = 0.8 / len(subs)
    for i, s in enumerate(subs):
        vals = piv[s].reindex(models).values
        ax.bar(x + (i - (len(subs) - 1) / 2) * w, vals, w, color=MOD_PAL[i],
               alpha=0.85, ec='white', label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=7.5)
    ax.set_ylabel('Macro-AUC')
    ax.set_ylim(max(0.3, piv.values.min() - 0.08), min(1.02, piv.values.max() + 0.08))
    ax.legend(frameon=False, loc='lower left', ncol=2)
    clean(ax); panel(ax, 'a')
    ax.set_title('Meta-learner benchmark', fontsize=10, color=DARK)

    ax = axes[1]
    if per_class is not None:
        pc = per_class.set_index('model')
        classes = [c for c in pc.columns]
        top5 = list(pc.index[:5])
        x = np.arange(len(top5))
        w = 0.8 / len(classes)
        for i, cls in enumerate(classes):
            vals = [pc.loc[m, cls] for m in top5]
            ax.bar(x + (i - (len(classes) - 1) / 2) * w, vals, w,
                   color=CLASS_PAL[i % len(CLASS_PAL)], alpha=0.85, ec='white',
                   label=cls)
        ax.set_xticks(x)
        ax.set_xticklabels(top5, rotation=35, ha='right', fontsize=7.5)
        ax.set_ylabel('AUC (one-vs-rest)')
        ax.legend(frameon=False)
        ax.set_ylim(0.3, 1.02)
    clean(ax); panel(ax, 'b')
    ax.set_title('Per-class AUC (top models)', fontsize=10, color=DARK)

    ax = axes[2]
    best = piv['all'].idxmax()
    best_auc = piv['all'].max()
    txt = ('Meta-Learner Summary\n\n'
           f'Best model: {best}\n(AUC = {best_auc:.3f})\n\n'
           f'{len(models)} models benchmarked\n'
           'with leak-free LOO\n\n'
           'Availability flags keep\n'
           'every subject in the model\n'
           '— zero sample loss')
    keypanel(ax, 'Key Message', txt)
    panel(ax, 'c')

    fig.suptitle('Availability-Aware Multimodal Integration — Model Benchmark',
                 fontweight='bold', fontsize=12, color=DARK, y=1.02)
    plt.tight_layout()
    save(fig, out_dir, 'fig1_meta_benchmark')
    return fig


def fig_biomarkers(out_dir):
    bio = _read(out_dir, 'biomarkers_top20_per_modality.tsv')
    diff = _read(out_dir, 'diff_abundance_all.tsv',
                 'diff_abundance_stool16S.tsv', 'diff_abundance_stoolSG.tsv',
                 'diff_abundance_saliva16S.tsv')
    if bio is None and diff is None:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white',
                             gridspec_kw={'width_ratios': [1.4, 1.2, 0.9]})

    ax = axes[0]
    if bio is not None:
        mods = sorted(bio['modality'].unique())
        m = mods[0]
        top = bio[bio['modality'] == m].sort_values('importance').head(12)
        yp = range(len(top))
        ax.barh(list(yp), top['importance'].values[::-1], color=DARK, alpha=0.85,
                ec='white')
        ax.set_yticks(list(yp))
        ax.set_yticklabels(top['feature'].values[::-1], fontsize=7.5)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title(f'Top biomarkers — {m}', fontsize=10, color=DARK)
        if len(mods) > 1:
            ax.text(0.98, 0.02, f'{len(mods)} modalities analysed',
                    transform=ax.transAxes, ha='right', fontsize=7,
                    color=MED, style='italic')
    clean(ax); panel(ax, 'a')

    ax = axes[1]
    if diff is not None:
        d = diff.copy()
        pcol = 'p_value' if 'p_value' in d else ('p' if 'p' in d else None)
        qcol = 'q_value' if 'q_value' in d else ('fdr_q' if 'fdr_q' in d else None)
        if pcol is not None and 'log2fc' in d:
            d = d.dropna(subset=[pcol, 'log2fc'])
            d = d[d[pcol] > 0]
            p = -np.log10(d[pcol].values)
            q = d[qcol].values < 0.05 if qcol is not None else d[pcol] < 0.05
            ax.scatter(d['log2fc'][~q], p[~q], c=LIGHT, s=4, alpha=0.25,
                       rasterized=True)
            up = q & (d['log2fc'] > 0)
            dn = q & (d['log2fc'] < 0)
            ax.scatter(d['log2fc'][up], p[up], c=RED, s=12, alpha=0.5,
                       label=f'Up ({up.sum()})', rasterized=True)
            ax.scatter(d['log2fc'][dn], p[dn], c=BLUE, s=12, alpha=0.5,
                       label=f'Down ({dn.sum()})', rasterized=True)
            ax.axhline(y=-np.log10(0.05), color=MED, lw=0.6, ls='--')
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel('log₂ fold change')
        ax.set_ylabel('−log₁₀(p)')
        ax.set_title('Differential abundance', fontsize=10, color=DARK)
    clean(ax); panel(ax, 'b')

    ax = axes[2]
    n_sig = 0
    if diff is not None:
        d2 = diff.copy()
        qcol = 'q_value' if 'q_value' in d2 else ('fdr_q' if 'fdr_q' in d2 else None)
        if qcol is not None:
            n_sig = int((d2[qcol] < 0.05).sum())
    txt = ('Biomarker Discovery\n\n'
           f'FDR-significant features:\n{n_sig}\n\n'
           'SHAP ranks predictive\n'
           'importance per modality\n\n'
           'Volcano: effect size vs\n'
           'statistical significance\n\n'
           '→ See biomarkers_literature.tsv\n'
           'for PubMed evidence')
    keypanel(ax, 'Key Message', txt)
    panel(ax, 'c')

    fig.suptitle('Biomarker Discovery — SHAP Importance & Differential Abundance',
                 fontweight='bold', fontsize=12, color=DARK, y=1.02)
    plt.tight_layout()
    save(fig, out_dir, 'fig2_biomarkers')
    return fig


def fig_continuum(out_dir):
    preds = _read(out_dir, 'loo_predictions.tsv', 'loo_predictions_3class.tsv')
    if preds is None:
        return None
    cols = [c for c in preds.columns if c.startswith('SimpleMean_')]
    if not cols:
        return None
    classes = [c.split('_', 1)[1] for c in cols]
    K = len(classes)
    if K < 2:
        return None
    scores = []
    for _, r in preds.iterrows():
        ps = np.array([r[c] for c in cols])
        if K == 2:
            scores.append(ps[1])
        else:
            scores.append(ps[K - 1] - ps[0])
    preds = preds.copy()
    preds['risk'] = scores
    y = preds['y'].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white',
                             gridspec_kw={'width_ratios': [1.4, 1.2, 0.9]})

    ax = axes[0]
    for k in range(K):
        d = preds.loc[preds['y'] == k, 'risk'].values
        if len(d):
            vp = ax.violinplot(d, [k], widths=0.7, showmedians=True)
            for b in vp['bodies']:
                b.set_facecolor(CLASS_PAL[k % len(CLASS_PAL)])
                b.set_alpha(0.6)
            vp['cmedians'].set_color(DARK)
            vp['cmedians'].set_linewidth(1.5)
    ax.set_xticks(range(K))
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax.set_ylabel(f'Risk score ({classes[-1]} − {classes[0]})')
    ax.set_title('Risk continuum', fontsize=10, color=DARK)
    clean(ax); panel(ax, 'a')

    ax = axes[1]
    avail = None
    bp_path = Path(out_dir) / 'base_predictions_generic.tsv'
    if bp_path.exists():
        bp = pd.read_csv(bp_path, sep='\t')
        avail = bp.groupby('patient')['modality'].count()
    if avail is not None:
        cnts = avail.value_counts().sort_index()
        ax.bar([str(k) for k in cnts.index], cnts.values, color=DARK, alpha=0.85,
               ec='white', width=0.55)
        ax.set_xlabel('# available modalities')
        ax.set_ylabel('# subjects')
        ax.set_title('Modality availability', fontsize=10, color=DARK)
    clean(ax); panel(ax, 'b')

    ax = axes[2]
    txt = ('Risk Continuum\n\n'
           'Probability-based risk score\n'
           'places every subject on the\n'
           f'{classes[0]} → {classes[-1]} spectrum\n\n'
           'Subjects contribute regardless\n'
           'of missing modalities\n\n'
           '→ availability-aware scoring')
    keypanel(ax, 'Key Message', txt)
    panel(ax, 'c')

    fig.suptitle('Ordinal Risk Continuum & Modality Availability',
                 fontweight='bold', fontsize=12, color=DARK, y=1.02)
    plt.tight_layout()
    save(fig, out_dir, 'fig3_continuum')
    return fig


def make_all_figures(out_dir):
    out_dir = Path(out_dir)
    made = []
    for fn in [fig_benchmark, fig_biomarkers, fig_continuum]:
        try:
            if fn(out_dir) is not None:
                made.append(fn.__name__)
        except Exception as e:
            print(f'  ! {fn.__name__} failed: {type(e).__name__}: {str(e)[:200]}',
                  flush=True)
    print(f'[figures] done: {made}', flush=True)
    return made
