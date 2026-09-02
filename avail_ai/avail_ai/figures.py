# -*- coding: utf-8 -*-
"""
avail_figures — publication-quality multi-panel figure generator for AVAIL-AI.

Bennett style (Brain Communications / high-impact journal aesthetic):
  - white background, Arial, no top/right spines, sns.despine()
  - pastel palette, panel labels (a/b/c), 600 DPI + vector PDF

Each figure reads the TSV outputs produced by the notebook and saves PNG + PDF.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

CPAL = ['#2E5A87', '#5B9279', '#C77D9D', '#B0A7C9', '#3D2E5F']
BRAND_PURPLE = '#6B5B95'
GRAY = '#888888'
MODEL_COLORS = {
    'SimpleMean': '#2E5A87', 'WSimpleMean': '#3D2E5F', 'ElasticNetLR': '#5B9279',
    'MLP': '#C77D9D', 'MoE': '#B0A7C9', 'GNN': '#6B5B95',
    'Transformer': '#8C6D46', 'XGBoost': '#4E8098', 'TabNet': '#A1674A',
    'TabPFN': '#5C6BC0', 'TabICL': '#7E57C2',
}

# Yeo-17 sub-network -> Yeo-7 canonical abbreviation (matches the previous paper).
from .networks import NETWORK_FAMILIES, abbrev_network, abbrev_brain_label


def set_style():
    sns.set_style("white")
    mpl.rcParams.update({
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 11, 'axes.labelsize': 11, 'axes.linewidth': 1.0,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def _save(fig, out, name):
    fig.savefig(Path(out) / f'{name}.png', dpi=600)
    fig.savefig(Path(out) / f'{name}.pdf')
    plt.close(fig)
    print(f'  [fig] {name}.png/.pdf')


def _panel(ax, label):
    ax.text(-0.12, 1.04, label, transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')


def _model_color(m):
    return MODEL_COLORS.get(m, '#888888')


def _short(s, n=32):
    """Truncate a long label so figure text never overlaps."""
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + '…'


# ----------------------------------------------------------------------
def fig_classification(out):
    """(a) model macro-AUC (bar), (b) per-class AUC (grouped bar)."""
    res = pd.read_csv(Path(out) / 'meta_benchmark_3class_results.tsv', sep='\t')
    all_ = res[res['subgroup'] == 'all'].sort_values('macro_auc')
    pc = pd.read_csv(Path(out) / 'per_class_auc.tsv', sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    colors = [_model_color(m) for m in all_['model']]
    ax.barh(all_['model'], all_['macro_auc'], color=colors, alpha=0.9)
    ax.axvline(0.5, color=GRAY, ls=':', lw=1.2)
    ax.text(0.5, len(all_) - 0.3, 'Chance', color=GRAY, fontsize=9,
            fontstyle='italic', va='top')
    ax.set_xlabel('macro-AUC'); ax.set_xlim(0.35, 0.9)
    _panel(ax, 'a')

    ax = axes[1]
    x = np.arange(3); w = 0.08
    for i, row in pc.iterrows():
        ax.bar(x + (i - len(pc) / 2) * w, [row['HC'], row['MCI'], row['PDD']],
               width=w, color=_model_color(row['model']), alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(['HC', 'MCI', 'PDD'])
    ax.axhline(0.5, color=GRAY, ls=':', lw=1.2)
    ax.set_ylabel('one-vs-rest AUC'); ax.set_ylim(0.2, 0.9)
    ax.legend(pc['model'], loc='lower left', ncol=2, fontsize=8)
    _panel(ax, 'b')
    fig.suptitle('Meta-learner comparison (HC vs MCI vs PDD)', fontsize=12)
    fig.tight_layout()
    _save(fig, out, 'fig_classification')


def fig_ordinal(out):
    """Severity score HC < MCI < PDD (risk continuum)."""
    oc = pd.read_csv(Path(out) / 'ordinal_continuum.tsv', sep='\t')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for i, row in oc.iterrows():
        ax.plot([0, 1, 2], [row['HC_med'], row['MCI_med'], row['PDD_med']],
                marker='o', color=_model_color(row['model']),
                alpha=0.7 if row['model'] not in ('SimpleMean', 'ElasticNetLR') else 1.0,
                lw=2.5 if row['model'] in ('SimpleMean', 'ElasticNetLR') else 1.2)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['HC', 'MCI', 'PDD'])
    ax.set_ylabel('severity (0=HC, 2=PDD)')
    ax.legend(oc['model'], ncol=2, fontsize=8)
    _panel(ax, 'a')

    ax = axes[1]
    rows = oc.sort_values('spearman_rho')
    colors = [_model_color(m) for m in rows['model']]
    ax.barh(rows['model'], rows['spearman_rho'], color=colors, alpha=0.9)
    ax.set_xlabel('Spearman ρ (score vs HC→MCI→PDD)')
    ax.axvline(0, color=GRAY, lw=1)
    _panel(ax, 'b')
    fig.suptitle('Cognitive decline continuum (MCI = intermediate stage)', fontsize=12)
    fig.tight_layout()
    _save(fig, out, 'fig_ordinal')


def fig_dropout(out):
    """Drop-out recovery: strategies as the missingness rate rises."""
    d = pd.read_csv(Path(out) / 'dropout_recovery.tsv', sep='\t')
    summ = d.groupby(['rate', 'strategy']).agg(auc=('auc', 'mean')).reset_index()
    strat_col = {'AVAIL': '#5B9279', '0.5-placeholder': '#C77D9D', 'Complete-case': '#A1674A'}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for s, c in strat_col.items():
        sub = summ[summ['strategy'] == s]
        ax.plot(sub['rate'], sub['auc'], marker='o', color=c, label=s, lw=2.5)
    base = summ[summ['rate'] == summ['rate'].min()]['auc'].max()
    ax.axhline(base, color='k', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Missingness rate'); ax.set_ylabel('macro-AUC')
    ax.legend(); ax.set_ylim(0.35, 0.9)
    ax.set_title('Drop-out recovery (synthetic drop-out)')
    fig.tight_layout()
    _save(fig, out, 'fig_dropout')


def fig_biomarker(out, modality='stool16S'):
    """(a) top SHAP taxa, (b) top differential abundance (volcano)."""
    try:
        bio = pd.read_csv(Path(out) / 'biomarkers_top20_per_modality.tsv', sep='\t')
        bio = bio[bio['modality'] == modality]
    except Exception:
        bio = None
    diff = pd.read_csv(Path(out) / f'diff_abundance_{modality}.tsv', sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax = axes[0]
    if bio is not None and len(bio):
        b = bio.head(10).iloc[::-1].copy()
        b['feature'] = b['feature'].map(lambda s: _short(s, 34))
        ax.barh(b['feature'], b['importance'], color=BRAND_PURPLE, alpha=0.9)
        ax.set_xlabel('mean |SHAP|')
        ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f'{modality} — multivariate importance (SHAP)', fontsize=10)
    _panel(ax, 'a')

    ax = axes[1]
    sig = diff[diff['fdr_q'] < 0.05]
    ns = diff[diff['fdr_q'] >= 0.05]
    ax.scatter(ns['log2fc'], -np.log10(ns['p']), s=12, color='#CCCCCC', alpha=0.6)
    ax.scatter(sig['log2fc'], -np.log10(sig['p']), s=25, color=CPAL[0], alpha=0.9)
    for k, (_, r) in enumerate(sig.head(6).iterrows()):
        ax.annotate(_short(r['feature'], 26), (r['log2fc'], -np.log10(r['p'])),
                    fontsize=6.5, xytext=(3 + 6 * (k % 2), 3 + 6 * (k // 2)),
                    textcoords='offset points', alpha=0.9)
    ax.axhline(-np.log10(0.05), color=GRAY, ls=':', lw=1)
    ax.set_xlabel('log2FC (PDD vs HC)'); ax.set_ylabel('-log10(p)')
    ax.set_title(f'{modality} — differential abundance (volcano)', fontsize=10)
    _panel(ax, 'b')
    fig.suptitle('Biomarker discovery (multivariate + univariate)', fontsize=12)
    fig.tight_layout()
    _save(fig, out, f'fig_biomarker_{modality}')


def _crossmodal_heatmap(out, fname, title, save_name, k_micro=20, k_brain=25):
    f = Path(out) / fname
    if not f.exists():
        print(f'  [fig] {fname} missing, skipped'); return
    d = pd.read_csv(f, sep='\t')
    if 'brain' in d.columns:
        d['brain'] = d['brain'].map(abbrev_brain_label)
        # collapse duplicate (microbe, brain) pairs after Yeo-7 mapping
        d = d.groupby(['microbe', 'brain'], as_index=False).agg(
            rho=('rho', 'mean'), fdr_q=('fdr_q', 'min'))
    top_m = d.groupby('microbe')['fdr_q'].min().sort_values().head(k_micro).index
    top_b = d.groupby('brain')['fdr_q'].min().sort_values().head(k_brain).index
    piv = d[d['microbe'].isin(top_m) & d['brain'].isin(top_b)].pivot(
        index='microbe', columns='brain', values='rho')
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(piv.values, cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([_short(l, 32) for l in piv.index], fontsize=8)
    plt.colorbar(im, label='Spearman ρ', shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out, save_name)


def fig_crossmodal(out, tag='16S_stool'):
    """Focused microbiome<->brain correlation heatmap."""
    _crossmodal_heatmap(out, f'crossmodal_focused_{tag}.tsv',
                        f'Oral-gut-brain axis: top biomarkers × brain networks ({tag})',
                        f'fig_crossmodal_focused_{tag}')


def fig_crossmodal_full(out, tag='16S_stool'):
    """Full pairwise microbiome<->brain correlation heatmap."""
    _crossmodal_heatmap(out, f'crossmodal_{tag}.tsv',
                        f'Oral-gut-brain axis: microbiome × brain networks ({tag})',
                        f'fig_crossmodal_{tag}')


def fig_shap_meta(out):
    """XGBoost meta-learner meta-feature importance (mean |SHAP|)."""
    f = Path(out) / 'shap_meta_features.tsv'
    if not f.exists():
        print('  [fig] shap_meta_features.tsv missing, skipped'); return
    d = pd.read_csv(f, sep='\t').sort_values('importance').tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(d['feature'].map(lambda s: _short(s, 26)), d['importance'],
            color=BRAND_PURPLE, alpha=0.9)
    ax.set_xlabel('mean |SHAP|')
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title('XGBoost meta-learner: meta-feature importance')
    fig.tight_layout()
    _save(fig, out, 'fig_shap_meta')


def fig_kegg_functional(out, tag='stool'):
    """KEGG functional module SHAP importance (mechanism layer)."""
    f = Path(out) / f'kegg_functional_{tag}.tsv'
    if not f.exists():
        print(f'  [fig] kegg_functional_{tag}.tsv missing, skipped'); return
    d = pd.read_csv(f, sep='\t').sort_values('importance').tail(15)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d['name'].map(lambda s: _short(s, 42)), d['importance'],
            color=CPAL[0], alpha=0.9)
    ax.set_xlabel('mean |SHAP|')
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(f'KEGG functional modules ({tag}) — SHAP importance')
    fig.tight_layout()
    _save(fig, out, f'fig_kegg_functional_{tag}')


def make_all_figures(out):
    set_style()
    print('Generating figures...')
    steps = [
        lambda: fig_classification(out),
        lambda: fig_ordinal(out),
        lambda: fig_dropout(out),
        lambda: fig_shap_meta(out),
    ]
    steps += [lambda m=m: fig_biomarker(out, modality=m) for m in ['stool16S', 'stoolSG', 'fMRI']]
    steps += [lambda t=t: fig_crossmodal(out, tag=t) for t in ['16S_stool', '16S_saliva']]
    steps += [lambda t=t: fig_crossmodal_full(out, tag=t)
              for t in ['16S_stool', '16S_saliva', 'KEGG_stool', 'KEGG_saliva']]
    steps += [lambda t=t: fig_kegg_functional(out, tag=t) for t in ['stool', 'saliva']]
    for s in steps:
        try:
            s()
        except Exception as e:
            print(f'  [fig] skipped: {type(e).__name__}: {str(e)[:120]}')
    print('All figures complete.')


if __name__ == '__main__':
    import sys
    make_all_figures(sys.argv[1] if len(sys.argv) > 1 else '.')
