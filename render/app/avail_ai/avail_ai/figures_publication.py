# -*- coding: utf-8 -*-
"""Publication figures — manuscript Bennett style (v13/v14), varied layouts.

Each figure uses a different visual grammar (bars+heatmap+radar,
lollipop+volcano+dot plot, ridgeline+donut+band) so the paper does not
repeat itself. All annotations are placed OUTSIDE data marks or with
dedicated offsets — labels never sit on top of the graphics.
"""
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'axes.labelsize': 15, 'axes.linewidth': 1.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'legend.fontsize': 13, 'legend.frameon': False,
    'lines.linewidth': 1.8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
sns.set_style("white")

CPAL = ['#2E5A87', '#5B9279', '#C77D9D', '#B0A7C9', '#3D2E5F',
        '#4393C3', '#4DAC26', '#D6604D', '#8C6D31', '#666666', '#762A83']
POS_C = '#B4525C'
NEG_C = '#2E5A87'
GOLD = '#E6AB02'


def _panel(ax, letter):
    ax.text(-0.10, 1.05, letter, transform=ax.transAxes, fontsize=19,
            fontweight='bold', va='bottom', ha='left')


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out_dir / f'{name}.png', bbox_inches='tight', dpi=600)
    fig.savefig(out_dir / f'{name}.tiff', bbox_inches='tight', dpi=600,
                pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print(f'  ✓ {name} (pdf/png/tiff)', flush=True)


def _read(out_dir, *names):
    for n in names:
        p = out_dir / n
        if p.exists():
            return pd.read_csv(p, sep='\t')
    return None


# ───────────────────────────────────────────── FIG 1 · bars + heatmap + radar
def fig_benchmark(out_dir):
    res = _read(out_dir, 'meta_benchmark_3class_results.tsv',
                'meta_benchmark_results.tsv')
    if res is None:
        return None
    per_class = _read(out_dir, 'per_class_auc.tsv')
    loo = _read(out_dir, 'loo_predictions.tsv', 'loo_predictions_3class.tsv')
    bp = _read(out_dir, 'base_predictions_generic.tsv',
               'base_predictions_3class_v3_kegg.tsv')

    subs = [s for s in ['all', 'complete5', 'complete', 'ge3mod', 'le2mod']
            if (res['subgroup'] == s).any()]
    piv = res.pivot(index='model', columns='subgroup', values='macro_auc')[subs]
    piv = piv.sort_values('all', ascending=False)
    models = list(piv.index)

    fig = plt.figure(figsize=(21, 7), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0], wspace=0.32)

    # a — horizontal bar (all subjects) with labels at bar ends
    ax = fig.add_subplot(gs[0, 0])
    vals = piv['all'].reindex(models).values
    yp = np.arange(len(models))[::-1]
    colors = [CPAL[i % len(CPAL)] for i in range(len(models))]
    ax.barh(yp, vals, color=colors, alpha=0.85, edgecolor='white',
            linewidth=0.5, height=0.62)
    xmax = max(vals) + 0.045
    ax.set_xlim(0, xmax)
    for y, v in zip(yp, vals):
        ax.text(v + 0.006, y, f'{v:.3f}', va='center', ha='left', fontsize=12,
                fontweight='bold', color='#2d2d2d')
    ax.set_yticks(yp)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Macro-AUC (all subjects)')
    ax.axvline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
    ax.set_title('Meta-learner ranking', fontsize=11)
    sns.despine(ax=ax)
    _panel(ax, 'a')

    # b — heatmap (models × subgroups)
    ax = fig.add_subplot(gs[0, 1])
    hm = piv.loc[models].clip(0.4, 1.0)
    sns.heatmap(hm, ax=ax, cmap='YlGnBu', vmin=0.45, vmax=1.0,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                linewidths=0.6, linecolor='white', cbar_kws={'shrink': 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel('')
    ax.set_title('Macro-AUC by subgroup', fontsize=11)
    _panel(ax, 'b')

    # c — radar of per-class AUC (top 3 models)
    ax = fig.add_subplot(gs[0, 2], projection='polar')
    if per_class is not None:
        pc = per_class.set_index('model')
        classes = [c for c in pc.columns]
        top3 = [m for m in models[:3] if m in pc.index]
        ang = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist()
        ang += ang[:1]
        for j, m in enumerate(top3):
            v = [max(pc.loc[m, c], 0.0) for c in classes]
            v += v[:1]
            ax.plot(ang, v, 'o-', color=CPAL[j], lw=2, ms=5, label=m)
            ax.fill(ang, v, color=CPAL[j], alpha=0.10)
        ax.set_xticks(ang[:-1])
        ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8,
                           color='grey')
        ax.tick_params(axis='y', colors='grey', labelsize=8)
        ax.legend(loc='lower right', bbox_to_anchor=(1.35, 0.02), frameon=False,
                  fontsize=9)
        ax.set_title('Per-class AUC (top models)', fontsize=11, pad=22)
    else:
        ax.set_axis_off()
    _panel(ax, 'c')

    fig.suptitle('Availability-aware multimodal integration — model benchmark',
                 fontsize=17, fontweight='bold')
    _save(fig, out_dir, 'fig1_meta_benchmark')
    return fig


# ─────────────────────────────────────── FIG 2 · lollipop + volcano + dot plot
def fig_biomarkers(out_dir):
    bio = _read(out_dir, 'biomarkers_top20_per_modality.tsv')
    diff = _read(out_dir, 'diff_abundance_all.tsv',
                 'diff_abundance_stool16S.tsv', 'diff_abundance_stoolSG.tsv',
                 'diff_abundance_saliva16S.tsv', 'diff_abundance_salivaSG.tsv')
    if bio is None and diff is None:
        return None

    fig = plt.figure(figsize=(21, 7), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.15, 1.0], wspace=0.34)

    # a — lollipop: top-14 SHAP (label every other item)
    ax = fig.add_subplot(gs[0, 0])
    if bio is not None:
        top = bio.sort_values('importance', ascending=False).head(14).iloc[::-1]
        yp = np.arange(len(top))
        imp = top['importance'].values
        ax.hlines(yp, 0, imp, colors='#b8c4d0', lw=1.2, zorder=1)
        ax.scatter(imp, yp, s=85, c=[NEG_C if i % 2 else POS_C
                                     for i in range(len(top))],
                   alpha=0.9, zorder=2, edgecolors='white', linewidths=0.6)
        xmax = max(imp) * 1.18
        ax.set_xlim(0, xmax)
        for i, (y, v) in enumerate(zip(yp, imp)):
            if i % 2 == 0:
                ax.text(v + max(imp) * 0.015, y, f'{v:.3f}', va='center',
                        ha='left', fontsize=9, color='#2d2d2d')
        ax.set_yticks(yp)
        ax.set_yticklabels([str(f)[:26] for f in top['feature']], fontsize=8.5)
        ax.set_xlabel('Mean |SHAP|')
        ax.set_title('Top SHAP biomarkers', fontsize=11)
    sns.despine(ax=ax)
    _panel(ax, 'a')

    # b — volcano with top-6/6 labelled
    ax = fig.add_subplot(gs[0, 1])
    if diff is not None:
        d = diff.copy()
        pcol = 'p_value' if 'p_value' in d else ('p' if 'p' in d else None)
        qcol = 'q_value' if 'q_value' in d else ('fdr_q' if 'fdr_q' in d else None)
        if pcol is not None and 'log2fc' in d:
            d = d.dropna(subset=[pcol, 'log2fc'])
            d = d[d[pcol] > 0]
            p = -np.log10(d[pcol].values)
            q = d[qcol].values < 0.05 if qcol is not None else d[pcol] < 0.05
            ax.scatter(d['log2fc'][~q], p[~q], c='#d5d5d5', s=5, alpha=0.35,
                       rasterized=True)
            up = q & (d['log2fc'] > 0)
            dn = q & (d['log2fc'] < 0)
            ax.scatter(d['log2fc'][up], p[up], c=POS_C, s=16, alpha=0.6,
                       rasterized=True)
            ax.scatter(d['log2fc'][dn], p[dn], c=NEG_C, s=16, alpha=0.6,
                       rasterized=True)
            ax.axhline(y=-np.log10(0.05), color='grey', lw=0.8, ls='--')
            lab = pd.concat([d[up].nlargest(6, pcol), d[dn].nlargest(6, pcol)])
            for i, (_, r) in enumerate(lab.iterrows()):
                dy = 0.35 if i % 2 == 0 else -0.35
                ax.annotate(str(r['feature'])[:20],
                            (r['log2fc'], -np.log10(r[pcol])),
                            xytext=(6, 8 * (1 if i % 2 == 0 else -1) + dy),
                            textcoords='offset points', fontsize=7.5,
                            color='#333333',
                            arrowprops=dict(arrowstyle='-', lw=0.4,
                                            color='#999999'))
        ax.set_xlabel('log$_2$ fold change')
        ax.set_ylabel('−log$_{10}$(p)')
        ax.set_title('Differential abundance (top 12 labelled)', fontsize=11)
    sns.despine(ax=ax)
    _panel(ax, 'b')

    # c — dot plot: top FDR-sig effect sizes
    ax = fig.add_subplot(gs[0, 2])
    if diff is not None:
        d = diff.copy()
        qcol = 'q_value' if 'q_value' in d else ('fdr_q' if 'fdr_q' in d else None)
        if qcol is not None and 'log2fc' in d:
            sig = d[d[qcol] < 0.05].copy()
            sig['abs'] = sig['log2fc'].abs()
            sig = sig.sort_values('abs', ascending=False).head(12).iloc[::-1]
            yp = np.arange(len(sig))
            ax.scatter(sig['log2fc'].values, yp,
                       c=[POS_C if v > 0 else NEG_C for v in sig['log2fc']],
                       s=95, alpha=0.9, edgecolors='white', linewidths=0.6)
            xmax = max(abs(sig['log2fc'].min()), abs(sig['log2fc'].max())) * 1.3
            ax.set_xlim(-xmax, xmax)
            for y, (_, r) in zip(yp, sig.iterrows()):
                ax.text(0, y, f"{r['log2fc']:+.2f}", va='center',
                        ha='left' if r['log2fc'] >= 0 else 'right',
                        fontsize=9, fontweight='bold',
                        color=POS_C if r['log2fc'] > 0 else NEG_C)
            ax.axvline(0, color='grey', lw=0.8)
            ax.set_yticks(yp)
            ax.set_yticklabels([str(f)[:22] for f in sig['feature']], fontsize=8.5)
            ax.set_xlabel('log$_2$ fold change (FDR < 0.05)')
            ax.set_title('Top differential features', fontsize=11)
    sns.despine(ax=ax)
    _panel(ax, 'c')

    fig.suptitle('Biomarker discovery — SHAP importance and differential abundance',
                 fontsize=17, fontweight='bold')
    _save(fig, out_dir, 'fig2_biomarkers')
    return fig


# ──────────────────────────────────── FIG 3 · ridgeline + donut + AUC band
def fig_continuum(out_dir):
    loo = _read(out_dir, 'loo_predictions.tsv', 'loo_predictions_3class.tsv')
    bp = _read(out_dir, 'base_predictions_generic.tsv',
               'base_predictions_3class_v3_kegg.tsv')
    res = _read(out_dir, 'meta_benchmark_3class_results.tsv',
                'meta_benchmark_results.tsv')
    if loo is None:
        return None
    cols = [c for c in loo.columns if c.startswith('SimpleMean_')]
    if not cols:
        return None
    classes = [c.split('_', 1)[1] for c in cols]
    K = len(classes)
    if K < 2:
        return None
    scores = []
    for _, r in loo.iterrows():
        ps = np.array([r[c] for c in cols])
        scores.append(ps[1] if K == 2 else ps[K - 1] - ps[0])
    loo = loo.copy()
    loo['risk'] = scores

    fig = plt.figure(figsize=(21, 7), facecolor='white')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.9, 1.15], wspace=0.34)

    # a — ridgeline of risk scores
    ax = fig.add_subplot(gs[0, 0])
    from scipy.stats import gaussian_kde
    xg = np.linspace(min(scores) - 0.25, max(scores) + 0.25, 300)
    for k in range(K):
        d = loo.loc[loo['y'] == k, 'risk'].values
        if len(d) < 3:
            continue
        dens = gaussian_kde(d)(xg)
        dens = dens / dens.max() * 0.85
        off = k
        ax.fill_between(xg, off, off + dens, color=CPAL[k], alpha=0.28,
                        lw=0)
        ax.plot(xg, off + dens, color=CPAL[k], lw=1.8)
        ax.text(min(scores) - 0.22, off + 0.35, classes[k], fontsize=10,
                fontweight='bold', color=CPAL[k], va='center')
    ax.set_yticks([])
    ax.set_xlabel('Risk score')
    ax.set_xlim(min(scores) - 0.25, max(scores) + 0.25)
    ax.set_title('Risk score distributions', fontsize=11)
    sns.despine(ax=ax, left=True)
    _panel(ax, 'a')

    # b — donut of modality availability
    ax = fig.add_subplot(gs[0, 1])
    if bp is not None:
        cnts = bp.groupby('patient')['modality'].count().value_counts().sort_index()
        wedges, texts, autotexts = ax.pie(
            cnts.values, labels=None, autopct='%1.0f%%', startangle=90,
            colors=CPAL[:len(cnts)], pctdistance=0.82,
            wedgeprops=dict(width=0.38, edgecolor='white', linewidth=1.5))
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight('bold')
            at.set_color('white')
        ax.text(0, 0.08, str(int(cnts.sum())), ha='center', fontsize=20,
                fontweight='bold')
        ax.text(0, -0.14, 'subjects', ha='center', fontsize=9, color='grey')
        labels = [f'{k} mod' for k in cnts.index]
        ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=9, frameon=False)
        ax.set_title('Modality availability', fontsize=11)
    else:
        ax.set_axis_off()
    _panel(ax, 'b')

    # c — AUC vs modality count (band across top-3 models)
    ax = fig.add_subplot(gs[0, 2])
    if loo is not None and bp is not None and res is not None:
        from sklearn.metrics import roc_auc_score
        top3 = res[res['subgroup'] == 'all'].sort_values(
            'macro_auc', ascending=False)['model'].head(3).tolist()
        counts = bp.groupby('patient')['modality'].count()
        loo2 = loo.copy()
        loo2['nm'] = loo2['patient'].map(counts).fillna(1).astype(int)
        korder = sorted(loo2['nm'].unique())
        means, lows, highs, kept = [], [], [], []
        for k in korder:
            mask = loo2['nm'] == k
            if mask.sum() < 5:
                continue
            yk = (loo2.loc[mask, 'y'] == loo2['y'].max()).astype(int)
            if len(np.unique(yk)) < 2:
                continue
            aucs = []
            for m in top3:
                ccol = [c for c in loo2.columns if c.startswith(f'{m}_')]
                if len(ccol) != K:
                    continue
                sc = loo2.loc[mask, ccol[K - 1]].values
                if K > 2:
                    sc = sc - loo2.loc[mask, ccol[0]].values
                aucs.append(roc_auc_score(yk, sc))
            if aucs:
                kept.append(k)
                means.append(np.mean(aucs))
                lows.append(min(aucs))
                highs.append(max(aucs))
        if means:
            ax.fill_between(kept, lows, highs, color=NEG_C, alpha=0.15, lw=0)
            ax.plot(kept, means, 'o-', color=NEG_C, lw=2, ms=6)
            ax.plot(kept, lows, ':', color=NEG_C, lw=1, alpha=0.6)
            ax.plot(kept, highs, ':', color=NEG_C, lw=1, alpha=0.6)
            ax.set_xticks(kept)
            ax.set_xlabel('# available modalities')
            ax.set_ylabel('AUC (top-3 models, mean ± range)')
            ax.set_ylim(0.3, 1.05)
            ax.axhline(0.5, color='grey', ls='--', lw=0.8, alpha=0.5)
            ax.set_title('Performance vs modality availability', fontsize=11)
    sns.despine(ax=ax)
    _panel(ax, 'c')

    fig.suptitle('Ordinal risk continuum, availability and robustness',
                 fontsize=17, fontweight='bold')
    _save(fig, out_dir, 'fig3_continuum')
    return fig




def fig_shap(out_dir):
    """Per-modality SHAP importance bars (publication-ready, 14pt)."""
    bio = _read(out_dir, "biomarkers_top20_per_modality.tsv")
    if bio is None:
        return None
    mods = sorted(bio["modality"].unique())
    ncol = 2
    nrow = int(np.ceil(min(len(mods), 4) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 5.5 * nrow),
                             facecolor="white", constrained_layout=True)
    if nrow * ncol == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)
    for i, m in enumerate(mods[:4]):
        ax = axes[i // ncol, i % ncol]
        top = bio[bio["modality"] == m].sort_values(
            "importance", ascending=False).head(12).iloc[::-1]
        yp = np.arange(len(top))
        ax.barh(yp, top["importance"].values,
                color=[CPAL[i % len(CPAL)]], alpha=0.85,
                edgecolor="white", linewidth=0.5)
        xmax = top["importance"].max() * 1.18
        ax.set_xlim(0, xmax)
        for y, v in zip(yp, top["importance"].values):
            ax.text(v + xmax * 0.012, y, f"{v:.2f}", va="center", ha="left",
                    fontsize=12, fontweight="bold", color="#2d2d2d")
        ax.set_yticks(yp)
        ax.set_yticklabels([str(f)[:26] for f in top["feature"]],
                           fontsize=11.5)
        ax.set_xlabel("Mean |SHAP|", fontsize=14)
        ax.set_title(f"SHAP importance - {m}", fontsize=15,
                     fontweight="bold", pad=10)
        sns.despine(ax=ax)
        _panel(ax, "abcd"[i])
    for j in range(len(mods), nrow * ncol):
        axes[j // ncol, j % ncol].set_axis_off()
    fig.suptitle("SHAP biomarker importance per modality",
                 fontsize=17, fontweight="bold")
    _save(fig, out_dir, "fig4_shap_importance")
    return fig

def make_all_figures(out_dir):
    out_dir = Path(out_dir)
    made = []
    for fn in [fig_benchmark, fig_biomarkers, fig_continuum, fig_shap]:
        try:
            if fn(out_dir) is not None:
                made.append(fn.__name__)
        except Exception as e:
            print(f'  ! {fn.__name__} failed: {type(e).__name__}: {str(e)[:200]}',
                  flush=True)
    print(f'[figures] done: {made}', flush=True)
    return made
