#!/usr/bin/env python3
"""
AVAIL-AI LLM advisor step (optional, automatic).

Reads the numeric outputs in OUT and produces a detailed LLM-driven report:
  - model recommendation per subgroup (JSON reasoning)
  - biomarker discovery summary WITH PubMed literature evidence (free,
    no API key) and an LLM synthesis of biomarker plausibility
  - a manuscript-ready interpretation paragraph

API key resolution (never printed/logged):
  1. env AVAIL_LLM_API_KEY
  2. file ~/.config/avail/llm_key (0600)

Usage:
    python run_llm_advisor.py --out ./avail_results [--context "Parkinson disease"]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


import re as _re

_NET_PHRASES = [
    ("Salience Ventral Attention", "salience ventral attention network"),
    ("Default", "default mode network"),
    ("Dorsal Attention", "dorsal attention network"),
    ("Somatomotor", "sensorimotor network"),
    ("Visual Central", "visual network"),
    ("Limbic", "limbic network"),
    ("Control", "frontoparietal network"),
    ("Temporoparietal", "temporoparietal network"),
]


def _literature_queries(feature):
    """Direct feature, then its components (network pairs split on __/-)."""
    f = str(feature)
    for pre in ("full_", "part_", "wb_", "net_"):
        if f.startswith(pre):
            f = f[len(pre):]
    f = f.replace("_between_", "__").replace("_within_", "__")
    parts = [p.strip() for p in _re.split(r"__|--|-", f) if p.strip()]
    comps = []
    for part in parts:
        part = _re.sub(r"^(between|within)_", "", part, flags=_re.I)
        pname = _re.sub(r"\s+[A-Z]$", "", part)
        for long, phrase in _NET_PHRASES:
            if pname.startswith(long):
                comps.append(phrase)
                break
        comps.append(pname)
    queries, seen = [], set()
    for cand in [str(feature)] + comps:
        k = cand.lower()
        if cand and k not in seen:
            seen.add(k)
            queries.append(cand)
    return queries

MAX_BIOMARKERS = 12


def run(out, context='Parkinson disease'):
    OUT = Path(out).resolve()

    api_key = os.environ.get('AVAIL_LLM_API_KEY')
    if not api_key:
        kf = Path.home() / '.config/avail/llm_key'
        if kf.exists():
            api_key = kf.read_text().strip()

    from .llm import MultiLLMClient, build_clients
    from .advisor import Advisor

    llm = None
    if api_key:
        llm = MultiLLMClient(build_clients())
        adv = Advisor(llm)
    else:
        print('[llm] no API key -> plain report without AI sections', flush=True)

    res_path = None
    for cand in ['meta_benchmark_3class_results.tsv', 'meta_benchmark_results.tsv']:
        p = OUT / cand
        if p.exists():
            res_path = p
            break
    if res_path is None:
        print('[llm] missing meta_benchmark results -> skipped', flush=True)
        return
    res = pd.read_csv(res_path, sep='\t')

    report = {}

    # ---------- 1) model recommendation ----------
    subs = [s for s in ['all', 'complete5', 'complete', 'ge3mod', 'le2mod']
            if (res['subgroup'] == s).any()] or ['all']
    if llm:
        import time as _t
        for sub in subs:
            try:
                rec = adv.recommend_model(res, subgroup=sub)
                report[f'recommend_{sub}'] = rec
                print(f'[llm] recommend({sub}) -> {rec.get("recommended_model")}',
                      flush=True)
            except Exception as e:
                report[f'recommend_{sub}'] = {'error': str(e)}
                print(f'[llm] recommend({sub}) FAILED: {str(e)[:200]}', flush=True)
            _t.sleep(2)

    # ---------- 2) biomarker discovery + PubMed literature ----------
    bio_path = OUT / 'biomarkers_top20_per_modality.tsv'
    lit_rows = []
    bio_summary = ''
    if bio_path.exists():
        bio = pd.read_csv(bio_path, sep='\t')

        def _category(mod):
            m = str(mod).lower()
            if 'fmri' in m or 'brain' in m:
                return 'fMRI functional network'
            if '16s' in m and 'stool' in m:
                return 'stool 16S genus'
            if '16s' in m and 'saliva' in m:
                return 'saliva 16S genus'
            if 'stool' in m and 'sg' in m:
                return 'stool shotgun MGS species'
            if 'saliva' in m and 'sg' in m:
                return 'saliva shotgun MGS species'
            if 'kegg' in m:
                return 'KEGG pathway'
            return str(mod)

        picks = []
        for mname, grp in bio.groupby('modality'):
            top = grp.sort_values('importance', ascending=False).head(4)
            picks.extend(top[['modality', 'feature', 'importance']]
                         .to_dict('records'))
        picks = sorted(picks, key=lambda r: -r['importance'])[:MAX_BIOMARKERS]

        from . import _net as _netmod
        if _netmod.offline():
            print("[pubmed] offline mode: literature lookups disabled", flush=True)
            picks_offline = True
        else:
            picks_offline = False
        from .pubmed import literature_for
        from .rag import multi_db_literature
        for pk in ([] if picks_offline else picks):
            feature = str(pk['feature'])
            articles, extra, used_q = [], [], feature
            for qq in _literature_queries(feature):
                articles = literature_for(qq, context, n=2)
                extra = multi_db_literature(qq, context, n=1)
                if articles or extra:
                    used_q = qq
                    break
            row = {'category': _category(pk['modality']),
                   'modality': pk['modality'], 'biomarker': feature,
                   'query_used': used_q,
                   'importance': round(pk['importance'], 4),
                   'n_pubmed': len(articles), 'articles': articles,
                   'extra': extra}
            lit_rows.append(row)
            suffix = "" if used_q == feature else f" [via '{used_q}']"
            print(f"[pubmed] [{_category(pk['modality'])}] {feature}: "
                  f"{len(articles)} hits (+{len(extra)} other databases)"
                  + suffix, flush=True)
            for a in articles:
                print(f'          - {a["year"]} {a["title"][:80]}', flush=True)
        lit_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'articles'}
                               for r in lit_rows])
        lit_df.to_csv(OUT / 'biomarkers_literature.tsv', sep='\t', index=False)

        bio_summary = lit_df.to_csv(index=False)
        if llm and not lit_df.empty:
            by_cat = {}
            for r in lit_rows:
                by_cat.setdefault(r['category'], []).append(
                    {'biomarker': r['biomarker'],
                     'articles': [a['title'] for a in r['articles']],
                     'extra': [(e['source'], e['title']) for e in r['extra']]})
            articles_txt = json.dumps(by_cat, ensure_ascii=False)
            try:
                synth = llm.complete(
                    "You are a biomarker expert for clinical multi-omics papers. "
                    "Only cite the provided PubMed records; never invent references.",
                    f"Top biomarkers grouped by functional category "
                    f"(context: {context}):\n{articles_txt}\n\n"
                    "Write one short paragraph per category assessing whether "
                    "these biomarkers are biologically plausible and supported by "
                    "the literature. Be honest when evidence is weak.")
                report['biomarker_synthesis'] = synth.strip()
                print('[llm] biomarker synthesis done', flush=True)
            except Exception as e:
                report['biomarker_synthesis'] = f'error: {str(e)[:200]}'

    # ---------- 3) interpretation ----------
    top = res[res.subgroup == 'all'].sort_values('macro_auc', ascending=False).head(5)
    summary = 'Meta-learner benchmark (macro-AUC, all subjects):\n' + \
        top[['model', 'macro_auc']].to_csv(index=False)
    for name in ['per_class_auc.tsv', 'ordinal_continuum.tsv', 'dropout_recovery.tsv']:
        p = OUT / name
        if p.exists():
            summary += f'\n{name}:\n' + p.read_text()[:4000]
    if llm:
        try:
            report['interpretation'] = adv.interpret(summary)
            print('[llm] interpretation done', flush=True)
        except Exception as e:
            report['interpretation'] = f'error: {str(e)[:200]}'
            print(f'[llm] interpretation FAILED: {str(e)[:200]}', flush=True)

    # ---------- write outputs ----------
    report['_ai_meta'] = {
        'providers': [getattr(c, 'provider', None) or getattr(c, 'model', '?')
                      for c in llm.clients] if llm else [],
        'grounding_sources': ['PubMed', 'EuropePMC', 'OpenAlex',
                              'Semantic Scholar'],
        'note': 'AI-generated; verify before publication.',
    }
    (OUT / 'llm_advisor.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False))

    md = ['# AVAIL-AI AI Advisor Report', '',
          '> Generated by an AI assistant grounded in the retrieved evidence '
          '(PubMed / EuropePMC / OpenAlex / Semantic Scholar). Review and '
          'verify all statements and citations before publication use.', '']
    for sub in subs:
        rec = report.get(f'recommend_{sub}', {})
        if isinstance(rec, dict) and 'recommended_model' in rec:
            md += [f'## Recommendation — subgroup: {sub}', '',
                   f'- **Model:** {rec.get("recommended_model")}',
                   f'- **Runner-up:** {rec.get("runner_up")}',
                   f'- **Rationale:** {rec.get("rationale")}',
                   f'- **Caveat:** {rec.get("caveat")}', '']
    fig_names = [f for f in ['fig1_meta_benchmark.png', 'fig2_biomarkers.png',
                             'fig3_continuum.png'] if (OUT / f).exists()]
    if fig_names:
        md += ['## Publication figures', '',
               'The manuscript-style figures generated for this run: ' +
               ', '.join(f'`{f}`' for f in fig_names) + '.', '']
    if bio_summary:
        md += ['## Biomarker discovery + PubMed literature', '',
               f'(context: {context}; full evidence table: '
               f'`biomarkers_literature.tsv`)', '']

        def _cite(a):
            first = a.get('authors', '').split(',')[0] or 'et al'
            return (f"{first} et al., *{a['journal']}*, {a['year']}; "
                    f"[PMID {a['pmid']}]({a['url']})")

        by_cat = {}
        for r in lit_rows:
            by_cat.setdefault(r['category'], []).append(r)
        for cat, rows in by_cat.items():
            md.append(f"### {cat}")
            for r in rows:
                if r['articles']:
                    cites = "; ".join(_cite(a) for a in r['articles'])
                    md.append(f"- **{r['biomarker']}** has been previously "
                              f"reported in {context} "
                              f"({cites}).")
                else:
                    md.append(f"- **{r['biomarker']}**: no direct PubMed record "
                              f"was found for this feature; it may be a "
                              f"data-driven or novel candidate.")
                if r.get('extra'):
                    md.append("  - Additional database evidence: " + "; ".join(
                        f"{e['title']} ({e['source']}, {e['year']})"
                        + (f", doi:{e['doi']}" if e.get('doi') else "")
                        for e in r['extra']))
            md.append('')
        if report.get('biomarker_synthesis'):
            md += ['### AI synthesis of biomarker plausibility', '',
                   report['biomarker_synthesis'], '']
    md += ['## Manuscript interpretation', '',
           report.get('interpretation', '') or '(no AI key configured)']
    (OUT / 'llm_advisor_report.md').write_text('\n'.join(md))
    print('[llm] wrote llm_advisor_report.md + llm_advisor.json', flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='directory with pipeline TSVs')
    ap.add_argument('--context', default='Parkinson disease',
                    help='disease/context for PubMed biomarker queries')
    args = ap.parse_args(argv)
    return run(args.out, args.context)


if __name__ == '__main__':
    sys.exit(main())
