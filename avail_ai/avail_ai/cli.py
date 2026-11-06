# -*- coding: utf-8 -*-
"""AVAIL-AI command-line interface."""
import argparse
import getpass
import json
import os
import sys
from pathlib import Path
from .pipeline import run_pipeline
from . import figures
from .llm import LLMClient, PROVIDERS, load_api_key
from .advisor import Advisor


def _add_common(p):
    p.add_argument('--data-root', required=True, help='root of the data package')
    p.add_argument('--out', default='avail_results', help='output directory')


def _cmd_run(args):
    out = run_pipeline(args.data_root, args.out, classes=tuple(args.classes),
                       residualize=args.residualize, use_cache=not args.no_cache,
                       meta_models=tuple(args.meta_models))
    res = out['results']
    print(res.pivot(index='model', columns='subgroup', values='macro_auc').round(3).to_string())
    if args.figures:
        figures.make_all_figures(args.out)
    print(f'Done. Results in {args.out}')
    return 0


def _cmd_advise(args):
    import pandas as pd
    res_path = f'{args.out}/meta_benchmark_results.tsv'
    df = pd.read_csv(res_path, sep='\t')
    llm = LLMClient(api_key=args.api_key, provider=args.provider,
                    base_url=args.base_url, model=args.model)
    advisor = Advisor(llm)
    rec = advisor.recommend_model(df, primary_metric=args.metric, subgroup=args.subgroup)
    print(json.dumps(rec, indent=2, ensure_ascii=False))
    return 0


def _cmd_hyper(args):
    from .advisor import Advisor
    llm = LLMClient(api_key=args.api_key, provider=args.provider,
                    base_url=args.base_url, model=args.model)
    advisor = Advisor(llm)
    space = json.loads(args.space) if args.space else {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 8, 12, 16],
        'C': [0.1, 0.3, 1.0, 3.0],
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    }
    history = json.loads(args.history) if args.history else []
    recs = advisor.suggest_hyperparams(space, history, n=args.n)
    print(json.dumps(recs, indent=2, ensure_ascii=False))
    return 0


def _cmd_run_generic(args):
    from .generic import run_generic_pipeline
    out = run_generic_pipeline(args.data_root, args.out, classes=tuple(args.classes)
                               if args.classes else None)
    res = out["results"]
    print(res.pivot(index="model", columns="subgroup", values="macro_auc")
          .round(3).to_string())
    print(f"Done. Results in {args.out}")
    return 0


def _cmd_figures(args):
    from .figures_publication import make_all_figures
    make_all_figures(args.out)
    print(f'Done. Publication figures written to {args.out}')
    return 0


def _cmd_robustness(args):
    from .robustness import label_permutation_test
    label_permutation_test(args.data_root, args.out,
                           n_perm=args.permutations, jobs=args.jobs,
                           models=tuple(args.models), seed=args.seed)
    return 0


def _cmd_worker(args):
    from .worker import DEFAULT_TOKEN_FILE, DEFAULT_WORKDIR, serve
    serve(Path(args.workdir or DEFAULT_WORKDIR), args.host, args.port,
          Path(args.token_file or DEFAULT_TOKEN_FILE), name=args.name,
          max_parallel=args.max_parallel, mem_gb=args.mem_gb, nice=args.nice)
    return 0


def _cmd_worker_token(args):
    from .worker import DEFAULT_TOKEN_FILE, load_token
    tok = load_token(Path(args.token_file or DEFAULT_TOKEN_FILE))
    print(tok)
    return 0


def _cmd_radiomics(args):
    from . import radiomics as R
    summary = R.extract_dataset(args.data_root, args.out, mode=args.mode,
                                maps=args.maps,
                                afni_bin=(args.afni_bin or None))
    print(f"[radiomics] done: {summary}")
    return 0


def _cmd_sensitivity(args):
    from .robustness import sensitivity_subsets
    sensitivity_subsets(args.data_root, args.out,
                        models=tuple(args.models) if args.models else None)
    return 0


def _cmd_set_key(args):
    key = args.key
    if args.clear:
        key = None
    elif not key:
        if sys.stdin.isatty():
            key = getpass.getpass('API key: ')
        else:
            key = sys.stdin.read().strip()
    path = Path.home() / '.config' / 'avail' / 'llm_key'
    if not key:
        if path.exists():
            path.unlink()
            print(f'API key removed ({path}).')
        else:
            print('No API key was stored.')
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key.strip() + '\n')
    path.chmod(0o600)
    print(f'API key saved to {path} (chmod 600). '
          'The pipeline, web app and CLI will use it automatically.')
    return 0


def _cmd_status(args):
    key = load_api_key()
    if key:
        print(f'API key: configured ({len(key)} chars)')
    else:
        print('API key: NOT configured. Run: avail set-key')
    return 0


def _cmd_report(args):
    from .advisor_report import run as report_run
    return report_run(args.out, args.context)


def _cmd_chat(args):
    from .chat import run as chat_run
    return chat_run(args.out, args.question)


def _cmd_plot(args):
    from . import viz
    import sys as _sys
    out = Path(args.out)
    dest = Path(args.out_dir) if args.out_dir else out / 'viz'
    kinds = [k.strip().lower() for k in args.type.split(',') if k.strip()]
    for kind in kinds:
        try:
            if kind == 'network':
                viz.plot_network(out, data_root=args.data_root,
                                 out_path=dest / 'network',
                                 top_n=args.top_n, threshold=args.threshold,
                                 largest_component=args.largest_component,
                                 method=args.method)
            elif kind in ('roc', 'calibration', 'roc_calibration'):
                viz.plot_roc_calibration(out, out_path=dest / 'roc_calibration',
                                         model=args.model)
            elif kind in ('availability', 'coverage'):
                if not args.data_root:
                    print('[viz] availability needs --data-root; skipped')
                    continue
                viz.plot_availability(args.data_root, out_path=dest / 'availability')
            elif kind == 'umap':
                if not args.data_root:
                    print('[viz] umap needs --data-root; skipped')
                    continue
                viz.plot_umap(args.data_root, out_path=dest / 'umap',
                              group_col=args.group_col, modality=args.modality)
            elif kind == 'volcano':
                viz.plot_volcano(out, out_path=dest / 'volcano',
                                 comparison=args.comparison,
                                 modality=args.modality,
                                 q_thresh=args.q_thresh,
                                 fc_thresh=args.fc_thresh)
            elif kind == 'forest':
                viz.plot_forest(out, out_path=dest / 'forest', table=args.table)
            elif kind in ('network_modules', 'modules'):
                if not args.data_root:
                    print('[viz] network_modules needs --data-root; skipped')
                    continue
                viz.plot_network_modules(
                    args.data_root, out_path=dest / 'network_modules',
                    modality=args.modality, top_n=args.top_n,
                    threshold=args.threshold, max_modules=args.max_modules,
                    taxonomy=args.taxonomy, hub_top=args.hub_top,
                    label_top=args.label_top, method=args.method,
                    combined=args.combined)
            else:
                print(f'[viz] unknown type {kind!r}; skipped')
        except Exception as e:
            print(f'[viz] {kind}: {type(e).__name__}: {str(e)[:250]}')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="avail-omics",
        description='AVAIL-omics: availability-aware multimodal integration & learning.')
    sub = p.add_subparsers(dest='command')

    k = sub.add_parser('set-key', help='save/update the LLM API key (chmod 600)')
    k.add_argument('--key', default=None, help='API key (prompts if omitted)')
    k.add_argument('--clear', action='store_true', help='remove the stored key')
    k.set_defaults(func=_cmd_set_key)

    s = sub.add_parser('status', help='show API key / environment status')
    s.set_defaults(func=_cmd_status)

    r = sub.add_parser('run', help='run the core classification pipeline')
    _add_common(r)
    r.add_argument('--classes', nargs='+', default=['HC', 'MCI', 'PDD'])
    r.add_argument('--meta-models', nargs='+',
                   default=['SimpleMean', 'WSimpleMean', 'ElasticNetLR'])
    r.add_argument('--residualize', action='store_true')
    r.add_argument('--no-cache', action='store_true')
    r.add_argument('--figures', action='store_true', help='generate publication figures')
    r.set_defaults(func=_cmd_run)

    g = sub.add_parser('run-generic',
                       help='full 11-model benchmark on ANY tabular multimodal data '
                            '(metadata.tsv + features/*.tsv)')
    g.add_argument('--data-root', required=True)
    g.add_argument('--out', default='avail_results')
    g.add_argument('--classes', nargs='+', default=None,
                   help='ordered class labels (default: 3 most frequent)')
    g.set_defaults(func=_cmd_run_generic)

    f = sub.add_parser('figures',
                       help='publication-style figures (600dpi PDF/PNG + 300dpi TIFF) '
                            'from an existing results folder')
    f.add_argument('--out', required=True, help='results directory')
    f.set_defaults(func=_cmd_figures)

    rb = sub.add_parser('robustness',
                        help='full-pipeline label-permutation test '
                             '(leakage control, H0: chance)')
    rb.add_argument('--data-root', required=True)
    rb.add_argument('--out', required=True)
    rb.add_argument('--permutations', type=int, default=25)
    rb.add_argument('--jobs', type=int, default=4)
    rb.add_argument('--models', nargs='+',
                    default=['SimpleMean', 'WSimpleMean', 'ElasticNetLR',
                             'MLP'])
    rb.add_argument('--seed', type=int, default=0)
    rb.set_defaults(func=_cmd_robustness)

    se = sub.add_parser('sensitivity',
                        help='modality sensitivity/ablation with the full '
                             'meta-benchmark (full, no_<m>, only_<m>)')
    se.add_argument('--data-root', required=True)
    se.add_argument('--out', required=True)
    se.add_argument('--models', nargs='+', default=None)
    se.set_defaults(func=_cmd_sensitivity)

    rd = sub.add_parser(
        'radiomics',
        help='optional radiomics features: T1 (whole-brain + 17 networks) '
             'and/or functional maps ALFF/fALFF/ReHo (needs AFNI); writes '
             'radiomics.tsv in the standard modality layout')
    rd.add_argument('--data-root', required=True)
    rd.add_argument('--out', required=True)
    rd.add_argument('--mode', choices=['wb', 'both'], default='wb')
    rd.add_argument('--maps', default='',
                    help='comma list: alff,falff,reho (requires '
                         'bold_clean.nii.gz and AFNI)')
    rd.add_argument('--afni-bin', default=os.environ.get('AFNI_BIN', ''))
    rd.set_defaults(func=_cmd_radiomics)

    w = sub.add_parser(
        'worker',
        help='run a secure job server on YOUR machine (bearer token, '
             '127.0.0.1 by default; see WORKER.md)')
    w.add_argument('--workdir', default=None,
                   help='job sandbox root (default: ~/.local/share/avail-worker)')
    w.add_argument('--host', default='127.0.0.1')
    w.add_argument('--port', type=int, default=8787)
    w.add_argument('--token-file', default=None)
    w.add_argument('--name', default='my-gpu')
    w.add_argument('--max-parallel', type=int, default=1)
    w.add_argument('--mem-gb', type=float, default=None)
    w.add_argument('--nice', type=int, default=10)
    w.set_defaults(func=_cmd_worker)

    wt = sub.add_parser('worker-token',
                        help='print (creating if needed) the worker bearer token')
    wt.add_argument('--token-file', default=None)
    wt.set_defaults(func=_cmd_worker_token)

    a = sub.add_parser('advise', help='LLM-driven model recommendation')
    _add_common(a)
    a.add_argument('--metric', default='macro_auc')
    a.add_argument('--subgroup', default='all')
    a.add_argument('--provider', default='gemini', choices=list(PROVIDERS))
    a.add_argument('--api-key', default=None)
    a.add_argument('--base-url', default=None)
    a.add_argument('--model', default=None)
    a.set_defaults(func=_cmd_advise)

    h = sub.add_parser('hyper', help='LLM-driven hyperparameter suggestions')
    h.add_argument('--space', default=None, help='JSON dict {param: [choices]}')
    h.add_argument('--history', default=None, help='JSON list [[params, metric], ...]')
    h.add_argument('--n', type=int, default=1)
    h.add_argument('--provider', default='gemini', choices=list(PROVIDERS))
    h.add_argument('--api-key', default=None)
    h.add_argument('--base-url', default=None)
    h.add_argument('--model', default=None)
    h.set_defaults(func=_cmd_hyper)

    rp = sub.add_parser('report',
                        help='AI advisor report (llm_advisor.md/json) for an '
                             'existing results directory; needs an LLM key')
    rp.add_argument('--out', required=True, help='results directory')
    rp.add_argument('--context', default='Parkinson disease',
                    help='disease/context for literature queries')
    rp.set_defaults(func=_cmd_report)

    ch = sub.add_parser('chat', aliases=['agent'],
                        help='results-grounded Dr. AVAIL chat agent '
                             '(needs an LLM key)')
    ch.add_argument('--out', required=True, help='results directory')
    ch.add_argument('--question', default=None,
                    help='ask one question and exit')
    ch.set_defaults(func=_cmd_chat)

    pl = sub.add_parser('plot', help='soft-palette figures: network, '
                                    'ROC/calibration, availability')
    pl.add_argument('--out', required=True, help='results directory')
    pl.add_argument('--data-root', default=None,
                    help='generic data root (metadata + features/) for network/availability')
    pl.add_argument('--out-dir', default=None,
                    help='output directory (default: <out>/viz)')
    pl.add_argument('--type', default='network,roc,availability,umap,volcano,forest',
                    help='comma list: network,network_modules,roc,availability,umap,volcano,forest')
    pl.add_argument('--model', default=None, help='model for ROC/calibration')
    pl.add_argument('--group-col', default=None, help='metadata group column')
    pl.add_argument('--modality', default=None, help='restrict UMAP/volcano to one modality')
    pl.add_argument('--comparison', default=None, help='volcano comparison')
    pl.add_argument('--q-thresh', type=float, default=0.05)
    pl.add_argument('--fc-thresh', type=float, default=1.0)
    pl.add_argument('--table', default=None, help='forest table (effect + CI)')
    pl.add_argument('--max-modules', type=int, default=6)
    pl.add_argument('--taxonomy', default=None, help='TSV mapping feature -> group/taxon')
    pl.add_argument('--hub-top', type=int, default=2)
    pl.add_argument('--label-top', type=int, default=6)
    pl.add_argument('--top-n', type=int, default=24)
    pl.add_argument('--threshold', type=float, default=0.30)
    pl.add_argument('--largest-component', action='store_true',
                    help='network: draw only the largest connected component')
    pl.add_argument('--method', default='sparcc',
                    choices=['sparcc', 'spearman', 'pearson'],
                    help='correlation method (sparcc = CLR approximation)')
    pl.add_argument('--combined', action='store_true',
                    help='network_modules: single combined network with cross-module edges')
    pl.set_defaults(func=_cmd_plot)

    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
