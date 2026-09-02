# -*- coding: utf-8 -*-
"""AVAIL-AI command-line interface."""
import argparse
import getpass
import json
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


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='avail',
        description='AVAIL-AI: availability-aware multimodal integration & learning.')
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

    args = p.parse_args(argv)
    if not args.command:
        p.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
