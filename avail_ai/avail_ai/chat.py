# -*- coding: utf-8 -*-
"""Results-grounded chat agent for AVAIL outputs.

This is a lightweight, package-level counterpart of the web Dr. AVAIL agent:
it reads the numeric outputs of a completed run and answers questions with a
user-supplied LLM key. It does not start jobs or execute code.
"""
import json
from pathlib import Path

from .llm import MultiLLMClient, build_clients, load_api_key

SYSTEM = """You are Dr. AVAIL, a careful research assistant for the
availability-aware multi-omics pipeline. Answer questions using ONLY the
provided run outputs. Never invent numbers, model names, files or citations.
If something is absent, say so. Keep answers concise, use plain text with at
most **bold** and `code`, and always add a one-line caveat that the results
are research outputs, not clinical advice. Answer in English."""


def _read(path, limit=6000):
    try:
        return Path(path).read_text(errors="replace")[:limit]
    except Exception:
        return ""


def build_context(out):
    out = Path(out)
    parts = []
    v = out / "validation.json"
    if v.exists():
        parts.append("VALIDATION\n" + _read(v, 3000))
    for name in ("meta_benchmark_3class_results.tsv",
                 "meta_benchmark_results.tsv"):
        p = out / name
        if p.exists():
            parts.append(f"BENCHMARK ({name})\n" + _read(p, 7000))
            break
    b = out / "biomarkers_top20_per_modality.tsv"
    if b.exists():
        parts.append("TOP BIOMARKERS\n" + _read(b, 6000))
    rep = out / "llm_advisor_report.md"
    if rep.exists():
        parts.append("EXISTING AI REPORT\n" + _read(rep, 6000))
    return "\n\n".join(parts) or "(no result files found in this folder)"


def answer(out, question):
    if not load_api_key():
        raise RuntimeError(
            "No LLM API key found. Run `avail-omics set-key` or set "
            "AVAIL_LLM_API_KEY.")
    llm = MultiLLMClient(build_clients())
    ctx = build_context(out)
    prompt = f"Run outputs:\n{ctx}\n\nQuestion: {question}"
    return llm.complete(SYSTEM, prompt, max_tokens=1200).strip()


def run(out, question=None):
    if not load_api_key():
        print("No LLM API key found. Run `avail-omics set-key` or set "
              "AVAIL_LLM_API_KEY; the core pipeline still works without it.")
        return 2
    if question:
        try:
            print(answer(out, question))
            return 0
        except Exception as e:
            print(f"[agent] error: {type(e).__name__}: {str(e)[:300]}")
            return 2
    print("Dr. AVAIL results agent — type a question, or 'exit'.")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in ("exit", "quit"):
            break
        try:
            print(answer(out, q))
        except Exception as e:
            print(f"[agent] error: {type(e).__name__}: {str(e)[:300]}")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="avail-omics chat")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--question", default=None,
                    help="ask one question and exit (non-interactive)")
    args = ap.parse_args(argv)
    return run(args.out, args.question)


if __name__ == "__main__":
    import sys
    sys.exit(main())
