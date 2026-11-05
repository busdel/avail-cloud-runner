#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AVAIL-AI generic pipeline (thin wrapper over avail_ai.generic).

    python run_generic.py --data-root mydata --out results
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "avail_ai"))

from avail_ai.generic import run_generic_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=None)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip the LLM advisor step")
    args = ap.parse_args()

    OUT = Path(args.out).resolve()
    out = run_generic_pipeline(args.data_root, OUT, classes=args.classes)

    if not args.no_llm:
        llm_runner = ROOT / "run_llm_advisor.py"
        if llm_runner.exists():
            subprocess.run([sys.executable, str(llm_runner), "--out", str(OUT)],
                           check=False)

    print(f"[generic] DONE. Outputs: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
