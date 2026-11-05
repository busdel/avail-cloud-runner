#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless agent evaluation #2: grounding audit of advisor reports.

Extracts every PMID cited in advisor reports, verifies (a) the PMID
resolves on PubMed (hallucination check) and (b) the citing sentence
shares content tokens with the fetched title (support heuristic).
"""
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/limbic/AVAIL_GPU")
sys.path.insert(0, str(ROOT / "avail_ai"))
OUT = ROOT / "experiments/results/agent_eval"
OUT.mkdir(parents=True, exist_ok=True)

import pandas as pd  # noqa: E402
from avail_omics.pubmed import fetch_summaries  # noqa: E402

REPORTS = list((ROOT / "uploads").glob("*/avail_results/llm_advisor_report.md")) + \
    [ROOT / "experiments/results/metacardis_results/llm_advisor_report.md"]
REPORTS = [p for p in REPORTS if p.exists()]

PMID_RE = re.compile(r"PMID[:\s]*(\d{4,9})")
STOP = {"with", "from", "that", "this", "have", "study", "patients",
        "using", "based", "results", "which", "these", "their", "data",
        "analysis", "clinical", "machine", "learning", "model", "models"}

def tokens(t):
    return {w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{4,}", t.lower())
            if w not in STOP}

rows = []
for rep in REPORTS:
    text = rep.read_text()
    for m in PMID_RE.finditer(text):
        pmid = m.group(1)
        lo = text.rfind(".", 0, m.start()) + 1
        hi = text.find(".", m.end())
        sent = text[lo:hi if hi > 0 else m.end() + 120].strip()
        rows.append({"report": rep.parent.parent.name if "uploads" in str(rep)
                     else "metacardis", "pmid": pmid, "sentence": sent[:300]})

if not rows:
    print("no PMIDs found"); sys.exit(0)
df = pd.DataFrame(rows).drop_duplicates(["report", "pmid"])
summ = fetch_summaries(sorted(df.pmid.unique().tolist()))
info = {str(s.get("pmid", "")): s for s in summ}
df["resolves"] = df.pmid.map(lambda p: p in info)
df["title"] = df.pmid.map(lambda p: (info.get(p) or {}).get("title", ""))
df["overlap"] = [len(tokens(s) & tokens(t)) for s, t in zip(df.sentence, df.title)]
df["supported_heuristic"] = df.overlap > 0
df.to_csv(OUT / "grounding_audit.tsv", sep="\t", index=False)

n, res, sup = len(df), int(df.resolves.sum()), int(df.supported_heuristic.sum())
print("=== GROUNDING AUDIT ===")
print(f"rapor: {df.report.nunique()} | benzersiz atıf satırı: {n}")
print(f"PMID çözülüyor: {res}/{n} ({res/n:.0%})  [çözülmeyen = halüsinasyon adayı]")
print(f"cümle-başlık örtüşmesi: {sup}/{n} ({sup/n:.0%})")
for _, r in df[~df.resolves].iterrows():
    print(f"  !! {r.report} PMID {r.pmid}: çözülmedi")
for _, r in df[df.resolves & ~df.supported_heuristic].iterrows():
    print(f"  ?? {r.report} PMID {r.pmid}: zayıf örtüşme — '{r.title[:60]}'")
