# -*- coding: utf-8 -*-
"""Lightweight RAG for the web "Ask the AI" feature.

Retrieval: TF-IDF cosine over a per-job corpus built from the result files
(benchmark tables, per-class AUC, LLM advisor report, biomarker literature
table, predictions summary) plus a global corpus (methods notes, external
validation results).

Enrichment: the question is scanned for biomarker names present in the
corpus; matching biomarkers get live PubMed evidence (abstracts via
efetch). A Semantic Scholar query (free, no key) adds recent literature.

Everything is in-process - no vector DB needed.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .pubmed import literature_for, fetch_summaries


def _chunk_text(tag, text, max_len=1600):
    text = text.strip()
    if not text:
        return []
    return [{"tag": tag, "text": text[i:i + max_len]}
            for i in range(0, len(text), max_len)]


def build_job_corpus(job_dir):
    """Chunks from a job's results folder."""
    chunks = []
    res = Path(job_dir) / "avail_results"
    if res.exists():
        for f in ["meta_benchmark_results.tsv", "meta_benchmark_3class_results.tsv"]:
            p = res / f
            if p.exists():
                try:
                    df = pd.read_csv(p, sep="\t")
                    chunks += _chunk_text("benchmark", df.head(40).to_csv(index=False))
                except Exception:
                    pass
        p = res / "per_class_auc.tsv"
        if p.exists():
            try:
                chunks += _chunk_text("per_class_auc",
                                      pd.read_csv(p, sep="\t").head(30).to_csv(index=False))
            except Exception:
                pass
        p = res / "llm_advisor_report.md"
        if p.exists():
            chunks += _chunk_text("advisor_report", p.read_text()[:8000])
        p = res / "biomarkers_literature.tsv"
        if p.exists():
            try:
                lit = pd.read_csv(p, sep="\t")
                lit_txt = "\n".join(
                    f"{r.biomarker} ({r.category}): {r.n_pubmed} PubMed hits"
                    for r in lit.itertuples())
                chunks += _chunk_text("biomarker_literature", lit_txt)
            except Exception:
                pass
        p = res / "tuning_results.tsv"
        if p.exists():
            try:
                chunks += _chunk_text("tuning",
                                      pd.read_csv(p, sep="\t").to_csv(index=False))
            except Exception:
                pass
    return chunks


GLOBAL_FILES = [
    "avail_results/biomarkers_literature.tsv",
    "avail_results/llm_advisor_report.md",
    "experiments/METHODS_NOTES.md",
    "experiments/results/imputation_benchmark_v4_results.tsv",
    "experiments/results/missingness_recovery.tsv",
    "experiments/results/unimix_ad_results.tsv",
    "experiments/results/transfer_wallen_results.tsv",
    "experiments/external_wallen/results/meta_benchmark_results.tsv",
]


def build_global_corpus(base):
    chunks = []
    for f in GLOBAL_FILES:
        p = base / f
        if not p.exists():
            continue
        try:
            if p.suffix == ".md":
                chunks += _chunk_text(f, p.read_text()[:9000])
            else:
                df = pd.read_csv(p, sep="\t")
                chunks += _chunk_text(f, df.head(30).to_csv(index=False))
        except Exception:
            pass
    return chunks


class TfidfRetriever:
    def __init__(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        if not self.texts:
            self.vec = None
            return
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(self.texts)

    def retrieve(self, question, k=4):
        if self.vec is None:
            return []
        q = self.vec.transform([question])
        sims = np.asarray((self.matrix @ q.T).toarray()).ravel()
        idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in idx if sims[i] > 0]


def _s2_search(query, n=3):
    """Semantic Scholar (free, no key): titles + abstracts."""
    url = ("https://api.semanticscholar.org/graph/v1/paper/search"
           "?query=" + urllib.parse.quote(query) + f"&limit={n}"
           "&fields=title,abstract,year,externalIds")
    req = urllib.request.Request(url, headers={"User-Agent": "avail-omics/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        out = []
        for p in d.get("data", []):
            out.append({
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "abstract": (p.get("abstract") or "")[:600],
                "doi": (p.get("externalIds") or {}).get("DOI", ""),
            })
        return out
    except Exception:
        return []


def _europepmc_search(query, n=3):
    """EuropePMC (free, no key): titles + plain abstracts."""
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" +
           urllib.parse.quote(query) + f"&resultType=core&format=json&pageSize={n}")
    req = urllib.request.Request(url, headers={"User-Agent": "avail-omics/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        out = []
        for p in d.get("resultList", {}).get("result", []):
            out.append({
                "title": p.get("title", ""),
                "year": p.get("pubYear", ""),
                "abstract": (p.get("abstractText") or "")[:600],
                "doi": p.get("doi", "") or p.get("pmid", ""),
            })
        return out
    except Exception:
        return []


def _openalex_abstract(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def _openalex_search(query, n=3):
    """OpenAlex (free, no key): titles + reconstructed abstracts."""
    url = ("https://api.openalex.org/works?search=" + urllib.parse.quote(query)
           + f"&per-page={n}&mailto=avail@example.org")
    req = urllib.request.Request(url, headers={"User-Agent": "avail-omics/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read().decode())
        out = []
        for p in d.get("results", []):
            out.append({
                "title": p.get("title", ""),
                "year": p.get("publication_year", ""),
                "abstract": _openalex_abstract(
                    p.get("abstract_inverted_index"))[:600],
                "doi": p.get("doi", "") or "",
            })
        return out
    except Exception:
        return []


def multi_db_literature(biomarker, context, n=1):
    from ._net import offline
    if offline():
        return []
    """One hit per free database (EuropePMC, OpenAlex, Semantic Scholar)."""
    q = f'"{biomarker}" {context}'
    out = []
    for name, fn in [("EuropePMC", _europepmc_search),
                     ("OpenAlex", _openalex_search),
                     ("Semantic Scholar", _s2_search)]:
        try:
            for h in fn(q, n=n)[:n]:
                out.append({"source": name, **h})
        except Exception:
            pass
        time.sleep(0.3)
    return out


def enrich_question(question, corpus, n_bio=2, n_s2=2):
    """Live evidence: PubMed for corpus biomarkers in the question + S2."""
    evidence = []
    bio_names = set()
    for c in corpus:
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s[A-Za-z]+)?)\b", c["text"]):
            w = m.group(1)
            if len(w) > 3:
                bio_names.add(w)
    hits = []
    for name in sorted(bio_names):
        if re.search(re.escape(name), question, re.IGNORECASE):
            arts = literature_for(name, "Parkinson disease", n=n_bio)
            for a in arts:
                hits.append(f"[PubMed] {a['title']} ({a['journal']}, {a['year']}, "
                            f"PMID {a['pmid']})")
            if len(hits) >= 4:
                break
    if hits:
        evidence.append("PubMed evidence: " + " | ".join(hits[:4]))
    time.sleep(0.4)
    for name, fn in [("EuropePMC", _europepmc_search),
                     ("OpenAlex", _openalex_search),
                     ("Semantic Scholar", _s2_search)]:
        try:
            res = fn(question, n=2)
            if res:
                evidence.append(
                    f"{name}: " + " | ".join(
                        f"{p['title']} ({p['year']}) {p['abstract'][:250]}"
                        for p in res))
        except Exception:
            pass
        time.sleep(0.3)
    return evidence


def build_context(question, job_dir, base):
    """Returns (context_text, sources) where sources lists the provenance
    of every part of the context (auditability)."""
    job_chunks = build_job_corpus(job_dir)
    global_chunks = build_global_corpus(base)
    ret = TfidfRetriever(job_chunks + global_chunks)
    top = ret.retrieve(question, k=5)
    ctx_parts = []
    sources = []
    for c in top:
        ctx_parts.append(f"[{c['tag']}]\n{c['text']}")
        if c["tag"] not in sources:
            sources.append(c["tag"])
    for ev in enrich_question(question, job_chunks + global_chunks):
        ctx_parts.append(ev)
        label = ev.split(":", 1)[0].strip()
        sources.append(label)
    return "\n\n".join(ctx_parts)[:9000], sources
