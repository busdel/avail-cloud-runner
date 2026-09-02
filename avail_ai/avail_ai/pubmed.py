# -*- coding: utf-8 -*-
"""Free PubMed (NCBI E-utilities) literature lookup — no API key required.

Rate limits: ~3 req/s without a key, 10 req/s with one. We are conservative
(0.4 s between requests) and cache results on disk.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CACHE_DIR = Path.home() / ".cache" / "avail" / "pubmed"
_SLEEP = 0.4
_TIMEOUT = 20


def _get(path, params):
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "avail-omics/0.1 (research tool)"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode())


def search_pubmed(query, retmax=5, use_cache=True):
    """Return top PMIDs for a PubMed query."""
    key = "q_" + query.replace(" ", "_")[:200]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{key}.json"
    if use_cache and cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    try:
        time.sleep(_SLEEP)
        d = _get("esearch.fcgi", {
            "db": "pubmed", "term": query, "retmode": "json",
            "retmax": retmax, "sort": "relevance"})
        ids = d.get("esearchresult", {}).get("idlist", [])
        if use_cache:
            cf.write_text(json.dumps(ids))
        return ids
    except Exception:
        return []


def fetch_summaries(pmids, use_cache=True):
    """Return article dicts (title, journal, year, pmid, authors) for PMIDs."""
    if not pmids:
        return []
    key = "s_" + "_".join(pmids[:20])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{key}.json"
    if use_cache and cf.exists():
        try:
            return json.loads(cf.read_text())
        except Exception:
            pass
    out = []
    try:
        time.sleep(_SLEEP)
        d = _get("esummary.fcgi", {
            "db": "pubmed", "id": ",".join(pmids), "retmode": "json"})
        for uid, it in d.get("result", {}).items():
            if uid == "uids":
                continue
            authors = [a["name"] for a in it.get("authors", [])][:3]
            out.append({
                "pmid": uid,
                "title": it.get("title", "").rstrip("."),
                "journal": it.get("fulljournalname") or it.get("source", ""),
                "year": (it.get("pubdate") or "")[:4],
                "authors": ", ".join(authors) + (" et al." if len(authors) == 3 else ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
            })
    except Exception:
        pass
    if use_cache and out:
        cf.write_text(json.dumps(out))
    return out


def literature_for(biomarker, context, n=3, use_cache=True):
    """PubMed evidence for a biomarker in a disease/context. Returns list of dicts."""
    if not biomarker or not context:
        return []
    q = f'"{biomarker}" AND {context}'
    pmids = search_pubmed(q, retmax=n, use_cache=use_cache)
    return fetch_summaries(pmids, use_cache=use_cache)
