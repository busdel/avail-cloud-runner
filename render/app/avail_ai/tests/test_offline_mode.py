"""Offline-mode tests (AVAIL_OFFLINE=1)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_llm_offline_raises(monkeypatch):
    monkeypatch.setenv("AVAIL_OFFLINE", "1")
    from avail_omics.llm import LLMClient
    with pytest.raises(RuntimeError, match="offline"):
        LLMClient(api_key="dummy").complete("s", "u")


def test_pubmed_offline_empty(monkeypatch):
    monkeypatch.setenv("AVAIL_OFFLINE", "1")
    from avail_omics.pubmed import (fetch_summaries, literature_for,
                                    search_pubmed)
    assert search_pubmed("parkinson") == []
    assert fetch_summaries(["123"]) == []
    assert literature_for("X", "Y") == []


def test_offline_flag_reads_env(monkeypatch):
    from avail_omics._net import offline
    monkeypatch.delenv("AVAIL_OFFLINE", raising=False)
    assert offline() is False
    monkeypatch.setenv("AVAIL_OFFLINE", "true")
    assert offline() is True
