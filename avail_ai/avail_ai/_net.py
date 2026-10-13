# -*- coding: utf-8 -*-
"""Offline-mode helper.

Setting AVAIL_OFFLINE=1 (or true/yes) hard-disables every outbound network
call the package would otherwise make: LLM providers, PubMed look-ups and
atlas downloads. Nothing leaves the machine; features that need the
network degrade gracefully or raise an explicit error.
"""
import os


def offline():
    return os.environ.get("AVAIL_OFFLINE", "0").strip().lower() in \
        ("1", "true", "yes", "on")
