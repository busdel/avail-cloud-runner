# -*- coding: utf-8 -*-
"""Backward-compatibility alias for the renamed ``avail_omics`` package.

Code written against the earlier ``avail_ai`` import name keeps working:
``import avail_ai`` and ``from avail_ai.generic import ...`` resolve to
``avail_omics``.
"""
import sys

import avail_omics as _core
from avail_omics import *  # noqa: F401,F403
from avail_omics import __version__, __all__  # noqa: F401

sys.modules[__name__] = _core

_SUBMODULES = [
    "preprocess", "data", "learners", "pipeline", "figures",
    "figures_publication", "networks", "llm", "advisor", "generic", "radiomics",
    "cli", "imaging", "biomarkers", "vmr", "rag", "pubmed", "meta_learners",
    "robustness",
]
for _sub in _SUBMODULES:
    try:
        _mod = __import__(f"avail_omics.{_sub}", fromlist=["*"])
        sys.modules[f"{__name__}.{_sub}"] = _mod
    except Exception:  # optional dependencies may be missing
        pass
