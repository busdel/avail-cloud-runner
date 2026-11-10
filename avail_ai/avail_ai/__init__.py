# -*- coding: utf-8 -*-
"""AVAIL-omics: Availability-Aware Multimodal Integration & Learning.

A framework for multimodal clinical cohorts with missing modalities:
each modality trains its own base learner on ALL available subjects,
a fixed-length meta-feature vector (base probabilities + availability flags)
is assembled per subject, and a meta-learner makes the joint decision — with
zero sample loss (no subject is dropped for missing modalities).

Install:
    pip install avail-omics          # core (CPU-only)
    pip install "avail-omics[deep]"  # + neural/foundation meta-learners
"""
from .preprocess import (CLRTransformer, NetworkFeatureExtractor, ANOVAFDRSelector,
                         RFImportanceCounter, LassoFeatureSelector,
                         ConfounderResidualizer, preproc, apply_preproc)
from .data import load_labels, load_confounders, load_modalities, load_named_features
from .learners import (macro_auc, base_predictions_loo, build_meta_features,
                       simple_mean_probs, weighted_simple_mean, elasticnet_loo,
                       run_meta_benchmark)
from .pipeline import run_pipeline
from . import figures
from .networks import NETWORK_FAMILIES, abbrev_network, abbrev_brain_label
from .llm import LLMClient, PROVIDERS
from .advisor import Advisor

__version__ = "1.9.2"
__all__ = [
    "run_pipeline", "figures", "LLMClient", "Advisor", "PROVIDERS",
    "NETWORK_FAMILIES", "abbrev_network", "abbrev_brain_label",
    "load_labels", "load_confounders", "load_modalities", "load_named_features",
    "CLRTransformer", "NetworkFeatureExtractor", "ANOVAFDRSelector",
    "RFImportanceCounter", "LassoFeatureSelector", "ConfounderResidualizer",
    "preproc", "apply_preproc", "macro_auc", "base_predictions_loo",
    "build_meta_features", "simple_mean_probs", "weighted_simple_mean",
    "elasticnet_loo", "run_meta_benchmark",
]
