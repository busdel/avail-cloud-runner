# -*- coding: utf-8 -*-
"""AVAIL-AI: Availability-Aware Multimodal Integration & Learning.

A framework for multimodal clinical cohorts with missing modalities:
each modality trains its own base learner on ALL available subjects,
a fixed-length meta-feature vector (base probabilities + availability flags)
is assembled per subject, and a meta-learner makes the joint decision — with
zero sample loss (no subject is dropped for missing modalities).

Reference: SPARTA / AVAIL-AI framework for the oral-gut-brain axis in
Parkinson's cognitive trajectory (HC vs MCI vs PDD).
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

__version__ = "0.1.0"
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
