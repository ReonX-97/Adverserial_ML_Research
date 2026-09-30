"""
GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs

A training-free, inference-time framework for analyzing safety alignment
in MoE LLMs. Implements gate-level profiling, expert-level localization,
and targeted safety removal as described in arXiv:2512.21008v2.

This implementation is for academic research purposes only.
"""

from gatebreaker.config import GateBreakerConfig
from gatebreaker.datasets import (
    PromptDataset,
    load_harmful_prompts,
    load_benign_prompts,
    get_profiling_datasets,
    mask_non_content_tokens,
)
from gatebreaker.model_patcher import MoEPatcher
from gatebreaker.gate_profiler import GateProfiler
from gatebreaker.expert_localizer import ExpertLocalizer
from gatebreaker.safety_remover import SafetyRemover
from gatebreaker.metrics import GateBreakerEvaluator, compute_asr

__all__ = [
    "GateBreakerConfig",
    "PromptDataset",
    "load_harmful_prompts",
    "load_benign_prompts",
    "get_profiling_datasets",
    "mask_non_content_tokens",
    "MoEPatcher",
    "GateProfiler",
    "ExpertLocalizer",
    "SafetyRemover",
    "GateBreakerEvaluator",
    "compute_asr",
]
