"""
Inference Module for FamilyOS Unified Encoder

This module provides high-level inference APIs for the unified multi-task model.

Components:
    - unified_output: UnifiedNLPOutput dataclass and sys_nlp_infer() function

Usage:
    from modeling_studio.inference import UnifiedNLPOutput, sys_nlp_infer

    outputs = sys_nlp_infer(
        texts=["Mom took Panda to the park"],
        capabilities=["ner_family", "sentiment", "safety_familyos"],
    )
"""

from modeling_studio.inference.unified_output import (
    Entity,
    Relation,
    UnifiedNLPOutput,
    get_unified_model,
    sys_nlp_infer,
)

__all__ = [
    "Entity",
    "Relation",
    "UnifiedNLPOutput",
    "get_unified_model",
    "sys_nlp_infer",
]
