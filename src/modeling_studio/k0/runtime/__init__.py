"""K0 runtime module for FamilyOS integration."""

from modeling_studio.k0.runtime.model_registry import (
    CAPABILITY_ALIASES,
    HEAD_REGISTRY,
    LEGACY_MODEL_MAPPING,
    MODEL_REGISTRY,
    Capability,
    HeadInfo,
    ModelInfo,
    clear_cache,
    get_capability_for_module,
    get_head_info,
    get_model_info,
    get_tokenizer,
    get_unified_model,
    list_capabilities,
    list_heads,
    list_models,
    migrate_legacy_model,
    register_capability_alias,
    register_head,
    register_model,
    resolve_capability,
)

__all__ = [
    # Registries
    "MODEL_REGISTRY",
    "HEAD_REGISTRY",
    "CAPABILITY_ALIASES",
    "LEGACY_MODEL_MAPPING",
    # Enums and dataclasses
    "Capability",
    "ModelInfo",
    "HeadInfo",
    # Core functions
    "resolve_capability",
    "get_model_info",
    "get_head_info",
    "get_unified_model",
    "get_tokenizer",
    "list_capabilities",
    "list_models",
    "list_heads",
    # Migration helpers
    "migrate_legacy_model",
    "get_capability_for_module",
    # Utilities
    "clear_cache",
    "register_model",
    "register_head",
    "register_capability_alias",
]
