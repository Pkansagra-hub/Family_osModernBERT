"""
FamilyOS UltraBERT v4 - Multi-Task Language Model for Family Communication

A high-performance NLP package providing 12 capabilities for analyzing
family communication: sentiment, emotions, safety, NER, intent, and more.

Quick Start (Recommended - with auto warmup):
    >>> from familyos_ultrabert import Client
    >>> client = Client()  # Auto warmup for consistent latency
    >>> result = client.analyze("Mom picked up Panda from school")
    >>> print(result.sentiment)      # "very_positive"
    >>> print(result.safety)         # "GREEN"
    >>> print(result.latency_ms)     # 7.5

Direct Model Access:
    >>> from familyos_ultrabert import UltraBERT
    >>> model = UltraBERT.load()
    >>> results = model.analyze("Mom picked up Panda from school")

v4 Features:
    - 12 NLP capabilities with GlobalPointer NER
    - Automatic weight downloading from HuggingFace Hub
    - NPU/GPU/CPU backend auto-detection
    - PyTorch (GPU) and ONNX (CPU/GPU/NPU) backends
    - Single encoder pass for multi-capability inference
    - Optimized for low latency (<10ms on GPU)
    - Auto warmup for consistent first-call latency

Capabilities:
    - sentiment: 5-class sentiment analysis
    - emotions: Multi-label emotion detection (44 emotions)
    - safety_familyos: Family-safe content band (GREEN/AMBER/RED/CRISIS)
    - safety_generic: Multi-label safety categories
    - intent: User intent classification
    - ingress: Message routing/category
    - ner_family: Family member entity recognition (GlobalPointer)
    - ner_general: General named entity recognition (GlobalPointer)
    - temporal: Temporal expression extraction (GlobalPointer)
    - relation: Relationship type classification
    - nli: Natural language inference
    - embedding: 768-dim sentence embeddings
    - relevance: Cross-encoder reranking via MGRH (46M params)

License: Proprietary - All Rights Reserved
"""

__version__ = "4.0.9"
__author__ = "FamilyOS Team"

from familyos_ultrabert.model import UltraBERT
from familyos_ultrabert.labels import CAPABILITIES, Capability
from familyos_ultrabert.client import Client, ClientResult, analyze

# Weight management
from familyos_ultrabert.weights_manager import (
    download_encoder,
    get_cache_dir,
    get_cache_size,
    clear_cache,
    is_cached,
    get_weights_info,
)

# Alias for backward compatibility
FamilyOSModel = UltraBERT

__all__ = [
    # Primary API (v2.0.1+)
    "Client",
    "ClientResult",
    "analyze",
    # Direct model access
    "UltraBERT",
    "FamilyOSModel",  # Backward compatibility
    "CAPABILITIES",
    "Capability",
    # Weight management
    "download_encoder",
    "get_cache_dir",
    "get_cache_size",
    "clear_cache",
    "is_cached",
    "get_weights_info",
    # Version
    "__version__",
]
