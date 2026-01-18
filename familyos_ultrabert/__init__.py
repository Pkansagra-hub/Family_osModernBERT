"""
FamilyOS UltraBERT v3 - Multi-Task Language Model for Family Communication

A high-performance NLP package providing 13 capabilities for analyzing
family communication: sentiment, emotions, safety, NER, intent, counterfactual
generation, and more.

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

v3 Features:
    - 13 NLP capabilities including counterfactual generation
    - Lazy decoder loading for memory efficiency
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
    - ner_family: Family member entity recognition
    - ner_general: General named entity recognition
    - temporal: Temporal expression extraction
    - relation: Relationship type classification
    - nli: Natural language inference
    - embedding: 768-dim sentence embeddings
    - counterfactual: Generate alternative phrasings (v3, requires decoder)

License: Proprietary - All Rights Reserved
"""

__version__ = "3.0.2"
__author__ = "FamilyOS Team"

from familyos_ultrabert.model import UltraBERT
from familyos_ultrabert.labels import CAPABILITIES, Capability, DECODER_CAPABILITIES
from familyos_ultrabert.client import Client, ClientResult, analyze

# v3: Weight management
from familyos_ultrabert.weights_manager import (
    download_encoder,
    download_decoder,
    get_cache_dir,
    get_cache_size,
    clear_cache,
    is_cached,
    get_weights_info,
)

# v3: Lazy decoder loading
from familyos_ultrabert.decoder_session import DecoderSession

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
    "DECODER_CAPABILITIES",
    # v3: Weight management
    "download_encoder",
    "download_decoder",
    "get_cache_dir",
    "get_cache_size",
    "clear_cache",
    "is_cached",
    "get_weights_info",
    # v3: Lazy decoder loading
    "DecoderSession",
    # Version
    "__version__",
]
