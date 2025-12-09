"""
FamilyOS UltraBERT v2 - Multi-Task Language Model for Family Communication

A high-performance NLP package providing 12 capabilities for analyzing
family communication: sentiment, emotions, safety, NER, intent, and more.

Quick Start:
    >>> from familyos_ultrabert import UltraBERT
    >>> model = UltraBERT.load()
    >>> results = model.analyze("Mom picked up Panda from school")
    >>> print(results["sentiment"])
    {'prediction': 'positive', 'confidence': 0.92}

Features:
    - 12 NLP capabilities in a single model
    - PyTorch (GPU) and ONNX (CPU/GPU) backends
    - Single encoder pass for multi-capability inference
    - Optimized for low latency (<15ms for 6 capabilities on GPU)

Capabilities:
    - sentiment: 5-class sentiment analysis
    - emotions: Multi-label emotion detection (28 emotions)
    - safety_familyos: Family-safe content band (GREEN/YELLOW/RED)
    - safety_generic: Multi-label safety categories
    - intent: User intent classification
    - ingress: Message routing/category
    - ner_family: Family member entity recognition
    - ner_general: General named entity recognition
    - temporal: Temporal expression extraction
    - relation: Relationship type classification
    - nli: Natural language inference
    - embedding: 768-dim sentence embeddings

License: Apache 2.0
"""

__version__ = "2.0.0"
__author__ = "FamilyOS Team"

from familyos_ultrabert.model import UltraBERT
from familyos_ultrabert.labels import CAPABILITIES, Capability

# Alias for backward compatibility
FamilyOSModel = UltraBERT

__all__ = [
    "UltraBERT",
    "FamilyOSModel",  # Backward compatibility
    "CAPABILITIES",
    "Capability",
    "__version__",
]
