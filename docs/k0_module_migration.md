# K0 Module Migration Guide

This guide provides step-by-step instructions for migrating K0 modules from their current multi-model architecture to the unified `familyos_unified_v2` model.

## Overview

### Resource Savings Summary

| Module | Current Memory | Unified Memory | Savings | Current Latency | Unified Latency | Speedup |
|--------|---------------|----------------|---------|-----------------|-----------------|---------|
| M02 (hippocampus) | 1,200 MB | 650 MB (shared) | 85% | 45 ms | 35 ms | 1.3x |
| M04 (affect) | 900 MB | 0 MB (shared) | 100% | 35 ms | 0 ms (cached) | ∞ |
| M10 (context) | 1,500 MB | 0 MB (shared) | 100% | 55 ms | 0 ms (cached) | ∞ |
| P08 (embedding) | 750 MB | 0 MB (shared) | 100% | 15 ms | 0 ms (cached) | ∞ |
| **Total** | **4,350 MB** | **650 MB** | **85%** | **150 ms** | **35 ms** | **4.3x** |

### Migration Phases

1. **Phase 1**: Import migration (no runtime changes)
2. **Phase 2**: Shadow mode (parallel execution)
3. **Phase 3**: Gradual rollout (traffic shifting)
4. **Phase 4**: Full migration (deprecate old models)

---

## Module Migrations

### M02: `hippocampus.semantic_project`

**Purpose**: Semantic understanding and entity extraction for family memories.

**Current Model**: `distilbert-base-uncased-finetuned-sst-2-english`

**New Capability**: `NER_FAMILY`

#### Before (Current Implementation)

```python
# k0/modules/M02_hippocampus/semantic_project.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SemanticProjector:
    """Extract semantic entities from family memories."""

    def __init__(self):
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract_entities(self, text: str) -> list[dict]:
        """Extract entities from text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Custom entity extraction logic...
        entities = self._parse_entities(outputs, text)
        return entities

    def _parse_entities(self, outputs, text: str) -> list[dict]:
        # Legacy parsing logic
        pass
```

#### After (Unified Implementation)

```python
# k0/modules/M02_hippocampus/semantic_project.py

from modeling_studio.inference.unified_output import sys_nlp_infer
from modeling_studio.k0.runtime import resolve_capability, Capability

class SemanticProjector:
    """Extract semantic entities from family memories using unified model."""

    def __init__(self):
        # Verify capability is available
        model_name, head_name = resolve_capability(Capability.NER_FAMILY)
        assert model_name == "familyos_unified_v2"
        self._capability = Capability.NER_FAMILY

    def extract_entities(self, text: str) -> list[dict]:
        """Extract family-specific entities from text."""
        # Use unified inference API
        result = sys_nlp_infer(
            text=text,
            capabilities=["ner_family"],
        )

        # Convert to legacy format for backward compatibility
        return [
            {
                "text": entity.text,
                "label": entity.label,
                "start": entity.start_char,
                "end": entity.end_char,
                "confidence": entity.confidence,
            }
            for entity in result.entities
        ]

    def extract_entities_batch(self, texts: list[str]) -> list[list[dict]]:
        """Batch entity extraction for improved throughput."""
        results = []
        for text in texts:
            results.append(self.extract_entities(text))
        return results
```

#### Migration Checklist for M02

- [ ] Update imports to use `modeling_studio.inference.unified_output`
- [ ] Replace model initialization with capability resolution
- [ ] Update `extract_entities()` to use `sys_nlp_infer()`
- [ ] Verify entity format matches legacy output
- [ ] Run shadow mode comparison tests
- [ ] Update unit tests
- [ ] Remove old model dependencies from requirements

---

### M04: `affect.analyze`

**Purpose**: Emotion and sentiment analysis for family communications.

**Current Models**:
- `j-hartmann/emotion-english-distilroberta-base` (emotions)
- `distilbert-base-uncased-finetuned-sst-2-english` (sentiment)

**New Capabilities**: `EMOTIONS`, `SENTIMENT`

#### Before (Current Implementation)

```python
# k0/modules/M04_affect/analyze.py

from transformers import pipeline
import torch

class AffectAnalyzer:
    """Analyze emotions and sentiment in family communications."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1

        # Load emotion model (400MB)
        self.emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=self.device,
            return_all_scores=True,
        )

        # Load sentiment model (250MB)
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
        )

    def analyze_emotion(self, text: str) -> dict:
        """Detect primary emotion in text."""
        results = self.emotion_pipe(text)[0]
        # Find highest scoring emotion
        primary = max(results, key=lambda x: x["score"])
        return {
            "emotion": primary["label"],
            "confidence": primary["score"],
            "all_scores": {r["label"]: r["score"] for r in results},
        }

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment polarity."""
        result = self.sentiment_pipe(text)[0]
        return {
            "sentiment": result["label"].lower(),
            "score": result["score"],
        }

    def full_analysis(self, text: str) -> dict:
        """Combined emotion and sentiment analysis."""
        return {
            "emotion": self.analyze_emotion(text),
            "sentiment": self.analyze_sentiment(text),
        }
```

#### After (Unified Implementation)

```python
# k0/modules/M04_affect/analyze.py

from modeling_studio.inference.unified_output import sys_nlp_infer
from modeling_studio.k0.runtime import resolve_capability, Capability

class AffectAnalyzer:
    """Analyze emotions and sentiment using unified model."""

    def __init__(self):
        # Verify capabilities
        for cap in [Capability.EMOTIONS, Capability.SENTIMENT]:
            model_name, _ = resolve_capability(cap)
            assert model_name == "familyos_unified_v2"

    def analyze_emotion(self, text: str) -> dict:
        """Detect primary emotion in text."""
        result = sys_nlp_infer(
            text=text,
            capabilities=["emotions"],
        )

        return {
            "emotion": result.emotions[0] if result.emotions else "neutral",
            "confidence": max(result.emotion_scores.values()) if result.emotion_scores else 0.0,
            "all_scores": result.emotion_scores,
        }

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment polarity."""
        result = sys_nlp_infer(
            text=text,
            capabilities=["sentiment"],
        )

        # Map 5-class sentiment to legacy binary format
        sentiment_map = {
            "very_negative": ("negative", 0.9),
            "negative": ("negative", 0.7),
            "neutral": ("neutral", 0.5),
            "positive": ("positive", 0.7),
            "very_positive": ("positive", 0.9),
        }

        label, base_score = sentiment_map.get(
            result.sentiment, ("neutral", 0.5)
        )

        return {
            "sentiment": label,
            "score": result.sentiment_score,
        }

    def full_analysis(self, text: str) -> dict:
        """Combined emotion and sentiment analysis in single inference."""
        # Single inference call for both capabilities
        result = sys_nlp_infer(
            text=text,
            capabilities=["emotions", "sentiment"],
        )

        return {
            "emotion": {
                "emotion": result.emotions[0] if result.emotions else "neutral",
                "confidence": max(result.emotion_scores.values()) if result.emotion_scores else 0.0,
                "all_scores": result.emotion_scores,
            },
            "sentiment": {
                "sentiment": result.sentiment,
                "score": result.sentiment_score,
            },
        }
```

#### Migration Checklist for M04

- [ ] Update imports to use `modeling_studio.inference.unified_output`
- [ ] Remove dual-pipeline initialization
- [ ] Update emotion analysis to use unified inference
- [ ] Update sentiment analysis with score mapping
- [ ] Combine multiple analyses into single inference call
- [ ] Verify output format compatibility
- [ ] Run shadow mode comparison tests
- [ ] Remove old model dependencies

---

### M10: `context.ingress_classify`

**Purpose**: Classify incoming messages by context type.

**Current Model**: `facebook/bart-large-mnli` (1.5GB)

**New Capability**: `INGRESS`

#### Before (Current Implementation)

```python
# k0/modules/M10_context/ingress_classify.py

from transformers import pipeline
import torch

class IngressClassifier:
    """Classify incoming messages by type and context."""

    LABELS = [
        "user_message",
        "system_event",
        "notification",
        "command",
        "query",
        "other",
    ]

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1

        # Zero-shot classification with BART (1.5GB model)
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device,
        )

    def classify(self, text: str) -> dict:
        """Classify message type using zero-shot."""
        result = self.classifier(
            text,
            candidate_labels=self.LABELS,
            multi_label=False,
        )

        return {
            "label": result["labels"][0],
            "score": result["scores"][0],
            "all_scores": dict(zip(result["labels"], result["scores"])),
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Batch classification."""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
```

#### After (Unified Implementation)

```python
# k0/modules/M10_context/ingress_classify.py

from modeling_studio.inference.unified_output import sys_nlp_infer
from modeling_studio.k0.runtime import resolve_capability, Capability

class IngressClassifier:
    """Classify incoming messages using unified model."""

    LABELS = [
        "user_message",
        "system_event",
        "notification",
        "command",
        "query",
        "other",
    ]

    def __init__(self):
        # Verify capability
        model_name, head_name = resolve_capability(Capability.INGRESS)
        assert model_name == "familyos_unified_v2"
        assert head_name == "ingress_head"

    def classify(self, text: str) -> dict:
        """Classify message type using fine-tuned head."""
        result = sys_nlp_infer(
            text=text,
            capabilities=["ingress"],
        )

        # The unified model has a dedicated ingress head
        # No zero-shot overhead - direct classification
        return {
            "label": result.ingress_type,
            "score": result.ingress_confidence,
            "all_scores": result.ingress_scores,
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        """Batch classification with single model load."""
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
```

#### Migration Checklist for M10

- [ ] Update imports to use unified inference
- [ ] Remove BART zero-shot pipeline
- [ ] Update classify() to use dedicated ingress head
- [ ] Verify label compatibility
- [ ] Run shadow mode comparison
- [ ] Update unit tests
- [ ] Remove facebook/bart-large-mnli dependency

---

### P08: Embedding Pipeline

**Purpose**: Generate embeddings for semantic search and similarity.

**Current Model**: `sentence-transformers/all-MiniLM-L6-v2` (90MB)

**New Capability**: `EMBEDDING`

#### Before (Current Implementation)

```python
# k0/pipelines/P08_embedding/embedder.py

from sentence_transformers import SentenceTransformer
import torch
import numpy as np

class FamilyEmbedder:
    """Generate embeddings for family content."""

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))
```

#### After (Unified Implementation)

```python
# k0/pipelines/P08_embedding/embedder.py

from modeling_studio.inference.unified_output import sys_nlp_infer
from modeling_studio.k0.runtime import resolve_capability, Capability
import numpy as np

class FamilyEmbedder:
    """Generate embeddings using unified model."""

    def __init__(self):
        # Verify capability
        model_name, head_name = resolve_capability(Capability.EMBEDDING)
        assert model_name == "familyos_unified_v2"
        assert head_name == "embedding_head"

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        result = sys_nlp_infer(
            text=text,
            capabilities=["embedding"],
        )

        # Embeddings are returned as numpy array
        embedding = np.array(result.embedding)

        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return np.stack(embeddings)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1, emb2))
```

#### Migration Checklist for P08

- [ ] Update imports to use unified inference
- [ ] Remove SentenceTransformer dependency
- [ ] Update embed() to use sys_nlp_infer
- [ ] Verify embedding dimensions match (may need projection layer)
- [ ] Benchmark embedding quality on family-specific content
- [ ] Run similarity comparison tests
- [ ] Update downstream consumers

---

## Shadow Mode Implementation

### Shadow Mode Wrapper

Use this wrapper to run both implementations in parallel and compare results:

```python
# k0/utils/shadow_mode.py

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

@dataclass
class ShadowResult:
    """Result from shadow mode comparison."""
    legacy_result: Any
    unified_result: Any
    legacy_latency_ms: float
    unified_latency_ms: float
    match: bool
    diff: dict | None = None

def shadow_compare(
    legacy_fn: Callable,
    unified_fn: Callable,
    *args,
    comparator: Callable[[Any, Any], tuple[bool, dict | None]] | None = None,
    **kwargs,
) -> ShadowResult:
    """
    Run both implementations and compare results.

    Args:
        legacy_fn: Original implementation function
        unified_fn: New unified implementation function
        comparator: Optional custom comparison function
        *args, **kwargs: Arguments to pass to both functions

    Returns:
        ShadowResult with comparison data
    """
    # Run legacy
    start = time.perf_counter()
    legacy_result = legacy_fn(*args, **kwargs)
    legacy_ms = (time.perf_counter() - start) * 1000

    # Run unified
    start = time.perf_counter()
    unified_result = unified_fn(*args, **kwargs)
    unified_ms = (time.perf_counter() - start) * 1000

    # Compare results
    if comparator:
        match, diff = comparator(legacy_result, unified_result)
    else:
        match = legacy_result == unified_result
        diff = None if match else {"legacy": legacy_result, "unified": unified_result}

    result = ShadowResult(
        legacy_result=legacy_result,
        unified_result=unified_result,
        legacy_latency_ms=legacy_ms,
        unified_latency_ms=unified_ms,
        match=match,
        diff=diff,
    )

    # Log comparison
    logger.info(
        f"Shadow comparison: match={match}, "
        f"legacy={legacy_ms:.2f}ms, unified={unified_ms:.2f}ms, "
        f"speedup={legacy_ms/unified_ms:.2f}x"
    )

    return result
```

### Using Shadow Mode

```python
# Example: M04 shadow mode testing

from k0.modules.M04_affect.analyze_legacy import AffectAnalyzer as LegacyAnalyzer
from k0.modules.M04_affect.analyze import AffectAnalyzer as UnifiedAnalyzer
from k0.utils.shadow_mode import shadow_compare

# Initialize both analyzers
legacy = LegacyAnalyzer()
unified = UnifiedAnalyzer()

# Test texts
test_texts = [
    "I'm so happy we're all going to grandma's house!",
    "I'm worried about the upcoming exam.",
    "The weather is nice today.",
]

# Run shadow comparison
for text in test_texts:
    result = shadow_compare(
        legacy.full_analysis,
        unified.full_analysis,
        text,
    )

    print(f"Text: {text[:50]}...")
    print(f"  Match: {result.match}")
    print(f"  Speedup: {result.legacy_latency_ms / result.unified_latency_ms:.2f}x")
    if not result.match:
        print(f"  Diff: {result.diff}")
```

---

## Rollback Procedures

### Automatic Rollback Triggers

```python
# k0/utils/rollback.py

from dataclasses import dataclass
from typing import Callable

@dataclass
class RollbackCriteria:
    """Criteria for automatic rollback."""
    crisis_recall_min: float = 0.95  # CRISIS recall must be >= 95%
    false_positive_max: float = 0.05  # FP rate must be <= 5%
    latency_p99_max_ms: float = 100.0  # P99 latency <= 100ms
    error_rate_max: float = 0.01  # Error rate <= 1%

def check_rollback_needed(
    metrics: dict,
    criteria: RollbackCriteria,
) -> tuple[bool, list[str]]:
    """
    Check if rollback is needed based on metrics.

    Returns:
        Tuple of (rollback_needed, list of violated criteria)
    """
    violations = []

    if metrics.get("crisis_recall", 1.0) < criteria.crisis_recall_min:
        violations.append(
            f"CRISIS recall {metrics['crisis_recall']:.2%} < {criteria.crisis_recall_min:.2%}"
        )

    if metrics.get("false_positive_rate", 0.0) > criteria.false_positive_max:
        violations.append(
            f"FP rate {metrics['false_positive_rate']:.2%} > {criteria.false_positive_max:.2%}"
        )

    if metrics.get("latency_p99_ms", 0.0) > criteria.latency_p99_max_ms:
        violations.append(
            f"P99 latency {metrics['latency_p99_ms']:.0f}ms > {criteria.latency_p99_max_ms:.0f}ms"
        )

    if metrics.get("error_rate", 0.0) > criteria.error_rate_max:
        violations.append(
            f"Error rate {metrics['error_rate']:.2%} > {criteria.error_rate_max:.2%}"
        )

    return len(violations) > 0, violations
```

### Manual Rollback

```bash
# Rollback to legacy models
export K0_USE_UNIFIED_MODEL=false
k0 service restart

# Verify rollback
k0 health check --module M02 M04 M10 P08
```

---

## Testing Guide

### Unit Test Updates

```python
# tests/test_m04_migration.py

import pytest
from k0.modules.M04_affect.analyze import AffectAnalyzer

class TestAffectAnalyzerMigration:
    """Test M04 module migration to unified model."""

    @pytest.fixture
    def analyzer(self):
        return AffectAnalyzer()

    def test_emotion_detection(self, analyzer):
        """Test emotion detection returns expected format."""
        result = analyzer.analyze_emotion("I'm so happy today!")

        assert "emotion" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sentiment_analysis(self, analyzer):
        """Test sentiment analysis returns expected format."""
        result = analyzer.analyze_sentiment("This is great!")

        assert "sentiment" in result
        assert "score" in result
        assert result["sentiment"] in ["positive", "negative", "neutral"]

    def test_full_analysis(self, analyzer):
        """Test combined analysis uses single inference."""
        result = analyzer.full_analysis("I love my family!")

        assert "emotion" in result
        assert "sentiment" in result

    def test_capability_resolution(self):
        """Test capability resolves to unified model."""
        from modeling_studio.k0.runtime import resolve_capability, Capability

        for cap in [Capability.EMOTIONS, Capability.SENTIMENT]:
            model_name, head_name = resolve_capability(cap)
            assert model_name == "familyos_unified_v2"
```

### Integration Test

```python
# tests/integration/test_unified_migration.py

import pytest
from modeling_studio.k0.runtime import (
    MODEL_REGISTRY,
    resolve_capability,
    get_unified_model,
    Capability,
)

class TestUnifiedMigration:
    """Integration tests for unified model migration."""

    def test_registry_contains_unified_model(self):
        """Verify unified model is registered."""
        assert "familyos_unified_v2" in MODEL_REGISTRY

    def test_all_capabilities_resolve(self):
        """All capabilities resolve to unified model."""
        for cap in Capability:
            model_name, head_name = resolve_capability(cap)
            assert model_name == "familyos_unified_v2"
            assert head_name is not None

    def test_capability_aliases(self):
        """Test capability aliases work correctly."""
        test_aliases = [
            ("ner", "familyos_unified_v2"),
            ("sentiment_analysis", "familyos_unified_v2"),
            ("embeddings", "familyos_unified_v2"),
        ]

        for alias, expected_model in test_aliases:
            model_name, _ = resolve_capability(alias)
            assert model_name == expected_model

    @pytest.mark.slow
    def test_model_loads(self):
        """Test model loads successfully."""
        model = get_unified_model()
        assert model is not None
```

---

## Dependency Cleanup

After successful migration, remove the following dependencies:

```toml
# pyproject.toml - Remove these after migration

# Remove from dependencies:
# sentence-transformers  # Replaced by unified embedding head
# transformers>=4.30  # Can downgrade version requirements

# These models will no longer be downloaded:
# - facebook/bart-large-mnli (1.5GB)
# - j-hartmann/emotion-english-distilroberta-base (400MB)
# - sentence-transformers/all-MiniLM-L6-v2 (90MB)
# - distilbert-base-uncased-finetuned-sst-2-english (250MB)
```

---

## Support and Troubleshooting

### Common Issues

1. **Import errors after migration**
   - Ensure `modeling_studio` is installed: `pip install -e .`
   - Check Python path includes src directory

2. **Model not found errors**
   - Run `python -c "from modeling_studio.k0.runtime import MODEL_REGISTRY; print(MODEL_REGISTRY.keys())"`
   - Verify model is registered

3. **Output format mismatches**
   - Use shadow mode to identify differences
   - Update downstream consumers to handle new format

4. **Performance regression**
   - Check if model is cached: first call is slower
   - Verify CUDA is available: `torch.cuda.is_available()`

### Getting Help

- File issues in the repository
- Contact: #nlp-platform Slack channel
- Documentation: https://internal.docs/modeling-studio
