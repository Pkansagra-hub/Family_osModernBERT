# FamilyOS Unified Encoder - Rollout Plan

> **Document Version:** 1.0
> **Last Updated:** November 2025
> **Status:** Draft

---

## Executive Summary

This document outlines the phased rollout plan for migrating from the current 9-model NLP zoo (4350MB) to the unified ModernBERT encoder (650MB). The rollout prioritizes **safety** and **stability**, with clear rollback criteria at each phase.

### Key Metrics

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Memory Footprint** | 4350MB | 650MB | 85% reduction |
| **Load Time** | ~62s | ~8s | 7.7x faster |
| **Per-Envelope Latency** | ~150ms | ~35ms | 4.3x faster |
| **Model Count** | 9 models | 1 unified | 9→1 |
| **Capabilities** | 9 tasks | 12 tasks | +3 new capabilities |

---

## Phase 1: Shadow Mode (Week 1-2)

### Objective

Deploy unified model alongside existing model zoo to validate accuracy and performance **without any user impact**.

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        K0 Runtime                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────────┐        ┌─────────────────┐           │
│   │  Current Zoo    │        │  Unified Model   │           │
│   │  (Production)   │        │  (Shadow)        │           │
│   │                 │        │                  │           │
│   │  • ner_bert     │        │  • modernbert    │           │
│   │  • go_emotions  │        │    + 12 heads    │           │
│   │  • vader        │        │                  │           │
│   │  • distilbert   │        │                  │           │
│   │  • bart-mnli    │        │                  │           │
│   └────────┬────────┘        └────────┬─────────┘           │
│            │                          │                     │
│            │                          │                     │
│            ▼                          ▼                     │
│   ┌─────────────────┐        ┌─────────────────┐           │
│   │  Primary Output │        │  Shadow Output   │           │
│   │  → User         │        │  → Logs Only     │           │
│   └─────────────────┘        └─────────────────┘           │
│                                       │                     │
│                                       ▼                     │
│                              ┌─────────────────┐           │
│                              │  Comparison     │           │
│                              │  Metrics Logger │           │
│                              └─────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Actions

1. **Deploy Shadow Instance**
   - Deploy unified model to production environment
   - Configure to run on same inputs as current zoo
   - Output to logging only (no user-facing changes)

2. **Implement Comparison Logging**
   ```python
   # In k0/runtime/shadow_comparison.py
   def log_comparison(
       input_text: str,
       zoo_output: dict,
       unified_output: UnifiedNLPOutput,
       latency_zoo_ms: float,
       latency_unified_ms: float,
   ) -> None:
       """Log comparison between zoo and unified model outputs."""
       divergence = compute_divergence(zoo_output, unified_output)

       logger.info(
           "shadow_comparison",
           input_hash=hash(input_text),
           divergence=divergence,
           latency_zoo_ms=latency_zoo_ms,
           latency_unified_ms=latency_unified_ms,
           safety_zoo=zoo_output.get("safety_band"),
           safety_unified=unified_output.safety_familyos,
       )
   ```

3. **Monitor Key Metrics**
   - Output divergence rate per capability
   - Safety band disagreement rate (critical)
   - Latency comparison (P50, P95, P99)
   - Memory usage comparison
   - Error rates

### Success Criteria for Phase 1→2 Progression

| Metric | Threshold | Priority |
|--------|-----------|----------|
| Safety band agreement | ≥ 95% | P0 (blocking) |
| CRISIS detection agreement | ≥ 98% | P0 (blocking) |
| NER entity overlap (F1) | ≥ 90% | P1 |
| Sentiment agreement | ≥ 85% | P1 |
| P95 latency | ≤ 50ms | P1 |
| Error rate | < 0.1% | P0 (blocking) |

### Rollback Trigger (Phase 1)

**Automatic rollback NOT required** - shadow mode has no user impact. However, discontinue shadow deployment if:

- Unified model error rate > 1%
- Memory pressure affecting production
- Significant infrastructure cost increase

---

## Phase 2: Gradual Migration (Week 3-4)

### Objective

Progressively route traffic to unified model, starting with low-risk capabilities and internal dev spaces.

### Migration Order

```
Week 3: Low-Risk Capabilities
├── embedding (P08) - Easiest, no classification
├── ingress (M10) - Replace heavy BART model
└── sentiment - Well-validated

Week 4: Medium-Risk Capabilities
├── emotions - Multi-label, needs validation
├── ner_general - Entity extraction
├── ner_family - FamilyOS-specific
└── nli - Pair classification

Week 4+: High-Risk Capabilities (after safety validation)
├── safety_generic - Toxicity detection
└── safety_familyos - Policy bands (most critical)
```

### Traffic Routing Strategy

```python
# k0/runtime/traffic_router.py

class TrafficRouter:
    """Route traffic between zoo and unified model."""

    def __init__(self):
        self.unified_percentage = {
            # Start with 0%, increase gradually
            "embedding": 0,
            "ingress": 0,
            "sentiment": 0,
            "emotions": 0,
            "ner_general": 0,
            "ner_family": 0,
            "nli": 0,
            "safety_generic": 0,
            "safety_familyos": 0,
        }

    def should_use_unified(self, capability: str, user_space: str) -> bool:
        """Determine if this request should use unified model."""
        # Internal dev spaces always use unified (canary)
        if user_space in INTERNAL_DEV_SPACES:
            return True

        # Random sample based on percentage
        return random.random() < self.unified_percentage[capability]

    def increase_traffic(self, capability: str, percentage: int) -> None:
        """Increase unified model traffic percentage."""
        self.unified_percentage[capability] = min(100, percentage)
```

### Gradual Rollout Schedule

| Day | Capability | Unified % | Notes |
|-----|------------|-----------|-------|
| D1 | embedding | 10% | Start with embeddings |
| D2 | embedding | 50% | Monitor quality |
| D3 | embedding | 100% | Full cutover |
| D4 | ingress | 25% | Replace BART |
| D5 | ingress | 100% | Monitor memory savings |
| D6 | sentiment | 25% | Start sentiment |
| D7 | sentiment | 100% | Full cutover |
| D8-10 | emotions, ner_* | 25%→100% | Monitor entity quality |
| D11-12 | nli | 25%→100% | Pair classification |
| D13-14 | safety_generic | 10%→50%→100% | Careful monitoring |
| D15+ | safety_familyos | 10%→25%→50%→100% | Most careful |

### Monitoring Dashboard

```yaml
# Grafana dashboard panels
panels:
  - title: "Unified Model Traffic %"
    type: gauge
    targets:
      - expr: unified_traffic_percentage{capability="$capability"}

  - title: "Safety Agreement Rate"
    type: timeseries
    targets:
      - expr: safety_agreement_rate{model="unified"}
    alert:
      threshold: 0.95
      action: page_oncall

  - title: "Latency Comparison"
    type: timeseries
    targets:
      - expr: histogram_quantile(0.95, latency_ms{model="zoo"})
      - expr: histogram_quantile(0.95, latency_ms{model="unified"})

  - title: "Error Rate"
    type: timeseries
    targets:
      - expr: rate(errors_total{model="unified"}[5m])
    alert:
      threshold: 0.01
      action: auto_rollback
```

### Success Criteria for Phase 2→3 Progression

| Metric | Threshold | Duration |
|--------|-----------|----------|
| All capabilities at 100% unified | Yes | 48 hours stable |
| CRISIS recall | ≥ 98% | Continuous |
| Cultural FP rate (Indian expressions) | ≤ 2% | Tested |
| P95 latency | ≤ 50ms | 24 hours |
| Error rate | < 0.1% | 48 hours |
| No user complaints | Yes | 48 hours |

---

## Phase 3: Full Rollout (Week 5+)

### Objective

Complete migration to unified model, deprecate old model zoo, realize memory and latency savings.

### Actions

1. **Deprecate Model Zoo**
   ```python
   # k0/runtime/model_registry.py

   DEPRECATED_MODELS = [
       "ner_transformer",      # → ner_general + ner_family
       "sentiment_transformer", # → sentiment
       "go_emotions",          # → emotions
       "clinical_safety",      # → safety_generic + safety_familyos
       "zero_shot_classifier", # → ingress
       "sentence_transformer", # → embedding
   ]

   # Keep these for fallback
   RETAINED_MODELS = [
       "spacy_nlp",    # Fallback NER
       "vader",        # Lexicon baseline
   ]
   ```

2. **Remove Zoo Loading**
   - Update K0 startup to skip deprecated models
   - Measure actual memory savings
   - Verify startup time improvement

3. **Archive Zoo Code**
   - Move zoo model code to `k0/legacy/`
   - Keep for reference but don't load
   - Document migration in changelog

### Post-Rollout Validation

```python
# scripts/validate_rollout.py

def validate_full_rollout():
    """Validate unified model is fully deployed."""

    # 1. Check memory usage
    memory_mb = get_k0_memory_usage()
    assert memory_mb < 800, f"Memory {memory_mb}MB exceeds target 800MB"

    # 2. Check startup time
    startup_s = measure_k0_startup()
    assert startup_s < 15, f"Startup {startup_s}s exceeds target 15s"

    # 3. Check all capabilities available
    model = get_unified_model()
    for cap in Capability:
        output = model.infer("Test text", capabilities=[cap.value])
        assert output is not None, f"Capability {cap} failed"

    # 4. Check safety sensitivity
    crisis_texts = load_crisis_test_set()
    for text in crisis_texts:
        output = model.infer(text, capabilities=["safety_familyos"])
        assert output.safety_familyos == "CRISIS", f"Missed CRISIS: {text}"

    print("✅ Full rollout validation passed")
```

### Success Metrics (Post-Rollout)

| Metric | Target | Measurement |
|--------|--------|-------------|
| Memory reduction | ≥ 80% (870MB saved) | `nvidia-smi` / `psutil` |
| Startup time reduction | ≥ 70% (54s saved) | Timing |
| Latency reduction | ≥ 70% (115ms saved) | P95 measurements |
| CRISIS recall | ≥ 98% | Weekly safety audit |
| User-reported issues | 0 safety-related | Support tickets |

---

## Rollback Criteria

### Automatic Rollback Triggers

The following conditions trigger **immediate automatic rollback** to model zoo:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| CRISIS recall drop | < 95% | Immediate rollback |
| Safety FP rate | > 5% | Immediate rollback |
| P95 latency | > 100ms | Immediate rollback |
| Error rate | > 1% | Immediate rollback |
| OOM events | Any | Immediate rollback |

### Rollback Procedure

```python
# k0/runtime/rollback.py

class RollbackManager:
    """Manage rollback to model zoo."""

    def __init__(self):
        self.rollback_triggered = False
        self.rollback_reason = None

    def check_rollback_conditions(self, metrics: dict) -> bool:
        """Check if rollback is needed."""
        if metrics["crisis_recall"] < 0.95:
            self.trigger_rollback("CRISIS recall below 95%")
            return True

        if metrics["safety_fp_rate"] > 0.05:
            self.trigger_rollback("Safety FP rate above 5%")
            return True

        if metrics["p95_latency_ms"] > 100:
            self.trigger_rollback("P95 latency above 100ms")
            return True

        if metrics["error_rate"] > 0.01:
            self.trigger_rollback("Error rate above 1%")
            return True

        return False

    def trigger_rollback(self, reason: str) -> None:
        """Execute rollback to model zoo."""
        self.rollback_triggered = True
        self.rollback_reason = reason

        logger.critical(f"ROLLBACK TRIGGERED: {reason}")

        # 1. Route all traffic to zoo
        traffic_router.set_all_to_zoo()

        # 2. Alert oncall
        pagerduty.alert(
            severity="critical",
            message=f"Unified model rollback: {reason}",
        )

        # 3. Log for post-mortem
        logger.info(
            "rollback_executed",
            reason=reason,
            timestamp=datetime.utcnow(),
        )

    def verify_rollback(self) -> bool:
        """Verify rollback succeeded."""
        # Check zoo is serving
        for cap in ALL_CAPABILITIES:
            response = test_capability(cap, use_zoo=True)
            if not response.success:
                return False
        return True
```

### Manual Rollback Triggers

The following require **manual investigation** before rollback decision:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| User complaints | > 3 safety-related | Investigate, consider rollback |
| Embedding quality drop | Spearman < 0.75 | Investigate |
| NER quality drop | F1 < 80% | Investigate |
| Unusual divergence patterns | Significant | Investigate |

---

## K0 Module Integration

### Module: M02 (hippocampus.semantic_project)

**Current Implementation:**
```python
# Uses: ner_transformer, spacy_nlp
ner_model = load_model("ner_transformer")
entities = ner_model(text)
```

**After Migration:**
```python
# Uses: unified model with 3 capabilities
from modeling_studio.inference import sys_nlp_infer

outputs = sys_nlp_infer(
    texts=[text],
    capabilities=["ner_general", "ner_family", "temporal"],
)
entities = outputs[0].ner_general + outputs[0].ner_family
temporal = outputs[0].temporal
```

**Integration Points:**
- `hippocampus/semantic_project.py`: Update `extract_entities()` function
- Remove `ner_transformer` model loading
- Keep `spacy_nlp` as fallback for complex parsing

---

### Module: M04 (affect.analyze)

**Current Implementation:**
```python
# Uses: vader, go_emotions, sentiment_transformer, clinical_safety
vader_scores = vader_analyzer(text)
emotions = go_emotions(text)
sentiment = sentiment_transformer(text)
safety = clinical_safety(text)
```

**After Migration:**
```python
# Uses: unified model with 5 capabilities
outputs = sys_nlp_infer(
    texts=[text],
    capabilities=["sentiment", "emotions", "safety_generic", "safety_familyos", "intent"],
)
sentiment = outputs[0].sentiment
emotions = outputs[0].emotions
safety_band = outputs[0].safety_familyos
intent = outputs[0].intent
```

**Integration Points:**
- `affect/analyze.py`: Update `analyze_affect()` function
- Remove 4 separate model loadings
- Keep `vader` as Tier-0 lexicon fallback

---

### Module: M10 (context.ingress_classify)

**Current Implementation:**
```python
# Uses: zero_shot_classifier (BART-large, 1.5GB!)
result = zero_shot_classifier(text, candidate_labels=ACTIVITY_TYPES)
```

**After Migration:**
```python
# Uses: unified model with 1 capability
outputs = sys_nlp_infer(
    texts=[text],
    capabilities=["ingress"],
)
activity_type = outputs[0].ingress
confidence = outputs[0].ingress_confidence
```

**Integration Points:**
- `context/ingress_classify.py`: Update `classify_ingress()` function
- Remove 1.5GB BART model (biggest memory savings!)
- Direct classification instead of zero-shot

---

### Module: P08 (embedding pipeline)

**Current Implementation:**
```python
# Uses: sentence_transformer (MiniLM, 384-dim)
embedding = sentence_transformer.encode(text)  # 384-dim
```

**After Migration:**
```python
# Uses: unified model embedding head (768-dim)
outputs = sys_nlp_infer(
    texts=[text],
    capabilities=["embedding"],
)
embedding = outputs[0].embedding  # 768-dim
```

**Integration Points:**
- `pipelines/embedding.py`: Update `embed_text()` function
- Update downstream consumers for 768-dim (vs 384-dim)
- Consider projection layer for backward compatibility if needed

---

## Monitoring & Alerting

### Key Metrics to Monitor

```yaml
# prometheus/alerts.yml

groups:
  - name: unified_model_alerts
    rules:
      # CRISIS recall alert (P0)
      - alert: CrisisRecallLow
        expr: crisis_recall < 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "CRISIS recall below 95%"
          action: "Trigger immediate rollback"

      # Safety FP rate alert (P0)
      - alert: SafetyFPHigh
        expr: safety_false_positive_rate > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Safety FP rate above 5%"
          action: "Trigger immediate rollback"

      # Latency alert (P1)
      - alert: LatencyHigh
        expr: histogram_quantile(0.95, latency_ms) > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 100ms"
          action: "Investigate, consider rollback"

      # Error rate alert (P0)
      - alert: ErrorRateHigh
        expr: rate(errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 1%"
          action: "Trigger immediate rollback"
```

### Dashboard Panels

1. **Traffic Split** - Current % unified vs zoo
2. **Safety Metrics** - CRISIS recall, FP rate, band distribution
3. **Latency** - P50, P95, P99 comparison
4. **Memory** - Unified model memory usage
5. **Errors** - Error rate and types
6. **Divergence** - Output agreement with zoo

---

## Timeline Summary

| Week | Phase | Key Actions | Exit Criteria |
|------|-------|-------------|---------------|
| 1-2 | Shadow Mode | Deploy shadow, log comparisons | ≥95% safety agreement |
| 3-4 | Gradual Migration | Route 10%→100% per capability | 48h stable at 100% |
| 5+ | Full Rollout | Deprecate zoo, measure savings | All metrics green |

---

## Appendix A: Capability Mapping

| Old Model | Memory | New Capability | Notes |
|-----------|--------|----------------|-------|
| ner_transformer | 500MB | ner_general + ner_family | Split into 2 heads |
| familyos_emotions | 500MB | emotions | 28→44 emotions (FamilyOS schema) |
| sentiment_transformer | 400MB | sentiment | 3→5 classes |
| clinical_safety | 300MB | safety_generic + safety_familyos | Split: generic + bands |
| zero_shot_classifier | 1500MB | ingress | 7→12 domains |
| sentence_transformer | 500MB | embedding | 384→768 dim |
| spacy_nlp | 100MB | (keep) | Fallback parsing |
| vader | 50MB | (keep) | Tier-0 lexicon |

---

## Appendix B: Rollback Checklist

- [ ] Traffic router set to 100% zoo
- [ ] Unified model unloaded from memory
- [ ] Alert sent to oncall
- [ ] Incident ticket created
- [ ] User-facing status updated
- [ ] Post-mortem scheduled

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | Nov 2025 | K0 Team | Initial draft |
