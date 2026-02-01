# Intent/Ingress V2 Release Integration Plan

**Document Version:** 1.1
**Created:** 2026-01-31
**Updated:** 2026-02-01
**Status:** Planning

---

## Overview

Integrate the new **Label-Description Embedding** based Intent V2 and Ingress V2 heads from the training repo (`src/modeling_studio`) into the release package (`familyos_ultrabert`).

### K1 Diagram Requirements

- **Intent Head V2 (MULTI-LABEL):** Returns `{primary: str, all: List[str], scores: Dict[str, float]}`
- **Ingress Head V2 (MULTI-LABEL):** Returns `{domains: List[str], scores: Dict[str, float]}`

### Architecture Summary

```text
[CLS] (768-dim)
      |
Query Projection (768 -> projection_dim)
      |
L2 Normalize
      |
Cosine Similarity with Learnable Label Embeddings
      |
Temperature Scaling (learnable)
      |
Sigmoid (multi-label) -> threshold -> primary + all[]
```

---

## Milestones

---

### Milestone 0: Training Repo Label Consistency (PRE-REQUISITE)

> Add V2 label schemas to training repo for consistency

**Status:** VERIFIED (2026-02-01)
**Estimated Effort:** 0.25 days
**Dependencies:** None
**Verification:** Training script ran successfully with 5 heads (3 NER + 2 V2 classification)

#### Current State (RESOLVED)

The training repo now has V2 label schemas in the central location:

- `INTENT_V2_LABELS` - 8 labels, `multi_label_classification`
- `INGRESS_V2_LABELS` - 12 labels, `multi_label_classification`
- `Capability.INTENT_V2` and `Capability.INGRESS_V2` added to enum
- Mappings added to `CAPABILITY_TO_LABELS` and `ALL_LABEL_SCHEMAS`
- Exports added to `__all__`

#### Epic 0.1: Add V2 Schemas to Training Repo

**Issue 0.1.1: Add INTENT_V2_LABELS to training repo**

- [x] Add `INTENT_V2_LABELS` to `src/modeling_studio/data/labels.py`
- [x] Set `problem_type="multi_label_classification"`

**File:** `src/modeling_studio/data/labels.py`
**Location:** After line 710 (after INTENT_LABELS)

```python
INTENT_V2_LABELS = LabelSchema(
    name="intent_v2",
    label2id={
        "log_memory": 0,
        "query_memory": 1,
        "set_reminder": 2,
        "express_feeling": 3,
        "seek_advice": 4,
        "share_news": 5,
        "reflect": 6,
        "other": 7,
    },
    problem_type="multi_label_classification",
    description="User intent classification V2 - multi-label (8 intents)",
)
```

---

**Issue 0.1.2: Add INGRESS_V2_LABELS to training repo**

- [x] Add `INGRESS_V2_LABELS` to `src/modeling_studio/data/labels.py`
- [x] Set `problem_type="multi_label_classification"`

**File:** `src/modeling_studio/data/labels.py`
**Location:** After line 580 (after INGRESS_LABELS)

```python
INGRESS_V2_LABELS = LabelSchema(
    name="ingress_v2",
    label2id={
        "DIARY": 0,
        "TASK": 1,
        "HEALTH": 2,
        "FINANCE": 3,
        "RELATIONSHIP": 4,
        "WORK": 5,
        "META": 6,
        "MEMORY": 7,
        "PLANNING": 8,
        "CELEBRATION": 9,
        "CONCERN": 10,
        "GRATITUDE": 11,
    },
    problem_type="multi_label_classification",
    description="Domain classification V2 - multi-label (12 domains)",
)
```

---

**Issue 0.1.3: Add V2 Capabilities to enum**

- [x] Add `INTENT_V2 = "intent_v2"` to `Capability` enum
- [x] Add `INGRESS_V2 = "ingress_v2"` to `Capability` enum

**File:** `src/modeling_studio/data/labels.py`
**Location:** Line 760-770 (Capability enum)

---

**Issue 0.1.4: Update CAPABILITY_TO_LABELS mapping**

- [x] Add `Capability.INTENT_V2: INTENT_V2_LABELS`
- [x] Add `Capability.INGRESS_V2: INGRESS_V2_LABELS`

**File:** `src/modeling_studio/data/labels.py`
**Location:** Line 780-795 (CAPABILITY_TO_LABELS dict)

---

**Issue 0.1.5: Update exports and ALL_LABEL_SCHEMAS**

- [x] Add to `__all__`: `"INTENT_V2_LABELS"`, `"INGRESS_V2_LABELS"`
- [x] Add to `ALL_LABEL_SCHEMAS` dict

**File:** `src/modeling_studio/data/labels.py`
**Location:** Lines 820-875

---

**Issue 0.1.6: Update training script to use central labels (optional)**

- [x] Refactored training script to import from labels.py instead of head classes
- [x] Updated `CLASSIFICATION_LABEL_CONFIGS` to use `INTENT_V2_LABELS.label2id` and `INGRESS_V2_LABELS.label2id`
- [x] Added `IntentHeadV2` and `IngressHeadV2` imports to `modernbert_multitask.py`
- [x] Added `Capability.INTENT_V2` and `Capability.INGRESS_V2` to `CAPABILITY_TO_HEAD_TYPE`
- [x] Added V2 capabilities to `TASK_GROUPS["sequence_tasks"]`

**Files Modified:**

- `scripts/training/train_globalpointer_unified.py` - Uses central labels now
- `src/modeling_studio/models/modernbert_multitask.py` - V2 head mappings added

---

### Milestone 1: Labels & Schema Layer (Release Repo)

> Define V2 label schemas with multi-label support in `familyos_ultrabert/labels.py`

**Status:** VERIFIED (2026-02-01)
**Estimated Effort:** 0.5 days
**Dependencies:** Milestone 0 (recommended but not blocking)
**Verification:** Imports work, backward compatibility confirmed

#### Epic 1.1: Add V2 Label Schemas

**Issue 1.1.1: Add INTENT_V2_LABELS schema**

- [x] Add new `LabelSchema` with `problem_type="multi_label_classification"`
- [x] Labels (8): `log_memory, query_memory, set_reminder, express_feeling, seek_advice, share_news, reflect, other`
- [x] Add descriptions for zero-shot init

**File:** `familyos_ultrabert/labels.py`
**Location:** After line 346 (after INTENT_LABELS)

```python
INTENT_V2_LABELS = LabelSchema(
    name="intent_v2",
    label2id={
        "log_memory": 0,
        "query_memory": 1,
        "set_reminder": 2,
        "express_feeling": 3,
        "seek_advice": 4,
        "share_news": 5,
        "reflect": 6,
        "other": 7,
    },
    problem_type="multi_label_classification",
    description="8 user intents (multi-label)",
)
```

---

**Issue 1.1.2: Add INGRESS_V2_LABELS schema**

- [x] Add new `LabelSchema` with `problem_type="multi_label_classification"`
- [x] Labels (12): `DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META, MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE`

**File:** `familyos_ultrabert/labels.py`
**Location:** After INTENT_V2_LABELS

```python
INGRESS_V2_LABELS = LabelSchema(
    name="ingress_v2",
    label2id={
        "DIARY": 0,
        "TASK": 1,
        "HEALTH": 2,
        "FINANCE": 3,
        "RELATIONSHIP": 4,
        "WORK": 5,
        "META": 6,
        "MEMORY": 7,
        "PLANNING": 8,
        "CELEBRATION": 9,
        "CONCERN": 10,
        "GRATITUDE": 11,
    },
    problem_type="multi_label_classification",
    description="12 domain categories (multi-label)",
)
```

---

**Issue 1.1.3: Add Capability enum entries**

- [x] Add `INTENT_V2 = "intent_v2"` to `Capability` enum
- [x] Add `INGRESS_V2 = "ingress_v2"` to `Capability` enum

**File:** `familyos_ultrabert/labels.py`
**Location:** Line 53-70 (Capability enum)

```python
class Capability(str, Enum):
    # ... existing ...
    INTENT = "intent"
    INTENT_V2 = "intent_v2"      # NEW
    INGRESS_V2 = "ingress_v2"    # NEW
```

---

#### Epic 1.2: Update Capability Mappings

**Issue 1.2.1: Update CAPABILITY_TO_LABELS mapping**

- [x] Add `Capability.INTENT_V2: INTENT_V2_LABELS` mapping
- [x] Add `Capability.INGRESS_V2: INGRESS_V2_LABELS` mapping

**File:** `familyos_ultrabert/labels.py`
**Location:** Line 351-370 (CAPABILITY_TO_LABELS dict)

---

**Issue 1.2.2: Backward compatibility validation**

- [x] Verify old `INTENT` and `INGRESS` capabilities still work
- [x] Ensure `Capability("intent")` still returns old labels
- [x] Write unit test for both old and new capabilities

---

### Milestone 2: Head Implementation Layer

> Port LabelDescriptionHead architecture to `familyos_ultrabert/models/heads.py`

**Status:** VERIFIED (2026-02-01)
**Estimated Effort:** 1.5 days
**Dependencies:** Milestone 1
**Verification:** Import test passed - all heads and mappings work

#### Epic 2.1: Port Head Classes

**Issue 2.1.1: Port LabelDescriptionHead base class**

- [x] Copy class from `src/modeling_studio/models/heads.py` lines 3685-3933
- [x] Include all methods:
  - `__init__` (query_proj, label_embeddings, log_temperature)
  - `temperature` property
  - `freeze/unfreeze/freeze_label_embeddings`
  - `init_label_embeddings_from_encoder` (zero-shot)
  - `forward` (cosine similarity, sigmoid/softmax)
  - `get_label_similarities`

**File:** `familyos_ultrabert/models/heads.py`
**Location:** Before line 3685 (before exports section)
**Lines to add:** ~250 lines

**Key Implementation Details:**

```python
class LabelDescriptionHead(nn.Module):
    INTENT_LABELS = ["log_memory", "query_memory", "set_reminder", ...]
    INGRESS_LABELS = ["DIARY", "TASK", "HEALTH", ...]

    def __init__(self, hidden_size=768, num_labels=8, projection_dim=768,
                 temperature=0.07, multi_label=False, dropout=0.1, label_names=None):
        # Query projection: [CLS] -> projection space
        self.query_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )
        # Learnable label embeddings
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, projection_dim) * 0.02)
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
```

---

**Issue 2.1.2: Port IntentHeadV2 class**

- [x] Copy class from `src/modeling_studio/models/heads.py` lines 3935-4033
- [x] Include `INTENT_DESCRIPTIONS` for zero-shot init
- [x] Include `init_from_descriptions` method

**File:** `familyos_ultrabert/models/heads.py`
**Location:** After LabelDescriptionHead
**Lines to add:** ~100 lines

**Key Details:**

- Inherits from `LabelDescriptionHead`
- 8 labels, multi_label=True by default
- Adds `intent_primary`, `intent_scores`, `low_confidence_mask` to output

---

**Issue 2.1.3: Port IngressHeadV2 class**

- [x] Copy class from `src/modeling_studio/models/heads.py` lines 4036-4145
- [x] Include `INGRESS_DESCRIPTIONS` for zero-shot init
- [x] Include `init_from_descriptions` method

**File:** `familyos_ultrabert/models/heads.py`
**Location:** After IntentHeadV2
**Lines to add:** ~100 lines

**Key Details:**

- Inherits from `LabelDescriptionHead`
- 12 labels, multi_label=True by default
- Adds `ingress_primary`, `ingress_scores`, `low_confidence_mask` to output

---

**Issue 2.1.4: Update **all** exports**

- [x] Add `"LabelDescriptionHead"` to `__all__`
- [x] Add `"IntentHeadV2"` to `__all__`
- [x] Add `"IngressHeadV2"` to `__all__`
- [x] Add `"create_label_description_head"` factory function

**File:** `familyos_ultrabert/models/heads.py`
**Location:** Line 3685-3700

---

#### Epic 2.2: Model Integration

**Issue 2.2.1: Update head imports in modernbert_multitask.py**

- [x] Add imports for new head classes

**File:** `familyos_ultrabert/models/modernbert_multitask.py`
**Location:** Line 71-78 (imports section)

```python
from familyos_ultrabert.models.heads import (
    # ... existing imports ...
    IntentHeadV2,      # NEW
    IngressHeadV2,     # NEW
    LabelDescriptionHead,  # NEW
)
```

---

**Issue 2.2.2: Update CAPABILITY_TO_HEAD_TYPE mapping**

- [x] Add `Capability.INTENT_V2: IntentHeadV2` mapping
- [x] Add `Capability.INGRESS_V2: IngressHeadV2` mapping

**File:** `familyos_ultrabert/models/modernbert_multitask.py`
**Location:** Line 140-156

```python
CAPABILITY_TO_HEAD_TYPE: dict[Capability, type[nn.Module]] = {
    # ... existing ...
    Capability.INTENT: IntentHead,
    Capability.INTENT_V2: IntentHeadV2,      # NEW
    Capability.INGRESS_V2: IngressHeadV2,    # NEW
}
```

---

**Issue 2.2.3: Update TASK_GROUPS configuration**

- [x] Added V2 heads to `"sequence_tasks"` group

**File:** `familyos_ultrabert/models/modernbert_multitask.py`
**Location:** Line 104-117

```python
TASK_GROUPS = {
    "sequence_tasks": [
        # ... existing ...
        Capability.INTENT,
        Capability.INTENT_V2,      # NEW
        Capability.INGRESS_V2,     # NEW
    ],
}
```

---

**Issue 2.2.4: Update _init_heads() for V2 heads**

- [x] Added special case handling for V2 heads (no problem_type arg)
- [x] Added multi_label=True for K1 requirement

**File:** `familyos_ultrabert/models/modernbert_multitask.py`

---

**Issue 2.2.5: Update familyos_ultrabert/data/labels.py**

- [x] Added INTENT_V2_LABELS and INGRESS_V2_LABELS schemas
- [x] Added Capability.INTENT_V2 and Capability.INGRESS_V2 to enum
- [x] Added V2 to CAPABILITY_TO_LABELS mapping

**Note:** This file is SEPARATE from `familyos_ultrabert/labels.py` - both needed updating.

---

### Milestone 3: Inference Layer

> Add multi-label postprocessing for V2 heads

**Status:** VERIFIED (2026-02-01)
**Estimated Effort:** 1 day
**Dependencies:** Milestone 2
**Verification:** Import test passed, mock data test passed with K1-compliant output

#### Epic 3.1: PyTorch Inference

**Issue 3.1.1: Add _postprocess_label_description function**

- [x] Create new function for multi-label V2 heads
- [x] Return K1-compliant format: `{primary, all, scores}` for intent, `{domains, scores}` for ingress

**File:** `familyos_ultrabert/pytorch_inference.py`
**Location:** After line 220 (_postprocess_safety)

```python
def _postprocess_label_description_intent(
    logits: torch.Tensor,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Multi-label intent classification with K1-compliant output."""
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    scores = {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    # Primary = highest score (always returned)
    primary_idx = int(np.argmax(probs))
    primary = schema.id2label[primary_idx]

    # All = labels above threshold
    all_labels = [schema.id2label[i] for i, p in enumerate(probs) if p >= threshold]

    return {
        "primary": primary,
        "all": all_labels,
        "scores": scores,
        "confidence": round(float(probs[primary_idx]), 4),
    }

def _postprocess_label_description_ingress(
    logits: torch.Tensor,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Multi-label ingress classification with K1-compliant output."""
    probs = torch.sigmoid(logits[0]).cpu().numpy()
    scores = {schema.id2label[i]: round(float(p), 4) for i, p in enumerate(probs)}

    # Domains = all labels above threshold
    domains = [schema.id2label[i] for i, p in enumerate(probs) if p >= threshold]

    return {
        "domains": domains,
        "scores": scores,
    }
```

---

**Issue 3.1.2: Update postprocess() routing**

- [x] Add routing for `intent_v2` and `ingress_v2`

**File:** `familyos_ultrabert/pytorch_inference.py`
**Location:** Line 323-338 (postprocess function)

```python
def postprocess(capability: str, logits: torch.Tensor, ...):
    # ... existing code ...

    elif capability == "intent_v2":
        return _postprocess_label_description_intent(logits, schema, threshold or 0.5)
    elif capability == "ingress_v2":
        return _postprocess_label_description_ingress(logits, schema, threshold or 0.5)
    else:
        return _postprocess_sequence_single(logits, schema)
```

---

**Issue 3.1.3: Add DEFAULT_THRESHOLDS for V2 heads**

- [x] Add default threshold values for V2 heads

**File:** `familyos_ultrabert/pytorch_inference.py`
**Location:** Line 35-45 (DEFAULT_THRESHOLDS dict)

```python
DEFAULT_THRESHOLDS = {
    "ner_general": -1.0,
    "ner_family": -0.7,
    "temporal": -1.9,
    "intent_v2": 0.5,      # NEW
    "ingress_v2": 0.5,     # NEW
}
```

---

#### Epic 3.2: ONNX Inference

**Issue 3.2.1: Add _postprocess_label_description functions (numpy)**

- [x] Mirror PyTorch functions using numpy instead of torch

**File:** `familyos_ultrabert/onnx_inference.py`
**Location:** After line 170 (_postprocess_safety)

```python
def _postprocess_label_description_intent(
    logits: np.ndarray,
    schema: LabelSchema,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Multi-label intent classification with K1-compliant output."""
    probs = _sigmoid(logits[0])
    # ... same logic as PyTorch version but with numpy ...
```

---

**Issue 3.2.2: Update postprocess() routing**

- [x] Add routing for `intent_v2` and `ingress_v2`

**File:** `familyos_ultrabert/onnx_inference.py`
**Location:** Line 261-291 (postprocess function)

---

**Issue 3.2.3: Add DEFAULT_THRESHOLDS for V2 heads**

- [x] Add default threshold values

**File:** `familyos_ultrabert/onnx_inference.py`
**Location:** Line 40-50 (DEFAULT_THRESHOLDS dict)

---

### Milestone 4: Client API Layer

> Expose V2 capabilities through public Client API

**Status:** VERIFIED (2026-02-01)
**Estimated Effort:** 1 day
**Dependencies:** Milestone 3
**Verification:** Import test passed, properties and methods work correctly

#### Epic 4.1: Client Methods

**Issue 4.1.1: Add get_intent_v2() method**

- [x] Add convenience method returning multi-label dict

**File:** `familyos_ultrabert/client.py`
**Location:** After line 347 (after get_intent)

```python
def get_intent_v2(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Multi-label intent classification (V2).

    Returns:
        Dict with:
        - primary: str - highest scoring intent
        - all: List[str] - all intents above threshold
        - scores: Dict[str, float] - all intent scores
        - confidence: float - confidence of primary
    """
    result = self.analyze(text, capabilities=["intent_v2"])
    return result._caps.get("intent_v2", {})
```

---

**Issue 4.1.2: Add get_ingress_v2() method**

- [x] Add convenience method returning multi-label dict

**File:** `familyos_ultrabert/client.py`
**Location:** After get_intent_v2

```python
def get_ingress_v2(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Multi-label domain classification (V2).

    Returns:
        Dict with:
        - domains: List[str] - all domains above threshold
        - scores: Dict[str, float] - all domain scores
    """
    result = self.analyze(text, capabilities=["ingress_v2"])
    return result._caps.get("ingress_v2", {})
```

---

#### Epic 4.2: ClientResult Properties

**Issue 4.2.1: Add intent_v2 properties**

- [x] Add properties: `intent_v2_primary`, `intent_v2_all`, `intent_v2_scores`, `intent_v2_confidence`

**File:** `familyos_ultrabert/client.py`
**Location:** After line 766 (after intent properties)

```python
# Intent V2 (multi-label)
@property
def intent_v2_primary(self) -> str:
    """Primary intent (highest score)."""
    return self._caps.get("intent_v2", {}).get("primary", "unknown")

@property
def intent_v2_all(self) -> List[str]:
    """All intents above threshold."""
    return self._caps.get("intent_v2", {}).get("all", [])

@property
def intent_v2_scores(self) -> Dict[str, float]:
    """All intent scores."""
    return self._caps.get("intent_v2", {}).get("scores", {})
```

---

**Issue 4.2.2: Add ingress_v2 properties**

- [x] Add properties: `ingress_v2_domains`, `ingress_v2_scores`

**File:** `familyos_ultrabert/client.py`
**Location:** After intent_v2 properties

```python
# Ingress V2 (multi-label)
@property
def ingress_v2_domains(self) -> List[str]:
    """All domains above threshold."""
    return self._caps.get("ingress_v2", {}).get("domains", [])

@property
def ingress_v2_scores(self) -> Dict[str, float]:
    """All domain scores."""
    return self._caps.get("ingress_v2", {}).get("scores", {})
```

---

**Issue 4.2.3: Update to_dict() method**

- [x] Add V2 fields to output dict

**File:** `familyos_ultrabert/client.py`
**Location:** Line 836-850 (to_dict method)

```python
def to_dict(self) -> Dict[str, Any]:
    return {
        # ... existing ...
        "intent": self.intent,
        "intent_v2": {
            "primary": self.intent_v2_primary,
            "all": self.intent_v2_all,
            "scores": self.intent_v2_scores,
        },
        "ingress": self.ingress,
        "ingress_v2": {
            "domains": self.ingress_v2_domains,
            "scores": self.ingress_v2_scores,
        },
    }
```

---

### Milestone 5: Documentation & Examples

> Update docs and add usage examples

**Status:** Not Started
**Estimated Effort:** 0.5 days
**Dependencies:** Milestone 4

#### Epic 5.1: API Documentation

**Issue 5.1.1: Update API.md capabilities section**

- [ ] Add `intent_v2` capability documentation
- [ ] Add `ingress_v2` capability documentation
- [ ] Document output format differences from V1

**File:** `familyos_ultrabert/API.md`
**Location:** After intent/ingress documentation

```markdown
### Intent V2 (Multi-Label)

Multi-label intent classification using label-description embeddings.

**Output:**
| Field | Type | Description |
|-------|------|-------------|
| `primary` | `str` | Highest scoring intent |
| `all` | `List[str]` | All intents above threshold |
| `scores` | `Dict[str, float]` | All intent scores |
| `confidence` | `float` | Confidence of primary |

**Labels:** log_memory, query_memory, set_reminder, express_feeling,
seek_advice, share_news, reflect, other

**Example:**
```python
result = client.analyze(text, capabilities=["intent_v2"])
print(result.intent_v2_primary)  # "log_memory"
print(result.intent_v2_all)      # ["log_memory", "reflect"]
```

```

---

**Issue 5.1.2: Update ClientResult documentation**

- [ ] Add V2 properties to ClientResult attribute table

**File:** `familyos_ultrabert/API.md`
**Location:** ClientResult Class section (line ~160)

---

#### Epic 5.2: Examples

**Issue 5.2.1: Update basic_usage.py**

- [ ] Add V2 intent/ingress examples

**File:** `familyos_ultrabert/examples/basic_usage.py`
**Location:** After line 50

```python
# V2 Multi-label intent
intent_v2 = client.get_intent_v2(text)
print(f"Primary intent: {intent_v2['primary']}")
print(f"All intents: {intent_v2['all']}")

# V2 Multi-label ingress
ingress_v2 = client.get_ingress_v2(text)
print(f"Domains: {ingress_v2['domains']}")
```

---

**Issue 5.2.2: Add V2-specific example file**

- [ ] Create `familyos_ultrabert/examples/intent_ingress_v2.py`
- [ ] Show multi-label use cases
- [ ] Show threshold configuration

---

## File Touchpoints Summary

### Training Repo (`src/modeling_studio`)

| File | Milestone | Changes | Lines Added |
|------|-----------|---------|-------------|
| `src/modeling_studio/data/labels.py` | M0 | Add V2 schemas, enum, mappings | ~60 |
| **TOTAL (Training)** | | | **~60 lines** |

### Release Repo (`familyos_ultrabert`)

| File | Milestone | Changes | Lines Added |
|------|-----------|---------|-------------|
| `familyos_ultrabert/labels.py` | M1 | Add schemas, enum, mappings | ~50 |
| `familyos_ultrabert/models/heads.py` | M2 | Port 3 classes | ~450 |
| `familyos_ultrabert/models/modernbert_multitask.py` | M2 | Add imports, mappings | ~20 |
| `familyos_ultrabert/pytorch_inference.py` | M3 | Add postprocessing | ~60 |
| `familyos_ultrabert/onnx_inference.py` | M3 | Mirror PyTorch | ~60 |
| `familyos_ultrabert/client.py` | M4 | Add methods, properties | ~80 |
| `familyos_ultrabert/API.md` | M5 | Documentation | ~100 |
| `familyos_ultrabert/examples/basic_usage.py` | M5 | Add examples | ~20 |
| **TOTAL (Release)** | | | **~840 lines** |

### Grand Total: ~900 lines across both repos

---

## Technical Details

### Source Files (Training Repo)

| File | Lines | Content |
|------|-------|---------|
| `src/modeling_studio/models/heads.py` | 3685-4150 | LabelDescriptionHead, IntentHeadV2, IngressHeadV2 |
| `src/modeling_studio/data/labels.py` | 550-720 | INTENT_LABELS, INGRESS_LABELS (single-label - need V2) |

### Label Schemas

**Intent V2 Labels (8):**

| ID | Label | Description |
|----|-------|-------------|
| 0 | log_memory | Record a memory, thought, or experience |
| 1 | query_memory | Retrieve or search past memories |
| 2 | set_reminder | Set a reminder, alarm, or scheduled task |
| 3 | express_feeling | Share emotions or feelings |
| 4 | seek_advice | Ask for guidance or recommendations |
| 5 | share_news | Share news, updates, or events |
| 6 | reflect | Reflect on past experiences or contemplate |
| 7 | other | General conversation or unclear intent |

**Ingress V2 Labels (12):**

| ID | Label | Description |
|----|-------|-------------|
| 0 | DIARY | Personal journal entries, daily reflections |
| 1 | TASK | To-do items, reminders, action items |
| 2 | HEALTH | Medical, wellness, fitness, mental health |
| 3 | FINANCE | Money, bills, budgets, expenses |
| 4 | RELATIONSHIP | Family, friends, social connections |
| 5 | WORK | Professional, career, job-related |
| 6 | META | Questions about the app, system, features |
| 7 | MEMORY | Recalling past events, nostalgia |
| 8 | PLANNING | Future plans, goals, scheduling |
| 9 | CELEBRATION | Achievements, milestones, happy events |
| 10 | CONCERN | Worries, problems, issues |
| 11 | GRATITUDE | Thanks, appreciation, positive acknowledgment |

### Output Format Comparison

**V1 (Single-Label):**

```python
{
    "prediction": "log_memory",
    "confidence": 0.92,
    "scores": {"log_memory": 0.92, "query_memory": 0.03, ...}
}
```

**V2 (Multi-Label) - Intent:**

```python
{
    "primary": "log_memory",
    "all": ["log_memory", "reflect"],
    "scores": {"log_memory": 0.92, "reflect": 0.71, ...},
    "confidence": 0.92
}
```

**V2 (Multi-Label) - Ingress:**

```python
{
    "domains": ["DIARY", "MEMORY"],
    "scores": {"DIARY": 0.88, "MEMORY": 0.65, ...}
}
```

---

## Testing Checklist

### Unit Tests

- [ ] `test_labels.py`: Test V2 label schemas
- [ ] `test_heads.py`: Test LabelDescriptionHead, IntentHeadV2, IngressHeadV2
- [ ] `test_inference.py`: Test postprocessing functions
- [ ] `test_client.py`: Test V2 methods and properties

### Integration Tests

- [ ] End-to-end analyze() with V2 capabilities
- [ ] Backward compatibility with V1 intent/ingress
- [ ] ONNX export and inference

### Validation

- [ ] Zero-shot initialization from descriptions
- [ ] Threshold sensitivity analysis
- [ ] Multi-label overlap analysis

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing intent/ingress users | HIGH | Keep V1 as default, V2 opt-in |
| ONNX export issues with new architecture | MEDIUM | Test early, may need tracing changes |
| Threshold calibration | LOW | Provide tunable defaults |

---

## Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| Add V2 as separate capabilities (not replace V1) | Backward compatibility | 2026-01-31 |
| Keep V2 in "sequence_tasks" group | Simplicity, no adapter changes needed | 2026-01-31 |
| Default threshold = 0.5 | Standard sigmoid cutoff, tunable | 2026-01-31 |

---

## Notes

- Training data already prepared: `data/processed/intent_unified/` and `data/processed/ingress_unified/`
- Training script extended: `scripts/training/train_globalpointer_unified.py`
- Test training successful with 5 heads (3 NER + 2 classification)
