# Issue 1.2.5 Completion: Hub-to-Capability Routing

**Status:** ✅ **COMPLETE**
**Date:** December 4, 2025
**Effort:** 3 hours (as estimated)
**Tests:** 8 new tests, 97 total passing

---

## Summary

Implemented hub-to-capability routing for ModernBERT v3.3 Ultra that directs hub token representations to appropriate capability heads with automatic gradient masking for selective training.

---

## Implementation

### Files Created

1. **`src/modeling_studio/models/routing_v3.py`** (311 lines)
   - `HubRouter`: Routes hub token representations to capability heads
   - `CapabilityHead`: Wrapper for capability heads with automatic routing
   - `create_hub_routing_info()`: Utility for getting routing information
   - `print_routing_table()`: Debug utility

### Files Extended

2. **`src/modeling_studio/models/hub_tokens.py`** (added 25 lines)
   - Added `CAPABILITY_HUB_ROUTING` mapping (reference table for all 12 capabilities)

---

## Key Features

### HubRouter

```python
class HubRouter(nn.Module):
    """
    Routes hub token representations to capability heads.

    Routing table maps all 12 capabilities to (pool_type, hub_token):
    - pool_type: "hub" for sequence-level, "token" for token-level
    - hub_token: The hub providing the representation (None for token-level)
    """
```

**Features:**
- **ROUTING_TABLE**: Complete mapping of all 12 capabilities
  - 9 hub-routed capabilities (sequence-level)
  - 3 token-level capabilities (per-token)
- **get_representation_for_capability()**: Returns correct representation + type
- **get_hub_gradient_mask()**: Creates masks for selective hub training

**Routing Strategy:**

| Hub Token | Capabilities | Count |
|-----------|--------------|-------|
| `[EMO]` | emotions, sentiment, safety_generic, safety_familyos | 4 |
| `[MEM]` | embedding | 1 |
| `[REL]` | nli, relation | 2 |
| `[TASK]` | intent, ingress | 2 |
| Token-level | ner_general, ner_family, temporal | 3 |

### CapabilityHead Wrapper

```python
class CapabilityHead(nn.Module):
    """
    Wrapper for capability heads with automatic hub routing.

    Automatically determines routing based on capability:
    - Hub capabilities: Extract hub token representation
    - Token capabilities: Pass full sequence
    """
```

**Key Innovation: Automatic Routing**
```python
def forward(self, hidden_states, pooled_outputs, **kwargs):
    if self.pool_type == "token":
        # Token-level head (NER, temporal)
        return self.head(hidden_states, **kwargs)
    else:
        # Hub-routed head
        representation = pooled_outputs[self.hub_token]
        return self.head(representation, **kwargs)
```

This eliminates manual routing logic in the model forward pass.

### Gradient Masking

**Purpose:** Enable selective hub training during multi-task batches

**Example:**
```python
# Training batch with emotions + sentiment (both use [EMO])
active_caps = ["emotions", "sentiment"]
masks = router.get_hub_gradient_mask(active_caps, batch_size=4, device="cpu")

# Result:
masks["[EMO]"]  = [1.0, 1.0, 1.0, 1.0]  # Train (capabilities active)
masks["[MEM]"]  = [0.0, 0.0, 0.0, 0.0]  # Freeze (no active capabilities)
masks["[REL]"]  = [0.0, 0.0, 0.0, 0.0]  # Freeze
masks["[TASK]"] = [0.0, 0.0, 0.0, 0.0]  # Freeze
```

This ensures gradients only flow through relevant hubs, preventing interference between unrelated capabilities.

---

## Tests Added (8 new tests)

All tests added to `tests/v3/test_hub_tokens.py`:

### HubRouter Tests (5 tests)

1. **`test_hub_router_routing_table`**
   - ✅ **Acceptance Criterion 1:** All 12 capabilities correctly mapped to routing types
   - Verifies ROUTING_TABLE has all capabilities
   - Checks correct (pool_type, hub_token) tuples

2. **`test_hub_router_get_representation_for_capability_hub`**
   - ✅ **Acceptance Criterion 2:** Hub capabilities (9) route through appropriate hub tokens
   - Tests EMO, MEM, REL, TASK routing
   - Verifies correct hub representation returned

3. **`test_hub_router_get_representation_for_capability_token`**
   - ✅ **Acceptance Criterion 3:** Token-level capabilities (3) receive full sequence representations
   - Tests ner_general, ner_family, temporal
   - Verifies full hidden_states returned (not pooled)

4. **`test_hub_router_get_hub_gradient_mask`**
   - ✅ **Acceptance Criterion 4:** Gradient masks correctly identify which hubs should be trained
   - Tests single hub active, multiple hubs active, no hubs active
   - Verifies mask values (1.0 for train, 0.0 for freeze)

5. **`test_hub_router_all_capabilities_mapped`**
   - Comprehensive test verifying all 12 capabilities
   - Checks counts by hub (4+1+2+2+3=12)

### CapabilityHead Tests (2 tests)

6. **`test_capability_head_hub_routing`**
   - ✅ **Acceptance Criterion 5:** CapabilityHead wrapper correctly handles hub routing
   - Tests hub-routed capability (emotions)
   - Verifies logits shape [batch, num_labels]

7. **`test_capability_head_token_routing`**
   - ✅ **Acceptance Criterion 5:** CapabilityHead wrapper correctly handles token-level routing
   - Tests token-level capability (ner_general)
   - Verifies logits shape [batch, seq_len, num_labels]

### Utility Tests (1 test)

8. **`test_create_hub_routing_info`**
   - Tests `create_hub_routing_info()` utility
   - Verifies all 12 capabilities return correct info
   - Checks hub_description present for hub capabilities

---

## Acceptance Criteria Verification

| Criterion | Status | Verified By |
|-----------|--------|-------------|
| ✅ All 12 capabilities correctly mapped to routing types | **PASS** | `test_hub_router_routing_table` |
| ✅ Hub capabilities (9) route through appropriate hub tokens | **PASS** | `test_hub_router_get_representation_for_capability_hub` |
| ✅ Token-level capabilities (3) receive full sequence representations | **PASS** | `test_hub_router_get_representation_for_capability_token` |
| ✅ Gradient masks correctly identify which hubs should be trained | **PASS** | `test_hub_router_get_hub_gradient_mask` |
| ✅ `CapabilityHead` wrapper correctly handles both routing types | **PASS** | `test_capability_head_hub_routing` + `test_capability_head_token_routing` |

---

## Test Results

```
pytest tests/v3/test_hub_tokens.py --tb=line -q

Results (12.31s):
      97 passed  ✅
```

**Breakdown:**
- 89 existing tests (Issues 1.2.1-1.2.4)
- **8 new routing tests (Issue 1.2.5)**
- All passing ✅

---

## Integration with v3 Architecture

### Complete Routing Table

| Capability | Hub Token | Pool Type | Position | Head Input Shape |
|------------|-----------|-----------|----------|------------------|
| emotions | `[EMO]` | hub | 1 | [batch, 768] |
| sentiment | `[EMO]` | hub | 1 | [batch, 768] |
| safety_generic | `[EMO]` | hub | 1 | [batch, 768] |
| safety_familyos | `[EMO]` | hub | 1 | [batch, 768] |
| embedding | `[MEM]` | hub | 2 | [batch, 768] |
| nli | `[REL]` | hub | 3 | [batch, 768] |
| relation | `[REL]` | hub | 3 | [batch, 768] |
| intent | `[TASK]` | hub | 4 | [batch, 768] |
| ingress | `[TASK]` | hub | 4 | [batch, 768] |
| ner_general | None | token | N/A | [batch, seq_len, 768] |
| ner_family | None | token | N/A | [batch, seq_len, 768] |
| temporal | None | token | N/A | [batch, seq_len, 768] |

### Usage in Model Forward Pass

```python
from modeling_studio.models.routing_v3 import HubRouter, CapabilityHead
from modeling_studio.models.poolers_v3 import CombinedPooler

class ModernBERTv3Ultra(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ModernBERTEncoderV3(config)
        self.pooler = CombinedPooler(config.hidden_size)
        self.router = HubRouter()

        # Wrap capability heads with automatic routing
        self.heads = nn.ModuleDict({
            "emotions": CapabilityHead(
                "emotions",
                EmotionHead(config.hidden_size, num_labels=44),
                config.hidden_size
            ),
            "ner_general": CapabilityHead(
                "ner_general",
                NERHead(config.hidden_size, num_labels=17),
                config.hidden_size
            ),
            # ... other capabilities
        })

    def forward(self, input_ids, attention_mask, capability=None):
        # Encoder
        hidden_states = self.encoder(input_ids, attention_mask)

        # Pooling
        pooled_outputs = self.pooler(hidden_states, attention_mask)

        # Route to capability head
        if capability:
            # Automatic routing via CapabilityHead wrapper
            logits = self.heads[capability](hidden_states, pooled_outputs)
            return logits

        return hidden_states, pooled_outputs
```

### Training with Gradient Masking

```python
# During training
router = HubRouter()

for batch in dataloader:
    input_ids, attention_mask, labels, capabilities = batch

    # Get active capabilities in this batch
    active_caps = list(set(capabilities))  # e.g., ["emotions", "sentiment"]

    # Create gradient masks
    masks = router.get_hub_gradient_mask(
        active_caps,
        batch_size=len(input_ids),
        device=input_ids.device
    )

    # Forward pass
    hidden_states = model.encoder(input_ids, attention_mask)
    pooled = model.pooler(hidden_states, attention_mask)

    # Apply gradient masks to hub tokens
    for hub_token, mask in masks.items():
        if mask.sum() == 0:  # Hub not active
            pooled[hub_token] = pooled[hub_token].detach()

    # Compute losses for active capabilities
    losses = []
    for cap in active_caps:
        logits = model.heads[cap](hidden_states, pooled)
        loss = criterion(logits, labels[cap])
        losses.append(loss)

    total_loss = sum(losses) / len(losses)
    total_loss.backward()
```

---

## Design Notes

### Why Separate HubRouter and CapabilityHead?

1. **Separation of Concerns:**
   - `HubRouter`: Routing logic (which hub, which representation)
   - `CapabilityHead`: Head wrapping (automatic routing application)

2. **Flexibility:**
   - Can use `HubRouter` standalone for custom routing
   - Can use `CapabilityHead` for automatic integration

3. **Testing:**
   - Easier to test routing logic separately
   - Easier to test head wrapping separately

### Gradient Masking Strategy

**Problem:** During multi-task training, not all hubs are used in every batch.

**Solution:** Gradient masking ensures only active hubs receive gradients:

```python
# Batch with emotions capability
masks["[EMO]"] = [1.0, 1.0, 1.0, 1.0]  # Train
masks["[MEM]"] = [0.0, 0.0, 0.0, 0.0]  # Freeze

# Apply mask (option 1: detach)
if mask.sum() == 0:
    pooled["[MEM]"] = pooled["[MEM]"].detach()

# Apply mask (option 2: multiply)
pooled["[MEM]"] = pooled["[MEM]"] * mask.unsqueeze(-1)
```

**Benefits:**
- Prevents interference between unrelated capabilities
- Improves training stability
- Enables efficient multi-task batching

### Extension to hub_tokens.py

Added `CAPABILITY_HUB_ROUTING` dictionary to `hub_tokens.py`:
- Provides static reference mapping
- Complements dynamic `get_hub_for_capability()` function
- Useful for documentation and validation

---

## Next Steps

### Milestone 1 Status: ✅ **COMPLETE**

All 5 issues in Epic 1.2 (Hub Token System) are now complete:

- ✅ Issue 1.2.1: Hub Token Registry
- ✅ Issue 1.2.2: Hub Token Injection Tokenizer
- ✅ Issue 1.2.3: Semantic Centroid Initialization
- ✅ Issue 1.2.4: Hub Token Pooler
- ✅ Issue 1.2.5: Hub-to-Capability Routing

**Total Tests:** 97 passing (across all issues)

### Milestone 2: v3 Attention & Transformer Layers (NEXT)

**Epic 2.1: Sliding Window Attention**
- Issue 2.1.1: Global-Local Attention Mask Creation
- Issue 2.1.2: Layer-wise Window Size Configuration
- Issue 2.1.3: MultiScaleAttentionWithGlobals
- Issue 2.1.4: Flash Attention 2 Integration (Safety Switch)

---

## Conclusion

Issue 1.2.5 is **complete** with all acceptance criteria met:

✅ All 12 capabilities correctly mapped
✅ Hub capabilities route through appropriate hubs
✅ Token-level capabilities receive full sequences
✅ Gradient masks enable selective hub training
✅ CapabilityHead wrapper handles both routing types

**Test Coverage:** 8 new tests, 97 total passing
**Integration:** Ready for use in ModernBERT v3.3 Ultra
**Next:** Milestone 2 - Attention implementation
