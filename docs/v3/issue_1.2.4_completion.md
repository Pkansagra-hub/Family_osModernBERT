# Issue 1.2.4 Completion: Hub Token Pooler

**Status:** ✅ **COMPLETE**
**Date:** December 4, 2025
**Effort:** 3 hours (as estimated)
**Tests:** 10 new tests, 89 total passing

---

## Summary

Implemented hub token pooling for ModernBERT v3.3 Ultra that extracts representations at positions 0-4 ([CLS] and 4 hub tokens) for routing to capability-specific heads.

---

## Implementation

### Files Created

1. **`src/modeling_studio/models/poolers_v3.py`** (216 lines)
   - `HubTokenPooler`: Extracts hub token representations from final hidden states
   - `CombinedPooler`: Provides CLS, Mean, and Hub token pooling in one forward pass

### Key Features

#### HubTokenPooler

```python
class HubTokenPooler(nn.Module):
    """
    Extracts hub token representations from the final hidden states.

    Given sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text> [SEP] [PAD]...
    Returns dict of hub token representations for routing to heads.
    """
```

**Features:**
- Extracts representations at fixed positions (0-4)
- Optional projection layers (Linear + Tanh) for each hub token
- `get_pooled_for_capability(capability)`: Routes capability to correct hub

**Usage:**
```python
pooler = HubTokenPooler(hidden_size=768, add_projection=False)
hidden_states = torch.randn(2, 128, 768)  # [batch, seq, hidden]
pooled = pooler(hidden_states)
# pooled = {"[CLS]": ..., "[EMO]": ..., "[MEM]": ..., "[REL]": ..., "[TASK]": ...}
```

#### CombinedPooler

```python
class CombinedPooler(nn.Module):
    """
    Combined pooler that provides CLS, Mean, and Hub token pooling.
    """
```

**Features:**
- Extracts all hub tokens via `HubTokenPooler`
- CLS projection with Tanh activation (BERT-style)
- Mean pooling that **excludes positions 0-4** (only text tokens)
- Proper padding mask handling

**Key Innovation: Special Token Masking**
```python
# Mask out CLS and hub tokens from mean pooling
mean_mask = attention_mask.clone()
mean_mask[:, :5] = 0  # Zero out [CLS] and 4 hub positions

# Compute mean only over text tokens
mask_expanded = mean_mask.unsqueeze(-1).float()
sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
pooled["mean"] = sum_hidden / sum_mask
```

This ensures mean pooling aggregates only actual text, not special tokens.

---

## Tests Added (10 new tests)

All tests added to `tests/v3/test_hub_tokens.py`:

### HubTokenPooler Tests (5 tests)

1. **`test_hub_token_pooler_basic`**
   - ✅ Verifies all hub tokens extracted
   - ✅ Checks correct output shapes [batch, hidden]

2. **`test_hub_token_pooler_extracts_correct_positions`**
   - ✅ **Acceptance Criterion 1:** Correctly extracts representations at positions 0-4
   - Sets unique values at each position and verifies correct extraction

3. **`test_hub_token_pooler_with_projection`**
   - ✅ **Acceptance Criterion 4:** Optional projection layer works correctly
   - Verifies projections are created and applied
   - Confirms projected values differ from raw values

4. **`test_hub_token_pooler_get_pooled_for_capability`**
   - ✅ **Acceptance Criterion 2:** `get_pooled_for_capability()` returns correct hub for each capability
   - Tests all 12 capabilities:
     - EMO hub: emotions, sentiment, safety_generic, safety_familyos
     - MEM hub: embedding
     - REL hub: nli, relation
     - TASK hub: intent, ingress
     - CLS fallback: ner_general, ner_family, temporal (token-level)

5. **`test_hub_token_pooler_variable_sequence_length`**
   - ✅ **Acceptance Criterion 5:** Handles variable sequence lengths
   - Tests seq_len ∈ {32, 64, 128, 256, 512}

### CombinedPooler Tests (5 tests)

6. **`test_combined_pooler_basic`**
   - Verifies all pooled representations present
   - Checks output shapes

7. **`test_combined_pooler_mean_excludes_cls_and_hub`**
   - ✅ **Acceptance Criterion 3:** Mean pooling excludes CLS and hub tokens
   - Sets positions 0-4 to 999.0, positions 5+ to 1.0
   - Verifies mean ≈ 1.0 (positions 0-4 excluded)

8. **`test_combined_pooler_mean_with_padding`**
   - Tests mean pooling with padding tokens
   - Verifies correct masking of padding + special tokens

9. **`test_combined_pooler_mean_without_attention_mask`**
   - Tests fallback behavior when attention_mask is None
   - Simple mean over positions 5+

10. **`test_combined_pooler_cls_projection`**
    - Verifies CLS projection applies Tanh
    - Checks output range ∈ [-1, 1]

---

## Acceptance Criteria Verification

| Criterion | Status | Verified By |
|-----------|--------|-------------|
| ✅ Correctly extracts representations at positions 0-4 | **PASS** | `test_hub_token_pooler_extracts_correct_positions` |
| ✅ `get_pooled_for_capability()` returns correct hub for each capability | **PASS** | `test_hub_token_pooler_get_pooled_for_capability` |
| ✅ Mean pooling excludes CLS and hub tokens | **PASS** | `test_combined_pooler_mean_excludes_cls_and_hub` |
| ✅ Optional projection layer works correctly | **PASS** | `test_hub_token_pooler_with_projection` |
| ✅ Handles variable sequence lengths | **PASS** | `test_hub_token_pooler_variable_sequence_length` |

---

## Test Results

```
pytest tests/v3/test_hub_tokens.py --tb=line -q

Results (16.83s):
      89 passed  ✅
```

**Breakdown:**
- 79 existing tests (Issues 1.2.1, 1.2.2, 1.2.3)
- **10 new pooler tests (Issue 1.2.4)**
- All passing ✅

---

## Integration with v3 Architecture

### Position Mapping

```
Sequence: [CLS] [EMO] [MEM] [REL] [TASK] <text tokens...> [SEP] [PAD]...
Position:   0     1     2     3     4      5+
```

### Capability Routing

| Capability | Hub Token | Position | Pooler Method |
|------------|-----------|----------|---------------|
| emotions | `[EMO]` | 1 | `get_pooled_for_capability("emotions")` |
| sentiment | `[EMO]` | 1 | `get_pooled_for_capability("sentiment")` |
| safety_generic | `[EMO]` | 1 | `get_pooled_for_capability("safety_generic")` |
| safety_familyos | `[EMO]` | 1 | `get_pooled_for_capability("safety_familyos")` |
| embedding | `[MEM]` | 2 | `get_pooled_for_capability("embedding")` |
| nli | `[REL]` | 3 | `get_pooled_for_capability("nli")` |
| relation | `[REL]` | 3 | `get_pooled_for_capability("relation")` |
| intent | `[TASK]` | 4 | `get_pooled_for_capability("intent")` |
| ingress | `[TASK]` | 4 | `get_pooled_for_capability("ingress")` |
| ner_general | `[CLS]` | 0 | Token-level (full sequence) |
| ner_family | `[CLS]` | 0 | Token-level (full sequence) |
| temporal | `[CLS]` | 0 | Token-level (full sequence) |

### Usage in Model Forward Pass

```python
from modeling_studio.models.poolers_v3 import HubTokenPooler, CombinedPooler

class ModernBERTv3Ultra(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ModernBERTEncoderV3(config)
        self.hub_pooler = HubTokenPooler(config.hidden_size)
        # or
        self.combined_pooler = CombinedPooler(config.hidden_size)

    def forward(self, input_ids, attention_mask, capability=None):
        hidden_states = self.encoder(input_ids, attention_mask)

        if capability:
            # Route to specific capability head
            pooled = self.hub_pooler.get_pooled_for_capability(
                hidden_states, capability
            )
            logits = self.heads[capability](pooled)
        else:
            # Get all pooled representations
            pooled = self.combined_pooler(hidden_states, attention_mask)

        return pooled
```

---

## Design Notes

### Why Separate HubTokenPooler and CombinedPooler?

1. **Modularity:** Different use cases
   - `HubTokenPooler`: When you only need hub tokens
   - `CombinedPooler`: When you need all pooling strategies

2. **Efficiency:** Avoid unnecessary computation
   - Don't compute mean pooling if you only need hub tokens

3. **Flexibility:** Easy to add more pooling strategies
   - Future: Max pooling over hub tokens
   - Future: Attention-weighted hub pooling

### Special Token Masking Strategy

**Problem:** Standard mean pooling would include [CLS] and hub tokens, diluting the text representation.

**Solution:** Explicitly zero out positions 0-4 in the attention mask before computing mean:

```python
mean_mask = attention_mask.clone()
mean_mask[:, :5] = 0  # Exclude [CLS] + 4 hub tokens
```

**Why positions 0-4?**
- Position 0: [CLS] (sequence classification token)
- Positions 1-4: [EMO], [MEM], [REL], [TASK] (hub tokens)
- Position 5+: Actual text tokens + [SEP]

This ensures the mean representation is **purely derived from text content**.

---

## Next Steps

### Issue 1.2.5: Hub-to-Capability Routing

With poolers implemented, the next step is to create the routing logic that:
1. Maps each capability to its hub token
2. Handles token-level vs sequence-level tasks
3. Integrates with the multi-head architecture

**File:** `src/modeling_studio/models/routing_v3.py`

**Key components:**
- `HubRouter`: Routes hub representations to capability heads
- `CapabilityHead`: Wrapper that handles both routing types
- Gradient masking for selective hub training

---

## Conclusion

Issue 1.2.4 is **complete** with all acceptance criteria met:

✅ Hub token extraction at correct positions
✅ Capability-aware pooling
✅ Mean pooling excludes special tokens
✅ Optional projection layers
✅ Variable sequence length support

**Test Coverage:** 10 new tests, 89 total passing
**Integration:** Ready for use in ModernBERT v3.3 Ultra forward pass
**Next:** Issue 1.2.5 (Hub-to-Capability Routing)
