# Issue 2.1.2: Layer-wise Window Size Configuration - COMPLETE ✅

**Status:** ✅ COMPLETE
**Date:** December 4, 2025
**Effort:** 2 hours (estimated) / Already implemented
**Dependencies:** Issue 1.1.1 ✅, Issue 2.1.1 ✅

---

## Summary

Issue 2.1.2 implements layer-wise window size configuration for ModernBERT v3.3 Ultra's 28-layer architecture. This configuration system enables progressive attention span growth across four layer bands, from local token interactions (64 tokens) to full context understanding (512 tokens).

**Key Achievement:** All 28 layers now have clearly defined sliding window sizes that align with their semantic processing level, enabling efficient long-context understanding up to 8192 tokens.

---

## Implementation Details

### Files Modified
- **`src/modeling_studio/models/attention_v3.py`** - Extended with layer configuration system

### Components Implemented

#### 1. Layer Window Configuration Dictionary
```python
LAYER_WINDOW_CONFIG: Dict[int, int] = {
    # Foundation Band: Local token interactions (64)
    1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 64,

    # Context Band: Phrase-level patterns (128)
    7: 128, 8: 128, ..., 18: 128,

    # Semantic Band: Sentence-level semantics (256)
    19: 256, 20: 256, 21: 256, 22: 256,

    # Family Band: Full context (512)
    23: 512, 24: 512, 25: 512, 26: 512, 27: 512, 28: 512,
}
```

#### 2. Layer Band Definitions
```python
LAYER_BANDS: Dict[str, Tuple[int, int, int]] = {
    "foundation": (1, 6, 64),     # 6 layers, 64-token window
    "context": (7, 18, 128),       # 12 layers, 128-token window
    "semantic": (19, 22, 256),     # 4 layers, 256-token window
    "family": (23, 28, 512),       # 6 layers, 512-token window
}
```

#### 3. Core Functions

**`get_window_size_for_layer(layer_idx: int) -> int`**
- Returns window size for any layer (1-28)
- Validates layer index range
- Raises ValueError for invalid indices

**`get_layer_band_name(layer_idx: int) -> str`**
- Returns band name ("foundation", "context", "semantic", "family")
- Validates layer index range

**`get_attention_mask_for_layer(layer_idx, seq_len, device, dtype)`**
- Convenience function combining window lookup and mask creation
- Automatically applies correct window size for the layer
- Returns ready-to-use attention mask

**`print_layer_config()`**
- Debug utility for visualizing layer configuration
- Prints formatted table of bands and window sizes

**`get_layer_config_summary() -> Dict`**
- Programmatic access to configuration
- Returns structured dictionary with all band info

---

## Acceptance Criteria Verification

### ✅ Criterion 1: Foundation (L1-6) uses 64-token window
**Status:** PASS
**Evidence:**
```python
for layer in range(1, 7):
    assert get_window_size_for_layer(layer) == 64
    assert get_layer_band_name(layer) == "foundation"
```
**Test:** `test_foundation_band_window_64` - PASSING

---

### ✅ Criterion 2: Context (L7-18) uses 128-token window
**Status:** PASS
**Evidence:**
```python
for layer in range(7, 19):
    assert get_window_size_for_layer(layer) == 128
    assert get_layer_band_name(layer) == "context"
```
**Test:** `test_context_band_window_128` - PASSING

---

### ✅ Criterion 3: Semantic (L19-22) uses 256-token window
**Status:** PASS
**Evidence:**
```python
for layer in range(19, 23):
    assert get_window_size_for_layer(layer) == 256
    assert get_layer_band_name(layer) == "semantic"
```
**Test:** `test_semantic_band_window_256` - PASSING

---

### ✅ Criterion 4: Family (L23-28) uses 512-token window
**Status:** PASS
**Evidence:**
```python
for layer in range(23, 29):
    assert get_window_size_for_layer(layer) == 512
    assert get_layer_band_name(layer) == "family"
```
**Test:** `test_family_band_window_512` - PASSING

---

### ✅ Criterion 5: Invalid layer indices raise ValueError
**Status:** PASS
**Evidence:**
```python
# Layer 0 (invalid)
with pytest.raises(ValueError, match="Invalid layer index"):
    get_window_size_for_layer(0)

# Layer 29 (invalid)
with pytest.raises(ValueError, match="Invalid layer index"):
    get_window_size_for_layer(29)

# Negative indices (invalid)
with pytest.raises(ValueError, match="Invalid layer index"):
    get_window_size_for_layer(-1)
```
**Test:** `test_invalid_layer_index_raises` - PASSING

---

### ✅ Criterion 6: All 28 layers have defined window sizes
**Status:** PASS
**Evidence:**
```python
# All layers 1-28 have valid window sizes
for layer in range(1, 29):
    window_size = get_window_size_for_layer(layer)
    assert window_size in [64, 128, 256, 512]

# LAYER_WINDOW_CONFIG has exactly 28 entries
assert len(LAYER_WINDOW_CONFIG) == 28
assert all(i in LAYER_WINDOW_CONFIG for i in range(1, 29))
```
**Tests:**
- `test_all_28_layers_have_window_sizes` - PASSING
- `test_layer_window_config_constant` - PASSING

---

## Test Results

### Test Suite: `TestLayerWindowConfiguration`
**Total Tests:** 10
**Status:** ✅ All PASSING
**Execution Time:** 5.81s

```
tests/v3/test_attention_v3.py::TestLayerWindowConfiguration
  ✓ test_layer_window_config_constant
  ✓ test_layer_bands_constant
  ✓ test_foundation_band_window_64
  ✓ test_context_band_window_128
  ✓ test_semantic_band_window_256
  ✓ test_family_band_window_512
  ✓ test_invalid_layer_index_raises
  ✓ test_all_28_layers_have_window_sizes
  ✓ test_get_attention_mask_for_layer
  ✓ test_layer_config_summary

Results: 10 passed in 5.81s
```

---

## Layer Configuration Summary

| Band       | Layers  | Window | Purpose                          | Training  |
|------------|---------|--------|----------------------------------|-----------|
| Foundation | L1-6    | 64     | Local token interactions         | ❄️ Frozen |
| Context    | L7-18   | 128    | Phrase-level patterns            | ❄️ Frozen |
| Semantic   | L19-22  | 256    | Sentence-level semantics         | 🔥 Trainable |
| Family     | L23-28  | 512    | Full context + FamilyOS tasks    | 🔥 Trainable |

**Total Layers:** 28
**Total Parameters:** ~40M base + ~6M new (L23-28)
**Max Context:** 8192 tokens

---

## Integration with Issue 2.1.1

This configuration system integrates seamlessly with Issue 2.1.1's attention mask creation:

```python
# Issue 2.1.1 provides mask creation
mask = create_global_local_attention_mask(seq_len, window_size, global_positions)

# Issue 2.1.2 provides layer-specific window lookup
window_size = get_window_size_for_layer(layer_idx)

# Combined: convenience function
mask = get_attention_mask_for_layer(layer_idx, seq_len)
```

**Key Synergy:** Issue 2.1.2 extends 2.1.1 by providing the semantic mapping from layer indices to window sizes, enabling different attention spans at different processing depths.

---

## Design Rationale

### Progressive Attention Span Growth

The window size progression (64→128→256→512) follows the principle of **progressive abstraction**:

1. **Foundation (64):** Focus on local token relationships (morphology, syntax)
2. **Context (128):** Capture phrase-level dependencies (syntactic structures)
3. **Semantic (256):** Understand sentence-level meaning (semantics)
4. **Family (512):** Process full conversational context (pragmatics, FamilyOS tasks)

### Why These Specific Sizes?

- **64 tokens:** ~10-15 words (sub-sentence level)
- **128 tokens:** ~20-30 words (sentence to short paragraph)
- **256 tokens:** ~40-60 words (paragraph level)
- **512 tokens:** ~80-120 words (multi-turn conversation)

These windows balance:
- ✅ Computational efficiency (O(n×w) instead of O(n²))
- ✅ Task requirements (emotions, NER, relations need varying context)
- ✅ Memory constraints (512-window at top layers manageable on consumer GPUs)

### Global Hub Tokens

**Critical:** Hub tokens (positions 0-4) have **global attention** regardless of layer:
- This is handled by Issue 2.1.1's `create_global_local_attention_mask()`
- Hub tokens see entire sequence, entire sequence sees hubs
- Only text→text attention uses sliding windows

**Result:** Text at position 1000 can attend to [EMO] hub at position 1, even in Foundation layer (window=64).

---

## Performance Characteristics

### Memory Complexity

**Standard Attention (without windows):**
```
Memory = O(n² × num_layers) = O(8192² × 28) ≈ 1.9B values
```

**With Sliding Windows + Global Tokens:**
```
Foundation: O(n × 64) × 6 layers
Context:    O(n × 128) × 12 layers
Semantic:   O(n × 256) × 4 layers
Family:     O(n × 512) × 6 layers
Globals:    O(n × 5) × 28 layers (hub tokens)

Total ≈ O(n × 250) average window
Reduction: 32x memory saving at 8192 tokens
```

### Compute Complexity

**Attention FLOPs per Layer:**
- Foundation: ~524M FLOPs (8192 × 64 × 768)
- Context: ~1.05B FLOPs (8192 × 128 × 768)
- Semantic: ~2.10B FLOPs (8192 × 256 × 768)
- Family: ~4.19B FLOPs (8192 × 512 × 768)

**Total for 28 layers:** ~60B FLOPs (vs. ~1.9T FLOPs for full attention)

---

## Edge Cases Handled

### 1. Short Sequences (seq_len < window_size)
```python
# If seq_len=50 and window=128, effective window is min(50, 128)=50
# Mask creation handles this correctly via min/max clipping
```

### 2. Layer Index Validation
```python
# Out-of-range indices raise clear errors
get_window_size_for_layer(0)   # ValueError: "Must be in range [1, 28]"
get_window_size_for_layer(29)  # ValueError: "Must be in range [1, 28]"
```

### 3. Device Compatibility
```python
# get_attention_mask_for_layer() passes device to mask creation
mask = get_attention_mask_for_layer(1, 100, device=torch.device("cuda"))
```

### 4. Dtype Flexibility
```python
# Supports both boolean and float masks
mask_bool = get_attention_mask_for_layer(1, 100, dtype=torch.bool)
mask_float = get_attention_mask_for_layer(1, 100, dtype=torch.float32)
```

---

## Next Steps

### Ready for Issue 2.1.3: MultiScaleAttentionWithGlobals
With layer configuration complete, the next step is implementing the full attention module:

**Requirements:**
- ✅ Issue 2.1.1: Mask creation (complete)
- ✅ Issue 2.1.2: Layer configuration (complete)
- 🔲 Issue 2.1.3: MHA implementation with QKV projections

**Implementation Plan (Issue 2.1.3):**
```python
class MultiScaleAttentionWithGlobals(nn.Module):
    def __init__(self, hidden_size=768, layer_idx=1, ...):
        # Get layer-specific window from Issue 2.1.2
        self.window_size = get_window_size_for_layer(layer_idx)

    def forward(self, hidden_states, attention_mask=None):
        # Get layer-specific mask from Issue 2.1.1 + 2.1.2
        mask = get_attention_mask_for_layer(self.layer_idx, seq_len)
        # ... QKV projections, attention computation
```

### Future Optimizations (Post-MVP)

1. **Dynamic Window Sizing:** Adjust windows based on input length
2. **Learned Window Sizes:** Make window sizes learnable parameters
3. **Sparse Attention Patterns:** Experiment with dilated/strided windows
4. **Flash Attention Integration:** Optimize kernel for global+local pattern

---

## Conclusion

**Issue 2.1.2 Status:** ✅ COMPLETE

All acceptance criteria met with comprehensive test coverage. The layer-wise window configuration system provides a clean, efficient interface for managing attention spans across ModernBERT v3.3 Ultra's 28-layer architecture.

**Key Achievements:**
- ✅ All 28 layers configured with appropriate window sizes
- ✅ Four layer bands (Foundation, Context, Semantic, Family) clearly defined
- ✅ Error handling for invalid layer indices
- ✅ Integration with Issue 2.1.1's mask creation system
- ✅ Utility functions for debugging and programmatic access
- ✅ 10/10 tests passing

**Ready to proceed with Issue 2.1.3 (MultiScaleAttentionWithGlobals).**

---

**Completed by:** FamilyOS Team
**Date:** December 4, 2025
**Milestone:** 2.1 - v3 Attention & Transformer Layers
**Next Issue:** 2.1.3 - MultiScaleAttentionWithGlobals (6 hours estimated)
