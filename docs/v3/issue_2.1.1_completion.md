# Issue 2.1.1 Completion: Global-Local Attention Mask Creation

**Status:** ✅ **COMPLETE**
**Date:** December 4, 2025
**Effort:** 4 hours (as estimated)
**Tests:** 37 tests, all passing

---

## Summary

Implemented global-local attention mask creation for ModernBERT v3.3 Ultra, solving the **"Blind Hub" problem** where hub tokens couldn't see beyond their local sliding window in long sequences.

---

## Implementation

### Files Created

1. **`src/modeling_studio/models/attention_v3.py`** (560 lines)
   - Complete attention mask creation system
   - Layer-wise window configuration
   - Utility functions for mask manipulation

### Key Components

#### 1. Global-Local Attention Mask

```python
def create_global_local_attention_mask(
    seq_len: int,
    window_size: int,
    global_positions: List[int] = [0, 1, 2, 3, 4],
    device: torch.device = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    Create attention mask with global tokens + sliding windows.

    Global tokens (positions 0-4): Can attend to ALL, are attended by ALL
    Text tokens (positions 5+): Use sliding windows + always see globals
    """
```

**Key Features:**
- Hub tokens at positions 1-4 ([EMO], [MEM], [REL], [TASK]) have global attention
- [CLS] at position 0 also gets global attention
- Text tokens use sliding windows but always attend to hub tokens
- Bidirectional: hubs see all, all see hubs

**Visual Example (seq_len=10, window=4):**
```
          0  1  2  3  4  5  6  7  8  9   (keys)
       +--------------------------------
   0   |  1  1  1  1  1  1  1  1  1  1   <- [CLS] global
   1   |  1  1  1  1  1  1  1  1  1  1   <- [EMO] global
   2   |  1  1  1  1  1  1  1  1  1  1   <- [MEM] global
   3   |  1  1  1  1  1  1  1  1  1  1   <- [REL] global
   4   |  1  1  1  1  1  1  1  1  1  1   <- [TASK] global
   5   |  1  1  1  1  1  1  1  1  0  0   <- text: globals + window
   6   |  1  1  1  1  1  1  1  1  1  0   <- text: globals + window
   7   |  1  1  1  1  1  1  1  1  1  1   <- text: globals + window
   8   |  1  1  1  1  1  0  1  1  1  1   <- text: globals + window
   9   |  1  1  1  1  1  0  0  1  1  1   <- text: globals + window
(queries)
```

#### 2. Layer-Wise Window Configuration

```python
LAYER_WINDOW_CONFIG = {
    # Foundation Band (L1-6): 64 tokens
    1: 64, 2: 64, 3: 64, 4: 64, 5: 64, 6: 64,

    # Context Band (L7-18): 128 tokens
    7: 128, 8: 128, ..., 18: 128,

    # Semantic Band (L19-22): 256 tokens
    19: 256, 20: 256, 21: 256, 22: 256,

    # Family Band (L23-28): 512 tokens
    23: 512, 24: 512, 25: 512, 26: 512, 27: 512, 28: 512,
}
```

**Layer Bands:**

| Band | Layers | Window | Purpose |
|------|--------|--------|---------|
| Foundation | L1-6 | 64 | Token-level patterns (morphology, subwords) |
| Context | L7-18 | 128 | Phrase patterns (entities, short phrases) |
| Semantic | L19-22 | 256 | Clause/sentence patterns (syntax, semantics) |
| Family | L23-28 | 512 | Full context (family dynamics, relationships) |

#### 3. Additional Functions

```python
# Causal masks (for decoder-style, if needed)
create_causal_global_local_mask(...)

# Batch expansion for multi-head attention
expand_mask_for_batch(mask, batch_size, num_heads)

# Convert boolean to additive mask
convert_mask_to_additive(mask)  # True -> 0.0, False -> -inf

# Layer configuration utilities
get_window_size_for_layer(layer_idx)
get_layer_band_name(layer_idx)
get_attention_mask_for_layer(layer_idx, seq_len)
print_layer_config()
get_layer_config_summary()

# Analysis utilities
visualize_attention_mask(mask)
count_attention_patterns(mask)
```

---

## Tests Created (37 tests, all passing)

### Test Structure

1. **TestGlobalLocalAttentionMask** (13 tests)
   - Basic mask creation
   - Global token properties
   - Text token sliding windows
   - Variable sequence lengths
   - Device and dtype handling
   - Visual example verification
   - Pattern counting

2. **TestCausalMask** (3 tests)
   - Causal mask creation
   - Causal constraints
   - Global tokens in causal mode

3. **TestLayerWindowConfiguration** (10 tests)
   - Layer window config constants
   - Band-specific window sizes
   - Invalid index handling
   - Convenience functions

4. **TestMaskUtilities** (2 tests)
   - Mask conversion
   - Batch expansion

5. **TestAttentionIntegration** (4 tests)
   - Full pipeline (mask → expand → convert)
   - Different layers produce different patterns
   - Long sequences (8192 tokens)

6. **TestBlindHubSolution** (5 tests)
   - Hub tokens see all positions
   - Text tokens see hub tokens
   - **Critical test:** Text at position 500 can see [EMO] at position 1

---

## Acceptance Criteria Verification

### ✅ All 5 Acceptance Criteria Met

#### Criterion 1: Global positions (0-4) have full row attention
**Test:** `test_global_positions_have_full_row_attention`
```python
for pos in GLOBAL_TOKEN_POSITIONS:
    row = mask[pos, :]
    assert row.all()  # All 1s
    assert row.sum() == seq_len
```
**Status:** ✅ PASS

#### Criterion 2: Global positions have full column attention
**Test:** `test_global_positions_have_full_column_attention`
```python
for pos in GLOBAL_TOKEN_POSITIONS:
    col = mask[:, pos]
    assert col.all()  # All 1s
    assert col.sum() == seq_len
```
**Status:** ✅ PASS

#### Criterion 3: Text tokens use sliding window for non-global positions
**Test:** `test_text_tokens_use_sliding_window`
```python
text_pos = 10
row = mask[text_pos, :]
# Check: attends to globals + window, not to distant tokens
```
**Status:** ✅ PASS

#### Criterion 4: Mask shape is [seq_len, seq_len] or [batch, heads, seq_len, seq_len]
**Test:** `test_mask_shape_batch_expansion`
```python
mask_2d = create_global_local_attention_mask(10, 4)
assert mask_2d.shape == (10, 10)

mask_4d = expand_mask_for_batch(mask_2d, 2, 12)
assert mask_4d.shape == (2, 12, 10, 10)
```
**Status:** ✅ PASS

#### Criterion 5: Works with variable sequence lengths
**Test:** `test_variable_sequence_lengths`
```python
for seq_len in [8, 16, 32, 64, 128, 256]:
    mask = create_global_local_attention_mask(seq_len, window_size)
    assert mask.shape == (seq_len, seq_len)
```
**Status:** ✅ PASS

---

## Blind Hub Problem - Solved ✅

### The Problem

In original design, hub tokens used sliding windows like text tokens. This meant:
- [EMO] at position 1 with 64-token window can only see positions 0-33
- In a 1000-token sequence, [EMO] is **blind** to positions 34-1000
- Hub tokens cannot aggregate information from long sequences
- Defeats the entire purpose of hub tokens

### The Solution

**Global Bidirectional Attention:**
1. Hub tokens attend to ALL positions (row = all 1s)
2. ALL positions attend to hub tokens (column = all 1s)
3. Cost: ~4 × seq_len additional attention (negligible vs N²)

### Critical Test (from implementation_plan_v3.md)

```python
def test_text_at_position_500_can_see_emo_hub():
    """
    MUST PASS: Text at position 500 can attend to [EMO] at position 1.
    """
    seq_len = 1000
    window_size = 64  # [EMO] at pos 1 is far outside window
    mask = create_global_local_attention_mask(seq_len, window_size)

    text_pos = 500
    emo_pos = 1

    assert mask[text_pos, emo_pos], "Blind Hub problem not solved!"
    assert mask[emo_pos, text_pos], "[EMO] cannot see distant text!"
```

**Result:** ✅ **PASS** - Blind Hub problem completely solved

---

## Layer Configuration Tests

### Window Size Configuration (Issue 2.1.2 partial)

All 28 layers correctly configured:

```python
def test_foundation_band_window_64():
    for layer in range(1, 7):
        assert get_window_size_for_layer(layer) == 64
        assert get_layer_band_name(layer) == "foundation"

def test_context_band_window_128():
    for layer in range(7, 19):
        assert get_window_size_for_layer(layer) == 128
        assert get_layer_band_name(layer) == "context"

def test_semantic_band_window_256():
    for layer in range(19, 23):
        assert get_window_size_for_layer(layer) == 256
        assert get_layer_band_name(layer) == "semantic"

def test_family_band_window_512():
    for layer in range(23, 29):
        assert get_window_size_for_layer(layer) == 512
        assert get_layer_band_name(layer) == "family"
```

**All tests:** ✅ PASS

---

## Integration Tests

### Long Sequence Test (8192 tokens)

```python
def test_long_sequence_8192_tokens():
    """Test with full v3 context length."""
    seq_len = 8192
    layer_idx = 28  # Family layer (512 window)

    mask = get_attention_mask_for_layer(layer_idx, seq_len)

    # Global tokens still work
    for pos in GLOBAL_TOKEN_POSITIONS:
        assert mask[pos, :].all()
        assert mask[:, pos].all()

    # Text token at position 4096
    mid_pos = 4096
    row = mask[mid_pos, :]

    # Attends to globals
    for global_pos in GLOBAL_TOKEN_POSITIONS:
        assert row[global_pos]

    # Attends within 512 window
    half_window = 512 // 2
    start = mid_pos - half_window
    end = mid_pos + half_window + 1
    for i in range(start, end):
        if 0 <= i < seq_len:
            assert row[i]
```

**Result:** ✅ PASS - Handles full 8k context

---

## Performance Characteristics

### Attention Pattern Density

For seq_len=100, window=64:
```python
stats = count_attention_patterns(mask)
# {
#     "seq_len": 100,
#     "global_tokens": 5,
#     "attended_by_all": 5,
#     "total_edges": ~6900,  # vs 10,000 for full attention
#     "density": ~0.69,      # 31% reduction
# }
```

### Cost Analysis

**Additional cost for global tokens:**
- Hub tokens: 4 × seq_len edges (hubs attend to all)
- Text to hubs: seq_len × 4 edges (all attend to hubs)
- Total: ~8 × seq_len additional edges
- vs N² for sliding window: **negligible** (0.08% overhead for seq_len=1000)

**Memory savings from sliding windows:**
- Full attention: seq_len²
- With windows: ~window_size × seq_len + 8 × seq_len
- For seq_len=8192, window=512: **93% memory reduction**

---

## Module Exports

```python
__all__ = [
    # Constants
    "GLOBAL_TOKEN_POSITIONS",
    "LAYER_WINDOW_CONFIG",
    "LAYER_BANDS",

    # Mask creation
    "create_global_local_attention_mask",
    "create_causal_global_local_mask",
    "expand_mask_for_batch",
    "convert_mask_to_additive",

    # Layer configuration
    "get_window_size_for_layer",
    "get_layer_band_name",
    "get_attention_mask_for_layer",
    "print_layer_config",
    "get_layer_config_summary",

    # Utilities
    "visualize_attention_mask",
    "count_attention_patterns",
]
```

---

## Next Steps

### Issue 2.1.2: Layer-wise Window Size Configuration
**Status:** ✅ **ALSO COMPLETE** (implemented together)
- All window size functions implemented
- All layer configuration tests passing

### Issue 2.1.3: MultiScaleAttentionWithGlobals (NEXT)
**File:** `src/modeling_studio/models/attention_v3.py` (extend)
**Effort:** 6 hours
**Dependencies:** Issues 2.1.1 ✅, 2.1.2 ✅

Ready to implement full MHA module with:
- QKV projections
- Multi-head reshape (12 heads × 64 dim)
- Global-local mask application
- Output projection

### Issue 2.1.4: Flash Attention 2 with Safety Switch
**File:** `src/modeling_studio/models/attention_v3.py` (extend)
**Effort:** 6 hours
**Dependencies:** Issue 2.1.3

---

## Test Results

```bash
pytest tests/v3/test_attention_v3.py -v

Results (8.26s):
      37 passed  ✅

Test Coverage:
- Global-local mask creation: 13 tests
- Causal masks: 3 tests
- Layer configuration: 10 tests
- Utilities: 2 tests
- Integration: 4 tests
- Blind Hub solution: 5 tests
```

---

## Conclusion

Issue 2.1.1 (and partial 2.1.2) is **complete** with all acceptance criteria met:

✅ Global positions have full row and column attention
✅ Text tokens use sliding windows correctly
✅ Mask shapes are correct for 2D and 4D
✅ Works with variable sequence lengths
✅ All 28 layers have correct window sizes
✅ **Blind Hub problem completely solved**

**Test Coverage:** 37 tests, 100% passing
**Integration:** Ready for MultiScaleAttentionWithGlobals implementation
**Documentation:** Complete with visual examples and usage patterns
