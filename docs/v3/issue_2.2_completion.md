# Epic 2.2: Loss Function - Completion Report

## Status: COMPLETED

## Summary

Implemented the SOTA GlobalPointer loss function (Multi-Label Categorical Cross-Entropy) for span-based NER training. This loss function is specifically designed for the GlobalPointer architecture and naturally handles extreme class imbalance in span detection.

## What Was Done

### Issue 2.2.1: Implement GlobalPointerLoss class

**File**: `src/modeling_studio/models/losses.py`

Added `GlobalPointerLoss` class implementing multi-label categorical cross-entropy:

```python
class GlobalPointerLoss(nn.Module):
    """
    Multi-Label Categorical Cross-Entropy Loss for GlobalPointer NER.

    Uses logsumexp trick for stable computation of circle-loss style
    separation between positive and negative predictions.
    """
```

Key features:
- **LogSumExp aggregation**: Stable numerical computation
- **Circle-loss style separation**: Pushes positive spans high, negative spans low
- **Automatic imbalance handling**: No need for explicit pos_weight tuning
- **Padding and lower-triangular masking**: Built-in
- **Float16 support**: Dynamic mask value selection to avoid overflow

### Issue 2.2.3: Implement FocalGlobalPointerLoss variant

**File**: `src/modeling_studio/models/losses.py`

Added `FocalGlobalPointerLoss` extending base with focal weighting:

```python
class FocalGlobalPointerLoss(GlobalPointerLoss):
    """
    GlobalPointer loss with focal loss weighting for extreme imbalance.
    Adds (1-p)^gamma weighting to down-weight easy examples.
    """
```

### Issue 2.2.2: Add loss_type parameter to GlobalPointerNERHead

**File**: `src/modeling_studio/models/heads.py`

Updated `GlobalPointerNERHead.__init__` to accept `loss_type` parameter:

```python
def __init__(
    self,
    hidden_size: int = 768,
    num_labels: int = 4,
    head_size: int = 64,
    dropout: float = 0.1,
    use_rope: bool = True,
    rope_base: float = 10000.0,
    loss_type: str = "globalpointer",  # NEW: "globalpointer" | "focal_globalpointer" | "bce"
):
```

Loss type options:
- `"globalpointer"` (default): SOTA multi-label categorical CE
- `"focal_globalpointer"`: With focal loss weighting
- `"bce"`: Legacy BCE fallback

### Issue 2.2.4: Write unit tests for loss functions

**File**: `tests/unit/test_globalpointer_loss.py`

Created comprehensive test suite with 25 tests covering:

- Output shape and scalar reduction
- Gradient flow verification
- Low loss for matching predictions
- High loss for inverted predictions
- Padding mask effect
- Lower triangular masking
- Edge cases (empty labels, all positive, single sample)
- Numerical stability with extreme logits
- Reduction modes (mean, sum, none)
- Float16 support
- Device compatibility (CPU/CUDA)

### Issue 2.2.5: Update GlobalPointerNERHead.compute_loss

**File**: `src/modeling_studio/models/heads.py`

Updated `compute_loss()` method to delegate to `loss_fn`:

```python
def compute_loss(self, scores, span_labels, attention_mask=None):
    if self.loss_fn is not None:
        return self.loss_fn(scores, span_labels, attention_mask)
    # Fallback to BCE...
```

Added tests for loss_type in `tests/unit/test_globalpointer_head.py`:
- Default loss type is globalpointer
- Explicit globalpointer works
- Focal globalpointer works
- BCE fallback works
- Different loss types give different values

## Test Results

```
tests/unit/test_globalpointer_loss.py: 25 passed
tests/unit/test_globalpointer_head.py: 40 passed (7 new for loss_type)
Total: 65 passed, 0 failed
```

## Files Modified

1. `src/modeling_studio/models/losses.py`:
   - Added `GlobalPointerLoss` class (~100 lines)
   - Added `FocalGlobalPointerLoss` class (~50 lines)
   - Updated `__all__` exports

2. `src/modeling_studio/models/heads.py`:
   - Added `loss_type` parameter to `GlobalPointerNERHead.__init__`
   - Updated `compute_loss()` to use selected loss function
   - Added loss function instantiation in `__init__`

3. `tests/unit/test_globalpointer_loss.py`:
   - Created new test file with 25 test cases

4. `tests/unit/test_globalpointer_head.py`:
   - Added `TestGlobalPointerNERHeadLossType` class with 7 tests

## Technical Details

### Why Multi-Label Categorical CE > BCE

The GlobalPointer loss function from the original paper (Su et al., 2022) uses a clever formulation:

```python
# Flip sign: positive classes get negative pred, negative get positive
y_pred = (1 - 2 * y_true) * y_pred

# Mask opposite predictions
y_pred_neg = y_pred - y_true * 1e12  # Keep negatives
y_pred_pos = y_pred - (1 - y_true) * 1e12  # Keep positives

# LogSumExp aggregation
neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
loss = neg_loss + pos_loss
```

Benefits over BCE:
1. **Circle-loss separation**: Naturally pushes positive/negative apart
2. **Imbalance handling**: LogSumExp focuses on hardest examples
3. **No hyperparameter tuning**: No pos_weight needed
4. **SOTA results**: 93%+ F1 on benchmarks

### Float16 Support

Used dynamic mask value to avoid overflow:
```python
mask_value = 1e4 if y_pred.dtype == torch.float16 else 1e12
```

## Next Steps

- **Epic 2.3**: Decoding logic (partially complete - already built into head)
- **Epic 3.1**: Training script for GlobalPointer heads
- **Milestone 4**: Training execution with frozen encoder

## Reference

- Su et al. "Global Pointer: Novel Efficient Span-based Approach for Named Entity Recognition" (arXiv:2208.03054)
- Reference implementation: https://github.com/xhw205/Efficient-GlobalPointer-torch
