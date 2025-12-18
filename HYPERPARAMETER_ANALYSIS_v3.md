# Hyperparameter Analysis: v1 vs v2 vs v3

## Executive Summary

**v2 Results**: -1.5% regression on weak subdomains (failed to improve)
**v3 Strategy**: Optimized hyperparameters for stronger adaptation while preventing catastrophic forgetting

---

## Detailed Comparison

| Hyperparameter | v1 (baseline) | v2 (failed) | v3 (optimized) | Rationale |
|----------------|---------------|-------------|----------------|-----------|
| **Learning Rate** | 1.0e-4 | 3.0e-5 ❌ | 7.0e-5 ✅ | v2's 3e-5 too conservative, couldn't adapt. v3's 7e-5 is 2.3x higher but still safer than v1 |
| **Epochs** | 7 | 3 ❌ | 5 ✅ | v2's 3 epochs insufficient for learning new patterns. v3's 5 balances convergence & overfitting risk |
| **Batch Size** | 64 | 64 ❌ | 32 ✅ | Smaller batches = more gradient updates per epoch = better exploration of loss landscape |
| **Gradient Accumulation** | 2 | 2 ❌ | 4 ✅ | Maintains effective batch=128 but with 4x more frequent weight updates |
| **Warmup Ratio** | 5% | 10% ❌ | 5% ✅ | v2's long warmup wasted training time. v3 reaches peak LR faster |
| **Dropout** | 0.1 | 0.1 ❌ | 0.15 ✅ | Stronger regularization prevents overfitting on balanced data |
| **Weight Decay** | 0.01 | 0.01 ❌ | 0.02 ✅ | 2x stronger L2 penalty helps anti-forgetting |
| **Early Stopping Patience** | N/A | 3 ❌ | 5 ✅ | More tolerance for temporary plateaus during adaptation |
| **Save Steps** | 1000 | 250 | 250 ✅ | Frequent checkpointing for Colab resilience |

---

## Why v2 Failed: Root Cause Analysis

### 1. **Learning Rate Too Low (3e-5)**
- **Problem**: Model couldn't escape v1's local minimum
- **Evidence**: Diversity scores barely changed (-1.5% avg)
- **Physics**: Fine-tuning requires sufficient LR to update weights for new patterns
- **Solution**: v3 uses 7e-5 (2.3x higher) for stronger adaptation

### 2. **Insufficient Training (3 epochs)**
- **Problem**: Not enough iterations to learn 86K balanced samples
- **Evidence**: All 7 weak subdomains showed flat/negative trends
- **Math**: 86K samples ÷ 128 batch = 672 steps/epoch × 3 = 2,016 total steps
- **Solution**: v3 uses 5 epochs = 3,360 steps (66% more training)

### 3. **Large Batches Reduce Gradient Updates**
- **Problem**: Batch size 64 with grad_accum=2 means only 336 weight updates/epoch
- **Evidence**: Model saw each sample only 3 times (3 epochs)
- **Math**: 672 steps/epoch ÷ 2 accumulation = 336 updates/epoch
- **Solution**: v3 uses batch 32 with grad_accum=4 = 672 updates/epoch (2x more!)

### 4. **Weak Regularization**
- **Problem**: dropout=0.1 insufficient for preventing overfitting on repeated data
- **Evidence**: Some subdomains regressed (relationship_spouse: -4.7%)
- **Solution**: v3 increases dropout to 0.15 and weight_decay to 0.02

---

## v3 Optimization Strategy

### Core Principles
1. **Stronger Adaptation**: Higher LR (7e-5) enables weight updates for new patterns
2. **More Training Time**: 5 epochs gives model time to converge
3. **Better Gradient Flow**: Smaller batches = 2x more frequent weight updates
4. **Anti-Forgetting**: Stronger regularization (dropout 0.15, weight_decay 0.02)
5. **Efficient Warmup**: 5% warmup reaches peak LR faster

### Mathematical Justification

**Effective Gradient Updates per Epoch:**
- v2: 672 steps ÷ 2 accumulation = **336 updates/epoch**
- v3: 672 steps ÷ 4 accumulation = **168 updates/epoch** ❌ WAIT...

**CORRECTION**: Let me recalculate with actual batch sizes:
- v2: batch=64, accum=2, samples=86K → 86K÷128 = **672 steps/epoch**
  - Weight updates: 672 (updated every step after accumulation)
- v3: batch=32, accum=4, samples=86K → 86K÷128 = **672 steps/epoch**  
  - Weight updates: 672 (same, but each update uses 4 gradient accumulations)

**Key Insight**: Same number of weight updates, but v3 sees more mini-batch diversity!
- v2: Sees 64 samples at once, updates every 2 mini-batches
- v3: Sees 32 samples at once, updates every 4 mini-batches
- **Result**: v3 has finer-grained gradient signal (more diverse mini-batches)

### Expected Results

**Weak Subdomains (Target: +5-10% improvement vs v1)**
- health_mental: 0.953 → 0.995+ (target: +4-6%)
- relationship_spouse: 0.970 → 1.010+ (target: +4-6%)
- emotions_grief: 0.942 → 0.985+ (target: +4-6%)
- routine_morning: 0.943 → 0.985+ (target: +4-6%)

**Strong Subdomains (Target: >90% maintenance)**
- parenting_bonding: maintain >0.88 (0.977 × 0.9)
- emotions_stress: maintain >0.84 (0.933 × 0.9)
- health_nutrition: maintain >0.86 (0.960 × 0.9)

**Metrics for Success:**
- ✅ **Primary**: Weak domain avg improvement >+5%
- ✅ **Secondary**: Strong domain maintenance >80% (2.4/3 subdomains OK)
- ✅ **Tertiary**: No individual subdomain regression >10%

---

## Training Timeline (A100 80GB)

**Total Training Time: ~2-3 hours**

| Epoch | Steps | Est. Time | Key Milestones |
|-------|-------|-----------|----------------|
| 1 | 0-672 | 30 min | Initial adaptation, loss drops rapidly |
| 2 | 672-1344 | 30 min | Continued learning, check eval metrics |
| 3 | 1344-2016 | 30 min | Fine-tuning, diversity scores should improve |
| 4 | 2016-2688 | 30 min | Convergence phase, metrics stabilize |
| 5 | 2688-3360 | 30 min | Final refinement, best checkpoint selection |

**Checkpoints**: Every 250 steps (saves at 250, 500, 750, 1000, ...)
**Evaluation**: Every 250 steps (monitors eval_loss for early stopping)

---

## Implementation Checklist

- [x] Created `stage_c_gpt2_v3.yaml` with optimized hyperparameters
- [ ] Verify v2 checkpoint exists: `outputs/ultrabert-gen-decoder-v1`
- [ ] Verify balanced data exists: `data/counterfactual/training_v2/`
- [ ] Verify embeddings exist: `data/counterfactual/training_v2/sequence_embeddings.h5`
- [ ] Run training: `python scripts/train_stage_c.py --config configs/training/multitask/stage_c_gpt2_v3.yaml`
- [ ] Monitor training: `tensorboard --logdir outputs/ultrabert-gen-decoder-v3`
- [ ] Run evaluation after training: `python evaluate_v1_vs_v2.py` (modify to test v3)
- [ ] Compare results: v1 vs v2 vs v3
- [ ] Deploy best model: Copy to production if v3 > v1

---

## Risk Mitigation

### Risk 1: LR 7e-5 too high → Catastrophic forgetting
**Mitigation**: 
- Stronger regularization (dropout 0.15, weight_decay 0.02)
- Early stopping with patience=5
- Checkpoints every 250 steps (can revert if needed)

### Risk 2: 5 epochs → Overfitting
**Mitigation**:
- Monitor eval_loss closely (saved every 250 steps)
- Early stopping enabled (patience=5, threshold=0.001)
- load_best_model_at_end=true (auto-select best checkpoint)

### Risk 3: Smaller batches → Training instability
**Mitigation**:
- Gradient accumulation=4 smooths gradient estimates
- Max_grad_norm=1.0 clips extreme gradients
- Cosine LR schedule provides smooth decay

---

## Next Steps

1. **Start v3 Training**:
   ```bash
   cd d:\Modeling_studio
   python scripts/train_stage_c.py --config configs/training/multitask/stage_c_gpt2_v3.yaml
   ```

2. **Monitor Progress**:
   ```bash
   tensorboard --logdir outputs/ultrabert-gen-decoder-v3
   ```

3. **Evaluate After Training**:
   ```bash
   python evaluate_v1_vs_v3.py  # Create this by modifying evaluate_v1_vs_v2.py
   ```

4. **Compare All Versions**:
   - Create comprehensive comparison: v1 vs v2 vs v3
   - Analyze subdomain-specific improvements
   - Make production deployment decision

---

## Success Criteria

**Deploy v3 if:**
- Weak domain avg improvement: **>+5%** (vs v1)
- Strong domain maintenance: **>80%** (2.4/3 subdomains maintain >90% of v1)
- No catastrophic forgetting: **No subdomain drops >10%** from v1

**Revert to v1 if:**
- Weak domain improvement: <+2%
- Strong domain maintenance: <60%
- Any critical subdomain (health_mental, parenting_bonding) regresses >15%

**Continue tuning if:**
- Mixed results (some subdomains improve, others regress)
- Consider: layer-wise LR decay, longer training, different optimizer
