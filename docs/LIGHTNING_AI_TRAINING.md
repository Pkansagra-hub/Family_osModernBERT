# Lightning AI Training Guide for ModernBERT Multi-Task

## Quick Start

### 1. Upload Project to Lightning AI

In Lightning AI Studio, open a terminal and clone/upload your project:

```bash
# Option A: Clone from GitHub (if you've pushed)
git clone https://github.com/YOUR_USERNAME/modeling-studio.git
cd modeling-studio

# Option B: Upload via Lightning AI file browser
# Just drag & drop the project folder
```

### 2. Install Dependencies

```bash
# Install requirements
pip install -r requirements-lightning.txt

# Install the package in editable mode
pip install -e .

# Optional: Install Flash Attention 2 for maximum performance on H100/A100
pip install flash-attn --no-build-isolation
```

### 3. Run Training

#### H100 / H200 (80GB VRAM) - Full Training
```bash
python scripts/lightning_train.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --bf16 \
    --batch-size 64 \
    --gradient-accumulation 4 \
    --flash-attn \
    --wandb \
    --wandb-project modernbert-multitask \
    --output-dir outputs/h100-full-run
```

#### A100 (40/80GB) - Full Training
```bash
python scripts/lightning_train.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --bf16 \
    --batch-size 48 \
    --gradient-accumulation 5 \
    --flash-attn \
    --wandb \
    --output-dir outputs/a100-full-run
```

#### L40S (48GB) - Full Training
```bash
python scripts/lightning_train.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --bf16 \
    --batch-size 32 \
    --gradient-accumulation 8 \
    --wandb \
    --output-dir outputs/l40s-full-run
```

#### T4 (16GB) - Budget Option
```bash
python scripts/lightning_train.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --fp16 \
    --batch-size 16 \
    --gradient-accumulation 16 \
    --wandb \
    --output-dir outputs/t4-full-run
```

### 4. Quick Test (Debug Mode)
```bash
# Test with small dataset first (any GPU)
python scripts/lightning_train.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --debug \
    --bf16 \
    --output-dir outputs/debug-test
```

## GPU Comparison

| GPU | VRAM | Batch Size | Est. Time (3 epochs) | Cost/hr |
|-----|------|------------|---------------------|---------|
| **H200** | 80GB | 64 | ~30 min | Premium |
| **H100** | 80GB | 64 | ~35 min | Premium |
| **A100** | 80GB | 48 | ~45 min | $$ |
| **L40S** | 48GB | 32 | ~1.5 hr | $ |
| **T4** | 16GB | 16 | ~4 hr | Free tier |

## Recommended Settings

### For Best Quality (H100/H200):
- 3-5 epochs
- Learning rate: 2e-5
- Batch size: 64
- Effective batch: 256
- Flash Attention 2
- BF16 precision

### For Cost Efficiency (L40S/T4):
- 3 epochs
- Learning rate: 2e-5
- Gradient checkpointing (if needed)
- FP16 (T4) or BF16 (L40S)

## Weights & Biases Integration

1. Get your W&B API key from https://wandb.ai/settings
2. Set it in Lightning AI:
   ```bash
   wandb login YOUR_API_KEY
   ```
3. Add `--wandb` flag to training command

## Downloading Results

After training, download your model:
```bash
# From Lightning AI terminal
zip -r modernbert-multitask.zip outputs/YOUR_RUN/

# Then download via Lightning AI file browser
```

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable gradient checkpointing:
  ```bash
  # Add to config or modify training args
  gradient_checkpointing: true
  ```

### Slow Training
- Install Flash Attention: `pip install flash-attn --no-build-isolation`
- Use `--flash-attn` flag
- Increase `--dataloader-num-workers`

### W&B Not Logging
```bash
# Re-authenticate
wandb login --relogin
```
