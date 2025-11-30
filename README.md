# FamilyOS ModernBERT

Multi-task encoder model based on ModernBERT for family assistant applications.

## Features

- **Multi-task Learning**: NER, Sentiment, Emotions, Safety, NLI, Embeddings
- **ModernBERT Base**: 151M parameters, optimized for modern GPUs
- **BFloat16 Training**: Native support for A100/H100/RTX 40xx/50xx
- **Production Ready**: Validated training pipeline with comprehensive tests

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
cd Family_osModernBERT

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: Flash Attention for A100/H100
pip install flash-attn --no-build-isolation
```

### Validate Before Training

**ALWAYS run validation first to catch errors early:**

```bash
python scripts/validate_full_pipeline.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --samples 200 --steps 20
```

### Training

```bash
# Stage A: Generic multi-task training (public datasets)
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml
```

## GCP Training

For training on Google Cloud Platform with A100/H100:

```bash
# 1. Create VM with Deep Learning image
# 2. Clone repo and run setup
./setup_gcp.sh

# 3. Validate
python scripts/validate_full_pipeline.py \
    --config configs/training/multitask/stage_a_generic.yaml \
    --samples 200 --steps 20

# 4. Train
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_generic.yaml
```

Estimated training time on A100 40GB: **2-3 hours** (~$12 cost)

## Tasks

| Task | Type | Dataset | Metrics |
|------|------|---------|---------|
| NER General | Token Classification | CoNLL-2003 | F1, Precision, Recall |
| Sentiment | Classification | SST-2 | Accuracy, F1 |
| Emotions | Multi-label | GoEmotions | Micro/Macro F1 |
| Safety | Multi-label | Civil Comments | Micro/Macro F1 |
| NLI | Classification | MNLI, SNLI | Accuracy |
| Embedding | Contrastive | STS-B, NLI pairs | Spearman correlation |

## Project Structure

```
Family_osModernBERT/
├── configs/
│   ├── data/multitask/          # Dataset configurations
│   └── training/multitask/      # Training configurations
├── scripts/
│   ├── train_stage_a.py         # Stage A training script
│   ├── validate_full_pipeline.py # Full pipeline validation
│   └── setup_gcp.sh             # GCP setup script
├── src/modeling_studio/
│   ├── data/                    # Data loading & processing
│   ├── models/                  # Model architectures
│   ├── trainers/                # Training logic
│   └── evaluation/              # Metrics & evaluation
├── requirements.txt             # Python dependencies
└── pyproject.toml              # Package configuration
```

## Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 11.8+ (for GPU training)
- 8GB+ VRAM (16GB+ recommended)

## License

MIT License
