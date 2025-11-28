# Modeling Studio

A unified repository for training and fine-tuning models from BERT to Small Language Models (SLMs) using multiple frameworks.

## Repository Structure

```
modeling_studio/
├── configs/                    # Training configurations
│   ├── model/                  # Model-specific configs
│   ├── training/               # Training hyperparameters
│   └── data/                   # Data processing configs
├── data/                       # Dataset storage
│   ├── raw/                    # Raw unprocessed data
│   ├── processed/              # Preprocessed data
│   └── cache/                  # Cached tokenized data
├── src/
│   └── modeling_studio/
│       ├── models/             # Model definitions & architectures
│       ├── trainers/           # Training loops & strategies
│       ├── data/               # Data loading & processing
│       ├── utils/              # Utility functions
│       └── evaluation/         # Evaluation metrics & scripts
├── scripts/                    # Training & utility scripts
├── notebooks/                  # Experimentation notebooks
├── checkpoints/                # Model checkpoints (gitignored)
├── outputs/                    # Training outputs (gitignored)
└── tests/                      # Unit tests
```

## Supported Frameworks

- **PyTorch** - Core deep learning
- **Hugging Face Transformers** - Pre-trained models & tokenizers
- **PEFT** - Parameter-efficient fine-tuning (LoRA, QLoRA, etc.)
- **DeepSpeed** - Distributed training & optimization
- **Accelerate** - Multi-GPU/TPU training
- **bitsandbytes** - Quantization
- **Unsloth** - Fast fine-tuning for LLMs

## Supported Model Types

- **Encoder Models**: BERT, RoBERTa, DeBERTa, ELECTRA
- **Decoder Models**: GPT-2, LLaMA, Mistral, Phi, Qwen
- **Encoder-Decoder**: T5, BART, FLAN-T5
- **Custom Architectures**: Define your own

## Quick Start

```bash
# Install dependencies
pip install -e .

# Train a model (example)
python scripts/train.py --config configs/training/default.yaml

# Fine-tune with LoRA
python scripts/finetune.py --config configs/training/lora.yaml
```

## Configuration System

All configurations are managed via YAML files in `configs/`. Override any parameter via CLI:

```bash
python scripts/train.py --config configs/training/default.yaml \
    --model.name_or_path bert-base-uncased \
    --training.learning_rate 2e-5
```

## Adding New Models

1. Add model config to `configs/model/`
2. If custom architecture, add to `src/modeling_studio/models/`
3. Create training config in `configs/training/`

## Adding New Datasets

1. Place raw data in `data/raw/`
2. Create data config in `configs/data/`
3. Processed data will be cached in `data/processed/`

## License

MIT
