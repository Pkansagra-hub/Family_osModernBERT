#!/bin/bash
# Lightning AI Setup Script
# Run this first to set up the environment

echo "=============================================="
echo "Setting up ModernBERT Multi-Task Training"
echo "=============================================="

# Check GPU
echo ""
echo "Detecting GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements-lightning.txt

# Install package
echo ""
echo "Installing modeling-studio package..."
pip install -q -e .

# Try to install Flash Attention (optional, may fail on some GPUs)
echo ""
echo "Attempting Flash Attention 2 install (optional)..."
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "Flash Attention not available, using SDPA instead"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Run training with:"
echo "  python scripts/lightning_train.py --config configs/training/multitask/stage_a_generic.yaml --bf16 --wandb"
echo ""
echo "Or test first with:"
echo "  python scripts/lightning_train.py --config configs/training/multitask/stage_a_generic.yaml --debug --bf16"
echo ""
