#!/bin/bash
# =============================================================================
# GCP A100 Training Setup Script
# =============================================================================
# Run this on a GCP Deep Learning VM with A100 GPU
#
# Usage:
#   chmod +x setup_gcp.sh
#   ./setup_gcp.sh
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "FamilyOS ModernBERT - GCP Setup"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Check GPU
# -----------------------------------------------------------------------------
echo ""
echo "[1/6] Checking GPU..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: No NVIDIA GPU detected!"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
echo "  ✓ GPU: $GPU_NAME"
echo "  ✓ Memory: $GPU_MEM"

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | cut -d',' -f1)
echo "  ✓ CUDA: $CUDA_VERSION"

# -----------------------------------------------------------------------------
# 2. Upgrade pip
# -----------------------------------------------------------------------------
echo ""
echo "[2/6] Upgrading pip..."
pip install --upgrade pip

# -----------------------------------------------------------------------------
# 3. Install requirements
# -----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing requirements..."
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# 4. Install package
# -----------------------------------------------------------------------------
echo ""
echo "[4/6] Installing modeling-studio package..."
pip install -e .

# -----------------------------------------------------------------------------
# 5. Install Flash Attention (optional but HIGHLY recommended for A100)
# -----------------------------------------------------------------------------
echo ""
echo "[5/6] Installing Flash Attention 2..."
if [[ "$GPU_NAME" == *"A100"* ]] || [[ "$GPU_NAME" == *"H100"* ]] || [[ "$GPU_NAME" == *"L40"* ]]; then
    echo "  A100/H100/L40 detected - installing flash-attn..."
    pip install flash-attn --no-build-isolation || echo "  ⚠ Flash Attention install failed (non-critical)"
else
    echo "  Skipping flash-attn (not A100/H100/L40)"
fi

# -----------------------------------------------------------------------------
# 6. Validate installation
# -----------------------------------------------------------------------------
echo ""
echo "[6/6] Validating installation..."
python -c "
import torch
import transformers
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel
from modeling_studio.trainers.multitask_trainer import MultiTaskTrainer
print('  ✓ PyTorch:', torch.__version__)
print('  ✓ Transformers:', transformers.__version__)
print('  ✓ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  ✓ GPU:', torch.cuda.get_device_name(0))
    print('  ✓ BF16 support:', torch.cuda.get_device_capability(0)[0] >= 8)
print('  ✓ modeling_studio imports OK')
"

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Validate pipeline (ALWAYS DO THIS FIRST!):"
echo "   python scripts/validate_full_pipeline.py \\"
echo "       --config configs/training/multitask/stage_a_generic.yaml \\"
echo "       --samples 200 --steps 20"
echo ""
echo "2. Run full training:"
echo "   python scripts/train_stage_a.py \\"
echo "       --config configs/training/multitask/stage_a_generic.yaml"
echo ""
echo "3. With W&B logging:"
echo "   wandb login"
echo "   python scripts/train_stage_a.py \\"
echo "       --config configs/training/multitask/stage_a_generic.yaml \\"
echo "       --wandb"
echo ""
