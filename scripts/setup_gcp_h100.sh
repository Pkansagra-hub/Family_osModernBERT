#!/bin/bash
# =============================================================================
# GCP H100 Training Setup Script
# =============================================================================
# This script creates an H100 VM in us-central1, sets up the environment,
# and starts Stage A training with ALL SOTA features.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - $300 credit available
#   - H100 quota in us-central1
#
# Usage:
#   chmod +x scripts/setup_gcp_h100.sh
#   ./scripts/setup_gcp_h100.sh
#
# Cost Estimate:
#   - H100-80GB: ~$3.50/hour (on-demand) or ~$1.50/hour (spot)
#   - Training time: ~4-6 hours for 10 epochs
#   - Total: ~$10-20 (spot) or ~$20-40 (on-demand)
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ID=$(gcloud config get-value project)
ZONE="us-central1-a"
INSTANCE_NAME="modernbert-h100-training"
MACHINE_TYPE="a3-highgpu-1g"  # 1x H100-80GB
DISK_SIZE="200"  # GB
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"

# Use spot/preemptible for cost savings (recommended for training)
USE_SPOT=true

echo "============================================================"
echo "GCP H100 Training Setup"
echo "============================================================"
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE (1x H100-80GB)"
echo "Spot instance: $USE_SPOT"
echo "============================================================"

# =============================================================================
# CHECK QUOTA
# =============================================================================
echo ""
echo "Checking H100 quota in $ZONE..."
gcloud compute regions describe us-central1 --format="table(quotas[].metric,quotas[].limit,quotas[].usage)" | grep -i gpu || true

# =============================================================================
# CREATE INSTANCE
# =============================================================================
echo ""
echo "Creating H100 instance..."

SPOT_FLAG=""
if [ "$USE_SPOT" = true ]; then
    SPOT_FLAG="--provisioning-model=SPOT --instance-termination-action=STOP"
    echo "Using SPOT instance for cost savings (~60% cheaper)"
fi

gcloud compute instances create $INSTANCE_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=type=nvidia-h100-80gb,count=1 \
    --maintenance-policy=TERMINATE \
    --boot-disk-size=${DISK_SIZE}GB \
    --boot-disk-type=pd-ssd \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --metadata="install-nvidia-driver=True" \
    --scopes=cloud-platform \
    $SPOT_FLAG

echo "Instance created! Waiting for it to be ready..."
sleep 30

# =============================================================================
# GET INSTANCE IP
# =============================================================================
INSTANCE_IP=$(gcloud compute instances describe $INSTANCE_NAME --zone=$ZONE --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
echo "Instance IP: $INSTANCE_IP"

# =============================================================================
# SETUP SCRIPT TO RUN ON INSTANCE
# =============================================================================
echo ""
echo "Creating setup script..."

cat > /tmp/h100_setup.sh << 'EOF'
#!/bin/bash
set -e

echo "============================================================"
echo "Setting up H100 Training Environment"
echo "============================================================"

# Update system
sudo apt-get update -qq

# Clone repo
cd /home/$USER
if [ ! -d "Family_osModernBERT" ]; then
    git clone https://github.com/Pkansagra-hub/Family_osModernBERT.git
fi
cd Family_osModernBERT

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 (H100 optimized)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install flash-attn for H100
pip install flash-attn --no-build-isolation

# Install project dependencies
pip install -e ".[dev]"

# Set environment variables for optimal H100 performance
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# Verify GPU
echo ""
echo "GPU Info:"
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "============================================================"
echo "Environment ready! Starting training..."
echo "============================================================"

# Start training with H100 config
python scripts/train_stage_a.py \
    --config configs/training/multitask/stage_a_h100.yaml \
    2>&1 | tee training_h100.log

echo ""
echo "Training complete! Logs saved to training_h100.log"
EOF

# =============================================================================
# COPY AND RUN SETUP
# =============================================================================
echo ""
echo "Copying setup script to instance..."
gcloud compute scp /tmp/h100_setup.sh $INSTANCE_NAME:/tmp/h100_setup.sh --zone=$ZONE

echo ""
echo "Running setup script on instance..."
echo "(This will take ~10-15 minutes for initial setup)"
echo ""

# Run in background with nohup so it continues if SSH disconnects
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command="chmod +x /tmp/h100_setup.sh && nohup /tmp/h100_setup.sh > /tmp/setup.log 2>&1 &"

echo ""
echo "============================================================"
echo "SETUP INITIATED!"
echo "============================================================"
echo ""
echo "The training is now running in the background."
echo ""
echo "To monitor progress:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='tail -f /home/\$USER/Family_osModernBERT/training_h100.log'"
echo ""
echo "To SSH into the instance:"
echo "  gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To stop the instance (saves cost):"
echo "  gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "To delete the instance:"
echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$ZONE"
echo ""
echo "Estimated training time: 4-6 hours"
echo "Estimated cost: \$10-20 (spot) or \$20-40 (on-demand)"
echo "============================================================"
