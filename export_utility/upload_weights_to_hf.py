#!/usr/bin/env python
"""
Upload UltraBERT weights to HuggingFace private repository.

This script handles:
1. Creating the private HuggingFace repo (if not exists)
2. Uploading encoder weights (PyTorch + configs)
3. Uploading ONNX weights (optional)
4. Uploading decoder weights (after Stage C training)

Usage:
    # First time setup - login to HuggingFace
    huggingface-cli login

    # Upload encoder weights
    python export_utility/upload_weights_to_hf.py --component encoder

    # Upload ONNX weights
    python export_utility/upload_weights_to_hf.py --component onnx

    # Upload decoder weights (after training)
    python export_utility/upload_weights_to_hf.py --component decoder

    # Upload all components
    python export_utility/upload_weights_to_hf.py --component all
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "Pkansagra/ultrabert-weights"  # Your personal HuggingFace namespace
HF_REPO_TYPE = "model"

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PATHS = {
    "encoder": PROJECT_ROOT / "familyos_ultrabert" / "weights" / "pytorch",
    "onnx": PROJECT_ROOT / "familyos_ultrabert" / "weights" / "onnx",
    "decoder": PROJECT_ROOT / "outputs" / "ultrabert-gen-decoder-v1",
}


def check_huggingface_login() -> bool:
    """Check if user is logged in to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user_info = api.whoami()
        logger.info(f"Logged in as: {user_info['name']}")
        return True
    except Exception as e:
        logger.error(f"Not logged in to HuggingFace: {e}")
        logger.error("Run: huggingface-cli login")
        return False


def create_repo_if_not_exists() -> bool:
    """Create private HuggingFace repo if it doesn't exist."""
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()

    try:
        # Check if repo exists
        api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        logger.info(f"Repository exists: {HF_REPO_ID}")
        return True
    except RepositoryNotFoundError:
        logger.info(f"Creating private repository: {HF_REPO_ID}")
        create_repo(
            repo_id=HF_REPO_ID,
            private=True,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
        )
        logger.info(f"Created private repository: {HF_REPO_ID}")
        return True
    except Exception as e:
        logger.error(f"Error checking/creating repo: {e}")
        return False


def upload_encoder(source_path: Optional[Path] = None) -> bool:
    """Upload encoder weights to HuggingFace."""
    from huggingface_hub import HfApi

    source_path = source_path or DEFAULT_PATHS["encoder"]

    if not source_path.exists():
        logger.error(f"Encoder path not found: {source_path}")
        return False

    # Check for required files
    required_files = ["model.safetensors", "config.json"]
    missing = [f for f in required_files if not (source_path / f).exists()]
    if missing:
        logger.error(f"Missing required files: {missing}")
        return False

    logger.info(f"Uploading encoder from: {source_path}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(source_path),
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo="encoder",
        commit_message="Upload encoder weights",
    )

    logger.info("Encoder weights uploaded successfully!")
    return True


def upload_onnx(source_path: Optional[Path] = None) -> bool:
    """Upload ONNX weights to HuggingFace."""
    from huggingface_hub import HfApi

    source_path = source_path or DEFAULT_PATHS["onnx"]

    if not source_path.exists():
        logger.error(f"ONNX path not found: {source_path}")
        return False

    # Check for ONNX files
    onnx_files = list(source_path.glob("*.onnx"))
    if not onnx_files:
        logger.error(f"No ONNX files found in: {source_path}")
        return False

    logger.info(f"Uploading {len(onnx_files)} ONNX models from: {source_path}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(source_path),
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo="onnx",
        commit_message="Upload ONNX weights",
    )

    logger.info("ONNX weights uploaded successfully!")
    return True


def upload_decoder(source_path: Optional[Path] = None) -> bool:
    """Upload decoder weights to HuggingFace."""
    from huggingface_hub import HfApi

    source_path = source_path or DEFAULT_PATHS["decoder"]

    if not source_path.exists():
        logger.error(f"Decoder path not found: {source_path}")
        logger.error("Train decoder first with: python scripts/train_stage_c.py")
        return False

    # Check for decoder files
    decoder_files = list(source_path.glob("*.safetensors")) + list(source_path.glob("*.bin"))
    if not decoder_files:
        logger.error(f"No decoder weight files found in: {source_path}")
        return False

    logger.info(f"Uploading decoder from: {source_path}")

    api = HfApi()
    api.upload_folder(
        folder_path=str(source_path),
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo="decoder",
        commit_message="Upload decoder weights",
    )

    logger.info("Decoder weights uploaded successfully!")
    return True


def create_model_card() -> bool:
    """Create/update model card (README.md) on HuggingFace."""
    from huggingface_hub import HfApi

    model_card = """---
license: other
license_name: proprietary
license_link: LICENSE
library_name: transformers
tags:
  - family
  - nlp
  - sentiment
  - safety
  - ner
  - multitask
private: true
---

# FamilyOS UltraBERT Weights

Private model weights for FamilyOS UltraBERT v3.

## Contents

- `encoder/` - Main encoder weights (592 MB)
  - `model.safetensors` - PyTorch weights
  - `config.json` - Model configuration
  - `capabilities.json` - Head configurations

- `decoder/` - Counterfactual decoder (240 MB)
  - `decoder.safetensors` - MoE decoder weights
  - `config.json` - Decoder configuration

- `onnx/` - Quantized ONNX models (optional)
  - `*_int8.onnx` - INT8 quantized models

## Usage

```python
from familyos_ultrabert import Client

# Weights downloaded automatically on first use
client = Client()
result = client.analyze("Mom picked up the kids!")
```

## License

Proprietary - All Rights Reserved.
"""

    logger.info("Creating model card...")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Update model card",
    )

    logger.info("Model card created successfully!")
    return True


def print_repo_status():
    """Print current status of HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()

    try:
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)

        print("\n" + "=" * 60)
        print(f"Repository: https://huggingface.co/{HF_REPO_ID}")
        print("=" * 60)

        # Group files by folder
        folders = {}
        for f in files:
            parts = f.split("/")
            folder = parts[0] if len(parts) > 1 else "root"
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(f)

        for folder, folder_files in sorted(folders.items()):
            print(f"\n{folder}/")
            for f in folder_files[:5]:  # Show first 5 files
                print(f"  - {f}")
            if len(folder_files) > 5:
                print(f"  ... and {len(folder_files) - 5} more files")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Error getting repo status: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload UltraBERT weights to HuggingFace private repo"
    )
    parser.add_argument(
        "--component",
        choices=["encoder", "onnx", "decoder", "all", "status"],
        required=True,
        help="Component to upload (encoder, onnx, decoder, all) or 'status' to check repo"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Custom path to weights folder (optional)"
    )
    parser.add_argument(
        "--create-card",
        action="store_true",
        help="Create/update model card on HuggingFace"
    )

    args = parser.parse_args()

    # Check login
    if not check_huggingface_login():
        return 1

    # Status check
    if args.component == "status":
        print_repo_status()
        return 0

    # Create repo if needed
    if not create_repo_if_not_exists():
        return 1

    # Upload components
    success = True

    if args.component in ["encoder", "all"]:
        success = upload_encoder(args.path) and success

    if args.component in ["onnx", "all"]:
        success = upload_onnx(args.path) and success

    if args.component in ["decoder", "all"]:
        success = upload_decoder(args.path) and success

    # Create model card
    if args.create_card or args.component == "all":
        success = create_model_card() and success

    # Print status
    if success:
        print_repo_status()
        print("\nDone! Weights uploaded to private HuggingFace repo.")
        print(f"View at: https://huggingface.co/{HF_REPO_ID}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
