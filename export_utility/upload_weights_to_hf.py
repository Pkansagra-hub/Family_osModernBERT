#!/usr/bin/env python
"""
Upload UltraBERT weights to HuggingFace private repository.

v3 Structure:
    Pkansagra/ultrabert-weights/
    ├── README.md
    ├── encoder/
    │   └── v1/
    │       ├── fp32/
    │       │   ├── model.safetensors
    │       │   └── config.json
    │       └── int8/
    │           └── *.onnx (quantized heads)
    └── decoder/
        └── v3/
            ├── fp32/
            │   └── model.safetensors
            └── int8/
                ├── prefix_encoder.onnx
                └── decoder_core.onnx

Usage:
    # First time setup - login to HuggingFace
    huggingface-cli login

    # Upload encoder weights (all quantizations)
    python export_utility/upload_weights_to_hf.py --component encoder --version v1

    # Upload decoder weights
    python export_utility/upload_weights_to_hf.py --component decoder --version v3

    # Upload all components
    python export_utility/upload_weights_to_hf.py --component all

    # Check repo status
    python export_utility/upload_weights_to_hf.py --component status
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
HF_REPO_ID = "Pkansagra/ultrabert-weights"
HF_REPO_TYPE = "model"

# Default paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PATHS = {
    "encoder_pytorch": PROJECT_ROOT / "familyos_ultrabert" / "weights" / "pytorch",
    "encoder_onnx": PROJECT_ROOT / "familyos_ultrabert" / "weights" / "onnx",
    "decoder_pytorch": PROJECT_ROOT / "outputs" / "ultrabert-gen-decoder-v3",
    "decoder_onnx": PROJECT_ROOT / "exports" / "onnx" / "decoder",
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


def upload_encoder(
    version: str = "v1",
    pytorch_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
) -> bool:
    """Upload encoder weights to HuggingFace with v3 structure.

    Uploads:
        encoder/{version}/fp32/ - PyTorch weights
        encoder/{version}/int8/ - Quantized ONNX heads
    """
    from huggingface_hub import HfApi

    pytorch_path = pytorch_path or DEFAULT_PATHS["encoder_pytorch"]
    onnx_path = onnx_path or DEFAULT_PATHS["encoder_onnx"]

    api = HfApi()
    success = True

    # Upload PyTorch weights (fp32)
    if pytorch_path.exists():
        required_files = ["model.safetensors", "config.json"]
        missing = [f for f in required_files if not (pytorch_path / f).exists()]
        if missing:
            logger.error(f"Missing required files in {pytorch_path}: {missing}")
            success = False
        else:
            logger.info(f"Uploading encoder fp32 from: {pytorch_path}")
            api.upload_folder(
                folder_path=str(pytorch_path),
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                path_in_repo=f"encoder/{version}/fp32",
                commit_message=f"Upload encoder {version} fp32 weights",
            )
            logger.info(f"Encoder fp32 uploaded to encoder/{version}/fp32/")
    else:
        logger.warning(f"PyTorch encoder path not found: {pytorch_path}")
        success = False

    # Upload ONNX weights (int8)
    if onnx_path.exists():
        onnx_files = list(onnx_path.glob("*.onnx"))
        if onnx_files:
            logger.info(f"Uploading {len(onnx_files)} ONNX heads from: {onnx_path}")
            api.upload_folder(
                folder_path=str(onnx_path),
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                path_in_repo=f"encoder/{version}/int8",
                commit_message=f"Upload encoder {version} int8 ONNX weights",
            )
            logger.info(f"Encoder int8 uploaded to encoder/{version}/int8/")
        else:
            logger.warning(f"No ONNX files found in: {onnx_path}")
    else:
        logger.warning(f"ONNX encoder path not found: {onnx_path}")

    return success


def upload_decoder(
    version: str = "v3",
    pytorch_path: Optional[Path] = None,
    onnx_path: Optional[Path] = None,
) -> bool:
    """Upload decoder weights to HuggingFace with v3 structure.

    Uploads:
        decoder/{version}/fp32/ - PyTorch weights
        decoder/{version}/int8/ - Quantized ONNX decoder
    """
    from huggingface_hub import HfApi

    pytorch_path = pytorch_path or DEFAULT_PATHS["decoder_pytorch"]
    onnx_path = onnx_path or DEFAULT_PATHS["decoder_onnx"]

    api = HfApi()
    success = True

    # Upload PyTorch weights (fp32)
    if pytorch_path.exists():
        decoder_files = list(pytorch_path.glob("*.safetensors")) + list(pytorch_path.glob("*.bin"))
        if not decoder_files:
            logger.error(f"No decoder weight files found in: {pytorch_path}")
            success = False
        else:
            logger.info(f"Uploading decoder fp32 from: {pytorch_path}")
            api.upload_folder(
                folder_path=str(pytorch_path),
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                path_in_repo=f"decoder/{version}/fp32",
                commit_message=f"Upload decoder {version} fp32 weights",
            )
            logger.info(f"Decoder fp32 uploaded to decoder/{version}/fp32/")
    else:
        logger.error(f"Decoder path not found: {pytorch_path}")
        logger.error("Train decoder first with: python scripts/train_stage_c.py")
        success = False

    # Upload ONNX weights (int8) if available
    if onnx_path.exists():
        onnx_files = list(onnx_path.glob("*.onnx"))
        if onnx_files:
            logger.info(f"Uploading decoder ONNX from: {onnx_path}")
            api.upload_folder(
                folder_path=str(onnx_path),
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                path_in_repo=f"decoder/{version}/int8",
                commit_message=f"Upload decoder {version} int8 ONNX weights",
            )
            logger.info(f"Decoder int8 uploaded to decoder/{version}/int8/")
        else:
            logger.warning(f"No ONNX files in {onnx_path}, skipping int8 upload")
    else:
        logger.warning(f"Decoder ONNX path not found: {onnx_path}")
        logger.info("Export ONNX first with: python export_utility/export_decoder_onnx.py")

    return success


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
  - counterfactual
  - decoder
private: true
---

# FamilyOS UltraBERT v3 Weights

Private model weights for FamilyOS UltraBERT v3.0.0.

## Repository Structure

```
ultrabert-weights/
├── encoder/
│   └── v1/
│       ├── fp32/
│       │   ├── model.safetensors (592 MB)
│       │   ├── config.json
│       │   ├── capabilities.json
│       │   └── tokenizer files
│       └── int8/
│           └── *_quantized_dynamic.onnx (175 MB total)
└── decoder/
    └── v3/
        ├── fp32/
        │   ├── model.safetensors (1.4 GB)
        │   └── config.json
        └── int8/
            ├── prefix_encoder.onnx (~5 MB)
            └── decoder_core.onnx (~350 MB)
```

## Usage

```python
from familyos_ultrabert import Client

# Weights downloaded automatically on first use
client = Client()
result = client.analyze("Mom picked up the kids from school!")
print(result.sentiment)  # "very_positive"
print(result.safety)     # "GREEN"

# Counterfactual generation (decoder loaded on demand)
with client.create_decoder_session() as decoder:
    output = client.encode("I felt overwhelmed today")
    suggestion = decoder.generate(output)
    print(suggestion)
```

## Memory Footprint

| Configuration | Memory |
|---------------|--------|
| Encoder only (INT8) | ~175 MB |
| Encoder + Decoder (INT8) | ~525 MB |
| Encoder only (FP32) | ~620 MB |
| Encoder + Decoder (FP32) | ~2020 MB |

## Version History

- **v3.0.0** - Edge-ready with lazy decoder loading, INT8 quantization
- **v2.x** - 12 capabilities, bundled weights
- **v1.x** - Initial release

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
        commit_message="Update model card for v3.0.0",
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
        description="Upload UltraBERT v3 weights to HuggingFace private repo"
    )
    parser.add_argument(
        "--component",
        choices=["encoder", "decoder", "all", "status"],
        required=True,
        help="Component to upload or 'status' to check repo"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version string (e.g., v1 for encoder, v3 for decoder)"
    )
    parser.add_argument(
        "--pytorch-path",
        type=Path,
        default=None,
        help="Custom path to PyTorch weights folder"
    )
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=None,
        help="Custom path to ONNX weights folder"
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
        version = args.version or "v1"
        success = upload_encoder(
            version=version,
            pytorch_path=args.pytorch_path,
            onnx_path=args.onnx_path,
        ) and success

    if args.component in ["decoder", "all"]:
        version = args.version or "v3"
        success = upload_decoder(
            version=version,
            pytorch_path=args.pytorch_path,
            onnx_path=args.onnx_path,
        ) and success

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
