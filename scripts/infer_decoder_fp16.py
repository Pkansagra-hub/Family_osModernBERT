"""
FP16 Inference Script for UltraBERT-Gen Decoder (13th Head).

Optimized for low-VRAM environments (~2-4GB).
Loads encoder + decoder only, skips all other heads to save memory.

Usage:
    python scripts/infer_decoder_fp16.py --text "My child refuses to eat vegetables"
    python scripts/infer_decoder_fp16.py --interactive
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get device, preferring CUDA if available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_fp16(
    checkpoint_path: str,
    device: torch.device,
):
    """
    Load encoder + decoder in FP16 mode.

    Only loads encoder and counterfactual decoder head to minimize memory.
    All other heads are skipped.
    """
    from transformers import AutoTokenizer, AutoConfig
    from modeling_studio.models.decoder_config import DecoderMoEConfig
    from modeling_studio.models.decoder_moe import CounterfactualDecoderHead

    # Import ModernBERT for encoder
    try:
        from transformers import ModernBertModel
    except ImportError:
        from transformers import AutoModel
        ModernBertModel = AutoModel

    logger.info(f"Loading from: {checkpoint_path}")
    checkpoint_path = Path(checkpoint_path)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Load state dict
    logger.info("Loading state dict...")
    state_dict_path = checkpoint_path / "pytorch_model.bin"
    if not state_dict_path.exists():
        # Check for safetensors
        safetensors_path = checkpoint_path / "model.safetensors"
        if safetensors_path.exists():
            from safetensors.torch import load_file
            state_dict = load_file(str(safetensors_path))
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
    else:
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)

    # Extract encoder weights
    logger.info("Extracting encoder weights...")
    encoder_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "", 1)
            encoder_state_dict[new_key] = value
        elif key.startswith("backbone."):
            new_key = key.replace("backbone.", "", 1)
            encoder_state_dict[new_key] = value

    # Extract decoder weights (head.counterfactual.*)
    logger.info("Extracting decoder weights...")
    decoder_state_dict = {}
    for key, value in state_dict.items():
        if "counterfactual" in key:
            # Remove the "heads.counterfactual." or "head.counterfactual." prefix
            if key.startswith("heads.counterfactual."):
                new_key = key.replace("heads.counterfactual.", "")
            elif key.startswith("head.counterfactual."):
                new_key = key.replace("head.counterfactual.", "")
            else:
                new_key = key
            decoder_state_dict[new_key] = value

    # Clear original state dict to save memory
    del state_dict
    clear_memory()

    # Load encoder in FP16
    logger.info("Loading encoder in FP16...")
    encoder = ModernBertModel.from_pretrained(
        "answerdotai/ModernBERT-base",
        torch_dtype=torch.float16,
    )

    # Try to load encoder weights if we have them
    if encoder_state_dict:
        try:
            missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing:
                logger.warning(f"Missing encoder keys: {len(missing)}")
            if unexpected:
                logger.warning(f"Unexpected encoder keys: {len(unexpected)}")
        except Exception as e:
            logger.warning(f"Could not load encoder weights: {e}, using pretrained")

    encoder = encoder.to(device).half()
    encoder.eval()

    # Free encoder state dict
    del encoder_state_dict
    clear_memory()

    # Create decoder config
    decoder_config = DecoderMoEConfig(
        hidden_size=1280,
        num_layers=8,
        vocab_size=config.vocab_size if hasattr(config, "vocab_size") else 50368,
        num_attention_heads=20,
        num_kv_heads=4,
        num_experts=8,
        num_experts_per_token=2,
        encoder_hidden_size=768,
    )

    # Load decoder in FP16
    logger.info("Loading decoder in FP16...")
    decoder = CounterfactualDecoderHead(
        config=decoder_config,
        encoder_hidden_size=768,
    )

    # Load decoder weights
    if decoder_state_dict:
        try:
            missing, unexpected = decoder.load_state_dict(decoder_state_dict, strict=False)
            logger.info(f"Decoder loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                logger.debug(f"Missing keys: {missing[:5]}...")
        except Exception as e:
            logger.error(f"Failed to load decoder weights: {e}")
            raise

    decoder = decoder.to(device).half()
    decoder.eval()

    # Free decoder state dict
    del decoder_state_dict
    clear_memory()

    # Report memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    return encoder, decoder, tokenizer


@torch.inference_mode()
def generate_counterfactual(
    text: str,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
) -> str:
    """
    Generate counterfactual response for input text.
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Encode input
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16):
        encoder_outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        encoder_hidden = encoder_outputs.last_hidden_state

        # Generate with decoder
        generated_ids = decoder.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.sep_token_id or tokenizer.eos_token_id or 50282,
            pad_token_id=tokenizer.pad_token_id or 50283,
        )

    # Decode output
    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return generated_text


def main():
    parser = argparse.ArgumentParser(description="FP16 Decoder Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v1",
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Input text to generate counterfactual for",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    args = parser.parse_args()

    # Get device
    device = torch.device("cpu") if args.cpu else get_device()
    logger.info(f"Using device: {device}")

    # Load model
    encoder, decoder, tokenizer = load_model_fp16(args.checkpoint, device)

    if args.interactive:
        # Interactive mode
        print("\n" + "=" * 60)
        print("UltraBERT-Gen Decoder - Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")

        while True:
            try:
                text = input("Input: ").strip()
                if text.lower() in ["quit", "exit", "q"]:
                    break
                if not text:
                    continue

                output = generate_counterfactual(
                    text=text,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(f"\nOutput: {output}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                continue

    elif args.text:
        # Single inference
        output = generate_counterfactual(
            text=args.text,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nInput: {args.text}")
        print(f"Output: {output}")

    else:
        # Demo with sample texts
        samples = [
            "My child refuses to eat vegetables",
            "I am feeling stressed about work-life balance",
            "My teenager keeps staying up too late",
        ]

        print("\n" + "=" * 60)
        print("Demo Counterfactual Generation")
        print("=" * 60)

        for text in samples:
            try:
                output = generate_counterfactual(
                    text=text,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    device=device,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(f"\nInput: {text}")
                print(f"Output: {output}")
            except Exception as e:
                logger.error(f"Error with '{text}': {e}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
