#!/usr/bin/env python
"""Complete end-to-end verification of Stage C training setup."""

import sys
sys.path.insert(0, ".")

from transformers import AutoTokenizer
from modeling_studio.data.counterfactual_dataset import CounterfactualDataset
from modeling_studio.models.decoder_config import DecoderMoEConfig

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
config = DecoderMoEConfig()

print("=" * 70)
print("COMPLETE END-TO-END VERIFICATION")
print("=" * 70)

errors = []

# 1. Config check
print()
print("[1] DECODER CONFIG")
vocab_match = config.vocab_size == len(tokenizer)
pad_match = config.pad_token_id == tokenizer.pad_token_id
eos_match = config.eos_token_id == tokenizer.sep_token_id

print(f"  vocab_size: {config.vocab_size} (tokenizer: {len(tokenizer)}) - {'OK' if vocab_match else 'ERROR'}")
print(f"  pad_token_id: {config.pad_token_id} (tokenizer: {tokenizer.pad_token_id}) - {'OK' if pad_match else 'ERROR'}")
print(f"  eos_token_id: {config.eos_token_id} (tokenizer SEP: {tokenizer.sep_token_id}) - {'OK' if eos_match else 'ERROR'}")

if not vocab_match:
    errors.append(f"vocab_size mismatch: {config.vocab_size} vs {len(tokenizer)}")
if not pad_match:
    errors.append(f"pad_token_id mismatch: {config.pad_token_id} vs {tokenizer.pad_token_id}")
if not eos_match:
    errors.append(f"eos_token_id mismatch: {config.eos_token_id} vs {tokenizer.sep_token_id}")

# 2. Dataset check
print()
print("[2] DATASET LOADING")
dataset = CounterfactualDataset(
    data_dir="data/counterfactual/training_test",
    tokenizer=tokenizer,
    mode="precomputed",
    split="train",
    full_sequence=True,
)
print(f"  Train samples: {len(dataset)}")

val_dataset = CounterfactualDataset(
    data_dir="data/counterfactual/training_test",
    tokenizer=tokenizer,
    mode="precomputed",
    split="val",
    full_sequence=True,
)
print(f"  Val samples: {len(val_dataset)}")

# 3. Sample check
print()
print("[3] SAMPLE FORMAT")
sample = dataset[0]
print(f"  encoder_embeddings: {sample['encoder_embeddings'].shape}")
print(f"  encoder_attention_mask: {sample['encoder_attention_mask'].shape}")
print(f"  decoder_input_ids: {sample['decoder_input_ids'].shape}")
print(f"  labels: {sample['labels'].shape}")

# 4. Labels check
print()
print("[4] LABELS CHECK (critical!)")
labels = sample["labels"]
input_ids = sample["decoder_input_ids"]
print(f"  First input_id: {input_ids[0].item()} ({tokenizer.decode([input_ids[0].item()])})")
print(f"  First label: {labels[0].item()} (should be -100)")
print(f"  Second label: {labels[1].item()} ({tokenizer.decode([labels[1].item()])})")
print(f"  Last label: {labels[-1].item()} ({tokenizer.decode([labels[-1].item()])})")

if labels[0].item() == -100:
    print("  [OK] First label is -100 (BOS masked)")
else:
    errors.append("First label should be -100!")
    print("  [ERROR] First label should be -100!")

# 5. Token range check
print()
print("[5] TOKEN RANGE CHECK")
max_token = input_ids.max().item()
min_token = input_ids.min().item()
print(f"  Token range: {min_token} - {max_token}")
print(f"  Vocab size: {config.vocab_size}")
if max_token < config.vocab_size:
    print("  [OK] All tokens within vocab range")
else:
    errors.append(f"Token {max_token} exceeds vocab size!")
    print(f"  [ERROR] Token {max_token} exceeds vocab size!")

# 6. Embedding dimensions check
print()
print("[6] EMBEDDING DIMENSIONS")
emb_dim = sample["encoder_embeddings"].shape[-1]
expected_dim = config.encoder_hidden_size
print(f"  Encoder embedding dim: {emb_dim}")
print(f"  Expected (encoder_hidden_size): {expected_dim}")
if emb_dim == expected_dim:
    print("  [OK] Embedding dimensions match")
else:
    errors.append(f"Embedding dim mismatch: {emb_dim} vs {expected_dim}")
    print(f"  [ERROR] Dimension mismatch!")

# Summary
print()
print("=" * 70)
if errors:
    print(f"VERIFICATION FAILED - {len(errors)} ERRORS:")
    for e in errors:
        print(f"  - {e}")
else:
    print("VERIFICATION PASSED - ALL CHECKS OK")
print("=" * 70)
