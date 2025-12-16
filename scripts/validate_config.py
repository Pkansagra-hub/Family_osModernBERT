#!/usr/bin/env python
"""Validate Stage C config for A100 40GB training."""

import yaml
from pathlib import Path

config_path = "configs/training/multitask/stage_c_decoder.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("FULL CONFIG VALIDATION FOR A100 40GB")
print("=" * 70)

errors = []
warnings = []

# Model section
print("\n[MODEL]")
m = config.get("model", {})
print(f"  checkpoint_path: {m.get('checkpoint_path')}")
print(f"  freeze_encoder: {m.get('freeze_encoder')}")
print(f"  freeze_existing_heads: {m.get('freeze_existing_heads')}")
print(f"  torch_dtype: {m.get('torch_dtype')}")

# Check checkpoint exists
cp = Path(m.get("checkpoint_path", ""))
if cp.exists():
    print("  [OK] Checkpoint exists")
else:
    errors.append(f"Checkpoint NOT FOUND: {cp}")
    print(f"  [ERROR] Checkpoint NOT FOUND: {cp}")

# Decoder section
print("\n[DECODER]")
d = config.get("decoder", {})
print(f"  hidden_size: {d.get('hidden_size')}")
print(f"  num_layers: {d.get('num_layers')}")
print(f"  vocab_size: {d.get('vocab_size')}")
if d.get("vocab_size") != 50368:
    errors.append(f"vocab_size should be 50368, got {d.get('vocab_size')}")
    print("  [ERROR] vocab_size should be 50368!")
else:
    print("  [OK] vocab_size correct")
print(f"  num_attention_heads: {d.get('num_attention_heads')}")
print(f"  num_kv_heads: {d.get('num_kv_heads')}")
print(f"  num_experts: {d.get('num_experts')}")
print(f"  num_experts_per_token: {d.get('num_experts_per_token')}")
print(f"  dense_layers: {d.get('dense_layers')}")
print(f"  moe_layers: {d.get('moe_layers')}")

# Validate layer assignments
dense = d.get("dense_layers", [])
moe = d.get("moe_layers", [])
num_layers = d.get("num_layers", 8)
all_layers = set(dense) | set(moe)
expected_layers = set(range(num_layers))
if all_layers != expected_layers:
    errors.append(f"Layer assignment mismatch: dense={dense}, moe={moe}, expected 0-{num_layers-1}")
    print(f"  [ERROR] Layer assignment mismatch!")
else:
    print(f"  [OK] All {num_layers} layers assigned")

# Data section
print("\n[DATA]")
data = config.get("data", {})
print(f"  train_path: {data.get('train_path')}")
print(f"  embeddings_mode: {data.get('embeddings_mode')}")
print(f"  max_input_length: {data.get('max_input_length')}")
print(f"  max_output_length: {data.get('max_output_length')}")
print(f"  num_workers: {data.get('num_workers')}")
print(f"  load_to_ram: {data.get('load_to_ram')}")

# Training section
print("\n[TRAINING] - CRITICAL FOR 40GB")
t = config.get("training", {})
train_bs = t.get("per_device_train_batch_size")
eval_bs = t.get("per_device_eval_batch_size")
grad_accum = t.get("gradient_accumulation_steps")
effective_bs = train_bs * grad_accum

print(f"  per_device_train_batch_size: {train_bs}")
print(f"  per_device_eval_batch_size: {eval_bs}")
print(f"  gradient_accumulation_steps: {grad_accum}")
print(f"  EFFECTIVE BATCH SIZE: {effective_bs}")

if train_bs > 32:
    warnings.append(f"train_batch_size={train_bs} > 32 may OOM on 40GB!")
    print("  [WARNING] train_batch_size > 32 may OOM on 40GB!")
else:
    print("  [OK] train_batch_size safe for 40GB")

print(f"  learning_rate: {t.get('learning_rate')}")
print(f"  warmup_ratio: {t.get('warmup_ratio')}")
print(f"  num_train_epochs: {t.get('num_train_epochs')}")
print(f"  gradient_checkpointing: {t.get('gradient_checkpointing')}")

if not t.get("gradient_checkpointing"):
    errors.append("gradient_checkpointing MUST be true for 40GB!")
    print("  [ERROR] gradient_checkpointing MUST be true for 40GB!")
else:
    print("  [OK] gradient_checkpointing enabled")

print(f"  bf16: {t.get('bf16')}")
print(f"  fp16: {t.get('fp16')}")
print(f"  optim: {t.get('optim')}")
print(f"  save_steps: {t.get('save_steps')}")
print(f"  eval_steps: {t.get('eval_steps')}")
print(f"  load_best_model_at_end: {t.get('load_best_model_at_end')}")

if t.get("load_best_model_at_end"):
    warnings.append("load_best_model_at_end=true causes double GPU memory!")
    print("  [WARNING] load_best_model_at_end=true causes double GPU memory!")
else:
    print("  [OK] load_best_model_at_end=false (no double loading)")

print(f"  dataloader_num_workers: {t.get('dataloader_num_workers')}")

# Memory estimate
print("\n[MEMORY ESTIMATE]")
print("  Model params (bf16): ~1.5 GB")
print("  Optimizer states: ~3 GB")
print("  Gradients: ~1.5 GB")
print(f"  Activations (checkpointed, bs={train_bs}): ~10-15 GB")
print("  Estimated total: ~16-21 GB")
print("  A100 40GB headroom: ~19-24 GB")
print("  [OK] Should fit comfortably")

# Summary
print("\n" + "=" * 70)
if errors:
    print(f"VALIDATION FAILED - {len(errors)} ERRORS:")
    for e in errors:
        print(f"  - {e}")
else:
    print("VALIDATION PASSED - NO ERRORS")

if warnings:
    print(f"\n{len(warnings)} WARNINGS:")
    for w in warnings:
        print(f"  - {w}")

print("=" * 70)
