#!/usr/bin/env python
"""
Test Zero-Shot Intent/Ingress Classification

This script validates that:
1. The encoder from checkpoint-8000 produces good semantic embeddings
2. Label description embeddings capture semantic meaning
3. Zero-shot classification works before any training

Usage:
    python scripts/test_zero_shot_intent_ingress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file

from modeling_studio.models.heads import IntentHeadV2, IngressHeadV2


def load_encoder_from_checkpoint(checkpoint_path: str | Path):
    """Load just the encoder from a checkpoint."""
    from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel, Capability

    checkpoint_path = Path(checkpoint_path)
    print(f"Loading encoder from {checkpoint_path}")

    # Load config
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)

    # Create model with minimal capabilities (just need encoder)
    model = ModernBertMultiTaskModel(
        config=config,
        capabilities=[Capability.EMBEDDING],  # Minimal - just need encoder
        freeze_encoder=False,
    )
    model._init_encoder()

    # Load weights
    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        weights_path = checkpoint_path / "pytorch_model.bin"
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Load encoder weights
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    model.encoder.load_state_dict(encoder_state, strict=True)
    print(f"  Loaded encoder: {len(encoder_state)} tensors")

    return model.encoder


def encode_texts(encoder, tokenizer, texts: list[str], device: str = "cpu"):
    """Encode texts and return [CLS] embeddings."""
    encoder.eval()
    encoder.to(device)

    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        outputs = encoder(**inputs)
        # Get [CLS] token embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0]

    return cls_embeddings


def test_zero_shot_intent(encoder, tokenizer, device: str = "cpu"):
    """Test zero-shot intent classification (MULTI-LABEL)."""
    print("\n" + "=" * 60)
    print("ZERO-SHOT INTENT CLASSIFICATION TEST (MULTI-LABEL)")
    print("=" * 60)

    # Create IntentHeadV2 with MULTI-LABEL enabled
    intent_head = IntentHeadV2(hidden_size=768, multi_label=True)
    intent_head.to(device)

    # Initialize label embeddings from descriptions
    print("\nInitializing label embeddings from descriptions...")
    intent_head.init_from_descriptions(encoder, tokenizer)

    # Check label similarity matrix
    sim_matrix = intent_head.get_label_similarities()
    print("\nLabel similarity matrix (after init from descriptions):")
    print("  Diagonal (self-sim):", [f"{x:.3f}" for x in torch.diag(sim_matrix).tolist()])

    # Find most similar pairs
    sim_matrix_no_diag = sim_matrix.clone()
    sim_matrix_no_diag.fill_diagonal_(-1)
    max_sim_idx = sim_matrix_no_diag.argmax()
    i, j = max_sim_idx // 8, max_sim_idx % 8
    print(f"  Most similar pair: {intent_head.label_names[i]} <-> {intent_head.label_names[j]} = {sim_matrix[i,j]:.3f}")

    # Test sentences - now with MULTIPLE expected intents
    test_cases = [
        ("Remember when we went to the beach last summer?", ["log_memory", "query_memory"]),  # recalling AND querying
        ("What did grandma say about the recipe?", ["query_memory"]),
        ("Remind me to call mom tomorrow at 5pm", ["set_reminder"]),
        ("I'm feeling so happy today!", ["express_feeling"]),
        ("Should I take the new job offer?", ["seek_advice"]),
        ("Dad got promoted at work!", ["share_news", "express_feeling"]),  # news + excitement
        ("I miss the old days when we were kids", ["reflect", "express_feeling"]),  # reflection + emotion
        ("Hello, how are you?", ["other"]),
        # Complex multi-intent cases
        ("Today we went to the park with the kids, it was wonderful", ["log_memory", "express_feeling"]),
        ("I'm worried about grandpa's health, should we take him to the doctor?", ["express_feeling", "seek_advice"]),
        ("Remember to pick up dad's medicine and also tell me what he said about Christmas", ["set_reminder", "query_memory"]),
    ]

    print("\nZero-shot MULTI-LABEL predictions:")
    print("-" * 90)

    threshold = 0.15  # For multi-label, we use sigmoid threshold

    for text, expected_list in test_cases:
        # Encode text
        hidden = encode_texts(encoder, tokenizer, [text], device)
        hidden = hidden.unsqueeze(1).expand(-1, 10, -1)  # Fake seq_len for head

        # Get prediction
        output = intent_head(hidden)
        probs = output["probabilities"][0]  # sigmoid probs for multi-label

        # Get all labels above threshold
        detected = [(intent_head.label_names[i], probs[i].item())
                    for i in range(len(intent_head.label_names))
                    if probs[i].item() > threshold]
        detected = sorted(detected, key=lambda x: -x[1])

        # Get primary (highest)
        primary_idx = probs.argmax().item()
        primary_label = intent_head.label_names[primary_idx]

        # Check if expected are covered
        detected_labels = {d[0] for d in detected}
        expected_set = set(expected_list)
        coverage = len(detected_labels & expected_set) / len(expected_set)

        print(f"\"{text[:70]}...\"")
        print(f"  Expected: {expected_list}")
        print(f"  Primary:  {primary_label} ({probs[primary_idx]:.2%})")
        print(f"  All [{threshold:.0%}]: {[(l, f'{p:.1%}') for l,p in detected]}")
        print(f"  Coverage: {coverage:.0%}")
        print()

    print("-" * 90)
    print("Multi-label mode shows ALL intents above threshold, not just top-1")

    return 0.0  # No accuracy for multi-label demo


def test_zero_shot_ingress(encoder, tokenizer, device: str = "cpu"):
    """Test zero-shot ingress classification (MULTI-LABEL)."""
    print("\n" + "=" * 60)
    print("ZERO-SHOT INGRESS CLASSIFICATION TEST (MULTI-LABEL)")
    print("=" * 60)

    # Create IngressHeadV2 with MULTI-LABEL enabled
    ingress_head = IngressHeadV2(hidden_size=768, multi_label=True)
    ingress_head.to(device)

    # Initialize label embeddings from descriptions
    print("\nInitializing label embeddings from descriptions...")
    ingress_head.init_from_descriptions(encoder, tokenizer)

    # Test sentences - now with MULTIPLE expected domains
    test_cases = [
        ("Woke up early, had coffee and journaled", ["DIARY"]),
        ("Need to pick up groceries and dry cleaning", ["TASK"]),
        ("My blood pressure reading was high today", ["HEALTH", "DIARY"]),  # health + daily log
        ("The electricity bill is due next week", ["FINANCE", "TASK"]),  # finance + action needed
        ("Called mom, she's doing well", ["RELATIONSHIP", "DIARY"]),  # family + daily log
        ("Had a productive meeting with my boss", ["WORK", "DIARY"]),  # work + daily log
        ("How do I change my password in the app?", ["META"]),
        ("Remember when we visited Paris in 2019?", ["MEMORY"]),
        ("Planning a surprise party for dad's birthday", ["PLANNING", "CELEBRATION", "RELATIONSHIP"]),
        ("My daughter just graduated from college!", ["CELEBRATION", "RELATIONSHIP"]),
        ("Worried about the economy these days", ["CONCERN", "FINANCE"]),  # worry + financial
        ("Thank you for always being there for me", ["GRATITUDE", "RELATIONSHIP"]),
        # Complex multi-domain cases
        ("Grandma's medical bills are piling up, I'm worried", ["HEALTH", "FINANCE", "CONCERN"]),
        ("Planning dad's retirement party, need to check the budget", ["PLANNING", "CELEBRATION", "FINANCE", "RELATIONSHIP"]),
        ("Mom's doctor said her diabetes is improving, such good news!", ["HEALTH", "RELATIONSHIP", "CELEBRATION"]),
    ]

    print("\nZero-shot MULTI-LABEL predictions:")
    print("-" * 90)

    threshold = 0.15  # For multi-label, we use sigmoid threshold

    for text, expected_list in test_cases:
        # Encode text
        hidden = encode_texts(encoder, tokenizer, [text], device)
        hidden = hidden.unsqueeze(1).expand(-1, 10, -1)  # Fake seq_len for head

        # Get prediction
        output = ingress_head(hidden)
        probs = output["probabilities"][0]  # sigmoid probs for multi-label

        # Get all labels above threshold
        detected = [(ingress_head.label_names[i], probs[i].item())
                    for i in range(len(ingress_head.label_names))
                    if probs[i].item() > threshold]
        detected = sorted(detected, key=lambda x: -x[1])

        # Get primary (highest)
        primary_idx = probs.argmax().item()
        primary_label = ingress_head.label_names[primary_idx]

        # Check if expected are covered
        detected_labels = {d[0] for d in detected}
        expected_set = set(expected_list)
        coverage = len(detected_labels & expected_set) / len(expected_set)

        print(f"\"{text[:70]}...\"")
        print(f"  Expected: {expected_list}")
        print(f"  Primary:  {primary_label} ({probs[primary_idx]:.2%})")
        print(f"  All [{threshold:.0%}]: {[(l, f'{p:.1%}') for l,p in detected]}")
        print(f"  Coverage: {coverage:.0%}")
        print()

    print("-" * 90)
    print("Multi-label mode shows ALL domains above threshold - essential for FamilyOS routing!")

    return 0.0  # No accuracy for multi-label demo


def main():
    """Run zero-shot tests."""
    checkpoint_path = Path("checkpoints/checkpoint-8000")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")

    # Load encoder
    encoder = load_encoder_from_checkpoint(checkpoint_path)
    encoder.to(device)
    encoder.eval()

    # Run tests
    test_zero_shot_intent(encoder, tokenizer, device)
    test_zero_shot_ingress(encoder, tokenizer, device)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Multi-label mode is essential for FamilyOS K1 routing:")
    print("  - Intent: Returns primary + all[] + scores{}")
    print("  - Ingress: Returns domains[] above threshold")
    print()
    print("These are ZERO-SHOT results (no training on intent/ingress data).")
    print("Training will significantly improve multi-label detection.")
    print("=" * 60)
    print("Training will significantly improve these numbers.")


if __name__ == "__main__":
    main()
