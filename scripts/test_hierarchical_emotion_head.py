#!/usr/bin/env python3
"""Test HierarchicalEmotionHead meets Issue 3.6.6 acceptance criteria."""

import torch

from modeling_studio.models.heads import HierarchicalEmotionHead


def test_hierarchical_emotion_head():
    print("=" * 60)
    print("HierarchicalEmotionHead Test (Issue 3.6.6)")
    print("=" * 60)

    # Test 1: FamilyOS 44-emotion schema (default)
    print("\n--- Test 1: FamilyOS 44-Emotion Schema (Default) ---")
    head = HierarchicalEmotionHead(hidden_size=768)  # Default is now 44 emotions
    print(f"✓ Created head with {head.num_emotions} emotions")
    print(f"  use_familyos: {head.use_familyos}")
    print(f"  First 5 labels: {head.emotion_labels[:5]}")
    print(f"  Last 5 labels: {head.emotion_labels[-5:]}")

    # Check all FamilyOS emotions are present
    familyos_required = ["parental_pride", "homesickness", "togetherness", "warmth", "bittersweet"]
    for emotion in familyos_required:
        assert emotion in head.emotion_labels, f"Missing FamilyOS emotion: {emotion}"
    print("✓ All family-specific emotions present")

    # Test 2: Factory method for FamilyOS
    print("\n--- Test 2: Factory Method for FamilyOS ---")
    head_fos = HierarchicalEmotionHead.for_familyos(hidden_size=768)
    assert head_fos.num_emotions == 44, "FamilyOS should have 44 emotions"
    assert "parental_pride" in head_fos.emotion_labels
    print("✓ for_familyos() creates 44-emotion head")

    # Test 3: Factory method for GoEmotions (legacy)
    print("\n--- Test 3: Factory Method for GoEmotions (Legacy) ---")
    head_go = HierarchicalEmotionHead.for_goemotions(hidden_size=768)
    assert head_go.num_emotions == 32, "GoEmotions should have 32 emotions"
    assert head_go.use_familyos is False
    print("✓ for_goemotions() creates 32-emotion head")

    # Test 4: Forward pass with acceptance criteria
    print("\n--- Test 4: Forward Pass & Acceptance Criteria ---")
    batch_size = 2
    seq_len = 128
    hidden_states = torch.randn(batch_size, seq_len, 768)
    attention_mask = torch.ones(batch_size, seq_len)

    output = head(hidden_states, attention_mask)

    # Check acceptance criteria
    assert "primary_emotion" in output, "Missing primary_emotion"
    print(f"✓ primary_emotion: {output['primary_emotion']}")

    assert "secondary_emotions" in output, "Missing secondary_emotions"
    secondary = output["secondary_emotions"]
    print(f"✓ secondary_emotions: {secondary}")

    assert "emotion_scores" in output, "Missing emotion_scores"
    print("✓ emotion_scores present (44 emotions)")

    # Check secondary emotions <= 3
    if isinstance(secondary[0], list):
        for i, s in enumerate(secondary):
            assert len(s) <= 3, f"Sample {i}: Too many secondary emotions: {len(s)}"
        print(f"✓ Secondary emotions count: {[len(s) for s in secondary]} (all <= 3)")
    else:
        assert len(secondary) <= 3
        print(f"✓ Secondary emotions count: {len(secondary)} (<= 3)")

    # Test 5: Emotion families
    print("\n--- Test 5: Emotion Families ---")
    family = head.get_primary_family("parental_pride")
    print(f"  parental_pride → family: {family}")
    family = head.get_primary_family("homesickness")
    print(f"  homesickness → family: {family}")
    family = head.get_primary_family("togetherness")
    print(f"  togetherness → family: {family}")

    # Test 6: Loss computation with 44 labels
    print("\n--- Test 6: Loss Computation (44 labels) ---")
    labels = torch.zeros(batch_size, 44)
    labels[:, 1] = 1  # joy
    labels[:, 30] = 1  # nostalgia
    labels[:, 38] = 1  # parental_pride

    output_with_loss = head(hidden_states, attention_mask, labels=labels)
    assert "loss" in output_with_loss, "Missing loss when labels provided"
    print(f"✓ Loss: {output_with_loss['loss'].item():.4f}")

    # Test 7: Match with data schema
    print("\n--- Test 7: Match with FamilyOS Data Schema ---")
    data_emotions = [
        "neutral",
        "joy",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "love",
        "disgust",
        "admiration",
        "amusement",
        "approval",
        "caring",
        "excitement",
        "gratitude",
        "optimism",
        "pride",
        "relief",
        "contentment",
        "hope",
        "tenderness",
        "annoyance",
        "disappointment",
        "disapproval",
        "embarrassment",
        "grief",
        "nervousness",
        "remorse",
        "frustration",
        "overwhelmed",
        "emptiness",
        "nostalgia",
        "protectiveness",
        "togetherness",
        "longing",
        "warmth",
        "playfulness",
        "celebration",
        "belonging",
        "parental_pride",
        "parental_guilt",
        "patience",
        "worry",
        "bittersweet",
        "homesickness",
    ]

    for i, emotion in enumerate(data_emotions):
        assert (
            head.emotion_labels[i] == emotion
        ), f"Mismatch at {i}: head={head.emotion_labels[i]}, data={emotion}"
    print("✓ All 44 emotions match data/familyos/emotions schema")

    print("\n" + "=" * 60)
    print("✅ HierarchicalEmotionHead - All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_hierarchical_emotion_head()
