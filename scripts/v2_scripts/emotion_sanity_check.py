#!/usr/bin/env python
"""Quick sanity check for emotion head inference after Stage A training."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modeling_studio.data.labels import EMOTIONS_FAMILYOS_LABELS, Capability
from modeling_studio.models.modernbert_multitask import ModernBertMultiTaskModel


def main():
    model_path = "outputs/modernbert-multitask-v0-stage-a"
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Read capabilities from saved config
    with open(Path(model_path) / "capabilities.json") as f:
        cap_info = json.load(f)
    capabilities = [Capability(c) for c in cap_info["capabilities"]]

    model = ModernBertMultiTaskModel.from_pretrained(
        model_path,
        capabilities=capabilities,
    )
    model.eval()

    print(f"Loaded heads: {list(model.heads.keys())}")

    # Load a few samples from gold validation
    gold_val = Path("data/familyos/emotions/gold/validation.jsonl")
    samples = []
    with gold_val.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            samples.append(json.loads(line))

    print("\n=== Emotion Head Inference Sanity Check ===")
    print(f"Label schema: {EMOTIONS_FAMILYOS_LABELS.num_labels} classes\n")

    all_probs = []
    for idx, sample in enumerate(samples):
        text = sample["text"]
        true_emotions = sample.get("emotions", [])

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs, capability="emotions")

        logits = outputs.logits[0]  # Shape: (num_labels,)
        probs = torch.sigmoid(logits)
        all_probs.append(probs)

        # Get top-5 predictions
        top_probs, top_indices = torch.topk(probs, 5)

        print(f"--- Sample {idx + 1} ---")
        print(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        print(f"True emotions: {true_emotions}")
        print("Top-5 predictions:")
        for p, i in zip(top_probs.tolist(), top_indices.tolist()):
            label = EMOTIONS_FAMILYOS_LABELS.id2label[i]
            marker = " ✓" if label in true_emotions else ""
            print(f"  {label}: {p:.4f}{marker}")

        # Check max probability
        max_prob = probs.max().item()
        above_05 = (probs > 0.5).sum().item()
        above_03 = (probs > 0.3).sum().item()
        print(f"Max prob: {max_prob:.4f}, >0.5: {above_05}, >0.3: {above_03}\n")

    # Overall statistics
    print("=== Overall Probability Distribution ===")
    stacked = torch.stack(all_probs)
    print(f"Mean prob across all classes: {stacked.mean():.4f}")
    print(f"Max prob across all samples: {stacked.max():.4f}")
    print(f"Min prob across all samples: {stacked.min():.4f}")
    print(f"Std dev: {stacked.std():.4f}")


if __name__ == "__main__":
    main()
    main()
