#!/usr/bin/env python
"""
Quick inference demo for GlobalPointer NER heads.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "training"))

from train_globalpointer_unified import (
    load_model_and_replace_heads,
    HEADS_TO_REPLACE,
    LABEL_CONFIGS,
)


def run_inference(model, tokenizer, text: str, head_name: str, device, threshold: float = 0.0):
    """Run inference on a single text."""
    model.eval()

    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True,
    )

    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Get encoder hidden states
    with torch.no_grad():
        encoder_output = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if hasattr(encoder_output, "last_hidden_state"):
            hidden_states = encoder_output.last_hidden_state
        else:
            hidden_states = encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output

        # Get head output
        head = model.heads[head_name]
        output = head(hidden_states=hidden_states, attention_mask=attention_mask)
        logits = output["logits"]

        # Decode
        id2label = {v: k for k, v in LABEL_CONFIGS[head_name].items()}
        entities = head.decode_batch_efficient(
            logits,
            attention_mask=attention_mask,
            threshold=threshold,
            id2label=id2label,
        )[0]

    # Map token positions back to text
    results = []
    for ent in entities:
        tok_start = ent["start"]
        tok_end = ent["end"]

        # Get character positions from offset mapping
        if tok_start < len(offset_mapping) and tok_end < len(offset_mapping):
            char_start = offset_mapping[tok_start][0]
            char_end = offset_mapping[tok_end][1]
            span_text = text[char_start:char_end]

            results.append({
                "text": span_text,
                "label": ent["label"],
                "score": ent["score"],
                "char_span": (char_start, char_end),
            })

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths
    base_checkpoint = "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
    trained_checkpoint = Path("D:/Modeling_studio/checkpoints/checkpoint-4000")

    # Load model
    print(f"Loading base model...")
    model = load_model_and_replace_heads(base_checkpoint, head_size=64, dropout=0.1)

    # Load trained GlobalPointer weights
    print(f"Loading trained weights from {trained_checkpoint}...")
    weights_path = trained_checkpoint / "model.safetensors"
    state_dict = load_file(str(weights_path))

    for head_name in HEADS_TO_REPLACE:
        head_prefix = f"heads.{head_name}."
        head_state = {
            k.replace(head_prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(head_prefix)
        }
        if head_state:
            model.heads[head_name].load_state_dict(head_state, strict=True)
            print(f"  Loaded {head_name}")

    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    print("\n" + "=" * 70)
    print("GLOBALPOINTER INFERENCE DEMO")
    print("=" * 70)

    # Test examples
    test_cases = [
        # ner_general examples
        ("ner_general", "Apple CEO Tim Cook announced the new iPhone in San Francisco."),
        ("ner_general", "President Biden met with Prime Minister Modi in New Delhi yesterday."),
        ("ner_general", "Microsoft acquired Activision Blizzard for $69 billion."),

        # ner_family examples
        ("ner_family", "My grandmother Sarah always makes the best apple pie for Thanksgiving."),
        ("ner_family", "Uncle Bob and Aunt Mary are celebrating their 50th wedding anniversary."),
        ("ner_family", "Little Timmy just started kindergarten and loves his new teacher."),

        # temporal examples
        ("temporal", "The meeting is scheduled for next Tuesday at 3pm."),
        ("temporal", "We visited Paris last summer and stayed for two weeks."),
        ("temporal", "The report is due by December 31st, 2024."),
    ]

    for head_name, text in test_cases:
        print(f"\n[{head_name.upper()}] {text}")
        print("-" * 70)

        entities = run_inference(model, tokenizer, text, head_name, device, threshold=-0.5)

        if entities:
            for ent in entities:
                print(f"  {ent['label']:15} | \"{ent['text']}\" (score: {ent['score']:.2f})")
        else:
            print("  (no entities found)")

    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Enter text to analyze (type 'quit' to exit)")
    print("Format: <head_name>: <text>")
    print("Heads: ner_general, ner_family, temporal")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n> ").strip()
            if user_input.lower() == "quit":
                break

            if ":" in user_input:
                head_name, text = user_input.split(":", 1)
                head_name = head_name.strip().lower()
                text = text.strip()
            else:
                head_name = "ner_general"
                text = user_input

            if head_name not in HEADS_TO_REPLACE:
                print(f"Unknown head: {head_name}. Use: {HEADS_TO_REPLACE}")
                continue

            entities = run_inference(model, tokenizer, text, head_name, device, threshold=0.0)

            if entities:
                for ent in entities:
                    print(f"  {ent['label']:15} | \"{ent['text']}\" (score: {ent['score']:.2f})")
            else:
                print("  (no entities found)")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
