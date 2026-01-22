"""Test loading GlobalPointer checkpoint with release package - Production Inference."""

import json
import torch
from familyos_ultrabert.models import ModernBertMultiTaskModel, GlobalPointerNERHead
from familyos_ultrabert.data.labels import Capability
from familyos_ultrabert.data.globalpointer_collator import (
    NER_GENERAL_LABELS, NER_FAMILY_LABELS, TEMPORAL_LABELS
)
from transformers import AutoTokenizer, AutoModel, AutoConfig

# Configuration
CHECKPOINT_PATH = "d:/Modeling_studio/checkpoints/checkpoint-3000"
ENCODER_PATH = "d:/Modeling_studio/outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
THRESHOLD = 0.0  # Logit threshold (0.0 = sigmoid > 0.5)

print(f"Loading GlobalPointer checkpoint: {CHECKPOINT_PATH.split('/')[-1]}")

# Load capabilities from checkpoint
with open(f"{CHECKPOINT_PATH}/capabilities.json") as f:
    cap_data = json.load(f)
cap_names = cap_data["capabilities"]
caps = [Capability[c.upper()] for c in cap_names]
print(f"Capabilities: {len(caps)} heads")

# Load encoder config
config = AutoConfig.from_pretrained(ENCODER_PATH)
hidden_size = config.hidden_size

# Create model with standard heads first
model = ModernBertMultiTaskModel(config=config, capabilities=caps)

# Initialize encoder (needed before load_state_dict)
model.encoder = AutoModel.from_config(config)
print(f"Created model with {len(model.heads)} heads")

# Replace NER heads with GlobalPointerNERHead (matching training config)
HEAD_CONFIGS = {
    "ner_general": {"labels": NER_GENERAL_LABELS, "head_size": 64},
    "ner_family": {"labels": NER_FAMILY_LABELS, "head_size": 64},
    "temporal": {"labels": TEMPORAL_LABELS, "head_size": 64},
}

for head_name, cfg in HEAD_CONFIGS.items():
    if head_name in model.heads:
        new_head = GlobalPointerNERHead(
            hidden_size=hidden_size,
            num_labels=len(cfg["labels"]),
            head_size=cfg["head_size"],
            use_rope=True,
        )
        model.heads[head_name] = new_head
        print(f"  {head_name}: GlobalPointerNERHead ({len(cfg['labels'])} labels)")

# Load the full trained state dict (encoder + heads)
state_dict = torch.load(f"{CHECKPOINT_PATH}/pytorch_model.bin", weights_only=True, map_location="cpu")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded weights: Missing={len(missing)}, Unexpected={len(unexpected)}")

# Load tokenizer from checkpoint
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
print("Tokenizer loaded")

model.eval()


def extract_entities(text: str, threshold: float = THRESHOLD) -> dict:
    """
    Production inference: Extract entities from text using all GlobalPointer heads.

    Returns dict with entities per head, including char spans and confidence.
    """
    # Tokenize with offset mapping for char spans
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()  # (L, 2)

    results = {}

    with torch.no_grad():
        hidden = model.encoder(**inputs).last_hidden_state

        for head_name, cfg in HEAD_CONFIGS.items():
            head = model.heads[head_name]
            id2label = {v: k for k, v in cfg["labels"].items()}

            # Forward pass
            out = head(hidden, attention_mask=inputs["attention_mask"])
            logits = out["logits"]  # (1, num_labels, L, L)

            # Decode with threshold
            preds = head.decode_batch_efficient(
                logits,
                inputs["attention_mask"],
                threshold=threshold,
                id2label=id2label
            )[0]  # Get first batch item

            # Convert to production format with char spans
            entities = []
            for ent in sorted(preds, key=lambda x: -x["score"]):
                tok_start, tok_end = ent["start"], ent["end"]

                # Token to char span mapping
                char_start = offset_mapping[tok_start][0]
                char_end = offset_mapping[tok_end][1]

                # Extract actual text span
                text_span = text[char_start:char_end]

                # Confidence = sigmoid(logit)
                confidence = torch.sigmoid(torch.tensor(ent["score"])).item()

                entities.append({
                    "text": text_span,
                    "label": ent["label"],
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": tok_start,
                    "token_end": tok_end,
                    "score": ent["score"],
                    "confidence": confidence,
                })

            results[head_name] = entities

    return results


def format_entity(ent: dict) -> str:
    """Format entity for display."""
    return f"{ent['label']}:'{ent['text']}'({ent['confidence']:.0%})"


# Test examples - diverse family/temporal contexts
TEST_EXAMPLES = [
    "My grandmother Sarah lives in New York since 1985.",
    "Uncle Bob and Aunt Mary are coming for Christmas dinner tomorrow.",
    "My daughter Emma celebrated her 5th birthday last week at home.",
    "Dad calls me 'little bear' and we have pizza night every Friday.",
    "Grandpa John passed down his watch to my brother Michael in 2020.",
    "Mom and I visit the family cabin in Colorado every summer vacation.",
    "My sister's dog Max is part of the family since we adopted him.",
    "We celebrate Thanksgiving with cousins at grandma's house annually.",
    "The Johnson family moved to 123 Oak Street in Boston last March.",
    "Every Sunday morning, grandma makes her famous pancakes for everyone.",
]

print("\n" + "="*80)
print("GLOBALPOINTER INFERENCE TEST - PRODUCTION FORMAT")
print(f"Threshold: {THRESHOLD} (logit), Checkpoint: {CHECKPOINT_PATH.split('/')[-1]}")
print("="*80)

# Stats
total_entities = {"ner_family": 0, "ner_general": 0, "temporal": 0}

for i, text in enumerate(TEST_EXAMPLES, 1):
    print(f"\n[{i}] {text}")

    results = extract_entities(text)

    for head_name in ["ner_family", "ner_general", "temporal"]:
        entities = results[head_name][:5]  # Top 5
        total_entities[head_name] += len(results[head_name])

        if entities:
            formatted = [format_entity(e) for e in entities]
            print(f"    {head_name}: {', '.join(formatted)}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total entities found across {len(TEST_EXAMPLES)} examples:")
for head, count in total_entities.items():
    print(f"  {head}: {count}")
print("\n=== SUCCESS ===")
