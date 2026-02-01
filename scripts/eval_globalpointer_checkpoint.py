#!/usr/bin/env python
"""
Evaluation script for GlobalPointer checkpoints with threshold and temperature sweep.

Evaluates:
- Span heads (ner_general, ner_family, temporal) with logit threshold sweep
- Classification heads (intent_v2, ingress_v2) with temperature sweep
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from modeling_studio.data.globalpointer_collator import (
    NER_GENERAL_LABELS,
    NER_FAMILY_LABELS,
    TEMPORAL_LABELS,
)
from modeling_studio.data.labels import INTENT_V2_LABELS, INGRESS_V2_LABELS

# Import from the training script
sys.path.insert(0, str(Path(__file__).parent / "training"))
from train_globalpointer_unified import (
    load_model_and_replace_heads,
    MultiHeadSpanDataset,
    MultiHeadCollator,
    MultiHeadClassificationDataset,
    MultiHeadClassificationCollator,
    get_data_paths,
    load_config,
    HEADS_TO_REPLACE,
    LABEL_CONFIGS,
    CLASSIFICATION_HEADS,
    CLASSIFICATION_LABEL_CONFIGS,
)

# V2 Head configurations
V2_HEADS = ["intent_v2", "ingress_v2"]
V2_LABEL_CONFIGS = {
    "intent_v2": INTENT_V2_LABELS.label2id,
    "ingress_v2": INGRESS_V2_LABELS.label2id,
}


def evaluate_with_threshold(
    model,
    val_loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    """Evaluate with a specific threshold."""
    model.eval()

    all_preds = {h: [] for h in HEADS_TO_REPLACE}
    all_golds = {h: [] for h in HEADS_TO_REPLACE}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            head_names = batch["head_names"]
            span_labels = {k: v.to(device) for k, v in batch["span_labels"].items()}

            # Get encoder hidden states
            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(encoder_output, "last_hidden_state"):
                hidden_states = encoder_output.last_hidden_state
            else:
                hidden_states = (
                    encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                )

            # Evaluate each head
            for head_name in HEADS_TO_REPLACE:
                head = model.heads[head_name]
                head_labels = span_labels[head_name]
                id2label = {v: k for k, v in LABEL_CONFIGS[head_name].items()}

                output = head(hidden_states=hidden_states, attention_mask=attention_mask)
                logits = output["logits"]

                # Decode with specified threshold
                preds = head.decode_batch_efficient(
                    logits,
                    attention_mask=attention_mask,
                    threshold=threshold,
                    id2label=id2label,
                )

                batch_size, num_labels, seq_len, _ = head_labels.shape
                for b in range(batch_size):
                    if head_names[b] != head_name:
                        continue

                    pred_set = set()
                    for entity in preds[b]:
                        pred_set.add((entity["start"], entity["end"], entity["label"]))

                    gold_set = set()
                    for label_id in range(num_labels):
                        positions = torch.where(head_labels[b, label_id] > 0)
                        for i, j in zip(positions[0].tolist(), positions[1].tolist()):
                            gold_set.add((i, j, id2label[label_id]))

                    all_preds[head_name].append(pred_set)
                    all_golds[head_name].append(gold_set)

    # Compute metrics
    results = {}
    total_pred = total_gold = total_correct = 0

    for head_name in HEADS_TO_REPLACE:
        preds_list = all_preds[head_name]
        golds_list = all_golds[head_name]

        if not preds_list:
            continue

        head_pred = sum(len(p) for p in preds_list)
        head_gold = sum(len(g) for g in golds_list)
        head_correct = sum(len(p & g) for p, g in zip(preds_list, golds_list))

        precision = head_correct / head_pred if head_pred > 0 else 0.0
        recall = head_correct / head_gold if head_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[head_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": head_pred,
            "gold": head_gold,
            "correct": head_correct,
        }

        total_pred += head_pred
        total_gold += head_gold
        total_correct += head_correct

    overall_p = total_correct / total_pred if total_pred > 0 else 0.0
    overall_r = total_correct / total_gold if total_gold > 0 else 0.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    )

    results["overall"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "predicted": total_pred,
        "gold": total_gold,
        "correct": total_correct,
    }

    return results


def evaluate_v2_with_temperature(
    model,
    val_loader: DataLoader,
    device: torch.device,
    temperature: float,
    cls_threshold: float = 0.5,
) -> dict:
    """
    Evaluate V2 classification heads with a specific temperature.

    Args:
        model: The multi-task model
        val_loader: DataLoader with classification data
        device: torch device
        temperature: Temperature for logit scaling (lower = sharper)
        cls_threshold: Probability threshold for multi-label predictions

    Returns:
        Dict with per-head and overall metrics
    """
    import math

    model.eval()

    all_preds = {h: [] for h in V2_HEADS}
    all_golds = {h: [] for h in V2_HEADS}

    # Store original log_temperature values to restore later
    original_log_temps = {}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            head_names = batch["head_names"]
            classification_labels = batch["classification_labels"]

            # Get encoder hidden states
            encoder_output = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if hasattr(encoder_output, "last_hidden_state"):
                hidden_states = encoder_output.last_hidden_state
            else:
                hidden_states = (
                    encoder_output[0] if isinstance(encoder_output, tuple) else encoder_output
                )

            # Evaluate each V2 head
            for head_name in V2_HEADS:
                if head_name not in model.heads:
                    continue

                head = model.heads[head_name]
                head_labels = classification_labels.get(head_name)
                if head_labels is None:
                    continue

                head_labels = head_labels.to(device)
                id2label = {v: k for k, v in V2_LABEL_CONFIGS[head_name].items()}

                # Temporarily override log_temperature (temperature property uses exp(log_temperature))
                if head_name not in original_log_temps:
                    original_log_temps[head_name] = head.log_temperature.clone()

                # Set new temperature: log_temperature = log(temperature)
                head.log_temperature.data.fill_(math.log(temperature))

                output = head(hidden_states=hidden_states, attention_mask=attention_mask)
                probs = output["probabilities"]  # (B, num_labels)

                batch_size = probs.shape[0]
                for b in range(batch_size):
                    if head_names[b] != head_name:
                        continue

                    # Predicted labels (above cls_threshold)
                    pred_set = set()
                    pred_indices = torch.where(probs[b] > cls_threshold)[0].tolist()
                    for idx in pred_indices:
                        pred_set.add(id2label[idx])

                    # Gold labels
                    gold_set = set()
                    gold_indices = torch.where(head_labels[b] > 0)[0].tolist()
                    for idx in gold_indices:
                        gold_set.add(id2label[idx])

                    all_preds[head_name].append(pred_set)
                    all_golds[head_name].append(gold_set)

    # Restore original temperatures
    for head_name, orig_log_temp in original_log_temps.items():
        model.heads[head_name].log_temperature.data.copy_(orig_log_temp)

    # Compute metrics
    results = {}
    total_pred = total_gold = total_correct = 0

    for head_name in V2_HEADS:
        preds_list = all_preds[head_name]
        golds_list = all_golds[head_name]

        if not preds_list:
            continue

        head_pred = sum(len(p) for p in preds_list)
        head_gold = sum(len(g) for g in golds_list)
        head_correct = sum(len(p & g) for p, g in zip(preds_list, golds_list))

        precision = head_correct / head_pred if head_pred > 0 else 0.0
        recall = head_correct / head_gold if head_gold > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[head_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": head_pred,
            "gold": head_gold,
            "correct": head_correct,
        }

        total_pred += head_pred
        total_gold += head_gold
        total_correct += head_correct

    if total_pred > 0 and total_gold > 0:
        overall_p = total_correct / total_pred
        overall_r = total_correct / total_gold
        overall_f1 = (
            2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
        )

        results["overall"] = {
            "precision": overall_p,
            "recall": overall_r,
            "f1": overall_f1,
            "predicted": total_pred,
            "gold": total_gold,
            "correct": total_correct,
        }

    return results


def main():
    import argparse
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/training/globalpointer_heads.yaml")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max val samples")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    config = load_config(args.config)
    data_config = config.get("data", {})
    data_root = Path(data_config.get("root", "data"))
    encoder_config = config.get("encoder", {})
    base_checkpoint = encoder_config.get(
        "checkpoint", "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
    )

    # Load BASE model first (with original heads)
    print(f"Loading base model: {base_checkpoint}")
    model = load_model_and_replace_heads(
        base_checkpoint,
        head_size=64,
        dropout=0.1,
    )

    # NOW load the trained GlobalPointer weights from checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading trained weights from: {checkpoint_path}")

    weights_path = checkpoint_path / "model.safetensors"
    if weights_path.exists():
        state_dict = load_file(str(weights_path))
    else:
        state_dict = torch.load(
            checkpoint_path / "pytorch_model.bin", map_location="cpu", weights_only=True
        )

    # Load GlobalPointer head weights
    for head_name in HEADS_TO_REPLACE:
        head_prefix = f"heads.{head_name}."
        head_state = {
            k.replace(head_prefix, ""): v
            for k, v in state_dict.items()
            if k.startswith(head_prefix)
        }
        if head_state:
            model.heads[head_name].load_state_dict(head_state, strict=True)
            print(f"  Loaded {head_name} head: {len(head_state)} tensors")

    # Move entire model to device AFTER loading weights
    model = model.to(device)
    model.eval()

    # Load tokenizer from base checkpoint
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)

    # Load validation data
    data_paths = get_data_paths(data_config, data_root)
    dataset = MultiHeadSpanDataset(
        data_paths=data_paths,
        max_samples_per_head=args.max_samples,
    )

    # Use 10% for validation
    val_size = len(dataset) // 10
    _, val_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collator = MultiHeadCollator(
        tokenizer=tokenizer,
        label_configs=LABEL_CONFIGS,
        max_length=256,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    print(f"Validation samples: {len(val_dataset)}")
    print()

    # Test different thresholds - granular 0.1 steps
    thresholds = [2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0]

    print("=" * 80)
    print("THRESHOLD COMPARISON (Overall)")
    print("=" * 80)
    print()
    print(
        f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Pred':>10} | {'Gold':>10} | {'Correct':>10}"
    )
    print("-" * 80)

    for threshold in thresholds:
        results = evaluate_with_threshold(model, val_loader, device, threshold)
        overall = results["overall"]

        print(
            f"{threshold:>10.1f} | {overall['precision']:>10.3f} | {overall['recall']:>10.3f} | {overall['f1']:>10.3f} | {overall['predicted']:>10} | {overall['gold']:>10} | {overall['correct']:>10}"
        )

    print()
    print("=" * 80)
    print("PER-HEAD THRESHOLD OPTIMIZATION (0.1 granularity)")
    print("=" * 80)
    print()

    # Find best threshold for each head independently - 0.1 granularity
    # More efficient: coarse search then fine search
    coarse_thresholds = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    best_per_head = {}

    for head_name in HEADS_TO_REPLACE:
        print(f"--- {head_name} ---")
        print(f"  {'Thresh':>6} | {'P':>6} | {'R':>6} | {'F1':>6} | {'pred':>6} | {'gold':>6}")
        print(f"  {'-'*50}")

        best_f1 = 0.0
        best_thresh = 0.0
        best_result = None

        # Coarse search first
        for thresh in coarse_thresholds:
            results = evaluate_with_threshold(model, val_loader, device, thresh)
            if head_name in results:
                r = results[head_name]
                print(
                    f"  {thresh:>6.1f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['predicted']:>6} | {r['gold']:>6}"
                )

                if r["f1"] > best_f1:
                    best_f1 = r["f1"]
                    best_thresh = thresh
                    best_result = r

        # Fine search around best threshold (+/- 0.4 in 0.1 steps)
        print(f"  --- Fine search around {best_thresh:.1f} ---")
        fine_thresholds = [
            best_thresh + delta for delta in [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]
        ]
        for thresh in fine_thresholds:
            results = evaluate_with_threshold(model, val_loader, device, thresh)
            if head_name in results:
                r = results[head_name]
                print(
                    f"  {thresh:>6.1f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['predicted']:>6} | {r['gold']:>6}"
                )

                if r["f1"] > best_f1:
                    best_f1 = r["f1"]
                    best_thresh = thresh
                    best_result = r

        best_per_head[head_name] = {
            "threshold": best_thresh,
            "f1": best_f1,
            "result": best_result,
        }
        print(f"  BEST: threshold={best_thresh:.1f} -> F1={best_f1:.3f}")
        print()

    print("=" * 80)
    print("OPTIMAL PER-HEAD THRESHOLDS")
    print("=" * 80)
    print()
    print("# Copy this to your inference config:")
    print("THRESHOLDS = {")
    for head_name, info in best_per_head.items():
        print(f"    '{head_name}': {info['threshold']:.1f},  # F1={info['f1']:.3f}")
    print("}")
    print()

    # Also evaluate with optimal per-head thresholds combined
    print("=" * 80)
    print("COMBINED OPTIMAL THRESHOLDS EVALUATION")
    print("=" * 80)
    print()

    # For combined evaluation, we'd need to modify evaluate_with_threshold to accept per-head thresholds
    # For now, just show the summary
    total_pred = total_gold = total_correct = 0
    for head_name, info in best_per_head.items():
        if info["result"]:
            r = info["result"]
            total_pred += r["predicted"]
            total_gold += r["gold"]
            total_correct += r["correct"]
            print(
                f"  {head_name:12}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} @ threshold={info['threshold']:.1f}"
            )

    if total_pred > 0 and total_gold > 0:
        combined_p = total_correct / total_pred
        combined_r = total_correct / total_gold
        combined_f1 = (
            2 * combined_p * combined_r / (combined_p + combined_r)
            if (combined_p + combined_r) > 0
            else 0.0
        )
        print()
        print(f"  COMBINED: P={combined_p:.3f} R={combined_r:.3f} F1={combined_f1:.3f}")
        print(f"            pred={total_pred} gold={total_gold} correct={total_correct}")

    # ==========================================================================
    # V2 CLASSIFICATION HEADS EVALUATION (Intent/Ingress with Temperature Sweep)
    # ==========================================================================
    print()
    print("=" * 80)
    print("V2 CLASSIFICATION HEADS - TEMPERATURE SWEEP")
    print("=" * 80)
    print()

    # Check if V2 heads exist in model
    v2_heads_present = [h for h in V2_HEADS if h in model.heads]
    if not v2_heads_present:
        print("No V2 heads found in model. Skipping V2 evaluation.")
    else:
        # Load V2 head weights
        for head_name in V2_HEADS:
            if head_name not in model.heads:
                continue
            head_prefix = f"heads.{head_name}."
            head_state = {
                k.replace(head_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(head_prefix)
            }
            if head_state:
                model.heads[head_name].load_state_dict(head_state, strict=True)
                print(f"  Loaded {head_name} head: {len(head_state)} tensors")
            else:
                print(f"  WARNING: No weights found for {head_name}")

        # Load V2 classification data
        v2_data_paths = {
            "intent_v2": [data_root / "processed" / "intent_unified" / "val.jsonl"],
            "ingress_v2": [data_root / "processed" / "ingress_unified" / "val.jsonl"],
        }

        # Filter to heads that exist in model
        v2_data_paths = {k: v for k, v in v2_data_paths.items() if k in model.heads}

        if not v2_data_paths:
            print("No V2 data files found. Skipping V2 evaluation.")
        else:
            v2_dataset = MultiHeadClassificationDataset(
                data_paths=v2_data_paths,
                label_configs=V2_LABEL_CONFIGS,
                max_samples_per_head=args.max_samples,
            )

            # Use 10% for validation (or all if small)
            v2_val_size = max(1, len(v2_dataset) // 10)
            _, v2_val_dataset = torch.utils.data.random_split(
                v2_dataset,
                [len(v2_dataset) - v2_val_size, v2_val_size],
                generator=torch.Generator().manual_seed(42),
            )

            v2_collator = MultiHeadClassificationCollator(
                tokenizer=tokenizer,
                label_configs=V2_LABEL_CONFIGS,
                max_length=256,
            )

            v2_val_loader = DataLoader(
                v2_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=v2_collator,
                num_workers=0,
            )

            print(f"V2 Validation samples: {len(v2_val_dataset)}")
            print()

            # Temperature sweep for V2 heads
            # Lower temperature = sharper distributions (more confident)
            # Higher temperature = softer distributions (more uncertain)
            temperatures = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]
            cls_thresholds = [0.3, 0.4, 0.5]  # Test a few classification thresholds

            print("=" * 80)
            print("TEMPERATURE x THRESHOLD GRID SEARCH (V2 Heads)")
            print("=" * 80)
            print()

            best_v2_per_head = {h: {"f1": 0, "temperature": 0.07, "threshold": 0.5} for h in v2_heads_present}

            for cls_thresh in cls_thresholds:
                print(f"--- Classification Threshold: {cls_thresh} ---")
                print(
                    f"  {'Temp':>6} | {'P':>6} | {'R':>6} | {'F1':>6} | {'pred':>6} | {'gold':>6}"
                )
                print(f"  {'-'*55}")

                for temp in temperatures:
                    results = evaluate_v2_with_temperature(
                        model, v2_val_loader, device, temp, cls_thresh
                    )

                    if "overall" in results:
                        r = results["overall"]
                        print(
                            f"  {temp:>6.3f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['predicted']:>6} | {r['gold']:>6}"
                        )

                    # Track best per head
                    for head_name in v2_heads_present:
                        if head_name in results:
                            hr = results[head_name]
                            if hr["f1"] > best_v2_per_head[head_name]["f1"]:
                                best_v2_per_head[head_name] = {
                                    "f1": hr["f1"],
                                    "precision": hr["precision"],
                                    "recall": hr["recall"],
                                    "temperature": temp,
                                    "threshold": cls_thresh,
                                    "predicted": hr["predicted"],
                                    "gold": hr["gold"],
                                }
                print()

            # Per-head detailed evaluation
            print("=" * 80)
            print("PER-HEAD V2 TEMPERATURE OPTIMIZATION")
            print("=" * 80)
            print()

            for head_name in v2_heads_present:
                print(f"--- {head_name} ---")
                print(f"  {'Temp':>6} | {'Thresh':>6} | {'P':>6} | {'R':>6} | {'F1':>6}")
                print(f"  {'-'*45}")

                best_f1 = 0
                best_temp = 0.07
                best_thresh = 0.5

                for cls_thresh in cls_thresholds:
                    for temp in temperatures:
                        results = evaluate_v2_with_temperature(
                            model, v2_val_loader, device, temp, cls_thresh
                        )
                        if head_name in results:
                            r = results[head_name]
                            if r["f1"] > best_f1:
                                best_f1 = r["f1"]
                                best_temp = temp
                                best_thresh = cls_thresh
                                print(
                                    f"  {temp:>6.3f} | {cls_thresh:>6.2f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} *"
                                )

                print(f"  BEST: temp={best_temp:.3f}, thresh={best_thresh:.2f} -> F1={best_f1:.3f}")
                print()

            # Summary
            print("=" * 80)
            print("OPTIMAL V2 HEAD CONFIGURATIONS")
            print("=" * 80)
            print()
            print("# Copy this to your inference config:")
            print("V2_CONFIG = {")
            for head_name, info in best_v2_per_head.items():
                print(f"    '{head_name}': {{'temperature': {info['temperature']:.3f}, 'threshold': {info['threshold']:.2f}}},  # F1={info['f1']:.3f}")
            print("}")
            print()

            # Final summary
            print("=" * 80)
            print("V2 HEADS BEST RESULTS SUMMARY")
            print("=" * 80)
            print()
            for head_name, info in best_v2_per_head.items():
                print(f"  {head_name:12}: P={info['precision']:.3f} R={info['recall']:.3f} F1={info['f1']:.3f}")
                print(f"                 @ temperature={info['temperature']:.3f}, threshold={info['threshold']:.2f}")
                print()

            # ==========================================================================
            # ZERO-SHOT MODE: Evaluate with description-based label embeddings
            # ==========================================================================
            print()
            print("=" * 80)
            print("ZERO-SHOT EVALUATION (Description-Based Label Embeddings)")
            print("=" * 80)
            print()
            print("This tests what happens when we replace trained label embeddings")
            print("with encoder-projected descriptions (for zero-shot transfer).")
            print()

            # Store original label embeddings to restore later
            original_embeddings = {}
            original_label_names = {}

            # Intent V2 descriptions (same labels, description-based)
            INTENT_DESCRIPTIONS = [
                "User wants to save a memory, log an event, or write a diary entry",
                "User is searching for or asking about past memories",
                "User wants to set a reminder, alarm, or schedule something",
                "User is expressing emotions or feelings",
                "User is asking for advice, help, or guidance",
                "User is sharing news, updates, or information",
                "User is reflecting on life, experiences, or the past",
                "General conversation, greeting, or unclear intent",
            ]

            # Ingress V2 descriptions (same labels, description-based)
            INGRESS_DESCRIPTIONS = [
                "Personal diary entries and daily reflections",
                "Tasks, to-do items, reminders, and action items",
                "Health, medical, wellness, fitness, and mental health",
                "Money, bills, budgets, expenses, and finances",
                "Family, friends, relationships, and social connections",
                "Professional, career, job, and work-related topics",
                "Questions about the app, system, or features",
                "Recalling past events, memories, and nostalgia",
                "Future plans, goals, scheduling, and planning",
                "Achievements, milestones, celebrations, and happy events",
                "Worries, problems, concerns, and issues",
                "Thanks, appreciation, gratitude, and positive acknowledgment",
            ]

            for head_name in v2_heads_present:
                head = model.heads[head_name]
                original_embeddings[head_name] = head.label_embeddings.data.clone()
                original_label_names[head_name] = head.label_names.copy() if hasattr(head, 'label_names') else None

                # Initialize from descriptions
                descriptions = INTENT_DESCRIPTIONS if head_name == "intent_v2" else INGRESS_DESCRIPTIONS
                head.init_label_embeddings_from_encoder(
                    model.encoder,
                    tokenizer,
                    descriptions,
                )
                print(f"  {head_name}: Replaced trained embeddings with description-based")

            print()

            # Temperature sweep for zero-shot (likely needs higher temperatures)
            zs_temperatures = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
            zs_thresholds = [0.2, 0.3, 0.4, 0.5]

            print("ZERO-SHOT TEMPERATURE x THRESHOLD GRID SEARCH")
            print("-" * 60)

            best_zs_per_head = {h: {"f1": 0, "temperature": 1.0, "threshold": 0.3} for h in v2_heads_present}

            for cls_thresh in zs_thresholds:
                print(f"\n--- Classification Threshold: {cls_thresh} ---")
                print(f"  {'Temp':>6} | {'P':>6} | {'R':>6} | {'F1':>6} | {'pred':>6} | {'gold':>6}")
                print(f"  {'-'*55}")

                for temp in zs_temperatures:
                    results = evaluate_v2_with_temperature(
                        model, v2_val_loader, device, temp, cls_thresh
                    )

                    if "overall" in results:
                        r = results["overall"]
                        print(
                            f"  {temp:>6.2f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['predicted']:>6} | {r['gold']:>6}"
                        )

                    # Track best per head
                    for head_name in v2_heads_present:
                        if head_name in results:
                            hr = results[head_name]
                            if hr["f1"] > best_zs_per_head[head_name]["f1"]:
                                best_zs_per_head[head_name] = {
                                    "f1": hr["f1"],
                                    "precision": hr["precision"],
                                    "recall": hr["recall"],
                                    "temperature": temp,
                                    "threshold": cls_thresh,
                                }

            print()
            print("=" * 80)
            print("ZERO-SHOT OPTIMAL CONFIGURATIONS")
            print("=" * 80)
            print()
            print("# Copy this for zero-shot inference:")
            print("ZERO_SHOT_CONFIG = {")
            for head_name, info in best_zs_per_head.items():
                print(f"    '{head_name}': {{'temperature': {info['temperature']:.2f}, 'threshold': {info['threshold']:.2f}}},  # F1={info['f1']:.3f}")
            print("}")
            print()

            # Compare trained vs zero-shot
            print("=" * 80)
            print("TRAINED vs ZERO-SHOT COMPARISON")
            print("=" * 80)
            print()
            print(f"  {'Head':12} | {'Trained F1':>12} | {'Zero-Shot F1':>12} | {'Delta':>8}")
            print(f"  {'-'*55}")
            for head_name in v2_heads_present:
                trained_f1 = best_v2_per_head[head_name]["f1"]
                zs_f1 = best_zs_per_head[head_name]["f1"]
                delta = zs_f1 - trained_f1
                print(f"  {head_name:12} | {trained_f1:>12.3f} | {zs_f1:>12.3f} | {delta:>+8.3f}")
            print()

            # Restore original embeddings
            for head_name in v2_heads_present:
                head = model.heads[head_name]
                head.label_embeddings.data.copy_(original_embeddings[head_name])
                if original_label_names[head_name]:
                    head.label_names = original_label_names[head_name]
            print("  (Restored original trained embeddings)")


if __name__ == "__main__":
    main()
