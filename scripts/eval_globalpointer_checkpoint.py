#!/usr/bin/env python
"""
Quick evaluation script for GlobalPointer checkpoints with different thresholds.
Tests the claim that lowering threshold from 2.0 to 0.0 will boost F1 significantly.
"""

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

# Import from the training script
sys.path.insert(0, str(Path(__file__).parent / "training"))
from train_globalpointer_unified import (
    load_model_and_replace_heads,
    MultiHeadSpanDataset,
    MultiHeadCollator,
    get_data_paths,
    load_config,
    HEADS_TO_REPLACE,
    LABEL_CONFIGS,
)


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


if __name__ == "__main__":
    main()
