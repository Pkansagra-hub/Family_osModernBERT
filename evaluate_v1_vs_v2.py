#!/usr/bin/env python3
"""Comprehensive evaluation of v1 vs v2 counterfactual decoder models."""

import json
import logging
from pathlib import Path
from collections import Counter

import torch
from transformers import AutoTokenizer

from modeling_studio.models import ModernBertMultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Previously weak subdomains (had <1000 samples in v1)
WEAK_SUBDOMAINS_TEST = {
    "health_mental": [
        "I've been feeling really overwhelmed and anxious lately, can't sleep properly and it's affecting my work.",
        "My depression has gotten worse and I'm struggling to get out of bed.",
        "I have severe social anxiety and avoid going out.",
    ],
    "relationship_spouse": [
        "My wife and I keep fighting about money and it's creating tension in our relationship.",
        "We haven't had intimate time in months and I feel disconnected from my spouse.",
        "My husband criticizes everything I do and it's making me feel worthless.",
    ],
    "relationship_inlaws": [
        "My mother-in-law constantly criticizes my parenting and it makes family gatherings uncomfortable.",
        "My in-laws are too controlling and don't respect our parenting decisions.",
        "I feel like my spouse takes their family's side over mine.",
    ],
    "emotions_grief": [
        "It's been 6 months since my mother passed away but I still cry every day.",
        "I lost my child and I don't know how to go on.",
        "My best friend died and I can't seem to accept it.",
    ],
    "routine_morning": [
        "Every morning is chaos getting the kids ready for school, we're always running late.",
        "I can't get my kids out of bed in the morning.",
        "Morning routines are a constant battle with my toddler.",
    ],
    "routine_commute": [
        "My 2-hour commute is exhausting and I barely have energy left for my family.",
        "Traffic is making me late to work and creating stress at home.",
        "I hate my long commute and the time away from my kids.",
    ],
    "relationship_communication": [
        "We don't talk about our feelings and everything stays unresolved.",
        "My partner shuts down whenever we try to discuss problems.",
        "We're always miscommunicating and it causes unnecessary conflicts.",
    ],
}

# Strong subdomains
STRONG_SUBDOMAINS_TEST = {
    "parenting_bonding": ["How can I strengthen my bond with my teenage daughter?", "I want to spend more quality time with my kids."],
    "emotions_stress": ["Work stress is affecting my family life.", "I'm feeling overwhelmed by everything."],
    "health_nutrition": ["How can I get my kids to eat healthier?", "I'm struggling with meal planning for the family."],
}


def generate_counterfactual(encoder, decoder_head, input_text, tokenizer, device):
    """Generate counterfactual using encoder + decoder."""
    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            encoder_output = encoder(**inputs)
            encoder_hidden = encoder_output.last_hidden_state
            encoder_mask = inputs.get("attention_mask")

            # Use decoder head's generate method (prefix injection)
            generated = decoder_head.generate(
                encoder_hidden_states=encoder_hidden,
                encoder_attention_mask=encoder_mask,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
            )

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return text, len(generated[0])
    except Exception as e:
        logger.warning(f"Generation error: {str(e)[:60]}")
        return "[Error]", 0


def evaluate_subdomains(encoder, decoder_head, subdomains_dict, tokenizer, device):
    """Evaluate model on subdomains."""
    results = {}

    for subdomain, scenarios in subdomains_dict.items():
        outputs = []
        lengths = []
        bigrams = Counter()

        for scenario in scenarios:
            text, length = generate_counterfactual(encoder, decoder_head, scenario, tokenizer, device)
            outputs.append(text)
            lengths.append(length)

            tokens = tokenizer.tokenize(text)
            for i in range(len(tokens) - 1):
                bigrams[(tokens[i], tokens[i+1])] += 1

        distinct_bigrams = len(bigrams)
        total_bigrams = sum(bigrams.values()) if bigrams else 1
        diversity = distinct_bigrams / total_bigrams if total_bigrams > 0 else 0.0

        results[subdomain] = {
            "num_tests": len(outputs),
            "avg_length": sum(lengths) / len(lengths) if lengths else 0,
            "total_tokens": sum(lengths),
            "distinct_bigrams": distinct_bigrams,
            "diversity_score": diversity,
            "samples": outputs[:2],
        }

    return results


def main():
    """Run comprehensive evaluation of v1 vs v2 models."""

    print("=" * 80)
    print("V1 vs V2 COUNTERFACTUAL DECODER EVALUATION")
    print("=" * 80)

    print("\nLoading models...")
    try:
        model_v1 = ModernBertMultiTaskModel.load_checkpoint(
            "outputs/ultrabert-gen-decoder-v1",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("  [OK] v1 loaded")
    except Exception as e:
        print(f"  [ERR] v1 error: {e}")
        return

    try:
        model_v2 = ModernBertMultiTaskModel.load_checkpoint(
            "outputs/ultrabert-gen-decoder-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("  [OK] v2 loaded")
    except Exception as e:
        print(f"  [ERR] v2 error: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {
        "timestamp": str(Path.cwd()),
        "device": device,
        "weak_subdomains": {},
        "strong_subdomains": {},
        "summary": {},
    }

    # 1. Evaluate weak subdomains
    print("\n" + "=" * 80)
    print("1. WEAK SUBDOMAIN EVALUATION (Expected to Improve)")
    print("=" * 80)

    print("\nTesting v1 on weak subdomains...")
    v1_weak = evaluate_subdomains(
        model_v1.encoder,
        model_v1.heads["counterfactual"],
        WEAK_SUBDOMAINS_TEST,
        tokenizer,
        device
    )

    print("Testing v2 on weak subdomains...")
    v2_weak = evaluate_subdomains(
        model_v2.encoder,
        model_v2.heads["counterfactual"],
        WEAK_SUBDOMAINS_TEST,
        tokenizer,
        device
    )

    print(f"\n{'Subdomain':35s} {'v1 Diversity':>15s} {'v2 Diversity':>15s} {'Change':>12s}")
    print("-" * 80)

    weak_improvements = []
    for subdomain in WEAK_SUBDOMAINS_TEST.keys():
        v1_div = v1_weak.get(subdomain, {}).get("diversity_score", 0)
        v2_div = v2_weak.get(subdomain, {}).get("diversity_score", 0)
        improvement = ((v2_div - v1_div) / max(v1_div, 0.01)) * 100 if v1_div > 0 else 0
        weak_improvements.append((subdomain, v1_div, v2_div, improvement))

        status = "up" if v2_div > v1_div else "flat" if abs(v2_div - v1_div) < 0.001 else "down"
        print(f"{subdomain:35s} {v1_div:15.3f} {v2_div:15.3f} {status:>5s} {improvement:+7.1f}%")

    results["weak_subdomains"] = {
        "v1": v1_weak,
        "v2": v2_weak,
        "improvements": [{"subdomain": s, "v1": v1, "v2": v2, "improvement_pct": imp} for s, v1, v2, imp in weak_improvements]
    }

    # 2. Evaluate strong subdomains (anti-forgetting)
    print("\n" + "=" * 80)
    print("2. STRONG SUBDOMAIN EVALUATION (Anti-Forgetting Check)")
    print("=" * 80)

    print("\nTesting v1 on strong subdomains...")
    v1_strong = evaluate_subdomains(
        model_v1.encoder,
        model_v1.heads["counterfactual"],
        STRONG_SUBDOMAINS_TEST,
        tokenizer,
        device
    )

    print("Testing v2 on strong subdomains...")
    v2_strong = evaluate_subdomains(
        model_v2.encoder,
        model_v2.heads["counterfactual"],
        STRONG_SUBDOMAINS_TEST,
        tokenizer,
        device
    )

    print(f"\n{'Subdomain':35s} {'v1 Diversity':>15s} {'v2 Diversity':>15s} {'Status':>12s}")
    print("-" * 80)

    strong_maintained = 0
    for subdomain in STRONG_SUBDOMAINS_TEST.keys():
        v1_div = v1_strong.get(subdomain, {}).get("diversity_score", 0)
        v2_div = v2_strong.get(subdomain, {}).get("diversity_score", 0)

        maintained = "OK" if v2_div >= v1_div * 0.9 else "REGRESSED"
        if v2_div >= v1_div * 0.9:
            strong_maintained += 1

        print(f"{subdomain:35s} {v1_div:15.3f} {v2_div:15.3f} {maintained:>12s}")

    results["strong_subdomains"] = {
        "v1": v1_strong,
        "v2": v2_strong,
        "maintained_ratio": strong_maintained / len(STRONG_SUBDOMAINS_TEST) if STRONG_SUBDOMAINS_TEST else 0,
    }

    # 3. Summary
    print("\n" + "=" * 80)
    print("3. SAMPLE OUTPUT (health_mental)")
    print("=" * 80)

    health_v1 = v1_weak.get("health_mental", {}).get("samples", [])
    health_v2 = v2_weak.get("health_mental", {}).get("samples", [])

    if health_v1:
        print(f"\nv1: {health_v1[0][:120]}...")
    if health_v2:
        print(f"v2: {health_v2[0][:120]}...")

    # 4. Final summary
    print("\n" + "=" * 80)
    print("4. SUMMARY")
    print("=" * 80)

    avg_weak_improvement = sum(imp for _, _, _, imp in weak_improvements) / len(weak_improvements) if weak_improvements else 0
    maintain_ratio = results["strong_subdomains"]["maintained_ratio"]

    print(f"\nWeak subdomain improvement (avg):    {avg_weak_improvement:+.1f}%")
    print(f"Strong subdomain maintenance:        {maintain_ratio * 100:.0f}% OK")

    results["summary"] = {
        "weak_domain_avg_improvement": avg_weak_improvement,
        "strong_domain_maintenance_ratio": maintain_ratio,
        "ready_for_production": avg_weak_improvement > 5 and maintain_ratio > 0.8,
    }

    print("\nDecision:")
    if avg_weak_improvement > 5 and maintain_ratio > 0.8:
        print("  [YES] v2 READY FOR PRODUCTION")
        print("        - Weak subdomains improved")
        print("        - Strong subdomains maintained")
    elif avg_weak_improvement > 0 and maintain_ratio > 0.5:
        print("  [MIXED] v2 shows potential")
    else:
        print("  [NO] v2 needs further tuning")

    # Save results
    output_file = Path("evaluation_v1_vs_v2.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
