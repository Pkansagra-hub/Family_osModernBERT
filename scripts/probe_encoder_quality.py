#!/usr/bin/env python
"""
Encoder Quality Probe for NER Head Diagnosis

PURPOSE: Determine if the encoder is the bottleneck or if head architecture is the issue.

HYPOTHESIS:
- If encoder representations for "learned" (verb) vs "graduation" (milestone) are SIMILAR:
  → Encoder is the problem, head improvements won't help much
- If encoder representations are DIFFERENT but head still fails:
  → Head is the bottleneck, improving head architecture will help

TESTS:
1. Verb vs Milestone distinction (NF-001)
2. Determiner vs Entity distinction (NF-004)
3. Adjective vs Entity distinction (NF-003)
4. Partial vs Full entity span (NG-004)

OUTPUT: Diagnostic report with cosine similarities and recommendations.

Usage:
    python scripts/probe_encoder_quality.py
    python scripts/probe_encoder_quality.py --checkpoint outputs/modernbert-v2-for-v3-transfer
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Default checkpoint
DEFAULT_CHECKPOINT = "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"


def load_model_and_tokenizer(checkpoint_path: str) -> Tuple:
    """Load the multi-task model and tokenizer."""
    from transformers import AutoModel, AutoConfig

    print(f"Loading model from: {checkpoint_path}")

    # Try to load from checkpoint first
    checkpoint = Path(checkpoint_path)
    if checkpoint.exists() and (checkpoint / "model.safetensors").exists():
        print(f"Loading trained encoder from checkpoint: {checkpoint_path}")
        # Load the config to check architecture
        config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            checkpoint_path,
            config=config,
            trust_remote_code=True
        )
    else:
        print("Checkpoint not found, loading base ModernBERT encoder...")
        model = AutoModel.from_pretrained(
            "answerdotai/ModernBERT-base",
            trust_remote_code=True
        )

    model.eval()

    # Load tokenizer from checkpoint if available, else from base
    if checkpoint.exists() and (checkpoint / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    return model, tokenizer


def get_token_hidden_states(
    model,
    tokenizer,
    text: str,
    target_word: str,
    layer: int = -1
) -> torch.Tensor:
    """
    Extract hidden states for a specific word in the text.

    Args:
        model: The encoder model
        tokenizer: Tokenizer
        text: Full sentence
        target_word: Word to extract representation for
        layer: Which layer to extract from (-1 = last, -6 = layer 17, etc.)

    Returns:
        Hidden state tensor for the target word (averaged over subwords)
    """
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]  # (1, seq_len, 768)

    # Find the target word in the text (case-insensitive search)
    text_lower = text.lower()
    target_lower = target_word.lower()
    target_start = text_lower.find(target_lower)

    if target_start == -1:
        raise ValueError(f"Target word '{target_word}' not found in text: {text}")
    target_end = target_start + len(target_word)

    # Find which tokens correspond to the target word
    token_indices = []
    for i, (start, end) in enumerate(offset_mapping):
        # Skip special tokens (offset 0,0)
        if start == 0 and end == 0:
            continue
        # Check if this token overlaps with our target word
        if start < target_end and end > target_start:
            token_indices.append(i)

    if not token_indices:
        # Debug: print what we found
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        print(f"    DEBUG: Looking for '{target_word}' (pos {target_start}-{target_end})")
        print(f"    DEBUG: Tokens: {list(zip(tokens, offset_mapping))}")
        raise ValueError(f"No tokens found for '{target_word}' in text: {text}")

    # Average the hidden states for all subword tokens
    target_hidden = hidden_states[0, token_indices, :].mean(dim=0)

    return target_hidden


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_probe_test(
    model,
    tokenizer,
    test_name: str,
    pairs: List[Tuple[str, str, str, str]],
    layer: int = -1
) -> Dict:
    """
    Run a probe test comparing pairs of examples.

    Args:
        pairs: List of (text1, word1, text2, word2) tuples
               where word1 should be DIFFERENT from word2 if encoder is good

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    similarities = []

    for text1, word1, text2, word2 in pairs:
        try:
            hidden1 = get_token_hidden_states(model, tokenizer, text1, word1, layer)
            hidden2 = get_token_hidden_states(model, tokenizer, text2, word2, layer)

            sim = cosine_similarity(hidden1, hidden2)
            similarities.append(sim)

            print(f"\n  '{word1}' in: \"{text1}\"")
            print(f"  '{word2}' in: \"{text2}\"")
            print(f"  Cosine Similarity: {sim:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    avg_sim = sum(similarities) / len(similarities) if similarities else 0

    print(f"\n  AVERAGE SIMILARITY: {avg_sim:.4f}")

    return {
        "test_name": test_name,
        "pairs_tested": len(pairs),
        "similarities": similarities,
        "avg_similarity": avg_sim
    }


def run_same_word_different_context_test(
    model,
    tokenizer,
    test_name: str,
    examples: List[Tuple[str, str, str, str, str]],
    layer: int = -1
) -> Dict:
    """
    Test if encoder distinguishes same word in different contexts.

    Args:
        examples: List of (text_verb, word, text_entity, word, expected_difference)
    """
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"{'='*60}")

    similarities = []

    for text_verb, text_entity, word, context_verb, context_entity in examples:
        try:
            hidden_verb = get_token_hidden_states(model, tokenizer, text_verb, word, layer)
            hidden_entity = get_token_hidden_states(model, tokenizer, text_entity, word, layer)

            sim = cosine_similarity(hidden_verb, hidden_entity)
            similarities.append(sim)

            print(f"\n  Word: '{word}'")
            print(f"  As {context_verb}: \"{text_verb}\"")
            print(f"  As {context_entity}: \"{text_entity}\"")
            print(f"  Cosine Similarity: {sim:.4f}")
            print(f"  Interpretation: {'SAME context' if sim > 0.9 else 'DIFFERENT context' if sim < 0.7 else 'AMBIGUOUS'}")

        except Exception as e:
            print(f"  ERROR: {e}")

    avg_sim = sum(similarities) / len(similarities) if similarities else 0

    print(f"\n  AVERAGE SIMILARITY: {avg_sim:.4f}")

    return {
        "test_name": test_name,
        "examples_tested": len(examples),
        "similarities": similarities,
        "avg_similarity": avg_sim
    }


def run_all_probes(model, tokenizer, layer: int = -1) -> Dict:
    """Run all diagnostic probes."""
    results = {}

    # =========================================================================
    # TEST 1: Verb vs Milestone (NF-001)
    # =========================================================================
    # If encoder is good: "learned" (verb) should be DIFFERENT from "graduation" (noun/event)
    verb_vs_milestone_pairs = [
        ("Emma learned to ride a bike", "learned",
         "Emma's graduation ceremony was beautiful", "graduation"),
        ("Sofia passed her driving test", "passed",
         "Sofia's wedding day was perfect", "wedding"),
        ("I finally got promoted at work", "promoted",
         "The birthday party was amazing", "birthday"),
        ("She accepted the job offer", "accepted",
         "The anniversary celebration lasted all day", "anniversary"),
    ]

    results["verb_vs_milestone"] = run_probe_test(
        model, tokenizer,
        "Verb vs Milestone Distinction (NF-001)",
        verb_vs_milestone_pairs,
        layer
    )

    # =========================================================================
    # TEST 2: Determiner vs Entity (NF-004)
    # =========================================================================
    # If encoder is good: "the" should be DIFFERENT from actual pet names
    determiner_vs_entity_pairs = [
        ("Fur the cat likes to sleep", "the",
         "Buddy is our golden retriever", "Buddy"),
        ("The dog ran across the yard", "The",
         "Max chased the ball", "Max"),
        ("I walked the puppy this morning", "the",
         "Bella loves her treats", "Bella"),
    ]

    results["determiner_vs_entity"] = run_probe_test(
        model, tokenizer,
        "Determiner vs Entity Distinction (NF-004)",
        determiner_vs_entity_pairs,
        layer
    )

    # =========================================================================
    # TEST 3: Adjective vs Entity (NF-003, NG-002)
    # =========================================================================
    # If encoder is good: "old" should be DIFFERENT from "ring"
    adjective_vs_entity_pairs = [
        ("Grandma's old ring from 1942", "old",
         "Grandma's antique ring from 1942", "ring"),
        ("Feeling anxious about the meeting", "anxious",
         "John was nervous about the meeting", "John"),
        ("The grateful family gathered together", "grateful",
         "The Smith family gathered together", "Smith"),
    ]

    results["adjective_vs_entity"] = run_probe_test(
        model, tokenizer,
        "Adjective vs Entity Distinction (NF-003, NG-002)",
        adjective_vs_entity_pairs,
        layer
    )

    # =========================================================================
    # TEST 4: Common Noun vs Organization (NG-003)
    # =========================================================================
    # If encoder is good: "meeting" should be DIFFERENT from "Microsoft"
    common_noun_vs_org_pairs = [
        ("Had a meeting about Q2 results", "meeting",
         "Had a call with Microsoft about Q2", "Microsoft"),
        ("Checked my email this morning", "email",
         "Checked my Google account this morning", "Google"),
        ("The afternoon was very busy", "afternoon",
         "The Amazon delivery arrived", "Amazon"),
    ]

    results["common_noun_vs_org"] = run_probe_test(
        model, tokenizer,
        "Common Noun vs Organization Distinction (NG-003)",
        common_noun_vs_org_pairs,
        layer
    )

    # =========================================================================
    # TEST 5: Same Word, Different POS Context
    # =========================================================================
    # Critical test: Does encoder distinguish "learned" as verb vs noun usage?
    same_word_different_pos = [
        ("Emma learned to ride a bike today",
         "The learned professor gave a lecture",
         "learned", "verb (past tense)", "adjective"),
        ("I met with John yesterday",
         "The Met museum is amazing",
         "met", "verb (past tense)", "proper noun"),
        ("She passed the exam easily",
         "The mountain pass was treacherous",
         "pass", "verb (past tense)", "noun"),
    ]

    results["same_word_different_pos"] = run_same_word_different_context_test(
        model, tokenizer,
        "Same Word, Different POS Context",
        same_word_different_pos,
        layer
    )

    # =========================================================================
    # TEST 6: Entity Span Coherence (NG-004)
    # =========================================================================
    # If encoder is good: "Lincoln" in "Lincoln School" should be similar to "School"
    # (indicating they're part of the same entity)
    span_coherence_tests = [
        ("We visited Lincoln School yesterday", "Lincoln",
         "We visited Lincoln School yesterday", "School"),
        ("I flew to San Francisco last week", "San",
         "I flew to San Francisco last week", "Francisco"),
        ("Dinner at Bella Notte restaurant", "Bella",
         "Dinner at Bella Notte restaurant", "Notte"),
    ]

    results["span_coherence"] = run_probe_test(
        model, tokenizer,
        "Entity Span Coherence (NG-004) - High similarity = good",
        span_coherence_tests,
        layer
    )

    return results


def analyze_results(results: Dict) -> None:
    """Analyze probe results and provide diagnosis."""
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)

    diagnosis = []

    # Analyze verb vs milestone
    if results["verb_vs_milestone"]["avg_similarity"] < 0.7:
        diagnosis.append("GOOD: Encoder distinguishes verbs from milestones")
    elif results["verb_vs_milestone"]["avg_similarity"] > 0.85:
        diagnosis.append("BAD: Encoder conflates verbs with milestones - ENCODER PROBLEM")
    else:
        diagnosis.append("AMBIGUOUS: Verb/milestone distinction is weak")

    # Analyze determiner vs entity
    if results["determiner_vs_entity"]["avg_similarity"] < 0.6:
        diagnosis.append("GOOD: Encoder distinguishes determiners from entities")
    else:
        diagnosis.append("BAD: Encoder conflates determiners with entities - ENCODER PROBLEM")

    # Analyze adjective vs entity
    if results["adjective_vs_entity"]["avg_similarity"] < 0.7:
        diagnosis.append("GOOD: Encoder distinguishes adjectives from entities")
    else:
        diagnosis.append("BAD: Encoder conflates adjectives with entities - ENCODER PROBLEM")

    # Analyze span coherence (should be HIGH for good encoder)
    if results["span_coherence"]["avg_similarity"] > 0.8:
        diagnosis.append("GOOD: Encoder maintains span coherence (multi-word entities)")
    else:
        diagnosis.append("WEAK: Encoder doesn't strongly link multi-word entity parts")

    # Analyze same word different POS
    if results["same_word_different_pos"]["avg_similarity"] < 0.8:
        diagnosis.append("GOOD: Encoder is context-sensitive (same word, different meaning)")
    else:
        diagnosis.append("BAD: Encoder ignores context (same word = same representation)")

    print("\nPer-Test Diagnosis:")
    for d in diagnosis:
        status = "+" if d.startswith("GOOD") else "-" if d.startswith("BAD") else "?"
        print(f"  [{status}] {d}")

    # Overall verdict
    bad_count = sum(1 for d in diagnosis if "BAD" in d)
    good_count = sum(1 for d in diagnosis if "GOOD" in d)

    print("\n" + "-"*60)
    print("OVERALL VERDICT:")
    print("-"*60)

    if bad_count >= 3:
        print("""
  ENCODER IS THE BOTTLENECK

  The encoder representations do not distinguish between:
  - Verbs and entities
  - Function words and content words
  - Different POS of the same word

  RECOMMENDATION:
  - Improving head architecture alone will NOT solve NER issues
  - Need to retrain encoder with POS-aware objectives
  - Or add auxiliary POS features at inference time
""")
    elif good_count >= 4:
        print("""
  HEAD IS THE BOTTLENECK

  The encoder produces good representations that distinguish:
  - Verbs from milestone nouns
  - Determiners from entity names
  - Same word in different contexts

  RECOMMENDATION:
  - Encoder is working well
  - Head architecture is too simple to use these representations
  - Improve head: Add CRF, span extraction, or context aggregation
  - This should significantly reduce the 66% garbage rate
""")
    else:
        print("""
  MIXED RESULTS

  Some distinctions are good, others are weak.

  RECOMMENDATION:
  - Consider both encoder fine-tuning AND head improvements
  - Start with head improvements (lower risk)
  - If garbage rate doesn't improve, revisit encoder training
""")


def main():
    parser = argparse.ArgumentParser(description="Probe encoder quality for NER diagnosis")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Which layer to probe (-1=last, -6=layer 17, etc.)"
    )
    args = parser.parse_args()

    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Falling back to base ModernBERT encoder...")
        args.checkpoint = "answerdotai/ModernBERT-base"

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # Run all probes
    print("\n" + "="*60)
    print("ENCODER QUALITY PROBE FOR NER DIAGNOSIS")
    print("="*60)
    print(f"Model: {args.checkpoint}")
    print(f"Layer: {args.layer} (negative = from end)")

    results = run_all_probes(model, tokenizer, args.layer)

    # Analyze and diagnose
    analyze_results(results)

    # Summary table
    print("\n" + "="*60)
    print("SIMILARITY SUMMARY TABLE")
    print("="*60)
    print(f"{'Test':<45} {'Avg Sim':<10} {'Interpretation'}")
    print("-"*60)

    for test_name, data in results.items():
        avg = data["avg_similarity"]
        if test_name == "span_coherence":
            # High is good for span coherence
            interp = "GOOD" if avg > 0.8 else "WEAK" if avg > 0.6 else "BAD"
        else:
            # Low is good for distinction tests
            interp = "GOOD" if avg < 0.7 else "BAD" if avg > 0.85 else "AMBIGUOUS"
        print(f"{test_name:<45} {avg:<10.4f} {interp}")


if __name__ == "__main__":
    main()
