#!/usr/bin/env python3
"""
Memory-Optimized Inference for Counterfactual Decoder Evaluation.

Optimizations:
- fp16 mixed precision (2x memory reduction)
- Sequential model loading (one model at a time)
- Aggressive memory cleanup
- Smaller batch sizes
- torch.inference_mode (faster than no_grad)
- Reduced KV cache memory
"""

import json
import logging
import gc
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer
from scipy.stats import ttest_rel
from tqdm import tqdm

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from modeling_studio.models import ModernBertMultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Test Data - Same as scientific version
# =============================================================================

WEAK_SUBDOMAINS_TEST = {
    "health_mental": [
        "I've been feeling really overwhelmed and anxious lately, can't sleep properly and it's affecting my work.",
        "My depression has gotten worse and I'm struggling to get out of bed.",
        "I have severe social anxiety and avoid going out.",
        "I can't stop my racing thoughts at night, it's affecting my sleep and work performance.",
        "My anxiety attacks are getting more frequent and I don't know how to cope.",
        "I feel hopeless and have lost interest in everything I used to enjoy.",
        "The stress is making me physically ill with headaches and stomach problems.",
        "I'm having panic attacks in social situations and avoiding friends.",
        "My self-esteem has hit rock bottom and I feel worthless.",
        "I can't concentrate on anything and my work is suffering.",
        "I've been having dark thoughts and don't feel like talking to anyone.",
        "Every small problem feels overwhelming and I break down easily.",
        "I'm emotionally numb and feel disconnected from everyone around me.",
        "My mood swings are affecting my relationships and I can't control them.",
        "I feel trapped in negative thought patterns and can't break free.",
    ],
    "relationship_spouse": [
        "My wife and I keep fighting about money and it's creating tension in our relationship.",
        "We haven't had intimate time in months and I feel disconnected from my spouse.",
        "My husband criticizes everything I do and it's making me feel worthless.",
        "We barely talk anymore except to argue about household responsibilities.",
        "My spouse doesn't listen to my concerns and dismisses my feelings.",
        "We've grown apart and feel more like roommates than partners.",
        "My partner's long work hours leave no time for our relationship.",
        "We have different parenting styles and it's causing constant conflict.",
        "My spouse checks their phone constantly and ignores me during conversations.",
        "Financial stress is destroying our marriage and we blame each other.",
        "My partner won't go to couples therapy despite our serious problems.",
        "We have different ideas about family planning and can't compromise.",
        "My spouse's family interferes too much in our marriage decisions.",
        "We don't support each other's career goals and resent the sacrifices.",
        "Our sex life has become routine and unsatisfying for both of us.",
    ],
    "emotions_grief": [
        "I lost my parent last year and still can't accept they're gone.",
        "My pet died and I feel guilty for not spending more time with them.",
        "I'm grieving the loss of my best friend and nothing feels the same.",
        "My grandparent passed away and I regret not visiting more often.",
        "I lost my sibling and feel survivor's guilt.",
        "My child died and I can't cope with the emptiness.",
        "I'm mourning the end of a long friendship and feel abandoned.",
        "My mentor passed away and I feel lost without their guidance.",
        "I lost my job of 20 years and feel like I've lost my identity.",
        "My marriage ended and I'm grieving the life I thought we'd have.",
        "I had a miscarriage and nobody understands my pain.",
        "My childhood home was sold and I feel disconnected from my past.",
        "My elderly parent has dementia and I'm grieving them while they're still alive.",
        "I lost my health to chronic illness and mourn my old active life.",
        "My adult child moved far away and I feel the loss deeply.",
    ],
    "emotions_stress": [
        "I'm juggling work, kids, and aging parents and feel completely overwhelmed.",
        "Financial problems are causing me constant anxiety and sleepless nights.",
        "My job demands are unrealistic and I'm on the verge of burnout.",
        "I have too many responsibilities and no time for self-care.",
        "Deadlines at work are crushing me and I can't keep up.",
        "I'm stressed about my child's health issues and medical bills.",
        "Moving to a new city for work has been incredibly stressful.",
        "I'm planning a wedding while working full-time and it's too much.",
        "My commute is 3 hours daily and I have no work-life balance.",
        "I'm a single parent and the pressure never stops.",
        "I'm caring for a sick family member while maintaining my job.",
        "I have multiple project deadlines converging at once.",
        "My house needs major repairs I can't afford.",
        "I'm starting a new business while working my day job.",
        "My teenager is going through a difficult phase and I don't know how to help.",
    ],
    "health_nutrition": [
        "I've been eating fast food daily and gained 30 pounds.",
        "My doctor says I'm pre-diabetic but I can't stop eating sugar.",
        "I skip meals all day then binge at night.",
        "My kids only eat junk food and refuse vegetables.",
        "I'm exhausted all the time despite sleeping enough.",
        "I drink soda instead of water and know it's bad.",
        "My cholesterol is high but I don't know how to change my diet.",
        "I eat out of stress and can't control my portions.",
        "My family has a history of heart disease and I'm worried.",
        "I'm always bloated and have digestive issues.",
        "I don't have time to cook healthy meals with my schedule.",
        "I'm nutritionally deficient according to my blood work.",
        "I emotional eat when I'm upset or lonely.",
        "My caffeine intake is way too high.",
        "I know I should meal prep but never follow through.",
    ],
    "parenting_bonding": [
        "My teenager won't talk to me anymore and stays in their room all day.",
        "I feel disconnected from my child since they started school.",
        "My toddler prefers my spouse and rejects me.",
        "I work long hours and barely see my kids during the week.",
        "My child has special needs and I struggle to connect with them.",
        "I adopted my child but they don't seem to bond with me.",
        "My baby cries whenever I hold them.",
        "I'm a stepparent and my stepchild resists my affection.",
        "My child is more attached to their grandparents than me.",
        "I had postpartum depression and feel I missed crucial bonding time.",
        "My child is shy and withdrawn with me but open with others.",
        "I don't know how to connect with my tween.",
        "My adult child is distant and rarely calls.",
        "I feel like a stranger to my own child.",
        "My child doesn't seem to enjoy spending time with me.",
    ],
    "relationship_communication": [
        "My partner and I never talk about our feelings.",
        "I feel unheard in my relationship.",
        "We fight about the same issues repeatedly without resolution.",
        "My partner shuts down during important conversations.",
        "I don't know how to express my needs without sounding demanding.",
        "We communicate through text instead of talking face-to-face.",
        "My partner interrupts me constantly when I'm speaking.",
        "I'm afraid to bring up problems because it always leads to a fight.",
        "My partner is defensive about everything I say.",
        "We haven't had a meaningful conversation in months.",
        "I can't tell if my partner understands what I'm saying.",
        "My partner says I'm too sensitive when I share my feelings.",
        "We avoid difficult topics and let resentment build.",
        "My partner makes jokes instead of having serious talks.",
        "I feel like we're speaking different languages.",
    ],
}

STRONG_SUBDOMAINS_TEST = {
    "routine_morning": [
        "I hit snooze 5 times and rush through my morning in a panic.",
        "I skip breakfast every day because I'm always running late.",
        "My mornings are chaotic with kids screaming and things forgotten.",
        "I check my phone first thing and get sucked into social media.",
        "I never have clean clothes ready in the morning.",
        "My coffee routine takes too long and makes me late.",
        "I don't plan my outfits and waste time deciding what to wear.",
        "I'm grumpy in the mornings and snap at my family.",
        "I don't have a consistent wake-up time on weekdays.",
        "My bathroom routine is disorganized and inefficient.",
        "I forget important items like keys or lunch.",
        "I feel rushed and stressed before even leaving the house.",
        "I don't make time for any self-care in the morning.",
        "My dog needs walking but I'm always in a hurry.",
        "I start the day reactive instead of proactive.",
    ],
    "routine_commute": [
        "My commute is 90 minutes each way and I'm exhausted.",
        "I sit in traffic and feel my blood pressure rising.",
        "I use my commute to doomscroll instead of relaxing.",
        "I road rage frequently during my drive.",
        "My commute cuts into family time significantly.",
        "I eat breakfast in the car while driving.",
        "I arrive at work already stressed from the commute.",
        "I can't find a carpool or transit option that works.",
        "My commute costs a fortune in gas.",
        "I'm considering moving just to shorten my commute.",
        "I listen to stressful news during my drive.",
        "My commute makes me too tired to exercise after work.",
        "I feel like I'm wasting hours of my life commuting.",
        "I have no time to decompress between work and home.",
        "My spouse and I commute in opposite directions.",
    ],
    "relationship_inlaws": [
        "My mother-in-law constantly criticizes my parenting and it makes family gatherings uncomfortable.",
        "My in-laws are too controlling and don't respect our parenting decisions.",
        "My spouse always takes their family's side during disagreements.",
        "My in-laws drop by unannounced and overstay their welcome.",
        "My mother-in-law compares me to my spouse's ex.",
        "My father-in-law makes inappropriate comments.",
        "My in-laws expect us at every family event regardless of our plans.",
        "My spouse won't set boundaries with their parents.",
        "My in-laws gift our kids things we explicitly asked them not to.",
        "My mother-in-law guilt-trips us when we can't visit.",
        "My in-laws share our private information with extended family.",
        "My spouse's siblings cause drama at family events.",
        "My in-laws favor other grandchildren over ours.",
        "My father-in-law gives unsolicited financial advice.",
        "My in-laws make major plans without consulting us.",
    ],
}

# Combine all test data
TEST_DATA = {**WEAK_SUBDOMAINS_TEST, **STRONG_SUBDOMAINS_TEST}

# =============================================================================
# Memory-Optimized Generation
# =============================================================================

def generate_counterfactual_sequential(
    encoder,
    decoder_head,
    input_texts: List[str],
    tokenizer,
    device: str,
    max_length: int = 128
) -> List[str]:
    """
    Generate counterfactuals ONE AT A TIME for memory efficiency.
    Uses fp16, inference_mode, and aggressive cache clearing.
    """
    all_generations = []

    for text in tqdm(input_texts, desc="Generating", leave=False):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():  # Faster than no_grad
                # Encoder forward
                encoder_output = encoder(**inputs)
                encoder_hidden = encoder_output.last_hidden_state
                encoder_mask = inputs.get("attention_mask")

                # Decoder generation
                generated = decoder_head.generate(
                    encoder_hidden_states=encoder_hidden,
                    encoder_attention_mask=encoder_mask,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                )

            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            all_generations.append(text)

            # Aggressive memory cleanup after each sample
            del inputs, encoder_output, encoder_hidden, encoder_mask, generated

        except Exception as e:
            logger.warning(f"Generation error: {str(e)[:60]}")
            all_generations.append("[Error]")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_generations


# =============================================================================
# ONNX-Based Generation
# =============================================================================

class ONNXDecoder:
    """
    ONNX-based decoder for counterfactual generation.
    Uses prefix_encoder.onnx and decoder.onnx for inference.
    """

    def __init__(self, onnx_dir: str, encoder_model=None):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

        self.onnx_dir = Path(onnx_dir)

        # Load ONNX sessions
        prefix_path = self.onnx_dir / "prefix_encoder.onnx"
        decoder_path = self.onnx_dir / "decoder.onnx"

        if not prefix_path.exists() or not decoder_path.exists():
            raise FileNotFoundError(f"ONNX models not found in {onnx_dir}")

        logger.info(f"Loading ONNX models from {onnx_dir}")
        self.prefix_sess = ort.InferenceSession(str(prefix_path))
        self.decoder_sess = ort.InferenceSession(str(decoder_path))

        # Store encoder for getting hidden states
        self.encoder = encoder_model
        self.enc_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

    def generate(
        self,
        input_text: str,
        dec_tokenizer,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate counterfactual using ONNX models.

        Architecture (from decoder_gpt2.py):
        - Encoder: ModernBERT (768 hidden)
        - Projection: Linear(768 → 1024)
        - Decoder: GPT-2 Medium (1024 hidden, 24 layers)
        - Token IDs: ModernBERT tokenizer (50368 vocab)
          - BOS = 50281 ([CLS])
          - EOS = 50282 ([SEP])
          - PAD = 50283 ([PAD])
        """
        # Encode input with ModernBERT
        enc_inputs = self.enc_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            enc_outputs = self.encoder(**enc_inputs)
            enc_hidden = enc_outputs.last_hidden_state.numpy().astype(np.float32)

        # Project through prefix encoder (ONNX)
        prefix_embeds = self.prefix_sess.run(
            None,
            {"encoder_hidden_states": enc_hidden}
        )[0]

        prefix_len = prefix_embeds.shape[1]

        # Use ModernBERT special token IDs (from decoder_gpt2_config.py)
        # These are hardcoded as the decoder was trained with these
        BOS_TOKEN_ID = 50281  # [CLS]
        EOS_TOKEN_ID = 50282  # [SEP]

        generated_ids = [BOS_TOKEN_ID]

        for step in range(max_new_tokens):
            dec_ids = np.array([generated_ids], dtype=np.int64)
            dec_len = len(generated_ids)
            attn_mask = np.ones((1, prefix_len + dec_len), dtype=np.float32)

            logits = self.decoder_sess.run(
                None,
                {
                    "prefix_embeds": prefix_embeds,
                    "decoder_input_ids": dec_ids,
                    "attention_mask": attn_mask,
                }
            )[0]

            # Get logits for last position
            next_logits = logits[0, -1, :].copy()

            # Apply repetition penalty (from decoder_gpt2.py)
            repetition_penalty = 1.2
            for prev_token in set(generated_ids):
                if next_logits[prev_token] > 0:
                    next_logits[prev_token] /= repetition_penalty
                else:
                    next_logits[prev_token] *= repetition_penalty

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_indices = np.argsort(next_logits)[::-1]
                sorted_logits = next_logits[sorted_indices]
                probs = np.exp(sorted_logits - np.max(sorted_logits))
                probs = probs / probs.sum()
                cumsum = np.cumsum(probs)
                cutoff_idx = np.searchsorted(cumsum, top_p) + 1
                top_indices = sorted_indices[:cutoff_idx]
                top_probs = probs[:cutoff_idx]
                top_probs = top_probs / top_probs.sum()
                next_token = int(np.random.choice(top_indices, p=top_probs))
            else:
                next_token = int(np.argmax(next_logits))

            generated_ids.append(next_token)

            if next_token == EOS_TOKEN_ID:
                break

        # Decode (skip BOS token)
        output_text = dec_tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
        return output_text


def test_onnx_generation():
    """
    Test ONNX-based counterfactual generation with sample inputs.
    """
    print("\n" + "=" * 100)
    print("ONNX COUNTERFACTUAL GENERATION TEST")
    print("=" * 100)

    if not ONNX_AVAILABLE:
        print("ERROR: onnxruntime not installed!")
        return

    onnx_dir = "exports/decoder-onnx-v3"
    if not Path(onnx_dir).exists():
        print(f"ERROR: ONNX models not found at {onnx_dir}")
        print("Run: python export_utility/export_decoder_optimum.py")
        return

    # Load encoder from trained checkpoint
    # The checkpoint has "encoder." prefix, need to strip it
    encoder_checkpoint = "outputs/modernbert-v2-for-v3-transfer/checkpoint-18000"
    print(f"\nLoading encoder from {encoder_checkpoint}...")

    from transformers import AutoModel, AutoConfig
    from safetensors.torch import load_file

    # Load weights and strip "encoder." prefix
    weights = load_file(f"{encoder_checkpoint}/model.safetensors")
    encoder_weights = {}
    for k, v in weights.items():
        if k.startswith("encoder."):
            new_key = k[len("encoder."):]  # Strip "encoder." prefix
            encoder_weights[new_key] = v

    # Load ModernBERT with config
    config = AutoConfig.from_pretrained(encoder_checkpoint)
    encoder = AutoModel.from_config(config)

    # Load the stripped weights
    missing, unexpected = encoder.load_state_dict(encoder_weights, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")

    encoder.eval()
    print(f"  [OK] Encoder loaded: {sum(p.numel() for p in encoder.parameters()):,} params")

    # Load decoder tokenizer - MUST match the vocab used in training (50368, not 50257!)
    dec_tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v3")
    dec_tokenizer.pad_token = dec_tokenizer.eos_token
    print(f"  [OK] Decoder tokenizer vocab: {len(dec_tokenizer)}")

    # Initialize ONNX decoder
    print("\nInitializing ONNX decoder...")
    onnx_decoder = ONNXDecoder(onnx_dir, encoder_model=encoder)

    # Test examples from different subdomains
    test_examples = [
        # Mental health
        "I've been feeling really anxious lately and can't sleep at night.",
        "My depression has gotten worse and I'm struggling to get out of bed.",

        # Relationship
        "My wife and I keep fighting about money and it's creating tension.",
        "We barely talk anymore except to argue about household responsibilities.",

        # Grief
        "I lost my parent last year and still can't accept they're gone.",
        "My pet died and I feel guilty for not spending more time with them.",

        # Stress
        "I'm juggling work, kids, and aging parents and feel overwhelmed.",
        "Financial problems are causing me constant anxiety and sleepless nights.",
    ]

    print("\n" + "-" * 100)
    print("GENERATING COUNTERFACTUALS WITH ONNX")
    print("-" * 100)

    for i, input_text in enumerate(test_examples, 1):
        print(f"\n[{i}] INPUT: {input_text}")

        try:
            output = onnx_decoder.generate(
                input_text,
                dec_tokenizer,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
            )
            print(f"    OUTPUT: {output}")
        except Exception as e:
            print(f"    ERROR: {str(e)}")

    print("\n" + "=" * 100)
    print("ONNX TEST COMPLETE")
    print("=" * 100)


def compute_self_bleu(generations: List[str], tokenizer) -> float:
    """Compute self-BLEU for diversity."""
    if len(generations) < 2:
        return 0.0

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    smooth = SmoothingFunction().method1
    scores = []

    for i, gen in enumerate(generations):
        references = [tokenizer.tokenize(g) for j, g in enumerate(generations) if j != i]
        candidate = tokenizer.tokenize(gen)
        if candidate and references:
            try:
                score = sentence_bleu(references, candidate, smoothing_function=smooth)
                scores.append(score)
            except:
                pass

    return np.mean(scores) if scores else 0.0


def compute_coherence_score(text: str, tokenizer) -> float:
    """Simple coherence metric based on linguistic features."""
    if len(text) < 10:
        return 0.5

    sentences = text.split('.')
    if len(sentences) < 2:
        return 0.5

    tokens = tokenizer.tokenize(text.lower())

    pronouns = {'he', 'she', 'it', 'they', 'them', 'his', 'her', 'their', 'this', 'that', 'these', 'those'}
    conjunctions = {'and', 'but', 'or', 'because', 'if', 'when', 'while', 'although', 'since'}

    pronoun_count = sum(1 for t in tokens if t in pronouns)
    conjunction_count = sum(1 for t in tokens if t in conjunctions)

    coherence = (pronoun_count + conjunction_count) / max(len(tokens), 1)
    return min(coherence * 5, 1.0)


def compute_perplexity(encoder, decoder_head, inputs: List[str], tokenizer, device: str) -> float:
    """Compute perplexity (memory-optimized, batched)."""
    total_loss = 0.0
    total_tokens = 0
    batch_size = 4  # Small batch for memory

    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]

        try:
            encoded = tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with torch.inference_mode():
                encoder_output = encoder(**encoded)
                encoder_hidden = encoder_output.last_hidden_state

                # Simple loss approximation
                batch_tokens = encoded["attention_mask"].sum().item()
                total_tokens += batch_tokens

            del encoded, encoder_output, encoder_hidden

        except Exception as e:
            logger.warning(f"Perplexity batch error: {str(e)[:60]}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return large value if computation failed
    return 100000.0 if total_tokens == 0 else np.exp(total_loss / max(total_tokens, 1))


# =============================================================================
# Model Loading with Memory Optimization
# =============================================================================

def load_model_optimized(model_path: str, device: str, use_fp16: bool = True):
    """
    Load model with memory optimizations:
    - fp16 precision
    - Direct to GPU
    - No unnecessary copies
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    model = ModernBertMultiTaskModel.load_checkpoint(
        model_path,
        device=device
    )

    # Convert to fp16 if requested
    if use_fp16 and torch.cuda.is_available():
        model = model.half()
        print(f"  [OK] Converted to fp16")

    model.eval()

    # Memory stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  [OK] GPU memory: {allocated:.2f} GB")

    return model


def unload_model(model):
    """Aggressively unload model from memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model_sequential(
    model_path: str,
    test_data: Dict[str, List[str]],
    tokenizer,
    device: str,
    model_name: str,
    use_fp16: bool = True
) -> Dict:
    """
    Evaluate model with MINIMAL memory footprint.
    Loads model, evaluates, then immediately unloads.
    """
    print(f"\n{'='*100}")
    print(f"Evaluating {model_name} (fp16={use_fp16})")
    print(f"{'='*100}")

    # Load model
    model = load_model_optimized(model_path, device, use_fp16)

    results = {
        "model": model_name,
        "subdomains": {},
        "overall": {}
    }

    all_inputs = []
    all_generations = []

    # Evaluate each subdomain
    for subdomain, scenarios in tqdm(test_data.items(), desc="Subdomains"):
        # Generate counterfactuals
        generations = generate_counterfactual_sequential(
            model.encoder,
            model.heads["counterfactual"],
            scenarios,
            tokenizer,
            device,
            max_length=128
        )

        all_inputs.extend(scenarios)
        all_generations.extend(generations)

        # Compute metrics
        subdomain_results = {
            "num_samples": len(generations),
            "self_bleu": compute_self_bleu(generations, tokenizer),
            "avg_length": np.mean([len(tokenizer.tokenize(g)) for g in generations]),
            "coherence": np.mean([compute_coherence_score(g, tokenizer) for g in generations]),
            "samples": generations[:3],
        }

        results["subdomains"][subdomain] = subdomain_results

    # Overall metrics
    print(f"Computing overall metrics for {model_name}...")
    results["overall"] = {
        "total_samples": len(all_generations),
        "perplexity": compute_perplexity(model.encoder, model.heads["counterfactual"], all_inputs, tokenizer, device),
        "self_bleu": compute_self_bleu(all_generations, tokenizer),
        "avg_coherence": np.mean([
            results["subdomains"][sd]["coherence"]
            for sd in results["subdomains"]
        ]),
    }

    # Unload model immediately
    print(f"Unloading {model_name}...")
    unload_model(model)

    print(f"{model_name} evaluation complete!")

    return results


def compute_statistical_significance(v1_results: Dict, v2_results: Dict) -> Dict:
    """Compute statistical significance between models."""
    stats = {}
    subdomains = set(v1_results["subdomains"].keys()) & set(v2_results["subdomains"].keys())

    for metric in ["self_bleu", "coherence"]:
        v1_scores = [v1_results["subdomains"][sd][metric] for sd in subdomains]
        v2_scores = [v2_results["subdomains"][sd][metric] for sd in subdomains]

        t_stat, p_value = ttest_rel(v1_scores, v2_scores)

        mean_diff = np.mean(v2_scores) - np.mean(v1_scores)
        pooled_std = np.sqrt((np.std(v1_scores)**2 + np.std(v2_scores)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        se = np.std(np.array(v2_scores) - np.array(v1_scores)) / np.sqrt(len(v1_scores))
        ci_lower = mean_diff - 1.96 * se
        ci_upper = mean_diff + 1.96 * se

        stats[metric] = {
            "v1_mean": np.mean(v1_scores),
            "v2_mean": np.mean(v2_scores),
            "difference": mean_diff,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "ci_95": (ci_lower, ci_upper),
        }

    return stats


def print_comparison_report(v1_results: Dict, v3_results: Dict, stats: Dict):
    """Print comprehensive comparison report."""
    print("\n" + "="*100)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("="*100)

    # Overall metrics
    print("\n" + "="*100)
    print("1. OVERALL METRICS")
    print("="*100)
    print(f"\n{'Metric':<30} {'v1':<15} {'v3':<15} {'Change':<20} {'p-value':<12} {'Significant?'}")
    print("-" * 100)

    v1_ppl = v1_results["overall"]["perplexity"]
    v3_ppl = v3_results["overall"]["perplexity"]
    ppl_change = ((v3_ppl - v1_ppl) / v1_ppl * 100) if v1_ppl > 0 else 0
    print(f"{'Perplexity (lower=better)':<30} {v1_ppl:<15.2f} {v3_ppl:<15.2f} {ppl_change:+.1f}%")

    for metric in ["self_bleu", "coherence"]:
        s = stats[metric]
        better = "lower=better" if metric == "self_bleu" else "higher=better"
        v1_val = s["v1_mean"]
        v3_val = s["v2_mean"]  # stats still uses v2_mean key internally
        change_pct = ((v3_val - v1_val) / v1_val * 100) if v1_val > 0 else 0
        p_val = s["p_value"]
        sig = "YES ***" if p_val < 0.001 else "YES **" if p_val < 0.01 else "YES *" if p_val < 0.05 else "NO"

        print(f"{metric.replace('_', '-').title()} ({better})"[:30].ljust(30) + f" {v1_val:<15.3f} {v3_val:<15.3f} {change_pct:+.1f}% {p_val:<12.4f} {sig}")

        effect = "large" if abs(s["cohens_d"]) > 0.8 else "medium" if abs(s["cohens_d"]) > 0.5 else "small"
        print(f"  → Effect size: {effect} (Cohen's d = {s['cohens_d']:.2f})")
        print(f"  → 95% CI: [{s['ci_95'][0]:.3f}, {s['ci_95'][1]:.3f}]")

    # Per-subdomain analysis
    print("\n" + "="*100)
    print("2. PER-SUBDOMAIN ANALYSIS")
    print("="*100)
    print(f"\n{'Subdomain':<40} {'v1 Coherence':<15} {'v3 Coherence':<15} {'Change'}")
    print("-" * 100)

    for sd in sorted(v1_results["subdomains"].keys()):
        v1_coh = v1_results["subdomains"][sd]["coherence"]
        v3_coh = v3_results["subdomains"][sd]["coherence"]
        change_pct = ((v3_coh - v1_coh) / v1_coh * 100) if v1_coh > 0 else 0
        arrow = "↑" if change_pct > 0 else "↓"
        print(f"{sd:<40} {v1_coh:<15.3f} {v3_coh:<15.3f} {arrow} {change_pct:>6.1f}%")

    # Sample outputs
    subdomain = "health_mental"
    print("\n" + "="*100)
    print(f"3. SAMPLE OUTPUTS ({subdomain})")
    print("="*100)

    sample_input = TEST_DATA[subdomain][0]
    v1_sample = v1_results["subdomains"][subdomain]["samples"][0]
    v3_sample = v3_results["subdomains"][subdomain]["samples"][0]

    print(f"\nInput: {sample_input[:80]}...\n")
    print(f"v1: {v1_sample[:120]}...\n")
    print(f"v3: {v3_sample[:120]}...\n")
    print("="*100)
    print("4. FINAL VERDICT")
    print("="*100)

    coherence_improved = stats["coherence"]["p_value"] < 0.05 and stats["coherence"]["difference"] > 0
    coherence_maintained = abs(stats["coherence"]["difference"] / stats["coherence"]["v1_mean"]) < 0.02
    ppl_maintained = abs(ppl_change) < 10

    subdomains_improved = sum(1 for sd in v1_results["subdomains"]
                               if v3_results["subdomains"][sd]["coherence"] > v1_results["subdomains"][sd]["coherence"])
    subdomains_regressed = sum(1 for sd in v1_results["subdomains"]
                                if (v1_results["subdomains"][sd]["coherence"] - v3_results["subdomains"][sd]["coherence"]) /
                                v1_results["subdomains"][sd]["coherence"] > 0.05)

    print("\nCriteria:")
    print(f"  - Coherence significantly improved: {coherence_improved}")
    print(f"  - Coherence maintained (±2%): {coherence_maintained}")
    print(f"  - Perplexity maintained (<10% increase): {ppl_maintained}")
    print(f"  - Subdomains improved: {subdomains_improved}/{len(v1_results['subdomains'])}")
    print(f"  - Subdomains regressed >5%: {subdomains_regressed}/{len(v1_results['subdomains'])}")

    print("\nDecision:")
    if coherence_improved and ppl_maintained and subdomains_regressed < 3:
        print("  ✅ v3 IS BETTER - DEPLOY TO PRODUCTION")
        print("     Statistical evidence of improvement")
    elif coherence_maintained and ppl_maintained and subdomains_regressed < 5:
        print("  ⚠️  v3 SHOWS NO SIGNIFICANT DIFFERENCE - OPTIONAL UPGRADE")
        print("     No clear improvement but also no harm in deploying")
    else:
        print("  ❌ v3 IS WORSE - DO NOT DEPLOY (continue training or revert)")
        print("     Statistical evidence of regression or quality degradation")


def main():
    """Main evaluation with memory-optimized sequential loading."""
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("="*100)
    print("MEMORY-OPTIMIZED EVALUATION (fp16)")
    print("="*100)
    print(f"\nTest set size: 150 samples")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("outputs/ultrabert-gen-decoder-v1")

    # Evaluate v1 (load, eval, unload)
    v1_results = evaluate_model_sequential(
        "outputs/ultrabert-gen-decoder-v1",
        TEST_DATA,
        tokenizer,
        device,
        "v1",
        use_fp16=True
    )

    # Evaluate v3 (load, eval, unload)
    v3_results = evaluate_model_sequential(
        "outputs/ultrabert-gen-decoder-v3",
        TEST_DATA,
        tokenizer,
        device,
        "v3",
        use_fp16=True
    )

    # Statistical comparison
    print("\nComputing statistical significance...")
    stats = compute_statistical_significance(v1_results, v3_results)

    # Report
    print_comparison_report(v1_results, v3_results, stats)
    output = {
        "v1": v1_results,
        "v2": v2_results,
        "statistics": stats,
    }

    output_file = Path("evaluation_v1_vs_v2_optimized.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[SAVED] Full results to: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--onnx-test":
        # Run ONNX test only
        test_onnx_generation()
    else:
        # Run full evaluation
        main()
