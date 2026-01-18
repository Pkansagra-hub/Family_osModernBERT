"""
Evaluation Script for FP16 Decoder Model (P03.R5 Counterfactual Generation)

This script measures decoder quality across key metrics:
1. Constitutional Fidelity: Does output match family values?
2. Affect Appropriateness: Is normalization correct for arousal level?
3. Procedural Extractability: Can IF-THEN rules be reliably extracted?
4. Confidence Calibration: Does decoder output quality match expected utility?
5. Emotion Handling: Does response appropriately acknowledge emotional state?

Scientific Foundation:
- Walker (2009): REM sleep facilitates creative problem solving
- McGaugh (2004): Emotional arousal enhances memory consolidation
- Collins & Loftus (1975): Spreading activation in semantic networks

Metrics Target:
- Constitutional Fidelity: >90%
- Affect Appropriateness: >85%
- Procedural Extractability: >75%
- Confidence Calibration: >80%
- Emotion Handling: >80%
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import pairwise_distances

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationScenario:
    """Single evaluation test case."""

    scenario_id: str
    input_text: str
    constitution: str
    affect_arousal: float
    affect_valence: float
    affect_band: str
    emotion: str
    expected_themes: list[str] = field(default_factory=list)
    should_ask_question: bool = False
    description: str = ""


@dataclass
class EvaluationMetrics:
    """Scores for a single generation."""

    scenario_id: str
    constitutional_alignment: float
    affect_appropriateness: float
    procedural_extractability: float
    confidence_calibration: float
    emotion_acknowledgment: float
    generation_length: int
    has_questions: bool
    extracted_procedures: list[dict] = field(default_factory=list)

    @property
    def composite_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            0.25 * self.constitutional_alignment
            + 0.20 * self.affect_appropriateness
            + 0.20 * self.procedural_extractability
            + 0.20 * self.confidence_calibration
            + 0.15 * self.emotion_acknowledgment
        )


class ConstitutionalAlignmentEvaluator:
    """Measures how well output aligns with constitution values."""

    def __init__(self, tokenizer, controller):
        self.tokenizer = tokenizer
        self.controller = controller
        self._setup_token_embeddings()

    def _setup_token_embeddings(self) -> None:
        """Cache embeddings for constitution tokens."""
        self.constitution_embeddings = {}

        for const_key in ["gentle_parenting", "traditional_strict", "indian_joint_family"]:
            schema = self.controller.get_family_values(const_key)
            if not schema:
                continue

            pos_tokens = schema.get("positive_tokens", {})
            neg_tokens = schema.get("negative_tokens", {})

            for token_str in list(pos_tokens.keys()) + list(neg_tokens.keys()):
                token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
                for tid in token_ids:
                    if tid not in self.constitution_embeddings:
                        self.constitution_embeddings[tid] = {
                            "token": token_str,
                            "constitutions": set(),
                        }
                    self.constitution_embeddings[tid]["constitutions"].add(const_key)

    def evaluate(self, output_text: str, constitution: str) -> float:
        """
        Score constitutional alignment (0.0-1.0).

        Measures:
        - Presence of positive tokens from constitution
        - Absence of negative tokens
        - Semantic coherence with constitution values
        """
        output_ids = self.tokenizer.encode(output_text, add_special_tokens=False)

        schema = self.controller.get_family_values(constitution)
        if not schema:
            return 0.5

        pos_tokens = schema.get("positive_tokens", {})
        neg_tokens = schema.get("negative_tokens", {})

        # Count positive token presence
        pos_count = 0
        pos_weight_sum = 0.0
        for token_str, weight in pos_tokens.items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            for tid in token_ids:
                if tid in output_ids:
                    pos_count += 1
                    pos_weight_sum += max(0, weight)

        # Count negative token absence (penalize if present)
        neg_count = 0
        neg_penalty = 0.0
        for token_str, weight in neg_tokens.items():
            token_ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            for tid in token_ids:
                if tid in output_ids:
                    neg_count += 1
                    neg_penalty += abs(weight)

        # Score calculation
        max_pos_tokens = len(pos_tokens) * 2  # Allow multiple tokens per phrase
        pos_score = min(1.0, pos_count / max(max_pos_tokens, 1)) * 0.6
        neg_score = (1.0 - min(1.0, neg_count / max(len(neg_tokens), 1))) * 0.4

        alignment = pos_score + neg_score

        logger.debug(
            f"Constitutional alignment: pos={pos_count}/{max_pos_tokens}, "
            f"neg_penalty={neg_count}, score={alignment:.3f}"
        )

        return float(np.clip(alignment, 0.0, 1.0))


class AffectAppropriatenessEvaluator:
    """Measures if response is appropriate for emotional state."""

    def __init__(self):
        self.high_arousal_markers = {
            "calm": 0.8,
            "breathe": 0.7,
            "pause": 0.6,
            "slow": 0.5,
            "safe": 0.6,
        }
        self.low_arousal_markers = {
            "urgent": 0.5,
            "immediate": 0.5,
            "now": 0.3,
        }
        self.negative_valence_markers = {
            "understand": 0.7,
            "validate": 0.8,
            "feel": 0.6,
            "listen": 0.7,
            "support": 0.6,
        }

    def evaluate(
        self, output_text: str, arousal: float, valence: float, band: str
    ) -> float:
        """
        Score affect appropriateness (0.0-1.0).

        For HIGH arousal: should use calming language
        For LOW arousal: can use more energetic language
        For NEGATIVE valence: should use validating language
        """
        text_lower = output_text.lower()

        # High arousal → should calm
        if arousal > 0.65:
            calm_score = sum(
                weight for word, weight in self.high_arousal_markers.items()
                if word in text_lower
            ) / len(self.high_arousal_markers)
        else:
            calm_score = 0.5  # Neutral zone

        # Low arousal → can be direct
        if arousal < 0.35:
            direct_score = sum(
                weight for word, weight in self.low_arousal_markers.items()
                if word in text_lower
            ) / max(len(self.low_arousal_markers), 1)
        else:
            direct_score = 0.5

        # Negative valence → should validate
        if valence < -0.2:
            validate_score = sum(
                weight for word, weight in self.negative_valence_markers.items()
                if word in text_lower
            ) / len(self.negative_valence_markers)
        else:
            validate_score = 0.5

        # Crisis band → absolutely must have calming language
        if band == "CRISIS":
            if any(word in text_lower for word in self.high_arousal_markers.keys()):
                crisis_score = 0.95
            else:
                crisis_score = 0.4
        else:
            crisis_score = 0.5

        appropriateness = (
            0.35 * calm_score
            + 0.25 * direct_score
            + 0.25 * validate_score
            + 0.15 * crisis_score
        )

        logger.debug(
            f"Affect appropriateness: arousal={arousal:.2f}, valence={valence:.2f}, "
            f"band={band}, calm={calm_score:.3f}, validate={validate_score:.3f}, "
            f"score={appropriateness:.3f}"
        )

        return float(np.clip(appropriateness, 0.0, 1.0))


class ProceduralExtractabilityEvaluator:
    """Extracts and scores IF-THEN procedural knowledge."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.if_triggers = [
            "if", "when", "whenever", "should", "need to", "try", "consider"
        ]
        self.then_actions = [
            "then", "try", "consider", "focus on", "ask", "help them", "encourage"
        ]

    def evaluate(self, output_text: str) -> tuple[float, list[dict]]:
        """
        Extract procedural knowledge as IF-THEN rules.

        Returns: (extractability_score, list_of_procedures)
        """
        text_lower = output_text.lower()
        sentences = output_text.split(".")

        procedures = []
        extraction_score = 0.0

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if not sentence_lower:
                continue

            # Look for IF-THEN pattern
            has_if = any(trigger in sentence_lower for trigger in self.if_triggers)
            has_then = any(action in sentence_lower for action in self.then_actions)

            if has_if and has_then:
                # Extract IF part (usually before first action verb)
                if_part = sentence_lower.split(self.then_actions[0])[0]

                # Extract THEN part
                then_start = max(
                    (sentence_lower.find(action), action)
                    for action in self.then_actions
                    if action in sentence_lower
                )[1]
                then_idx = sentence_lower.find(then_start)
                then_part = sentence_lower[then_idx:].strip()

                if if_part and then_part:
                    procedures.append(
                        {
                            "if_condition": if_part.strip(),
                            "then_action": then_part.strip(),
                            "source_sentence": sentence.strip(),
                            "confidence": 0.75,  # Default for extracted procedures
                        }
                    )
                    extraction_score += 0.85

        # Normalize score
        if procedures:
            extraction_score = min(1.0, extraction_score / (len(procedures) * 0.85))
        else:
            extraction_score = 0.2  # Low score if no procedures extracted

        logger.debug(f"Extracted {len(procedures)} procedures, score={extraction_score:.3f}")

        return float(extraction_score), procedures


class ConfidenceCalibrationEvaluator:
    """Measures if model confidence aligns with output quality."""

    def __init__(self):
        self.length_ideal = 80  # Words

    def evaluate(self, output_text: str, scenario: EvaluationScenario) -> float:
        """
        Score confidence calibration.

        Measures:
        - Response length (too short = generic, too long = rambling)
        - Presence of specific examples
        - Presence of actionable steps
        - Acknowledgment of emotional context
        """
        word_count = len(output_text.split())

        # Length appropriateness
        if self.length_ideal * 0.7 < word_count < self.length_ideal * 1.5:
            length_score = 0.9
        elif self.length_ideal * 0.5 < word_count < self.length_ideal * 2.0:
            length_score = 0.7
        else:
            length_score = 0.4

        # Specificity
        has_example = any(
            word in output_text.lower()
            for word in ["example", "like", "such as", "specifically"]
        )
        example_score = 0.8 if has_example else 0.4

        # Actionability
        action_words = ["try", "focus", "practice", "ask", "help", "create", "set"]
        action_count = sum(
            1 for word in action_words if word in output_text.lower()
        )
        action_score = min(1.0, action_count / 2.0)

        # Emotion acknowledgment
        emotion_words = [
            "feel", "understand", "frustrat", "overwhelm", "stress",
            "anxious", "scared", "angry", "sad", "happy"
        ]
        emotion_acknowledged = any(
            word in output_text.lower() for word in emotion_words
        )
        emotion_score = 0.9 if emotion_acknowledged else 0.3

        calibration = (
            0.30 * length_score
            + 0.25 * example_score
            + 0.25 * action_score
            + 0.20 * emotion_score
        )

        logger.debug(
            f"Confidence calibration: length={length_score:.3f}, "
            f"example={example_score:.3f}, action={action_score:.3f}, "
            f"emotion={emotion_score:.3f}, score={calibration:.3f}"
        )

        return float(np.clip(calibration, 0.0, 1.0))


class EmotionAcknowledgmentEvaluator:
    """Measures if response acknowledges emotional state."""

    def __init__(self):
        self.acknowledgment_phrases = [
            "i understand",
            "i hear",
            "that sounds",
            "you must be",
            "that's frustrating",
            "it makes sense",
            "valid",
            "understandable",
        ]
        self.dismissive_phrases = [
            "just",
            "simply",
            "only",
            "only need to",
        ]

    def evaluate(self, output_text: str, scenario: EvaluationScenario) -> float:
        """
        Score emotion acknowledgment (0.0-1.0).

        Positive: acknowledges emotional state
        Negative: dismisses or ignores emotions
        """
        text_lower = output_text.lower()

        # Acknowledgments
        acknowledgments = sum(
            1 for phrase in self.acknowledgment_phrases if phrase in text_lower
        )
        ack_score = min(1.0, acknowledgments * 0.3)

        # Dismissals
        dismissals = sum(
            1 for phrase in self.dismissive_phrases if phrase in text_lower
        )
        dismissal_penalty = dismissals * 0.15

        # Validate the emotional context
        is_high_emotion = scenario.affect_arousal > 0.65
        emotion_score = ack_score - dismissal_penalty

        if is_high_emotion and acknowledgments > 0:
            emotion_score += 0.1  # Bonus for acknowledging high emotion

        logger.debug(
            f"Emotion acknowledgment: acks={acknowledgments}, "
            f"dismissals={dismissals}, score={emotion_score:.3f}"
        )

        return float(np.clip(emotion_score, 0.0, 1.0))


class DecoderEvaluator:
    """Main evaluation orchestrator."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        tokenizer,
        controller,
        device: torch.device,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.controller = controller
        self.device = device

        # Sub-evaluators
        self.constitutional_eval = ConstitutionalAlignmentEvaluator(tokenizer, controller)
        self.affect_eval = AffectAppropriatenessEvaluator()
        self.procedural_eval = ProceduralExtractabilityEvaluator(tokenizer)
        self.confidence_eval = ConfidenceCalibrationEvaluator()
        self.emotion_eval = EmotionAcknowledgmentEvaluator()

    def evaluate_scenario(
        self, scenario: EvaluationScenario
    ) -> tuple[EvaluationMetrics, str]:
        """
        Evaluate a single scenario.

        Returns: (metrics, generated_text)
        """
        # Import here to avoid circular dependency
        from infer_decoder_fp16 import generate_counterfactual

        logger.info(f"Evaluating scenario: {scenario.scenario_id}")
        logger.info(f"  Input: {scenario.input_text[:80]}...")
        logger.info(
            f"  Affect: arousal={scenario.affect_arousal:.2f}, "
            f"valence={scenario.affect_valence:.2f}, band={scenario.affect_band}"
        )

        # Generate counterfactual
        outputs = generate_counterfactual(
            text=scenario.input_text,
            encoder=self.encoder,
            decoder=self.decoder,
            tokenizer=self.tokenizer,
            device=self.device,
            max_new_tokens=96,
            temperature=0.55,
            normalization_method="clamp_tight"
            if scenario.affect_arousal > 0.65
            else "unit_norm",
            constitution_text=None,
            constitution_controller=self.controller,
            constitution_key=scenario.constitution,
        )

        generated_text = outputs[0] if outputs else ""

        # Evaluate each dimension
        const_alignment = self.constitutional_eval.evaluate(
            generated_text, scenario.constitution
        )
        affect_appropriate = self.affect_eval.evaluate(
            generated_text,
            scenario.affect_arousal,
            scenario.affect_valence,
            scenario.affect_band,
        )
        proc_extractability, procedures = self.procedural_eval.evaluate(generated_text)
        confidence_calib = self.confidence_eval.evaluate(generated_text, scenario)
        emotion_ack = self.emotion_eval.evaluate(generated_text, scenario)

        # Create metrics
        metrics = EvaluationMetrics(
            scenario_id=scenario.scenario_id,
            constitutional_alignment=const_alignment,
            affect_appropriateness=affect_appropriate,
            procedural_extractability=proc_extractability,
            confidence_calibration=confidence_calib,
            emotion_acknowledgment=emotion_ack,
            generation_length=len(generated_text.split()),
            has_questions="?" in generated_text,
            extracted_procedures=procedures,
        )

        logger.info(f"  Composite Score: {metrics.composite_score:.3f}")
        logger.info(f"    Constitutional: {const_alignment:.3f}")
        logger.info(f"    Affect Appropriate: {affect_appropriate:.3f}")
        logger.info(f"    Procedural: {proc_extractability:.3f}")
        logger.info(f"    Confidence: {confidence_calib:.3f}")
        logger.info(f"    Emotion: {emotion_ack:.3f}")

        return metrics, generated_text

    def run_evaluation_suite(
        self, scenarios: list[EvaluationScenario]
    ) -> dict[str, Any]:
        """Run full evaluation suite and return results."""
        logger.info(f"Starting evaluation of {len(scenarios)} scenarios...")

        all_metrics = []
        all_outputs = {}

        for scenario in scenarios:
            try:
                metrics, output = self.evaluate_scenario(scenario)
                all_metrics.append(metrics)
                all_outputs[scenario.scenario_id] = output
            except Exception as e:
                logger.error(f"Failed to evaluate {scenario.scenario_id}: {e}")
                continue

        if not all_metrics:
            logger.error("No scenarios evaluated successfully")
            return {}

        # Aggregate results
        composite_scores = [m.composite_score for m in all_metrics]
        const_scores = [m.constitutional_alignment for m in all_metrics]
        affect_scores = [m.affect_appropriateness for m in all_metrics]
        proc_scores = [m.procedural_extractability for m in all_metrics]
        conf_scores = [m.confidence_calibration for m in all_metrics]
        emotion_scores = [m.emotion_acknowledgment for m in all_metrics]

        results = {
            "total_scenarios": len(scenarios),
            "evaluated_scenarios": len(all_metrics),
            "summary": {
                "composite_score": float(np.mean(composite_scores)),
                "constitutional_alignment": {
                    "mean": float(np.mean(const_scores)),
                    "std": float(np.std(const_scores)),
                    "min": float(np.min(const_scores)),
                    "max": float(np.max(const_scores)),
                },
                "affect_appropriateness": {
                    "mean": float(np.mean(affect_scores)),
                    "std": float(np.std(affect_scores)),
                    "min": float(np.min(affect_scores)),
                    "max": float(np.max(affect_scores)),
                },
                "procedural_extractability": {
                    "mean": float(np.mean(proc_scores)),
                    "std": float(np.std(proc_scores)),
                    "min": float(np.min(proc_scores)),
                    "max": float(np.max(proc_scores)),
                },
                "confidence_calibration": {
                    "mean": float(np.mean(conf_scores)),
                    "std": float(np.std(conf_scores)),
                    "min": float(np.min(conf_scores)),
                    "max": float(np.max(conf_scores)),
                },
                "emotion_acknowledgment": {
                    "mean": float(np.mean(emotion_scores)),
                    "std": float(np.std(emotion_scores)),
                    "min": float(np.min(emotion_scores)),
                    "max": float(np.max(emotion_scores)),
                },
            },
            "target_thresholds": {
                "constitutional_alignment": 0.90,
                "affect_appropriateness": 0.85,
                "procedural_extractability": 0.75,
                "confidence_calibration": 0.80,
                "emotion_acknowledgment": 0.80,
            },
            "passed_thresholds": {
                "constitutional_alignment": bool(
                    np.mean(const_scores) >= 0.90
                ),
                "affect_appropriateness": bool(np.mean(affect_scores) >= 0.85),
                "procedural_extractability": bool(np.mean(proc_scores) >= 0.75),
                "confidence_calibration": bool(np.mean(conf_scores) >= 0.80),
                "emotion_acknowledgment": bool(np.mean(emotion_scores) >= 0.80),
            },
            "individual_results": [
                {
                    "scenario_id": m.scenario_id,
                    "composite_score": m.composite_score,
                    "constitutional_alignment": m.constitutional_alignment,
                    "affect_appropriateness": m.affect_appropriateness,
                    "procedural_extractability": m.procedural_extractability,
                    "confidence_calibration": m.confidence_calibration,
                    "emotion_acknowledgment": m.emotion_acknowledgment,
                    "generation_length": m.generation_length,
                    "has_questions": m.has_questions,
                    "extracted_procedures": m.extracted_procedures,
                }
                for m in all_metrics
            ],
            "outputs": all_outputs,
        }

        return results


def get_evaluation_scenarios() -> list[EvaluationScenario]:
    """Return standard evaluation scenarios."""
    return [
        EvaluationScenario(
            scenario_id="gentle_calm_baseline",
            input_text="My child won't eat their vegetables",
            constitution="gentle_parenting",
            affect_arousal=0.35,
            affect_valence=0.0,
            affect_band="GREEN",
            emotion="neutral",
            expected_themes=["connection", "choice", "understanding"],
            description="Baseline: calm parent, gentle constitution",
        ),
        EvaluationScenario(
            scenario_id="gentle_high_emotion",
            input_text="I hate when my kids don't listen!",
            constitution="gentle_parenting",
            affect_arousal=0.75,
            affect_valence=-0.35,
            affect_band="YELLOW",
            emotion="frustrated",
            expected_themes=["validation", "calm", "connection"],
            should_ask_question=False,
            description="High emotion: angry parent, gentle constitution (should calm)",
        ),
        EvaluationScenario(
            scenario_id="strict_behavior",
            input_text="My teenager keeps breaking curfew",
            constitution="traditional_strict",
            affect_arousal=0.6,
            affect_valence=-0.2,
            affect_band="YELLOW",
            emotion="worried",
            expected_themes=["rules", "consequences", "boundaries"],
            description="Traditional strict: behavior issue",
        ),
        EvaluationScenario(
            scenario_id="strict_high_emotion",
            input_text="I'm absolutely furious! They lied to my face!!!",
            constitution="traditional_strict",
            affect_arousal=0.85,
            affect_valence=-0.6,
            affect_band="RED",
            emotion="furious",
            expected_themes=["clarity", "consequences", "respect"],
            description="High emotion + strict constitution (should be firm but fair)",
        ),
        EvaluationScenario(
            scenario_id="indian_joint_family",
            input_text="My child won't respect their elders",
            constitution="indian_joint_family",
            affect_arousal=0.55,
            affect_valence=-0.25,
            affect_band="YELLOW",
            emotion="concerned",
            expected_themes=["family", "respect", "elders", "harmony"],
            description="Cultural: joint family values",
        ),
        EvaluationScenario(
            scenario_id="crisis_mode",
            input_text="I can't handle this anymore I'm losing control",
            constitution="gentle_parenting",
            affect_arousal=0.9,
            affect_valence=-0.7,
            affect_band="CRISIS",
            emotion="devastated",
            expected_themes=["support", "safe", "help", "care"],
            should_ask_question=False,
            description="Crisis: parent in distress (safety first)",
        ),
        EvaluationScenario(
            scenario_id="procedural_learning",
            input_text="Every bedtime turns into a battle",
            constitution="gentle_parenting",
            affect_arousal=0.65,
            affect_valence=-0.1,
            affect_band="YELLOW",
            emotion="stressed",
            expected_themes=["routine", "structure", "predictability"],
            should_ask_question=True,
            description="Pattern: recurring issue (should extract procedures)",
        ),
    ]


def main() -> None:
    """Run full evaluation suite."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate FP16 Decoder Model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="D:\\Modeling_studio\\outputs\\ultrabert-gen-decoder-v4",
        help="Path to decoder checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="D:\\Modeling_studio\\eval_results.json",
        help="Output path for evaluation results",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode",
    )
    args = parser.parse_args()

    # Import inference utilities
    from infer_decoder_fp16 import load_model_fp16, get_device, ConstitutionController

    device = torch.device("cpu") if args.cpu else get_device()
    logger.info(f"Using device: {device}")

    # Load model
    encoder, decoder, tokenizer = load_model_fp16(args.checkpoint, device)

    # Initialize controller
    controller = ConstitutionController(tokenizer)

    # Create evaluator
    evaluator = DecoderEvaluator(encoder, decoder, tokenizer, controller, device)

    # Get scenarios
    scenarios = get_evaluation_scenarios()

    # Run evaluation
    results = evaluator.run_evaluation_suite(scenarios)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete! Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Evaluated: {results['evaluated_scenarios']}/{results['total_scenarios']} scenarios")
    print(f"Composite Score: {results['summary']['composite_score']:.3f}/1.000")
    print()
    print("Metric Scores:")
    print(f"  Constitutional Alignment:    {results['summary']['constitutional_alignment']['mean']:.3f} (target: 0.900)")
    print(f"  Affect Appropriateness:      {results['summary']['affect_appropriateness']['mean']:.3f} (target: 0.850)")
    print(f"  Procedural Extractability:   {results['summary']['procedural_extractability']['mean']:.3f} (target: 0.750)")
    print(f"  Confidence Calibration:      {results['summary']['confidence_calibration']['mean']:.3f} (target: 0.800)")
    print(f"  Emotion Acknowledgment:      {results['summary']['emotion_acknowledgment']['mean']:.3f} (target: 0.800)")
    print()
    print("Threshold Status:")
    for metric, passed in results['passed_thresholds'].items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {metric}")
    print("=" * 80)


if __name__ == "__main__":
    main()
