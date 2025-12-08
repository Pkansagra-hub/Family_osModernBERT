"""
Comprehensive Data Quality Audit for FamilyOS Dataset

This script implements ALL audit rules defined in docs/DATA_QUALITY_AUDIT.md
Run from project root: python scripts/comprehensive_data_audit.py
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any


# =============================================================================
# CONFIGURATION
# =============================================================================

POSITIVE_SENTIMENTS = {"positive", "very_positive"}
NEGATIVE_SENTIMENTS = {"negative", "very_negative"}
POSITIVE_EMOTIONS = {
    "joy",
    "excitement",
    "love",
    "pride",
    "gratitude",
    "happiness",
    "celebration",
    "relief",
}
NEGATIVE_EMOTIONS = {
    "sadness",
    "grief",
    "frustration",
    "anger",
    "disappointment",
    "worry",
    "overwhelmed",
    "annoyance",
}
MEMORY_INTENTS = {"query_memory", "log_memory", "reflect"}
TASK_INTENTS = {"set_reminder", "seek_advice"}


# =============================================================================
# AUDIT RESULT TRACKING
# =============================================================================


class AuditResults:
    """Track all audit findings."""

    def __init__(self):
        self.violations = defaultdict(list)
        self.counts = Counter()
        self.sample_count = 0

    def add_violation(self, rule_id: str, sample_id: str, details: str):
        self.violations[rule_id].append({"sample_id": sample_id, "details": details})
        self.counts[rule_id] += 1

    def get_summary(self) -> dict:
        return {
            "total_samples": self.sample_count,
            "violations_by_rule": dict(self.counts),
            "total_violations": sum(self.counts.values()),
        }


# =============================================================================
# CATEGORY 1: Hub Routing vs Task Fields
# =============================================================================


def check_emo_routing_vs_emotions(sample: dict, results: AuditResults):
    """Rule 1.1: EMO=False should not have non-neutral emotions."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    emotions = tasks.get("emotions", [])
    sample_id = sample.get("id", "unknown")

    if not hub.get("EMO", True):  # EMO is False
        non_neutral = [e for e in emotions if e != "neutral"]
        if non_neutral:
            results.add_violation(
                "1.1_EMO_False_With_Emotions", sample_id, f"EMO=False but emotions={non_neutral}"
            )


def check_rel_routing_vs_relations(sample: dict, results: AuditResults):
    """Rule 1.2 & 1.3: REL routing should match relations presence."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    relations = tasks.get("relations", [])
    sample_id = sample.get("id", "unknown")

    if hub.get("REL", False) and not relations:
        results.add_violation("1.2_REL_True_No_Relations", sample_id, "REL=True but relations=[]")

    if not hub.get("REL", False) and relations:
        results.add_violation(
            "1.3_REL_False_Has_Relations", sample_id, f"REL=False but has relations={relations}"
        )


def check_mem_routing_vs_intent(sample: dict, results: AuditResults):
    """Rule 1.4: MEM routing should match memory intents."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    intent = tasks.get("intent", "")
    sample_id = sample.get("id", "unknown")

    if (
        hub.get("MEM", False)
        and intent not in MEMORY_INTENTS
        and intent not in {"other", "share_news", "express_feeling"}
    ):
        results.add_violation(
            "1.4_MEM_True_Non_Memory_Intent", sample_id, f"MEM=True but intent={intent}"
        )


# =============================================================================
# CATEGORY 2: Sentiment vs Emotions
# =============================================================================


def check_sentiment_emotion_consistency(sample: dict, results: AuditResults):
    """Rule 2.1 & 2.2: Sentiment should align with emotions."""
    tasks = sample.get("tasks", {})
    sentiment = tasks.get("sentiment", "")
    emotions = set(tasks.get("emotions", []))
    sample_id = sample.get("id", "unknown")

    # Rule 2.1: Positive sentiment with negative emotions
    if sentiment in POSITIVE_SENTIMENTS:
        neg_found = emotions & NEGATIVE_EMOTIONS
        if neg_found and "bittersweet" not in emotions:
            results.add_violation(
                "2.1_Positive_Sentiment_Negative_Emotions",
                sample_id,
                f"sentiment={sentiment} but has negative emotions={neg_found}",
            )

    # Rule 2.2: Negative sentiment with positive emotions
    if sentiment in NEGATIVE_SENTIMENTS:
        pos_found = emotions & POSITIVE_EMOTIONS
        if pos_found and "bittersweet" not in emotions:
            results.add_violation(
                "2.2_Negative_Sentiment_Positive_Emotions",
                sample_id,
                f"sentiment={sentiment} but has positive emotions={pos_found}",
            )


# =============================================================================
# CATEGORY 3: Intent vs Hub Routing
# =============================================================================


def check_intent_routing_consistency(sample: dict, results: AuditResults):
    """Rules 3.1-3.4: Intent should match routing flags."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    intent = tasks.get("intent", "")
    sample_id = sample.get("id", "unknown")

    # Rule 3.1: query_memory but MEM=False
    if intent == "query_memory" and not hub.get("MEM", True):
        results.add_violation(
            "3.1_Query_Memory_MEM_False", sample_id, "intent=query_memory but MEM=False"
        )

    # Rule 3.2: log_memory but MEM=False
    if intent == "log_memory" and not hub.get("MEM", True):
        results.add_violation(
            "3.2_Log_Memory_MEM_False", sample_id, "intent=log_memory but MEM=False"
        )

    # Rule 3.3: set_reminder but TASK=False
    if intent == "set_reminder" and not hub.get("TASK", True):
        results.add_violation(
            "3.3_Set_Reminder_TASK_False", sample_id, "intent=set_reminder but TASK=False"
        )

    # Rule 3.4: express_feeling but EMO=False
    if intent == "express_feeling" and not hub.get("EMO", True):
        results.add_violation(
            "3.4_Express_Feeling_EMO_False", sample_id, "intent=express_feeling but EMO=False"
        )


# =============================================================================
# CATEGORY 4: NER vs Relations
# =============================================================================


def check_ner_relations_consistency(sample: dict, results: AuditResults):
    """Rule 4.1-4.4: NER entities should match relations."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    ner = tasks.get("ner_family", [])
    relations = tasks.get("relations", [])
    sample_id = sample.get("id", "unknown")

    kinship_tokens = [e for e in ner if e.get("label") == "KINSHIP"]
    person_tokens = [e for e in ner if e.get("label") == "PERSON"]

    # Rule 4.1: KINSHIP token but no relation (skip generic terms)
    generic_kinship = {"family", "kids", "children", "parents"}
    specific_kinship = [
        e for e in kinship_tokens if e.get("token", "").lower() not in generic_kinship
    ]

    if specific_kinship and not relations and hub.get("REL", False):
        results.add_violation(
            "4.1_Kinship_No_Relation",
            sample_id,
            f"KINSHIP tokens={[e['token'] for e in specific_kinship]} but no relations",
        )

    # Rule 4.4: Relation object not in NER
    all_tokens = {(e.get("token") or "").lower() for e in ner}
    for rel in relations:
        obj_raw = rel.get("object")
        if obj_raw is None:
            continue
        obj = obj_raw.lower()
        if obj and obj not in all_tokens and obj != "user":
            # Check partial match
            if not any(obj in tok or tok in obj for tok in all_tokens):
                results.add_violation(
                    "4.4_Relation_Object_Not_In_NER",
                    sample_id,
                    f"relation object='{obj_raw}' not found in NER",
                )


# =============================================================================
# CATEGORY 7: ID and Deduplication
# =============================================================================


def check_duplicate_ids(samples: list, results: AuditResults):
    """Rule 7.1: Check for duplicate IDs."""
    id_counts = Counter(s.get("id", "unknown") for s in samples)
    for sample_id, count in id_counts.items():
        if count > 1:
            results.add_violation("7.1_Duplicate_ID", sample_id, f"ID appears {count} times")


# =============================================================================
# CATEGORY 8: Multi-Label Emotion Conflicts
# =============================================================================


def check_neutral_infection(sample: dict, results: AuditResults):
    """Rule 8.3: Neutral should be mutually exclusive."""
    tasks = sample.get("tasks", {})
    emotions = tasks.get("emotions", [])
    sample_id = sample.get("id", "unknown")

    if len(emotions) > 1 and "neutral" in emotions:
        other_emotions = [e for e in emotions if e != "neutral"]
        results.add_violation(
            "8.3_Neutral_Infection", sample_id, f"neutral mixed with {other_emotions}"
        )


def check_contradictory_emotions(sample: dict, results: AuditResults):
    """Rule 8.1: Check for contradictory emotion pairs."""
    tasks = sample.get("tasks", {})
    emotions = set(tasks.get("emotions", []))
    sample_id = sample.get("id", "unknown")

    contradictions = [
        ({"joy", "happiness"}, {"sadness", "grief"}),
        ({"excitement"}, {"boredom"}),
        ({"love", "affection"}, {"hatred", "resentment"}),
    ]

    for pos_set, neg_set in contradictions:
        if (emotions & pos_set) and (emotions & neg_set) and "bittersweet" not in emotions:
            results.add_violation(
                "8.1_Contradictory_Emotions",
                sample_id,
                f"contradictory emotions: {emotions & pos_set} vs {emotions & neg_set}",
            )


# =============================================================================
# CATEGORY 10 & 11: Span Validation
# =============================================================================


def check_span_alignment(sample: dict, results: AuditResults):
    """Rule 10.1, 10.2, 11.1: Validate NER and temporal spans."""
    text = sample.get("text", "")
    tasks = sample.get("tasks", {})
    sample_id = sample.get("id", "unknown")

    all_entities = tasks.get("ner_family", []) + tasks.get("temporal", [])

    for entity in all_entities:
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        token = entity.get("token", "")
        label = entity.get("label", "")

        # Rule 10.1: Valid indices
        if start < 0 or end < 0:
            results.add_violation(
                "10.1_Negative_Index", sample_id, f"Negative index: start={start}, end={end}"
            )
            continue

        if end > len(text):
            results.add_violation(
                "10.1_Index_Out_Of_Bounds", sample_id, f"end={end} > text_len={len(text)}"
            )
            continue

        if start >= end:
            results.add_violation("10.1_Invalid_Range", sample_id, f"start={start} >= end={end}")
            continue

        # Rule 10.2 & 11.1: Token match
        extracted = text[start:end]
        if extracted != token:
            results.add_violation(
                "11.1_Token_Mismatch",
                sample_id,
                f"label={label}: text[{start}:{end}]='{extracted}' != token='{token}'",
            )


# =============================================================================
# CATEGORY 12: The "Caring" Label Problem
# =============================================================================


def check_caring_problem(sample: dict, results: AuditResults):
    """Rule 12.1-12.3: Caring label misuse."""
    hub = sample.get("hub_routing", {})
    tasks = sample.get("tasks", {})
    emotions = tasks.get("emotions", [])
    intent = tasks.get("intent", "")
    sample_id = sample.get("id", "unknown")

    if "caring" in emotions:
        # Rule 12.2: Caring with EMO=False
        if not hub.get("EMO", True):
            results.add_violation(
                "12.2_Caring_EMO_False", sample_id, "caring emotion but EMO=False"
            )

        # Rule 12.3: Caring on task-only intents
        if intent == "set_reminder" and not hub.get("EMO", True):
            results.add_violation(
                "12.3_Caring_Task_Only", sample_id, "caring on set_reminder with EMO=False"
            )


# =============================================================================
# MAIN AUDIT FUNCTION
# =============================================================================


def audit_sample(sample: dict, results: AuditResults):
    """Run all audit checks on a single sample."""
    results.sample_count += 1

    # Category 1: Hub Routing
    check_emo_routing_vs_emotions(sample, results)
    check_rel_routing_vs_relations(sample, results)
    check_mem_routing_vs_intent(sample, results)

    # Category 2: Sentiment vs Emotions
    check_sentiment_emotion_consistency(sample, results)

    # Category 3: Intent vs Routing
    check_intent_routing_consistency(sample, results)

    # Category 4: NER vs Relations
    check_ner_relations_consistency(sample, results)

    # Category 8: Multi-label conflicts
    check_neutral_infection(sample, results)
    check_contradictory_emotions(sample, results)

    # Category 10 & 11: Span validation
    check_span_alignment(sample, results)

    # Category 12: Caring problem
    check_caring_problem(sample, results)


def load_jsonl_files(folder_path: Path) -> list:
    """Load all JSONL files from a folder."""
    samples = []
    files = sorted(folder_path.glob("shard_*.jsonl"))

    for file_path in files:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return samples


def print_report(results: AuditResults, dataset_name: str):
    """Print formatted audit report."""
    print(f"\n{'='*70}")
    print(f"AUDIT REPORT: {dataset_name}")
    print(f"{'='*70}")

    summary = results.get_summary()
    print(f"\nTotal Samples: {summary['total_samples']:,}")
    print(f"Total Violations: {summary['total_violations']:,}")
    print(f"Violation Rate: {summary['total_violations']/max(1,summary['total_samples'])*100:.2f}%")

    print(f"\n{'--- Violations by Rule ---'}")

    # Sort by count descending
    sorted_rules = sorted(summary["violations_by_rule"].items(), key=lambda x: x[1], reverse=True)

    for rule_id, count in sorted_rules:
        pct = count / summary["total_samples"] * 100
        severity = "CRITICAL" if pct > 10 else "HIGH" if pct > 5 else "MEDIUM" if pct > 1 else "LOW"
        print(f"  [{severity:8}] {rule_id}: {count:,} ({pct:.2f}%)")

    # Show examples for top violations
    print(f"\n{'--- Example Violations (Top 5 Rules) ---'}")
    for rule_id, _ in sorted_rules[:5]:
        examples = results.violations[rule_id][:3]
        print(f"\n  {rule_id}:")
        for ex in examples:
            print(f"    - {ex['sample_id']}: {ex['details'][:80]}")


def main():
    import sys

    base_path = Path(r"D:\Modeling_studio\data\familyos\unified")

    # Check for --healed flag
    use_healed = "--healed" in sys.argv

    if use_healed:
        print("=" * 70)
        print("AUDITING HEALED DATA")
        print("=" * 70)
        output_path = base_path / "output_healed"
        synthetic_path = base_path / "output_synthetic_healed"
    else:
        output_path = base_path / "output"
        synthetic_path = base_path / "output_synthetic"

    # Audit real data
    print(f"Loading {'HEALED ' if use_healed else ''}OUTPUT (real) data...")
    output_samples = load_jsonl_files(output_path)
    output_results = AuditResults()

    for sample in output_samples:
        audit_sample(sample, output_results)

    check_duplicate_ids(output_samples, output_results)
    print_report(output_results, "OUTPUT (Real Data)")

    # Audit synthetic data
    print("\nLoading OUTPUT_SYNTHETIC data...")
    synthetic_samples = load_jsonl_files(synthetic_path)
    synthetic_results = AuditResults()

    for sample in synthetic_samples:
        audit_sample(sample, synthetic_results)

    check_duplicate_ids(synthetic_samples, synthetic_results)
    print_report(synthetic_results, "OUTPUT_SYNTHETIC")

    # Combined summary
    print(f"\n{'='*70}")
    print("COMBINED SUMMARY")
    print(f"{'='*70}")

    total_samples = output_results.sample_count + synthetic_results.sample_count
    combined_counts = Counter()
    combined_counts.update(output_results.counts)
    combined_counts.update(synthetic_results.counts)

    print(f"\nTotal Samples: {total_samples:,}")
    print(
        f"Real Data: {output_results.sample_count:,} ({output_results.sample_count/total_samples*100:.1f}%)"
    )
    print(
        f"Synthetic: {synthetic_results.sample_count:,} ({synthetic_results.sample_count/total_samples*100:.1f}%)"
    )

    print(f"\n{'--- Critical Issues (Block Training) ---'}")
    critical_rules = [
        "8.3_Neutral_Infection",
        "11.1_Token_Mismatch",
        "1.1_EMO_False_With_Emotions",
        "7.1_Duplicate_ID",
        "3.1_Query_Memory_MEM_False",
        "3.2_Log_Memory_MEM_False",
        "3.4_Express_Feeling_EMO_False",
    ]

    for rule in critical_rules:
        count = combined_counts.get(rule, 0)
        pct = count / total_samples * 100
        status = "FAIL" if count > 0 else "PASS"
        print(f"  [{status}] {rule}: {count:,} ({pct:.2f}%)")

    print(f"\n{'--- Training Recommendation ---'}")
    critical_count = sum(combined_counts.get(r, 0) for r in critical_rules)
    if critical_count > total_samples * 0.01:
        print("  STATUS: DO NOT TRAIN - Critical issues exceed 1% threshold")
        print("  ACTION: Run data cleanup pipeline first")
    else:
        print("  STATUS: PROCEED WITH CAUTION")
        print("  ACTION: Apply gated training strategy")


if __name__ == "__main__":
    main()
