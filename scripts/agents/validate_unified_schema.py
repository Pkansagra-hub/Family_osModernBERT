"""
Validate Unified Data Schema

Checks that all samples follow the strict schema:
- NER: Only 10 allowed labels
- Temporal: Only 6 allowed labels
- Relations: Only 15 allowed predicates
- All required fields present

Usage:
    python validate_unified_schema.py
"""

import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
UNIFIED_DIR = BASE_DIR / "data" / "familyos" / "unified" / "output"

# Valid schemas
VALID_NER_LABELS = {
    "PERSON",
    "KINSHIP",
    "NICKNAME",
    "PET",
    "HOME_LOC",
    "FAMILY_EVENT",
    "ROUTINE",
    "TRADITION",
    "MILESTONE",
    "HEIRLOOM",
}

VALID_TEMPORAL_LABELS = {"DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"}

VALID_RELATIONS = {
    "no_relation",
    "parent_of",
    "child_of",
    "spouse_of",
    "sibling_of",
    "grandparent_of",
    "grandchild_of",
    "aunt_uncle_of",
    "niece_nephew_of",
    "cousin_of",
    "pet_of",
    "friend_of",
    "colleague_of",
    "lives_at",
    "owns",
}

VALID_EMOTIONS = {
    # Core (8)
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "love",
    "disgust",
    # Positive (12)
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "optimism",
    "pride",
    "relief",
    "contentment",
    "hope",
    "tenderness",
    # Negative (10)
    "annoyance",
    "disappointment",
    "disapproval",
    "embarrassment",
    "grief",
    "nervousness",
    "remorse",
    "frustration",
    "overwhelmed",
    "emptiness",
    # Family (14)
    "nostalgia",
    "protectiveness",
    "togetherness",
    "longing",
    "warmth",
    "playfulness",
    "celebration",
    "belonging",
    "parental_pride",
    "parental_guilt",
    "patience",
    "worry",
    "bittersweet",
    "homesickness",
}

VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}

VALID_SAFETY = {"GREEN", "AMBER", "RED", "CRISIS"}

VALID_INTENTS = {
    "log_memory",
    "query_memory",
    "set_reminder",
    "express_feeling",
    "seek_advice",
    "share_news",
    "reflect",
    "other",
}

VALID_INGRESS = {
    "DIARY",
    "TASK",
    "HEALTH",
    "FINANCE",
    "RELATIONSHIP",
    "WORK",
    "META",
    "MEMORY",
    "PLANNING",
    "CELEBRATION",
    "CONCERN",
    "GRATITUDE",
}


class SchemaValidator:
    """Validate unified data schema compliance."""

    def __init__(self):
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_ner": 0,
            "invalid_temporal": 0,
            "invalid_relations": 0,
            "invalid_emotions": 0,
            "invalid_sentiment": 0,
            "invalid_safety": 0,
            "invalid_intent": 0,
            "invalid_ingress": 0,
            "invalid_ner_labels": Counter(),
            "invalid_temporal_labels": Counter(),
            "invalid_relation_predicates": Counter(),
            "invalid_emotions_labels": Counter(),
        }

    def validate_ner(self, sample: dict) -> bool:
        """Validate NER annotations."""
        tasks = sample.get("tasks", {})
        ner_entities = tasks.get("ner_family", [])

        valid = True
        for entity in ner_entities:
            label = entity.get("label", "")
            if label not in VALID_NER_LABELS:
                self.stats["invalid_ner_labels"][label] += 1
                valid = False

        return valid

    def validate_temporal(self, sample: dict) -> bool:
        """Validate temporal annotations."""
        tasks = sample.get("tasks", {})
        temporal_entities = tasks.get("temporal", [])

        valid = True
        for entity in temporal_entities:
            label = entity.get("label", "")
            if label not in VALID_TEMPORAL_LABELS:
                self.stats["invalid_temporal_labels"][label] += 1
                valid = False

        return valid

    def validate_relations(self, sample: dict) -> bool:
        """Validate relation annotations."""
        tasks = sample.get("tasks", {})
        relations = tasks.get("relations", [])

        valid = True
        for relation in relations:
            predicate = relation.get("predicate", "")
            if predicate not in VALID_RELATIONS:
                self.stats["invalid_relation_predicates"][predicate] += 1
                valid = False

        return valid

    def validate_emotions(self, sample: dict) -> bool:
        """Validate emotion annotations."""
        tasks = sample.get("tasks", {})
        emotions = tasks.get("emotions", [])

        valid = True
        for emotion in emotions:
            if emotion not in VALID_EMOTIONS:
                self.stats["invalid_emotions_labels"][emotion] += 1
                valid = False

        return valid

    def validate_sentiment(self, sample: dict) -> bool:
        """Validate sentiment annotation."""
        tasks = sample.get("tasks", {})
        sentiment = tasks.get("sentiment", "")
        return sentiment in VALID_SENTIMENTS

    def validate_safety(self, sample: dict) -> bool:
        """Validate safety annotation."""
        tasks = sample.get("tasks", {})
        safety = tasks.get("safety_familyos", "")
        return safety in VALID_SAFETY

    def validate_intent(self, sample: dict) -> bool:
        """Validate intent annotation."""
        tasks = sample.get("tasks", {})
        intent = tasks.get("intent", "")
        return intent in VALID_INTENTS

    def validate_ingress(self, sample: dict) -> bool:
        """Validate ingress annotation."""
        tasks = sample.get("tasks", {})
        ingress = tasks.get("ingress", "")
        return ingress in VALID_INGRESS

    def validate_sample(self, sample: dict) -> bool:
        """Validate all annotations in a sample."""
        is_valid = True

        if not self.validate_ner(sample):
            self.stats["invalid_ner"] += 1
            is_valid = False

        if not self.validate_temporal(sample):
            self.stats["invalid_temporal"] += 1
            is_valid = False

        if not self.validate_relations(sample):
            self.stats["invalid_relations"] += 1
            is_valid = False

        if not self.validate_emotions(sample):
            self.stats["invalid_emotions"] += 1
            is_valid = False

        if not self.validate_sentiment(sample):
            self.stats["invalid_sentiment"] += 1
            is_valid = False

        if not self.validate_safety(sample):
            self.stats["invalid_safety"] += 1
            is_valid = False

        if not self.validate_intent(sample):
            self.stats["invalid_intent"] += 1
            is_valid = False

        if not self.validate_ingress(sample):
            self.stats["invalid_ingress"] += 1
            is_valid = False

        return is_valid

    def validate_shard(self, shard_path: Path) -> None:
        """Validate a single shard file."""
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    self.stats["total_samples"] += 1

                    if self.validate_sample(sample):
                        self.stats["valid_samples"] += 1

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line in {shard_path.name}")
                    continue

    def run(self) -> None:
        """Run validation on all shards."""
        shard_files = sorted(UNIFIED_DIR.glob("shard_*.jsonl"))

        if not shard_files:
            logger.error(f"No shard files found in {UNIFIED_DIR}")
            return

        logger.info(f"Validating {len(shard_files)} shard files...")

        for shard_path in shard_files:
            self.validate_shard(shard_path)

        self.print_stats()

    def print_stats(self) -> None:
        """Print validation statistics."""
        print("\n" + "=" * 70)
        print("SCHEMA VALIDATION REPORT")
        print("=" * 70)

        total = self.stats["total_samples"]
        valid = self.stats["valid_samples"]
        invalid = total - valid

        print(f"\n📊 Overall:")
        print(f"   Total samples: {total:,}")
        print(f"   Valid samples: {valid:,} ({100*valid/total:.1f}%)")
        print(f"   Invalid samples: {invalid:,} ({100*invalid/total:.1f}%)")

        print(f"\n❌ Validation Errors:")
        print(f"   NER violations: {self.stats['invalid_ner']:,}")
        print(f"   Temporal violations: {self.stats['invalid_temporal']:,}")
        print(f"   Relation violations: {self.stats['invalid_relations']:,}")
        print(f"   Emotion violations: {self.stats['invalid_emotions']:,}")
        print(f"   Sentiment violations: {self.stats['invalid_sentiment']:,}")
        print(f"   Safety violations: {self.stats['invalid_safety']:,}")
        print(f"   Intent violations: {self.stats['invalid_intent']:,}")
        print(f"   Ingress violations: {self.stats['invalid_ingress']:,}")

        if self.stats["invalid_ner_labels"]:
            print(f"\n🚫 Invalid NER Labels (top 10):")
            for label, count in self.stats["invalid_ner_labels"].most_common(10):
                print(f"   {label:20s} {count:6,} occurrences")

        if self.stats["invalid_temporal_labels"]:
            print(f"\n🚫 Invalid Temporal Labels (top 10):")
            for label, count in self.stats["invalid_temporal_labels"].most_common(10):
                print(f"   {label:20s} {count:6,} occurrences")

        if self.stats["invalid_relation_predicates"]:
            print(f"\n🚫 Invalid Relation Predicates (top 10):")
            for pred, count in self.stats["invalid_relation_predicates"].most_common(10):
                print(f"   {pred:20s} {count:6,} occurrences")

        if self.stats["invalid_emotions_labels"]:
            print(f"\n🚫 Invalid Emotion Labels (top 10):")
            for emotion, count in self.stats["invalid_emotions_labels"].most_common(10):
                print(f"   {emotion:20s} {count:6,} occurrences")

        print("=" * 70)

        if invalid == 0:
            print("\n✅ All samples are valid! Schema compliance: 100%")
        else:
            print(f"\n⚠️  {invalid:,} samples have schema violations")
            print("Run clean_unified_data.py to fix them")


def main():
    validator = SchemaValidator()
    validator.run()


if __name__ == "__main__":
    main()
