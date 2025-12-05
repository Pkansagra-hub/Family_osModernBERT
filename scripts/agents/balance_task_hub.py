"""
Balance TASK Hub Distribution

Problem: TASK hub is only 10.3% of data (7,486/72,451 samples)
Target: Increase to ~25-30% by upsampling task-oriented samples

Strategy:
1. Identify all TASK=true samples
2. Duplicate task-oriented samples with variations
3. Synthesize new task samples from templates

Usage:
    python balance_task_hub.py --target-ratio 0.25 --dry-run
    python balance_task_hub.py --target-ratio 0.25 --execute
"""

import argparse
import json
import logging
import random
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

# Task-oriented intents that should trigger TASK hub
TASK_INTENTS = {"set_reminder", "query_memory"}

# Task-oriented ingress domains
TASK_INGRESS = {"TASK", "PLANNING", "META"}


class TaskHubBalancer:
    """Balance TASK hub by upsampling task-oriented samples."""

    def __init__(self, target_ratio: float = 0.25, dry_run: bool = True):
        self.target_ratio = target_ratio
        self.dry_run = dry_run
        self.task_samples = []
        self.non_task_samples = []
        self.stats = {
            "total_samples": 0,
            "task_samples": 0,
            "non_task_samples": 0,
            "samples_to_add": 0,
            "intent_distribution": Counter(),
            "ingress_distribution": Counter(),
        }

    def load_samples(self) -> None:
        """Load and categorize all samples."""
        shard_files = sorted(UNIFIED_DIR.glob("shard_*.jsonl"))

        for shard_path in shard_files:
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.stats["total_samples"] += 1

                        # Check if TASK hub is active
                        hub_routing = sample.get("hub_routing", {})
                        is_task = hub_routing.get("TASK", False)

                        if is_task:
                            self.task_samples.append(sample)
                            self.stats["task_samples"] += 1
                        else:
                            self.non_task_samples.append(sample)
                            self.stats["non_task_samples"] += 1

                        # Track intent/ingress
                        intent = sample.get("tasks", {}).get("intent", "")
                        ingress = sample.get("tasks", {}).get("ingress", "")
                        self.stats["intent_distribution"][intent] += 1
                        self.stats["ingress_distribution"][ingress] += 1

                    except json.JSONDecodeError:
                        continue

        logger.info(f"Loaded {self.stats['total_samples']:,} samples")
        logger.info(
            f"  TASK=true: {self.stats['task_samples']:,} ({100*self.stats['task_samples']/self.stats['total_samples']:.1f}%)"
        )
        logger.info(
            f"  TASK=false: {self.stats['non_task_samples']:,} ({100*self.stats['non_task_samples']/self.stats['total_samples']:.1f}%)"
        )

    def calculate_upsampling(self) -> None:
        """Calculate how many samples to add."""
        current_ratio = self.stats["task_samples"] / self.stats["total_samples"]

        # Calculate target number of task samples
        # If we want target_ratio = 0.25:
        # task_samples + new_samples = 0.25 * (total_samples + new_samples)
        # Solving: new_samples = (0.25 * total_samples - task_samples) / (1 - 0.25)

        target_task_count = self.target_ratio * self.stats["total_samples"]
        samples_needed = int(target_task_count - self.stats["task_samples"])

        if samples_needed <= 0:
            logger.info(f"Target ratio {self.target_ratio:.1%} already achieved!")
            return

        self.stats["samples_to_add"] = samples_needed
        logger.info(f"\nTarget ratio: {self.target_ratio:.1%}")
        logger.info(f"Current ratio: {current_ratio:.1%}")
        logger.info(f"Samples to add: {samples_needed:,}")

    def upsample_task_samples(self) -> list[dict]:
        """Create additional task samples through upsampling."""
        if self.stats["samples_to_add"] <= 0:
            return []

        # Calculate how many times to replicate each sample
        replication_factor = self.stats["samples_to_add"] / len(self.task_samples)

        new_samples = []

        # Prioritize samples with task-oriented intents
        high_priority = [
            s for s in self.task_samples if s.get("tasks", {}).get("intent") in TASK_INTENTS
        ]
        low_priority = [
            s for s in self.task_samples if s.get("tasks", {}).get("intent") not in TASK_INTENTS
        ]

        logger.info(f"  High priority (task intents): {len(high_priority)}")
        logger.info(f"  Low priority (other): {len(low_priority)}")

        # Upsample high priority samples more
        high_priority_samples_needed = int(self.stats["samples_to_add"] * 0.7)
        low_priority_samples_needed = self.stats["samples_to_add"] - high_priority_samples_needed

        # Add high priority samples
        if high_priority:
            for _ in range(high_priority_samples_needed):
                sample = random.choice(high_priority)
                new_samples.append(sample.copy())

        # Add low priority samples
        if low_priority:
            for _ in range(low_priority_samples_needed):
                sample = random.choice(low_priority)
                new_samples.append(sample.copy())

        return new_samples

    def write_balanced_shards(self, new_samples: list[dict]) -> None:
        """Write balanced dataset to new shards."""
        all_samples = self.non_task_samples + self.task_samples + new_samples
        random.shuffle(all_samples)

        output_dir = UNIFIED_DIR.parent / "output_balanced"
        output_dir.mkdir(exist_ok=True)

        shard_size = 5000
        for i in range(0, len(all_samples), shard_size):
            shard_samples = all_samples[i : i + shard_size]
            shard_path = output_dir / f"shard_{i//shard_size:04d}.jsonl"

            with open(shard_path, "w", encoding="utf-8") as f:
                for sample in shard_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"\nWrote {len(all_samples):,} samples to {output_dir}")

        # Calculate final ratio
        task_count = sum(1 for s in all_samples if s.get("hub_routing", {}).get("TASK", False))
        final_ratio = task_count / len(all_samples)
        logger.info(f"Final TASK ratio: {final_ratio:.1%} ({task_count:,}/{len(all_samples):,})")

    def run(self) -> None:
        """Run the balancing process."""
        logger.info("=" * 70)
        logger.info("TASK HUB BALANCING")
        logger.info("=" * 70)

        self.load_samples()
        self.calculate_upsampling()

        if self.stats["samples_to_add"] <= 0:
            return

        logger.info("\n📊 Intent Distribution in TASK samples:")
        for intent, count in self.stats["intent_distribution"].most_common():
            if any(
                s.get("tasks", {}).get("intent") == intent
                and s.get("hub_routing", {}).get("TASK", False)
                for s in self.task_samples
            ):
                print(f"   {intent:20s} {count:6,}")

        if self.dry_run:
            logger.info("\n⚠️  DRY RUN - No changes will be made")
            logger.info(f"Would add {self.stats['samples_to_add']:,} samples via upsampling")
            return

        logger.info("\nUpsampling TASK samples...")
        new_samples = self.upsample_task_samples()
        logger.info(f"Generated {len(new_samples):,} new samples")

        self.write_balanced_shards(new_samples)
        logger.info("\n✅ Balancing complete!")


def main():
    parser = argparse.ArgumentParser(description="Balance TASK hub distribution")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=0.25,
        help="Target ratio for TASK samples (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without applying them"
    )
    parser.add_argument("--execute", action="store_true", help="Apply changes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    random.seed(args.seed)

    # Default to dry-run
    if not args.execute and not args.dry_run:
        args.dry_run = True

    balancer = TaskHubBalancer(target_ratio=args.target_ratio, dry_run=args.dry_run)
    balancer.run()


if __name__ == "__main__":
    main()
