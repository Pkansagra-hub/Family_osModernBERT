"""
Embedding Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic embedding training data
for the FamilyOS Embedding dataset.

Data Format:
- Triplets: (anchor, positive, negative) for contrastive learning
- Pairs: (text1, text2, similarity) for similarity learning

Clusters:
- immigration_docs, child_activities, health_records, financial_planning
- family_memories, daily_routines, family_traditions, emotional_support
"""

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_KEY = "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"  # Replace with your key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast:free"

# Rate limiting settings
REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 900
DELAY_BETWEEN_REQUESTS = 6.0

# Output settings
DATA_DIR = Path("D:/Modeling_studio/data/familyos/embeddings")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000

SAMPLES_PER_REQUEST = 200  # Triplets per API call

# =============================================================================
# Cluster Definitions (from README)
# =============================================================================

CLUSTERS = {
    "immigration_docs": {
        "description": "H-1B visa, USCIS, immigration-related texts common in Indian-American families",
        "keywords": [
            "H-1B",
            "USCIS",
            "I-140",
            "visa",
            "green card",
            "immigration",
            "EAD",
            "OPT",
            "petition",
        ],
        "examples": [
            "Need to file the I-140 petition by next month",
            "USCIS receipt notice came in the mail today",
            "H-1B extension is pending, waiting anxiously",
        ],
    },
    "child_activities": {
        "description": "School, activities, homework, playdates for children",
        "keywords": [
            "school",
            "homework",
            "soccer",
            "piano",
            "playdate",
            "practice",
            "tutor",
            "class",
        ],
        "examples": [
            "Emma has soccer practice at 4pm today",
            "Need to help Panda with her math homework",
            "Planning a playdate with the neighbors' kids",
        ],
    },
    "health_records": {
        "description": "Medical appointments, symptoms, medications",
        "keywords": [
            "doctor",
            "appointment",
            "medication",
            "checkup",
            "prescription",
            "symptoms",
            "hospital",
        ],
        "examples": [
            "Doctor's appointment scheduled for Thursday",
            "Need to refill the blood pressure medication",
            "Kids' annual checkup is overdue",
        ],
    },
    "financial_planning": {
        "description": "Budgets, investments, bills, financial decisions",
        "keywords": [
            "401k",
            "savings",
            "budget",
            "tax",
            "mortgage",
            "investment",
            "college fund",
            "bills",
        ],
        "examples": [
            "Reviewed the 401k allocations today",
            "Property tax bill is due next week",
            "Need to start saving for college fund",
        ],
    },
    "family_memories": {
        "description": "Photos, trips, celebrations, nostalgic content",
        "keywords": [
            "photos",
            "trip",
            "vacation",
            "remember",
            "anniversary",
            "wedding",
            "birthday",
            "childhood",
        ],
        "examples": [
            "Looking at old photos from the Goa trip",
            "Remember when Emma took her first steps?",
            "Can't believe it's been 10 years since our wedding",
        ],
    },
    "daily_routines": {
        "description": "Morning routines, school runs, meal times",
        "keywords": [
            "morning",
            "breakfast",
            "school run",
            "dinner",
            "bedtime",
            "routine",
            "lunch",
            "wake up",
        ],
        "examples": [
            "Morning school run was hectic today",
            "Dinner time is always chaotic with the kids",
            "Bedtime story took forever tonight",
        ],
    },
    "family_traditions": {
        "description": "Weekly/annual family customs and rituals",
        "keywords": [
            "Sunday",
            "Diwali",
            "Christmas",
            "tradition",
            "annual",
            "weekly",
            "celebration",
            "ritual",
        ],
        "examples": [
            "Sunday brunch at grandma's house",
            "Getting ready for our annual Diwali celebration",
            "Movie night this Friday with the whole family",
        ],
    },
    "emotional_support": {
        "description": "Comfort, advice, family support conversations",
        "keywords": [
            "support",
            "comfort",
            "advice",
            "help",
            "worried",
            "proud",
            "love",
            "care",
            "grateful",
        ],
        "examples": [
            "Mom always knows what to say when I'm stressed",
            "So proud of how the kids handled the move",
            "Grateful for the family support during tough times",
        ],
    },
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (
    """You are an expert data annotator for a family-focused embedding training task.
Your job is to generate high-quality training triplets for contrastive learning.

## Triplet Format
Each triplet has:
- **anchor**: A sentence about a family topic
- **positive**: A semantically SIMILAR sentence (same topic/meaning, different words)
- **negative**: A semantically DIFFERENT sentence (different topic entirely)

## Cluster Categories
"""
    + "\n".join([f"- **{k}**: {v['description']}" for k, v in CLUSTERS.items()])
    + """

## Quality Requirements

1. **Positives must be semantically similar but lexically different**
   - Same topic, different phrasing
   - NOT just word substitutions
   - Should capture the same meaning/intent

2. **Negatives must be clearly from a DIFFERENT cluster**
   - If anchor is about "child_activities", negative should be about "financial_planning" or "health_records"
   - Hard negatives (same domain but different topic) are valuable

3. **Use realistic family language**
   - Mix of Western and Indian family contexts
   - Include nicknames (Panda, Bunny), kinship terms (didi, bhai, nana)
   - Natural, conversational tone

4. **Diversity**
   - Vary sentence length (short notes to longer entries)
   - Mix text types: diary entries, reminders, conversations, notes

## Output Format
Output ONLY valid JSONL. Each line must be:
{"anchor": "...", "positive": "...", "negative": "...", "anchor_cluster": "cluster_name", "negative_cluster": "cluster_name"}

## Examples

{"anchor": "Need to file the I-140 petition before the deadline", "positive": "I-140 application has to be submitted by end of month", "negative": "Emma has piano lessons every Tuesday after school", "anchor_cluster": "immigration_docs", "negative_cluster": "child_activities"}
{"anchor": "Bedtime routine with the kids is exhausting", "positive": "Getting the children to sleep takes forever every night", "negative": "Property tax payment is due next Friday", "anchor_cluster": "daily_routines", "negative_cluster": "financial_planning"}
{"anchor": "Diwali celebration at nani's house this year", "positive": "We're doing the annual Diwali party at grandma's place", "negative": "Doctor appointment for dad's checkup tomorrow", "anchor_cluster": "family_traditions", "negative_cluster": "health_records"}

Now generate the requested number of triplets. Output JSONL only:"""
)


def get_user_prompt(num_samples: int, focus_clusters: list[str], batch_id: int) -> str:
    """Generate diverse user prompts."""
    cluster_info = "\n".join([f"- {c}: {CLUSTERS[c]['description']}" for c in focus_clusters])

    styles = [
        "diary entries and personal reflections",
        "reminder notes and to-do items",
        "text messages between family members",
        "photo captions and memory notes",
        "calendar entries and scheduling notes",
        "voice assistant queries",
    ]
    style = styles[batch_id % len(styles)]

    return f"""Generate {num_samples} embedding triplets.

Focus on these clusters for anchors:
{cluster_info}

Style: {style}

Requirements:
- Each triplet: anchor, positive (similar meaning, different words), negative (different topic)
- Include Indian family contexts (didi, bhai, nana, Diwali, etc.) in ~30% of samples
- Vary sentence length and complexity
- Make positives semantically similar but NOT just word swaps

Output JSONL only:"""


# =============================================================================
# Validation
# =============================================================================


def validate_triplet(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single triplet."""
    required = ["anchor", "positive", "negative"]

    for field in required:
        if field not in sample:
            return False, f"Missing '{field}' field"
        if not isinstance(sample[field], str):
            return False, f"'{field}' must be a string"
        if len(sample[field].strip()) < 10:
            return False, f"'{field}' too short"

    # Check anchor != positive (should be different phrasing)
    if sample["anchor"].lower() == sample["positive"].lower():
        return False, "Anchor and positive are identical"

    # Check anchor != negative
    if sample["anchor"].lower() == sample["negative"].lower():
        return False, "Anchor and negative are identical"

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    """Compute hash for deduplication."""
    text = sample["anchor"].lower() + sample["positive"].lower()
    return hashlib.md5(text.encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response."""
    valid_samples = []

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue

        try:
            # Try to extract JSON object
            json_match = re.search(r'\{[^{}]*"anchor"[^{}]*\}', line)
            if json_match:
                sample = json.loads(json_match.group())
            else:
                sample = json.loads(line)

            is_valid, error = validate_triplet(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")

        except json.JSONDecodeError:
            continue

    return valid_samples


# =============================================================================
# OpenRouter Client
# =============================================================================


class OpenRouterClient:
    """Client for OpenRouter API with rate limiting."""

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        requests_per_day: int = REQUESTS_PER_DAY,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day

        self.request_times: list[datetime] = []
        self.daily_count = 0
        self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)

        self.client = httpx.Client(timeout=120.0)

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = datetime.now()

        if now >= self.daily_reset:
            self.daily_count = 0
            self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            logger.info("Daily rate limit reset")

        if self.daily_count >= self.requests_per_day:
            raise RuntimeError("Daily rate limit reached")

        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]

        if len(self.request_times) >= self.requests_per_minute:
            oldest = min(self.request_times)
            wait_seconds = (oldest + timedelta(minutes=1) - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"Rate limiting: waiting {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 30000,
    ) -> str:
        """Generate text using OpenRouter API."""
        self._wait_for_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/modeling-studio",
            "X-Title": "FamilyOS Embedding Data Generator",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            self.request_times.append(datetime.now())
            self.daily_count += 1

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            logger.info(f"API call successful (daily: {self.daily_count}/{self.requests_per_day})")
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429:
                logger.warning("Rate limited by API. Waiting 60 seconds...")
                time.sleep(60)
            raise

    def close(self):
        self.client.close()


# =============================================================================
# Silver Data Manager
# =============================================================================


class SilverDataManager:
    """Manages sharded silver data storage for embeddings."""

    def __init__(self, silver_dir: Path = SILVER_DIR, shard_size: int = SHARD_SIZE):
        self.silver_dir = silver_dir
        self.shard_size = shard_size
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self.seen_hashes: set[str] = set()
        self._load_existing_hashes()

        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

    def _load_existing_hashes(self) -> None:
        """Load hashes from existing shards."""
        for shard_file in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.seen_hashes.add(compute_sample_hash(sample))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(self.seen_hashes)} existing hashes")

    def _get_shard_path(self, shard_id: int) -> Path:
        return self.silver_dir / f"shard_{shard_id:04d}.jsonl"

    def _count_shard_samples(self, shard_id: int) -> int:
        shard_path = self._get_shard_path(shard_id)
        if not shard_path.exists():
            return 0
        with open(shard_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _get_next_shard_id(self) -> int:
        existing = list(self.silver_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
        if self._count_shard_samples(max_id) >= self.shard_size:
            return max_id + 1
        return max_id

    def add_samples(self, samples: list[dict[str, Any]]) -> int:
        """Add samples to silver storage."""
        added = 0

        for sample in samples:
            sample_hash = compute_sample_hash(sample)
            if sample_hash in self.seen_hashes:
                continue

            if self.current_shard_count >= self.shard_size:
                self.current_shard_id += 1
                self.current_shard_count = 0
                logger.info(f"Started new shard: shard_{self.current_shard_id:04d}")

            shard_path = self._get_shard_path(self.current_shard_id)
            with open(shard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            self.seen_hashes.add(sample_hash)
            self.current_shard_count += 1
            added += 1

        return added

    def get_total_samples(self) -> int:
        return len(self.seen_hashes)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about silver data."""
        cluster_counts: dict[str, int] = defaultdict(int)
        total = 0

        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        if "anchor_cluster" in sample:
                            cluster_counts[sample["anchor_cluster"]] += 1
                        total += 1
                    except json.JSONDecodeError:
                        continue

        return {
            "total_samples": total,
            "num_shards": len(list(self.silver_dir.glob("shard_*.jsonl"))),
            "cluster_counts": dict(cluster_counts),
        }


# =============================================================================
# Data Generator Agent
# =============================================================================


class EmbeddingDataGeneratorAgent:
    """Agent that generates embedding training data."""

    def __init__(
        self,
        api_key: str = OPENROUTER_API_KEY,
        samples_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
    ):
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests

        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()

        self.cluster_counts: dict[str, int] = defaultdict(int)
        self._load_existing_counts()

        self.batch_id = 0

    def _load_existing_counts(self) -> None:
        """Load existing cluster counts."""
        stats = self.silver_manager.get_stats()
        self.cluster_counts = defaultdict(int, stats.get("cluster_counts", {}))
        logger.info(f"Existing cluster counts: {dict(self.cluster_counts)}")

    def _get_underrepresented_clusters(self, n: int = 3) -> list[str]:
        """Get the n most underrepresented clusters."""
        all_clusters = list(CLUSTERS.keys())
        sorted_clusters = sorted(all_clusters, key=lambda c: self.cluster_counts.get(c, 0))
        return sorted_clusters[:n]

    def generate_batch(self) -> int:
        """Generate a batch of samples."""
        focus = self._get_underrepresented_clusters(3)

        user_prompt = get_user_prompt(
            self.samples_per_request,
            focus_clusters=focus,
            batch_id=self.batch_id,
        )

        try:
            response = self.client.generate(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            # Update cluster counts
            for sample in samples:
                if "anchor_cluster" in sample:
                    self.cluster_counts[sample["anchor_cluster"]] += 1

            self.batch_id += 1

            logger.info(
                f"Generated {len(samples)} valid, added {added} new. "
                f"Focus: {focus}. Total: {self.silver_manager.get_total_samples()}"
            )

            return added

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return 0

    def run(
        self,
        target_samples: int | None = None,
        max_requests: int | None = None,
        run_time_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Run the data generation agent."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=run_time_minutes) if run_time_minutes else None

        existing_count = self.silver_manager.get_total_samples()

        stats = {
            "start_time": start_time.isoformat(),
            "existing_samples": existing_count,
            "new_samples": 0,
            "requests": 0,
            "errors": 0,
        }

        logger.info(f"Starting embedding data generation. Existing: {existing_count}")

        try:
            while True:
                if target_samples and stats["new_samples"] >= target_samples:
                    logger.info(f"Reached target of {target_samples} new samples")
                    break

                if max_requests and stats["requests"] >= max_requests:
                    logger.info(f"Reached max requests: {max_requests}")
                    break

                if end_time and datetime.now() >= end_time:
                    logger.info(f"Reached time limit: {run_time_minutes} minutes")
                    break

                stats["requests"] += 1
                added = self.generate_batch()

                if added > 0:
                    stats["new_samples"] += added
                else:
                    stats["errors"] += 1

                total = existing_count + stats["new_samples"]
                logger.info(
                    f"Progress: {stats['new_samples']} new ({total} total) "
                    f"from {stats['requests']} requests"
                )

                time.sleep(self.delay_between_requests)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except RuntimeError as e:
            if "rate limit" in str(e).lower():
                logger.warning("Stopping due to rate limit")
            else:
                raise
        finally:
            self.client.close()

        stats["end_time"] = datetime.now().isoformat()
        stats["final_cluster_counts"] = dict(self.cluster_counts)
        stats["total_samples"] = self.silver_manager.get_total_samples()

        logger.info(f"Generation complete. Stats: {json.dumps(stats, indent=2)}")
        return stats


# =============================================================================
# Export to Training Format
# =============================================================================


def export_triplets_for_training(
    output_file: Path | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Export silver triplets to training format."""
    silver_manager = SilverDataManager()

    samples = []
    for shard_path in sorted(SILVER_DIR.glob("shard_*.jsonl")):
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Convert to training format
                    triplet = {
                        "anchor": sample["anchor"],
                        "positive": sample["positive"],
                        "negative": sample["negative"],
                    }
                    samples.append(triplet)

                    if max_samples and len(samples) >= max_samples:
                        break
                except json.JSONDecodeError:
                    continue

        if max_samples and len(samples) >= max_samples:
            break

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Exported {len(samples)} triplets to {output_file}")

    return samples


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Run the embedding data generator agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate embedding training data")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new silver data")
    gen_parser.add_argument("--target", type=int, default=None, help="Target number of new samples")
    gen_parser.add_argument("--max-requests", type=int, default=None, help="Maximum API requests")
    gen_parser.add_argument("--run-time", type=int, default=None, help="Run time in minutes")
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=30, help="Samples per API call"
    )
    gen_parser.add_argument(
        "--delay", type=float, default=6.0, help="Delay between requests (seconds)"
    )
    gen_parser.add_argument("--api-key", type=str, default=None, help="OpenRouter API key")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show silver data statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to training format")
    export_parser.add_argument("--output", type=str, required=True, help="Output file path")
    export_parser.add_argument("--max", type=int, default=None, help="Max samples to export")

    args = parser.parse_args()

    if args.command == "generate":
        api_key = args.api_key or OPENROUTER_API_KEY
        if "YOUR_KEY_HERE" in api_key:
            print("ERROR: Please provide an API key via --api-key or edit the script")
            return

        agent = EmbeddingDataGeneratorAgent(
            api_key=api_key,
            samples_per_request=args.samples_per_request,
            delay_between_requests=args.delay,
        )
        stats = agent.run(
            target_samples=args.target,
            max_requests=args.max_requests,
            run_time_minutes=args.run_time,
        )
        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2))

    elif args.command == "stats":
        manager = SilverDataManager()
        stats = manager.get_stats()
        print("\n=== Silver Data Statistics ===")
        print(json.dumps(stats, indent=2))

    elif args.command == "export":
        output = Path(args.output)
        samples = export_triplets_for_training(output_file=output, max_samples=args.max)
        print(f"\nExported {len(samples)} triplets to {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
