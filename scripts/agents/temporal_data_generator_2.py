"""
Temporal Expression Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic temporal extraction data
for the FamilyOS Temporal dataset.

Schema: 13 BIO tags (6 entity types)
- DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast:free"

REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 900
DELAY_BETWEEN_REQUESTS = 6.0

DATA_DIR = Path("D:/Modeling_studio/data/familyos/temporal")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000
SAMPLES_PER_REQUEST = 300

# =============================================================================
# Temporal Schema (13 BIO tags)
# =============================================================================

TEMPORAL_LABELS = {
    0: "O",
    1: "B-DATE_ABS",
    2: "I-DATE_ABS",  # "January 15", "2024", "March 5th"
    3: "B-DATE_REL",
    4: "I-DATE_REL",  # "yesterday", "last week", "next month"
    5: "B-TIME",
    6: "I-TIME",  # "3pm", "morning", "at noon"
    7: "B-DURATION",
    8: "I-DURATION",  # "for 2 hours", "all day"
    9: "B-FREQUENCY",
    10: "I-FREQUENCY",  # "every Sunday", "weekly"
    11: "B-AGE",
    12: "I-AGE",  # "when she was 5", "in my 20s"
}

TAG_INFO = {
    0: ("O", True),
    1: ("DATE_ABS", True),
    2: ("DATE_ABS", False),
    3: ("DATE_REL", True),
    4: ("DATE_REL", False),
    5: ("TIME", True),
    6: ("TIME", False),
    7: ("DURATION", True),
    8: ("DURATION", False),
    9: ("FREQUENCY", True),
    10: ("FREQUENCY", False),
    11: ("AGE", True),
    12: ("AGE", False),
}

ENTITY_EXAMPLES = {
    "DATE_ABS": ["January 15", "March 5th", "2024", "December 25", "April 10, 2023"],
    "DATE_REL": ["yesterday", "today", "tomorrow", "last week", "next month", "this weekend"],
    "TIME": ["3pm", "morning", "at noon", "8:30 AM", "evening", "at night", "this afternoon"],
    "DURATION": ["for 2 hours", "all day", "5 minutes", "a week", "three months"],
    "FREQUENCY": ["every Sunday", "weekly", "daily", "twice a day", "every month", "annually"],
    "AGE": ["when she was 5", "at age 10", "in my 20s", "as a child", "when he turned 18"],
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for temporal expression extraction.
Your job is to generate realistic sentences with temporal expressions and annotate them with BIO tags.

## Temporal Entity Types (6 types, 13 BIO tags)

| ID | Tag | Description | Examples |
|----|-----|-------------|----------|
| 0 | O | Outside any temporal entity | Regular words |
| 1 | B-DATE_ABS | Begin absolute date | "January 15", "2024", "March 5th" |
| 2 | I-DATE_ABS | Inside absolute date | Multi-word dates |
| 3 | B-DATE_REL | Begin relative date | "yesterday", "last week", "next month" |
| 4 | I-DATE_REL | Inside relative date | "last week" (week is I-DATE_REL) |
| 5 | B-TIME | Begin time expression | "3pm", "morning", "at noon" |
| 6 | I-TIME | Inside time expression | "3 o'clock" (o'clock is I-TIME) |
| 7 | B-DURATION | Begin duration | "for 2 hours", "all day" |
| 8 | I-DURATION | Inside duration | "2 hours" (hours is I-DURATION) |
| 9 | B-FREQUENCY | Begin frequency | "every Sunday", "weekly" |
| 10 | I-FREQUENCY | Inside frequency | "every Sunday" (Sunday is I-FREQUENCY) |
| 11 | B-AGE | Begin age/life period | "when she was 5", "at age 10" |
| 12 | I-AGE | Inside age | "she was 5" (was 5 is I-AGE) |

## BIO Tagging Rules

1. **B- tag marks the BEGINNING of an entity**
2. **I- tag marks tokens INSIDE an entity (continuation)**
3. **Every I- must follow a B- of the same type**
4. **Single-word entities only get B- tag**

## Examples with proper BIO tagging

1. "We went to the park yesterday morning"
   tokens: ["We", "went", "to", "the", "park", "yesterday", "morning"]
   tags: [0, 0, 0, 0, 0, 3, 5]
   (yesterday=B-DATE_REL, morning=B-TIME - separate entities)

2. "Emma's birthday is on January 15"
   tokens: ["Emma", "'s", "birthday", "is", "on", "January", "15"]
   tags: [0, 0, 0, 0, 0, 1, 2]
   (January=B-DATE_ABS, 15=I-DATE_ABS - multi-word date)

3. "We visit grandma every Sunday"
   tokens: ["We", "visit", "grandma", "every", "Sunday"]
   tags: [0, 0, 0, 9, 10]
   (every=B-FREQUENCY, Sunday=I-FREQUENCY)

4. "The meeting lasted for 2 hours"
   tokens: ["The", "meeting", "lasted", "for", "2", "hours"]
   tags: [0, 0, 0, 7, 8, 8]
   (for=B-DURATION, 2=I-DURATION, hours=I-DURATION)

5. "When she was 5, she started piano"
   tokens: ["When", "she", "was", "5", ",", "she", "started", "piano"]
   tags: [11, 12, 12, 12, 0, 0, 0, 0]
   (When=B-AGE, she was 5=I-AGE, I-AGE, I-AGE)

## Quality Requirements

1. **Tokens and tags must have same length**
2. **Use proper BIO sequencing** - no orphan I- tags
3. **Include family contexts** - appointments, activities, milestones
4. **Mix entity types** - some sentences have multiple temporal expressions
5. **Tokenize properly** - split punctuation, contractions

## Output Format
Output ONLY valid JSONL. Each line must be:
{"tokens": ["list", "of", "tokens"], "temporal_tags": [0, 0, 0]}

Now generate the requested samples. Output JSONL only:"""


def get_user_prompt(num_samples: int, focus_entities: list[str], batch_id: int) -> str:
    """Generate diverse user prompts."""
    entity_info = "\n".join([f"- {e}: {', '.join(ENTITY_EXAMPLES[e][:3])}" for e in focus_entities])

    contexts = [
        "family schedules and appointments",
        "milestone memories (birthdays, first steps)",
        "daily routines (morning, bedtime, meals)",
        "recurring activities (weekly, monthly events)",
        "duration of activities (how long things lasted)",
        "age-related memories (when kids were young)",
        "holiday and celebration dates",
        "health appointments and check-ups",
    ]
    context = contexts[batch_id % len(contexts)]

    return f"""Generate {num_samples} temporal expression samples.

Focus on these entity types:
{entity_info}

Context: {context}
Include ~30% Indian family contexts.
Mix single and multiple temporal expressions per sentence.

Output JSONL only:"""


# =============================================================================
# Validation & Utilities
# =============================================================================


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single temporal sample."""
    if "tokens" not in sample or "temporal_tags" not in sample:
        return False, "Missing 'tokens' or 'temporal_tags'"

    tokens = sample["tokens"]
    tags = sample["temporal_tags"]

    if not isinstance(tokens, list) or not isinstance(tags, list):
        return False, "tokens and temporal_tags must be lists"

    if len(tokens) != len(tags):
        return False, f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags"

    if len(tokens) == 0:
        return False, "Empty sample"

    if not all(isinstance(t, str) for t in tokens):
        return False, "All tokens must be strings"

    if not all(isinstance(t, int) and 0 <= t <= 12 for t in tags):
        return False, f"Invalid tag values: {tags}"

    # Check BIO consistency
    prev_entity = None
    for i, tag in enumerate(tags):
        if tag == 0:
            prev_entity = None
        elif tag in TAG_INFO:
            entity, is_beginning = TAG_INFO[tag]
            if is_beginning:
                prev_entity = entity
            else:
                if prev_entity != entity:
                    return False, f"Orphan I-{entity} tag at position {i}"

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    text = " ".join(sample["tokens"]).lower()
    return hashlib.md5(text.encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response."""
    valid_samples = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            json_match = re.search(r'\{[^{}]*"tokens"[^{}]*"temporal_tags"[^{}]*\}', line)
            if json_match:
                sample = json.loads(json_match.group())
            else:
                sample = json.loads(line)

            is_valid, _ = validate_sample(sample)
            if is_valid:
                valid_samples.append(sample)
        except json.JSONDecodeError:
            continue
    return valid_samples


def get_entity_coverage(sample: dict[str, Any]) -> dict[str, int]:
    """Count entity types in a sample."""
    counts: dict[str, int] = defaultdict(int)
    for tag in sample.get("temporal_tags", []):
        if tag in TAG_INFO and tag != 0:
            entity, is_beginning = TAG_INFO[tag]
            if is_beginning:
                counts[entity] += 1
    return dict(counts)


# =============================================================================
# OpenRouter Client
# =============================================================================


class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.request_times: list[datetime] = []
        self.daily_count = 0
        self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        self.client = httpx.Client(timeout=120.0)

    def _wait_for_rate_limit(self) -> None:
        now = datetime.now()
        if now >= self.daily_reset:
            self.daily_count = 0
            self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        if self.daily_count >= REQUESTS_PER_DAY:
            raise RuntimeError("Daily rate limit reached")
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        if len(self.request_times) >= REQUESTS_PER_MINUTE:
            oldest = min(self.request_times)
            wait_seconds = (oldest + timedelta(minutes=1) - now).total_seconds()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._wait_for_rate_limit()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 30000,
        }
        response = self.client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        self.request_times.append(datetime.now())
        self.daily_count += 1
        return response.json()["choices"][0]["message"]["content"]

    def close(self):
        self.client.close()


# =============================================================================
# Silver Data Manager
# =============================================================================


class SilverDataManager:
    def __init__(self):
        self.silver_dir = SILVER_DIR
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: set[str] = set()
        self._load_existing_hashes()
        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

    def _load_existing_hashes(self):
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
        return sum(1 for _ in open(shard_path)) if shard_path.exists() else 0

    def _get_next_shard_id(self) -> int:
        # INSTANCE 2: Start from shard 1000
        existing = [int(p.stem.split("_")[1]) for p in self.silver_dir.glob("shard_1*.jsonl")]
        if not existing:
            return 1000
        max_id = max(existing)
        return max_id + 1 if self._count_shard_samples(max_id) >= SHARD_SIZE else max_id

    def add_samples(self, samples: list[dict[str, Any]]) -> int:
        added = 0
        for sample in samples:
            sample_hash = compute_sample_hash(sample)
            if sample_hash in self.seen_hashes:
                continue
            if self.current_shard_count >= SHARD_SIZE:
                self.current_shard_id += 1
                self.current_shard_count = 0
            with open(self._get_shard_path(self.current_shard_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            self.seen_hashes.add(sample_hash)
            self.current_shard_count += 1
            added += 1
        return added

    def get_total_samples(self) -> int:
        return len(self.seen_hashes)

    def get_stats(self) -> dict[str, Any]:
        entity_counts: dict[str, int] = defaultdict(int)
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        for entity, count in get_entity_coverage(sample).items():
                            entity_counts[entity] += count
                    except json.JSONDecodeError:
                        continue
        return {"total_samples": len(self.seen_hashes), "entity_counts": dict(entity_counts)}


# =============================================================================
# Data Generator Agent
# =============================================================================


class TemporalDataGeneratorAgent:
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()
        self.entity_counts: dict[str, int] = defaultdict(int)
        self.batch_id = 0

    def _get_underrepresented_entities(self, n: int = 3) -> list[str]:
        all_entities = ["DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"]
        sorted_entities = sorted(all_entities, key=lambda e: self.entity_counts.get(e, 0))
        return sorted_entities[:n]

    def generate_batch(self) -> int:
        focus = self._get_underrepresented_entities(3)
        user_prompt = get_user_prompt(
            SAMPLES_PER_REQUEST, focus_entities=focus, batch_id=self.batch_id
        )

        try:
            response = self.client.generate(SYSTEM_PROMPT, user_prompt)
            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            for sample in samples:
                for entity, count in get_entity_coverage(sample).items():
                    self.entity_counts[entity] += count

            self.batch_id += 1
            logger.info(
                f"Generated {len(samples)}, added {added}. Total: {self.silver_manager.get_total_samples()}"
            )
            return added
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            return 0

    def run(
        self, target_samples: int | None = None, max_requests: int | None = None
    ) -> dict[str, Any]:
        start_time = datetime.now()
        existing = self.silver_manager.get_total_samples()
        stats = {"existing": existing, "new_samples": 0, "requests": 0}

        try:
            while True:
                if target_samples and stats["new_samples"] >= target_samples:
                    break
                if max_requests and stats["requests"] >= max_requests:
                    break

                stats["requests"] += 1
                stats["new_samples"] += self.generate_batch()
                time.sleep(DELAY_BETWEEN_REQUESTS)
        except KeyboardInterrupt:
            logger.info("Interrupted")
        finally:
            self.client.close()

        stats["total"] = self.silver_manager.get_total_samples()
        stats["duration"] = str(datetime.now() - start_time)
        return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate temporal expression data")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--target", type=int, default=None)
    gen_parser.add_argument("--max-requests", type=int, default=None)
    gen_parser.add_argument("--api-key", type=str, default=None)

    subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.command == "generate":
        api_key = args.api_key or OPENROUTER_API_KEY
        if "YOUR_KEY_HERE" in api_key:
            print("ERROR: Provide API key via --api-key")
            return
        agent = TemporalDataGeneratorAgent(api_key=api_key)
        print(
            json.dumps(
                agent.run(target_samples=args.target, max_requests=args.max_requests), indent=2
            )
        )
    elif args.command == "stats":
        print(json.dumps(SilverDataManager().get_stats(), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
