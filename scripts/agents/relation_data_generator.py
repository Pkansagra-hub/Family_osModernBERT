"""
Relation Extraction Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic relation extraction data
for the FamilyOS Relations dataset.

Schema: 15 relation types
- no_relation, parent_of, child_of, spouse_of, sibling_of
- grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of
- pet_of, friend_of, colleague_of, lives_at, owns
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

DATA_DIR = Path("D:/Modeling_studio/data/familyos/relations")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000
SAMPLES_PER_REQUEST = 100

# =============================================================================
# Relation Schema
# =============================================================================

RELATION_LABELS = {
    0: {"name": "no_relation", "description": "No relationship between entities"},
    1: {
        "name": "parent_of",
        "description": "X is parent of Y",
        "example": "Mom took Emma to school",
    },
    2: {"name": "child_of", "description": "X is child of Y", "example": "Panda loves her mom"},
    3: {
        "name": "spouse_of",
        "description": "X is married to Y",
        "example": "I went with my husband to dinner",
    },
    4: {
        "name": "sibling_of",
        "description": "X is sibling of Y",
        "example": "Bhai and I played cricket",
    },
    5: {
        "name": "grandparent_of",
        "description": "X is grandparent of Y",
        "example": "Nani is visiting Emma",
    },
    6: {
        "name": "grandchild_of",
        "description": "X is grandchild of Y",
        "example": "Kids visited dada yesterday",
    },
    7: {
        "name": "aunt_uncle_of",
        "description": "X is aunt/uncle of Y",
        "example": "Chacha brought gifts for the kids",
    },
    8: {
        "name": "niece_nephew_of",
        "description": "X is niece/nephew of Y",
        "example": "Emma loves visiting her masi",
    },
    9: {
        "name": "cousin_of",
        "description": "X is cousin of Y",
        "example": "Played with cousin Rohan today",
    },
    10: {"name": "pet_of", "description": "X is pet of Y", "example": "Max is the family dog"},
    11: {
        "name": "friend_of",
        "description": "X is friend of Y",
        "example": "Sarah is Emma's best friend",
    },
    12: {
        "name": "colleague_of",
        "description": "X works with Y",
        "example": "Had lunch with my colleague Raj",
    },
    13: {
        "name": "lives_at",
        "description": "X lives at Y (location)",
        "example": "Grandma lives in Mumbai",
    },
    14: {
        "name": "owns",
        "description": "X owns Y (heirloom)",
        "example": "Dad inherited grandpa's watch",
    },
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for a family-focused relation extraction task.
Your job is to generate realistic sentences with two entities and their relationship.

## Relation Schema (15 types)

| ID | Relation | Description |
|----|----------|-------------|
| 0 | no_relation | No relationship between entities |
| 1 | parent_of | X is parent of Y |
| 2 | child_of | X is child of Y |
| 3 | spouse_of | X is married to Y |
| 4 | sibling_of | X is sibling of Y |
| 5 | grandparent_of | X is grandparent of Y |
| 6 | grandchild_of | X is grandchild of Y |
| 7 | aunt_uncle_of | X is aunt/uncle of Y |
| 8 | niece_nephew_of | X is niece/nephew of Y |
| 9 | cousin_of | X is cousin of Y |
| 10 | pet_of | X is pet of Y |
| 11 | friend_of | X is friend of Y |
| 12 | colleague_of | X works with Y |
| 13 | lives_at | X lives at Y (location) |
| 14 | owns | X owns Y (heirloom/object) |

## Quality Requirements

1. **Entity mentions must appear in the text exactly as specified**
   - entity1 and entity2 must be substrings of text

2. **Use diverse entity types**
   - Names: Emma, Panda, John, Sarah, Raj, Priya
   - Kinship terms: mom, dad, didi, bhai, nana, nani, chacha, masi
   - Nicknames: Panda, Bunny, Sweetie
   - Pets: Max, Whiskers, Buddy
   - Locations: Mumbai, kitchen, grandma's house

3. **Include Indian family contexts**
   - Kinship terms: didi (sister), bhai (brother), nana/nani (maternal grandparents), dada/dadi (paternal grandparents)
   - chacha/chachi (uncle/aunt), masi/mausa (maternal aunt/uncle)

4. **Relation direction matters**
   - parent_of: (Mom, parent_of, Emma) - Mom is the parent
   - child_of: (Emma, child_of, Mom) - Emma is the child

## Output Format
Output ONLY valid JSONL. Each line must be:
{"text": "...", "entity1": "...", "entity2": "...", "relation": <int 0-14>}

## Examples

{"text": "Mom took Emma to school this morning", "entity1": "Mom", "entity2": "Emma", "relation": 1}
{"text": "Panda loves spending time with her nani", "entity1": "Panda", "entity2": "nani", "relation": 6}
{"text": "Chacha brought gifts for the kids from his trip", "entity1": "Chacha", "entity2": "kids", "relation": 7}
{"text": "Max the dog follows Dad everywhere around the house", "entity1": "Max", "entity2": "Dad", "relation": 10}
{"text": "Grandma lives in Mumbai with dada", "entity1": "Grandma", "entity2": "Mumbai", "relation": 13}
{"text": "Sarah and Emma are classmates at school", "entity1": "Sarah", "entity2": "Emma", "relation": 11}

Now generate the requested samples. Output JSONL only:"""


def get_user_prompt(num_samples: int, focus_relations: list[int], batch_id: int) -> str:
    """Generate diverse user prompts."""
    relation_info = "\n".join(
        [
            f"- {RELATION_LABELS[r]['name']}: {RELATION_LABELS[r]['description']}"
            for r in focus_relations
        ]
    )

    contexts = [
        "immediate family interactions (parents, kids, siblings)",
        "extended family gatherings (grandparents, aunts, uncles, cousins)",
        "pets and family members",
        "friends and social relationships",
        "family heirlooms and possessions",
        "family members and locations they live",
        "Indian joint family dynamics",
        "Western nuclear family scenarios",
    ]
    context = contexts[batch_id % len(contexts)]

    return f"""Generate {num_samples} relation extraction samples.

Focus on these relations:
{relation_info}

Context: {context}
Include ~40% Indian family contexts (kinship terms like didi, bhai, nana, chacha, masi, etc.)

Output JSONL only:"""


# =============================================================================
# Validation & Utilities
# =============================================================================


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single sample."""
    required = ["text", "entity1", "entity2", "relation"]
    for field in required:
        if field not in sample:
            return False, f"Missing '{field}'"

    if not isinstance(sample["text"], str) or len(sample["text"]) < 10:
        return False, "Text too short"

    # Check entities appear in text
    text_lower = sample["text"].lower()
    if sample["entity1"].lower() not in text_lower:
        return False, f"entity1 '{sample['entity1']}' not in text"
    if sample["entity2"].lower() not in text_lower:
        return False, f"entity2 '{sample['entity2']}' not in text"

    if not isinstance(sample["relation"], int) or sample["relation"] not in range(15):
        return False, f"Invalid relation: {sample.get('relation')}"

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    text = f"{sample['text'].lower()}|{sample['entity1'].lower()}|{sample['entity2'].lower()}"
    return hashlib.md5(text.encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response."""
    valid_samples = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            json_match = re.search(r'\{[^{}]*"text"[^{}]*"relation"[^{}]*\}', line)
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
        existing = list(self.silver_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
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
        relation_counts: dict[int, int] = defaultdict(int)
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        relation_counts[sample.get("relation", -1)] += 1
                    except json.JSONDecodeError:
                        continue
        return {
            "total_samples": len(self.seen_hashes),
            "relation_counts": {
                RELATION_LABELS[k]["name"]: v
                for k, v in relation_counts.items()
                if k in RELATION_LABELS
            },
        }


# =============================================================================
# Data Generator Agent
# =============================================================================


class RelationDataGeneratorAgent:
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()
        self.relation_counts: dict[int, int] = defaultdict(int)
        self.batch_id = 0

    def _get_underrepresented_relations(self, n: int = 4) -> list[int]:
        all_relations = list(RELATION_LABELS.keys())
        sorted_relations = sorted(all_relations, key=lambda r: self.relation_counts.get(r, 0))
        return sorted_relations[:n]

    def generate_batch(self) -> int:
        focus = self._get_underrepresented_relations(4)
        user_prompt = get_user_prompt(
            SAMPLES_PER_REQUEST, focus_relations=focus, batch_id=self.batch_id
        )

        try:
            response = self.client.generate(SYSTEM_PROMPT, user_prompt)
            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            for sample in samples:
                self.relation_counts[sample.get("relation", -1)] += 1

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

    parser = argparse.ArgumentParser(description="Generate relation extraction data")
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
        agent = RelationDataGeneratorAgent(api_key=api_key)
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
