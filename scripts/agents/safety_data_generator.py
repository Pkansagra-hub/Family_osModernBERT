"""
Safety Classification Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic safety classification data
for the FamilyOS Safety dataset.

Schema: 4 policy bands
- GREEN (safe), AMBER (mild concern), RED (serious concern), CRISIS (emergency)
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

DATA_DIR = Path("D:/Modeling_studio/data/familyos/safety")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000
SAMPLES_PER_REQUEST = 100

# =============================================================================
# Safety Schema
# =============================================================================

SAFETY_LABELS = {
    0: {
        "name": "GREEN",
        "description": "Safe, routine content - process normally",
        "subcategories": ["daily_life", "positive_events", "neutral_updates", "family_activities"],
        "examples": [
            "Had a lovely dinner with the family tonight",
            "Kids played in the park after school",
            "Finished the quarterly report at work",
        ],
    },
    1: {
        "name": "AMBER",
        "description": "Needs attention, mild concern - flag for review",
        "subcategories": [
            "stress",
            "mild_sadness",
            "frustration",
            "health_mention",
            "minor_conflict",
        ],
        "examples": [
            "Feeling stressed about the upcoming presentation",
            "Had a fight with my spouse about finances",
            "Kids are being really difficult lately",
            "My back has been hurting all week",
        ],
    },
    2: {
        "name": "RED",
        "description": "Serious concern - escalate to K1, careful response",
        "subcategories": [
            "persistent_sadness",
            "isolation",
            "hopelessness",
            "substance",
            "severe_anxiety",
        ],
        "examples": [
            "I've been feeling empty for weeks now",
            "Don't feel like seeing anyone anymore",
            "Nothing ever seems to work out for me",
            "Had too much to drink again last night",
        ],
    },
    3: {
        "name": "CRISIS",
        "description": "Immediate intervention needed - emergency protocol",
        "subcategories": [
            "self_harm_ideation",
            "suicide_ideation",
            "harm_to_others",
            "abuse_disclosure",
        ],
        "examples": [
            "I don't want to be here anymore",
            "Sometimes I think everyone would be better off without me",
            "I want to hurt myself",
        ],
    },
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for a family well-being safety classification task.
Your job is to generate realistic text samples that families might write, labeled by safety policy band.

## Safety Policy Bands (4 levels)

| ID | Band | Description | System Response |
|----|------|-------------|-----------------|
| 0 | GREEN | Safe, routine content | Process normally |
| 1 | AMBER | Mild concern indicators | Flag for review, monitor |
| 2 | RED | Serious concern | Escalate, careful response |
| 3 | CRISIS | Immediate intervention needed | Emergency protocol |

### GREEN (ID: 0)
Safe, everyday family content. No special handling needed.
- Daily activities, positive events, neutral updates
- Work, school, hobbies, family activities
- Examples: "Kids had fun at the park", "Made dinner for the family", "Finished a project at work"

### AMBER (ID: 1)
Mild concern indicators that warrant monitoring but not immediate action.
- Subcategories: stress, mild_sadness, frustration, health_mention, minor_conflict
- Temporary low mood, work/life stress, family tensions
- Examples: "Feeling overwhelmed with deadlines", "Had a rough day", "Arguing with spouse about money"

### RED (ID: 2)
Serious concern requiring escalation and careful response.
- Subcategories: persistent_sadness, isolation, hopelessness, substance
- Ongoing depression indicators, social withdrawal, negative outlook
- Examples: "Been feeling empty for weeks", "Don't want to see anyone", "Nothing works out for me"

### CRISIS (ID: 3)
Immediate intervention required - emergency protocols.
- Subcategories: self_harm_ideation, suicide_ideation, harm_to_others, abuse_disclosure
- ONLY use for explicit mentions of self-harm, suicide, violence, or abuse
- Examples: "I don't want to be here anymore", "Want to hurt myself", "Everyone better off without me"

## IMPORTANT GUIDELINES

1. **GREEN should be most common (~50% of samples)**
   - Everyday family life: meals, activities, work, school, pets, routines

2. **AMBER should be second most common (~30%)**
   - Temporary stress, minor conflicts, small health issues
   - NOT ongoing depression or serious issues

3. **RED should be less common (~15%)**
   - Persistent negative patterns, isolation, hopelessness
   - NOT explicit self-harm mentions

4. **CRISIS should be rare (~5%)**
   - ONLY explicit self-harm/suicide ideation or abuse disclosure
   - Do NOT over-label mildly negative content as CRISIS

5. **Use realistic family language**
   - Include Indian family contexts (didi, bhai, nana, mummy, papa)
   - Natural, conversational tone

## Output Format
Output ONLY valid JSONL. Each line must be:
{"text": "...", "label": <int 0-3>}

## Examples

{"text": "Had a wonderful family dinner at nani's house today", "label": 0}
{"text": "Emma's soccer game was so exciting, they won 3-2!", "label": 0}
{"text": "Work has been really stressful this week with all the deadlines", "label": 1}
{"text": "Feeling a bit down today, nothing specific", "label": 1}
{"text": "I've been feeling so isolated lately, don't want to talk to anyone", "label": 2}
{"text": "Nothing I do ever seems to matter anymore", "label": 2}
{"text": "Sometimes I wonder if everyone would be happier without me around", "label": 3}

Now generate the requested samples. Output JSONL only:"""


def get_user_prompt(num_samples: int, focus_bands: list[int], batch_id: int) -> str:
    """Generate diverse user prompts."""
    band_info = "\n".join(
        [f"- {SAFETY_LABELS[b]['name']}: {SAFETY_LABELS[b]['description']}" for b in focus_bands]
    )

    contexts = [
        "daily family activities and routines (GREEN focus)",
        "work and career discussions (mix of GREEN/AMBER)",
        "parenting challenges and joys (mix of GREEN/AMBER)",
        "health and wellness topics (mix of GREEN/AMBER)",
        "emotional expressions and feelings (mix of bands)",
        "family conflicts and resolution (AMBER focus)",
        "stress and coping (AMBER/RED focus)",
        "mental health awareness (RED focus)",
    ]
    context = contexts[batch_id % len(contexts)]

    # Distribution guidance
    distribution = """
Distribution for this batch:
- GREEN (0): ~50% - everyday safe content
- AMBER (1): ~30% - mild concerns
- RED (2): ~15% - serious concerns
- CRISIS (3): ~5% - only explicit self-harm/suicide mentions"""

    return f"""Generate {num_samples} safety classification samples.

Focus on these bands:
{band_info}

Context: {context}
{distribution}

Include ~30% Indian family contexts.
Output JSONL only:"""


# =============================================================================
# Validation & Utilities
# =============================================================================


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single sample."""
    if "text" not in sample or "label" not in sample:
        return False, "Missing 'text' or 'label'"
    if not isinstance(sample["text"], str) or len(sample["text"].strip()) < 10:
        return False, "Text too short"
    if not isinstance(sample["label"], int) or sample["label"] not in range(4):
        return False, f"Invalid label: {sample.get('label')}"
    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    return hashlib.md5(sample["text"].lower().encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response."""
    valid_samples = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            json_match = re.search(r'\{[^{}]*"text"[^{}]*"label"[^{}]*\}', line)
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
        label_counts: dict[int, int] = defaultdict(int)
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        label_counts[sample.get("label", -1)] += 1
                    except json.JSONDecodeError:
                        continue
        return {
            "total_samples": len(self.seen_hashes),
            "label_counts": {
                SAFETY_LABELS[k]["name"]: v for k, v in label_counts.items() if k in SAFETY_LABELS
            },
        }


# =============================================================================
# Data Generator Agent
# =============================================================================


class SafetyDataGeneratorAgent:
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()
        self.label_counts: dict[int, int] = defaultdict(int)
        self.batch_id = 0

    def _get_focus_bands(self) -> list[int]:
        # Weight towards underrepresented but maintain distribution
        all_bands = [0, 1, 2, 3]
        weights = [50, 30, 15, 5]  # Target distribution
        actual = [self.label_counts.get(b, 0) for b in all_bands]
        total = sum(actual) or 1

        # Find bands below target percentage
        focus = []
        for b, target_pct in zip(all_bands, weights):
            actual_pct = (actual[b] / total) * 100 if total > 0 else 0
            if actual_pct < target_pct * 0.8:  # 80% of target
                focus.append(b)

        return focus if focus else [0, 1]  # Default to GREEN/AMBER

    def generate_batch(self) -> int:
        focus = self._get_focus_bands()
        user_prompt = get_user_prompt(
            SAMPLES_PER_REQUEST, focus_bands=focus, batch_id=self.batch_id
        )

        try:
            response = self.client.generate(SYSTEM_PROMPT, user_prompt)
            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            for sample in samples:
                self.label_counts[sample.get("label", -1)] += 1

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

    parser = argparse.ArgumentParser(description="Generate safety classification data")
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
        agent = SafetyDataGeneratorAgent(api_key=api_key)
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
