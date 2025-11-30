"""
Intent Classification Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic intent classification data
for the FamilyOS Intent dataset.

Schema: 8 intent types
- log_memory, query_memory, set_reminder, express_feeling
- seek_advice, share_news, reflect, other
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

OPENROUTER_API_KEY = "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"  # Replace with your key
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast:free"

REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 900
DELAY_BETWEEN_REQUESTS = 6.0

DATA_DIR = Path("D:/Modeling_studio/data/familyos/intents")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000
SAMPLES_PER_REQUEST = 200

# =============================================================================
# Intent Schema
# =============================================================================

INTENT_LABELS = {
    0: {
        "name": "log_memory",
        "description": "Store/record information",
        "examples": [
            "Had dinner with family tonight",
            "Emma took her first steps today",
            "Visited grandma in the hospital this afternoon",
        ],
    },
    1: {
        "name": "query_memory",
        "description": "Retrieve past information",
        "examples": [
            "What did we do last Sunday?",
            "When was Emma's first birthday party?",
            "Who came to Diwali dinner last year?",
        ],
    },
    2: {
        "name": "set_reminder",
        "description": "Create a reminder/task",
        "examples": [
            "Remind me to call mom tomorrow",
            "Set a reminder for Emma's piano class at 4pm",
            "Don't let me forget dad's medication refill",
        ],
    },
    3: {
        "name": "express_feeling",
        "description": "Share emotions/feelings",
        "examples": [
            "Feeling grateful today",
            "I'm so proud of Panda's grades",
            "Missing nani a lot lately",
        ],
    },
    4: {
        "name": "seek_advice",
        "description": "Ask for guidance/help",
        "examples": [
            "What should I do about Emma's tantrums?",
            "How do I handle work-life balance better?",
            "Any tips for the kids' screen time?",
        ],
    },
    5: {
        "name": "share_news",
        "description": "Announce/share updates",
        "examples": [
            "Guess what happened today!",
            "Great news - got the promotion!",
            "Panda made the soccer team!",
        ],
    },
    6: {
        "name": "reflect",
        "description": "Contemplation/musing",
        "examples": [
            "Thinking about the past...",
            "Life changes so fast with kids",
            "Wonder what the future holds for our family",
        ],
    },
    7: {
        "name": "other",
        "description": "Catch-all for misc",
        "examples": [
            "Hello!",
            "Thanks for everything",
            "Okay, sounds good",
        ],
    },
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for a family-focused intent classification task.
Your job is to generate realistic text samples that families might say or write, labeled with their intent.

## Intent Schema (8 types)

| ID | Intent | Description |
|----|--------|-------------|
| 0 | log_memory | Store/record information - User is recording a memory or event |
| 1 | query_memory | Retrieve past information - User is asking about past events |
| 2 | set_reminder | Create a reminder/task - User wants to be reminded of something |
| 3 | express_feeling | Share emotions/feelings - User is sharing how they feel |
| 4 | seek_advice | Ask for guidance/help - User wants advice or suggestions |
| 5 | share_news | Announce/share updates - User is announcing news or updates |
| 6 | reflect | Contemplation/musing - User is thinking deeply about something |
| 7 | other | Catch-all for misc - Greetings, acknowledgments, etc. |

## Quality Requirements

1. **Natural family language**
   - Mix of Western and Indian family contexts
   - Include nicknames, kinship terms (didi, bhai, nana, mummy, papa)
   - Casual, conversational tone

2. **Clear intent signals**
   - log_memory: Past tense, recording events ("Had dinner...", "Today we...")
   - query_memory: Questions about past ("What did...", "When was...")
   - set_reminder: Future actions, reminders ("Remind me...", "Don't forget...")
   - express_feeling: Emotional expressions ("Feeling...", "So happy...", "Miss...")
   - seek_advice: Questions seeking help ("What should I...", "How do I...")
   - share_news: Announcements ("Guess what!", "Great news!", "Just found out...")
   - reflect: Contemplative statements ("Thinking about...", "Life is...", "Wonder...")
   - other: Short, non-specific ("Thanks", "Okay", "Hello")

3. **Diversity**
   - Vary sentence length
   - Mix text types: diary entries, voice commands, text messages
   - Cover different family situations

## Output Format
Output ONLY valid JSONL. Each line must be:
{"text": "...", "label": <int 0-7>}

## Examples

{"text": "Had a lovely dinner with the whole family at nani's house", "label": 0}
{"text": "What did we do for Panda's 5th birthday?", "label": 1}
{"text": "Remind me to pick up Emma from soccer practice at 5pm", "label": 2}
{"text": "Feeling so blessed to have such a supportive family", "label": 3}
{"text": "How should I handle the kids fighting over screen time?", "label": 4}
{"text": "Big news - didi is getting married next spring!", "label": 5}
{"text": "Sometimes I wonder how different life would be without the kids", "label": 6}
{"text": "Thanks for the update!", "label": 7}

Now generate the requested samples. Output JSONL only:"""


def get_user_prompt(num_samples: int, focus_intents: list[int], batch_id: int) -> str:
    """Generate diverse user prompts."""
    intent_info = "\n".join(
        [f"- {INTENT_LABELS[i]['name']}: {INTENT_LABELS[i]['description']}" for i in focus_intents]
    )

    contexts = [
        "morning family conversations",
        "evening wind-down time",
        "weekend family activities",
        "school-related discussions",
        "health and wellness conversations",
        "financial discussions",
        "holiday and celebration planning",
        "daily routines and tasks",
    ]
    context = contexts[batch_id % len(contexts)]

    return f"""Generate {num_samples} intent classification samples.

Focus on these intents:
{intent_info}

Context: {context}
Include ~30% Indian family contexts (kinship terms, festivals, etc.)

Output JSONL only:"""


# =============================================================================
# Validation & Utilities
# =============================================================================


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single sample."""
    if "text" not in sample or "label" not in sample:
        return False, "Missing 'text' or 'label'"
    if not isinstance(sample["text"], str) or len(sample["text"].strip()) < 5:
        return False, "Text too short"
    if not isinstance(sample["label"], int) or sample["label"] not in range(8):
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
# OpenRouter Client (same as others)
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
                logger.info(f"Rate limiting: waiting {wait_seconds:.1f}s")
                time.sleep(wait_seconds)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self._wait_for_rate_limit()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/modeling-studio",
            "X-Title": "FamilyOS Intent Data Generator",
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
        if not shard_path.exists():
            return 0
        with open(shard_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _get_next_shard_id(self) -> int:
        existing = list(self.silver_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
        if self._count_shard_samples(max_id) >= SHARD_SIZE:
            return max_id + 1
        return max_id

    def add_samples(self, samples: list[dict[str, Any]]) -> int:
        added = 0
        for sample in samples:
            sample_hash = compute_sample_hash(sample)
            if sample_hash in self.seen_hashes:
                continue
            if self.current_shard_count >= SHARD_SIZE:
                self.current_shard_id += 1
                self.current_shard_count = 0
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
        label_counts: dict[int, int] = defaultdict(int)
        total = 0
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        label_counts[sample.get("label", -1)] += 1
                        total += 1
                    except json.JSONDecodeError:
                        continue
        return {
            "total_samples": total,
            "label_counts": {
                INTENT_LABELS[k]["name"]: v for k, v in label_counts.items() if k in INTENT_LABELS
            },
        }


# =============================================================================
# Data Generator Agent
# =============================================================================


class IntentDataGeneratorAgent:
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()
        self.label_counts: dict[int, int] = defaultdict(int)
        self._load_existing_counts()
        self.batch_id = 0

    def _load_existing_counts(self):
        for shard_path in SILVER_DIR.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.label_counts[sample.get("label", -1)] += 1
                    except json.JSONDecodeError:
                        continue

    def _get_underrepresented_labels(self, n: int = 3) -> list[int]:
        all_labels = list(INTENT_LABELS.keys())
        sorted_labels = sorted(all_labels, key=lambda l: self.label_counts.get(l, 0))
        return sorted_labels[:n]

    def generate_batch(self) -> int:
        focus = self._get_underrepresented_labels(3)
        user_prompt = get_user_prompt(
            SAMPLES_PER_REQUEST, focus_intents=focus, batch_id=self.batch_id
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
                added = self.generate_batch()
                stats["new_samples"] += added

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

    parser = argparse.ArgumentParser(description="Generate intent classification data")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--target", type=int, default=None)
    gen_parser.add_argument("--max-requests", type=int, default=None)
    gen_parser.add_argument("--api-key", type=str, default=None)

    stats_parser = subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.command == "generate":
        api_key = args.api_key or OPENROUTER_API_KEY
        if "YOUR_KEY_HERE" in api_key:
            print("ERROR: Provide API key via --api-key")
            return
        agent = IntentDataGeneratorAgent(api_key=api_key)
        stats = agent.run(target_samples=args.target, max_requests=args.max_requests)
        print(json.dumps(stats, indent=2))

    elif args.command == "stats":
        manager = SilverDataManager()
        print(json.dumps(manager.get_stats(), indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
