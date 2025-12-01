"""
Ingress Domain Data Generator Agent

Uses OpenRouter API with Grok to generate synthetic domain classification data
for the FamilyOS Ingress dataset.

Schema: 12 domain labels for routing user input to appropriate handlers.
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

DATA_DIR = Path("D:/Modeling_studio/data/familyos/ingress")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000
SAMPLES_PER_REQUEST = 200

# =============================================================================
# Ingress Labels (12 domains)
# =============================================================================

INGRESS_LABELS = {
    "DIARY": 0,  # Personal journal entries, reflections, daily logs
    "TASK": 1,  # To-do items, reminders, action items
    "HEALTH": 2,  # Medical, fitness, wellness topics
    "FINANCE": 3,  # Money, budgets, expenses, savings
    "RELATIONSHIP": 4,  # Family dynamics, friendships, social
    "WORK": 5,  # Professional, career, job-related
    "META": 6,  # Questions about the AI itself, settings
    "MEMORY": 7,  # Requests to recall or retrieve past info
    "PLANNING": 8,  # Future events, scheduling, coordination
    "CELEBRATION": 9,  # Achievements, milestones, happy events
    "CONCERN": 10,  # Worries, anxieties, problems
    "GRATITUDE": 11,  # Appreciation, thankfulness
}

LABEL_DESCRIPTIONS = {
    "DIARY": "Personal journal entries, reflections on the day, emotional processing",
    "TASK": "To-do items, reminders, chores, action items to complete",
    "HEALTH": "Medical appointments, symptoms, fitness goals, wellness topics",
    "FINANCE": "Money matters, budgets, expenses, savings, purchases",
    "RELATIONSHIP": "Family dynamics, friendships, conflicts, social connections",
    "WORK": "Professional matters, career, job, meetings, projects",
    "META": "Questions about the AI assistant, settings, capabilities",
    "MEMORY": "Requests to recall or retrieve past conversations or information",
    "PLANNING": "Future events, scheduling, vacation planning, coordination",
    "CELEBRATION": "Achievements, milestones, birthdays, happy announcements",
    "CONCERN": "Worries, anxieties, problems seeking advice or venting",
    "GRATITUDE": "Expressions of appreciation, thankfulness, positive moments",
}

LABEL_EXAMPLES = {
    "DIARY": [
        "Today was a really long day. I couldn't stop thinking about...",
        "Just reflecting on how much has changed this year.",
        "Had a weird dream last night, need to write it down.",
    ],
    "TASK": [
        "Remind me to pick up groceries tomorrow.",
        "Add 'call dentist' to my to-do list.",
        "I need to finish the laundry before 5pm.",
    ],
    "HEALTH": [
        "Priya has a doctor's appointment next Tuesday.",
        "I've been having headaches all week.",
        "Need to track our family's vaccinations.",
    ],
    "FINANCE": [
        "We spent way too much on eating out this month.",
        "Should we increase our emergency fund?",
        "The electric bill was higher than usual.",
    ],
    "RELATIONSHIP": [
        "Arjun and his sister got into another fight.",
        "I'm worried about how little time we spend together.",
        "Grandma called today, she seemed lonely.",
    ],
    "WORK": [
        "Big meeting tomorrow with the whole team.",
        "My boss gave me some tough feedback today.",
        "Thinking about asking for a promotion.",
    ],
    "META": [
        "Can you remember what I told you last week?",
        "How do I change my notification settings?",
        "What kind of things can you help me with?",
    ],
    "MEMORY": [
        "What did we decide about summer vacation?",
        "When was the last time Priya got sick?",
        "Remind me what we talked about on Monday.",
    ],
    "PLANNING": [
        "Let's plan Diwali celebrations for the family.",
        "We need to coordinate the kids' pickup schedule.",
        "What should we do for our anniversary?",
    ],
    "CELEBRATION": [
        "Arjun got an A on his math test!",
        "Our baby took her first steps today!",
        "Just got promoted at work!",
    ],
    "CONCERN": [
        "I'm really worried about my mom's health.",
        "The kids have been fighting a lot lately.",
        "I don't know how we'll afford this.",
    ],
    "GRATITUDE": [
        "So thankful for how supportive my partner has been.",
        "The kids made me breakfast in bed today!",
        "I'm grateful we have such good neighbors.",
    ],
}

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for family message domain classification.
Your job is to generate realistic family diary entries and classify them into domains.

## Domain Labels (12 categories)

| Label | ID | Description | Examples |
|-------|-----|-------------|----------|
| DIARY | 0 | Personal journal entries, reflections | "Today was a really long day..." |
| TASK | 1 | To-do items, reminders, chores | "Remind me to pick up groceries" |
| HEALTH | 2 | Medical, fitness, wellness | "Priya has a doctor's appointment" |
| FINANCE | 3 | Money, budgets, expenses | "We spent too much this month" |
| RELATIONSHIP | 4 | Family dynamics, friendships | "Arjun and his sister fought again" |
| WORK | 5 | Professional, career, job | "Big meeting tomorrow" |
| META | 6 | Questions about the AI itself | "Can you remember what I said?" |
| MEMORY | 7 | Recall/retrieve past info | "What did we decide about vacation?" |
| PLANNING | 8 | Future events, scheduling | "Let's plan Diwali celebrations" |
| CELEBRATION | 9 | Achievements, milestones | "Arjun got an A on his test!" |
| CONCERN | 10 | Worries, anxieties, problems | "I'm worried about mom's health" |
| GRATITUDE | 11 | Appreciation, thankfulness | "So thankful for my partner" |

## Quality Requirements

1. **Single primary domain per message** - choose the dominant one
2. **Natural family contexts** - use realistic names, situations
3. **Include ~30% Indian family contexts** - Diwali, chai, family names like Priya, Arjun, Amma
4. **Vary message length** - short (1 sentence) to medium (2-3 sentences)
5. **Mix emotional tones** - positive, neutral, concerned
6. **Avoid overlapping edge cases** - be clear about primary intent

## Output Format
Output ONLY valid JSONL. Each line must be:
{"text": "The user's message...", "label": "LABEL_NAME"}

Now generate the requested samples. Output JSONL only:"""


def get_user_prompt(num_samples: int, focus_labels: list[str], batch_id: int) -> str:
    """Generate diverse user prompts focusing on underrepresented labels."""
    label_info = "\n".join([f"- {label}: {LABEL_DESCRIPTIONS[label]}" for label in focus_labels])

    scenarios = [
        "weekday mornings (school, work prep)",
        "evening family time",
        "weekend activities",
        "holiday seasons (Diwali, Christmas, birthdays)",
        "stressful periods (illness, deadlines)",
        "quiet reflective moments",
        "busy multitasking parents",
        "milestone events (graduations, first days)",
        "family meals and cooking together",
        "parent-teacher meetings or school events",
        "family travel or road trips",
        "unexpected emergencies (lost items, urgent calls)",
        "celebrating achievements (awards, promotions)",
        "dealing with household chores and repairs",
        "family health routines (medication, exercise)",
        "budgeting and financial planning sessions",
        "conflicts and resolutions among siblings",
        "welcoming guests or relatives",
        "planning for future (vacations, studies)",
        "expressing gratitude or appreciation",
        "navigating technology issues at home",
        "supporting children with homework",
        "discussing family traditions or rituals",
        "managing remote work and home balance",
        "reflecting on family memories and milestones",
    ]
    scenario = scenarios[batch_id % len(scenarios)]

    return f"""Generate {num_samples} family message samples.

Focus especially on these labels (they need more samples):
{label_info}

Scenario context: {scenario}
Include ~30% Indian family contexts (names: Priya, Arjun, Amma; events: Diwali, puja).
Mix short (1 sentence) and medium (2-3 sentences) messages.

Output JSONL only:"""


# =============================================================================
# Validation & Utilities
# =============================================================================


def validate_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single ingress sample."""
    if "text" not in sample or "label" not in sample:
        return False, "Missing 'text' or 'label'"

    text = sample["text"]
    label = sample["label"]

    if not isinstance(text, str) or len(text.strip()) < 10:
        return False, "Text too short or invalid"

    if label not in INGRESS_LABELS:
        return False, f"Invalid label: {label}"

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    text = sample["text"].lower().strip()
    return hashlib.md5(text.encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response."""
    valid_samples = []
    for line in response_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            json_match = re.search(
                r'\{[^{}]*"text"[^{}]*"label"[^{}]*\}|\{[^{}]*"label"[^{}]*"text"[^{}]*\}', line
            )
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
        label_counts: dict[str, int] = defaultdict(int)
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        label_counts[sample["label"]] += 1
                    except (json.JSONDecodeError, KeyError):
                        continue
        return {"total_samples": len(self.seen_hashes), "label_counts": dict(label_counts)}


# =============================================================================
# Data Generator Agent
# =============================================================================


class IngressDataGeneratorAgent:
    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        self.client = OpenRouterClient(api_key=api_key)
        self.silver_manager = SilverDataManager()
        self.label_counts: dict[str, int] = defaultdict(int)
        self.batch_id = 0

    def _get_underrepresented_labels(self, n: int = 4) -> list[str]:
        all_labels = list(INGRESS_LABELS.keys())
        sorted_labels = sorted(all_labels, key=lambda l: self.label_counts.get(l, 0))
        return sorted_labels[:n]

    def generate_batch(self) -> int:
        focus = self._get_underrepresented_labels(4)
        user_prompt = get_user_prompt(
            SAMPLES_PER_REQUEST, focus_labels=focus, batch_id=self.batch_id
        )

        try:
            response = self.client.generate(SYSTEM_PROMPT, user_prompt)
            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            for sample in samples:
                self.label_counts[sample["label"]] += 1

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

    parser = argparse.ArgumentParser(description="Generate ingress domain data")
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
        agent = IngressDataGeneratorAgent(api_key=api_key)
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
