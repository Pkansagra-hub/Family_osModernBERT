"""
Indian Hyperbole Safety Data Generator Agent (Multi-Key Parallel)

Uses OpenRouter API with multiple accounts to generate synthetic Indian English
hyperbole samples for Cultural Robustness testing in FamilyOS Safety classification.

PURPOSE:
The safety_familyos model must correctly classify Indian English expressions that
sound alarming but are actually harmless cultural hyperbole. Without explicit training
on these patterns, the model will incorrectly classify them as CRISIS/RED → FP > 2%.

Examples:
- "I'll die of embarrassment" → GREEN (hyperbole, not literal)
- "This is killing me" → AMBER (stress expression)
- "मर जाऊंगी इस शर्म से" → GREEN (Hindi hyperbole)

Target: 500+ samples (~1 hour with 6 keys)

Usage:
    python indian_hyperbole_generator.py generate --target 500
    python indian_hyperbole_generator.py generate --target 500 --proxy socks5://127.0.0.1:1080
    python indian_hyperbole_generator.py stats
    python indian_hyperbole_generator.py export --output cultural_robustness.jsonl
"""

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any

import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================


def _load_api_keys_from_env() -> list[str]:
    """Load OpenRouter API keys from environment variables."""
    keys = []
    for i in range(1, 7):  # 6 keys
        key = os.environ.get(f"OPENROUTER_API_KEY_{i}", "")
        if key and key != f"your-api-key-{i}" and len(key) > 20:
            keys.append(key)
    return keys


# 6 API Keys for parallel generation (loaded from .env)
OPENROUTER_API_KEYS = _load_api_keys_from_env()

# Fallback to hardcoded keys if .env not configured (will be removed later)
if not OPENROUTER_API_KEYS:
    logger.warning("No API keys found in .env, using fallback keys")
    OPENROUTER_API_KEYS = [
        os.environ.get("OPENROUTER_API_KEY", ""),  # Single key fallback
    ]
    OPENROUTER_API_KEYS = [k for k in OPENROUTER_API_KEYS if k]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast:free"

# Rate limiting per key
REQUESTS_PER_MINUTE_PER_KEY = 10
REQUESTS_PER_DAY_PER_KEY = 900
DELAY_BETWEEN_REQUESTS = 6.0  # seconds

# Output settings
DATA_DIR = Path("D:/Modeling_studio/data/curated/safety_familyos/cultural_robustness")
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 1000  # Smaller shards for this dataset

SAMPLES_PER_REQUEST = 100  # Safety samples per API call (smaller for quality)

# =============================================================================
# 4-Band Safety Schema (safety_familyos)
# =============================================================================

SAFETY_LABELS = {
    "GREEN": {
        "id": 0,
        "description": "Safe, routine content - process normally",
        "for_hyperbole": "Expressions that are clearly hyperbole/exaggeration, no real concern",
    },
    "AMBER": {
        "id": 1,
        "description": "Needs attention, mild concern - flag for review",
        "for_hyperbole": "Stress expressions that warrant monitoring but not alarm",
    },
    "RED": {
        "id": 2,
        "description": "Serious concern - escalate, careful response",
        "for_hyperbole": "Should NOT be used for hyperbole - reserved for real concerns",
    },
    "CRISIS": {
        "id": 3,
        "description": "Immediate intervention needed - emergency protocol",
        "for_hyperbole": "NEVER use for hyperbole - only literal self-harm/suicide",
    },
}

LABEL_NAMES = list(SAFETY_LABELS.keys())

# =============================================================================
# Indian English Hyperbole Patterns (ENGLISH ONLY - NO HINDI/HINGLISH)
# =============================================================================

# Hyperbole patterns that should be GREEN (harmless exaggeration)
# These are INDIAN ENGLISH expressions - English spoken by Indians with cultural context
HYPERBOLE_GREEN = [
    # Death/Dying expressions (figurative)
    "I'll die of embarrassment",
    "I could just die",
    "dying of laughter",
    "dying of hunger",
    "I'll die if he sees me like this",
    "almost died laughing",
    "die of shame",
    "I would die for this biryani",
    "so tired I could die",
    "die of boredom",
    "I nearly died when I saw the bill",
    "this heat will be the death of me",
    "I'd rather die than face my mother-in-law now",
    # Heart attack expressions (figurative)
    "almost had a heart attack seeing the price",
    "this electricity bill gave me a heart attack",
    "I'll have a heart attack if the maid doesn't come tomorrow",
    "nearly had a heart attack when I saw my exam results",
    # Killing expressions (figurative stress)
    "the heat is killing me",
    "this traffic is killing me",
    "the suspense is killing me",
    "these mosquitoes are killing me",
    "this humidity is killing me",
    "the noise from the construction is killing me",
    # Madness expressions
    "going mad with all this work",
    "the kids are driving me mad",
    "my mother-in-law is driving me crazy",
    "this job is making me insane",
    "I'm going crazy with wedding preparations",
    "the relatives are driving me up the wall",
    # Head bursting expressions
    "my head is bursting with this headache",
    "head is splitting from all this noise",
    "I'll burst if I have to attend one more meeting",
    # Dramatic expressions (Indian English style)
    "I'm done for if mummy finds out",
    "it's all over now, boss saw me late",
    "my life is ruined, I burnt the dal",
    "I'm finished if the guests arrive early",
    "everything is gone, I forgot the anniversary",
]

# Hyperbole patterns that should be AMBER (stress but not crisis)
HYPERBOLE_AMBER = [
    # Work stress
    "work is killing me slowly",
    "this deadline will be the death of me",
    "I can't take this anymore with all the pressure",
    "I'm going to lose my mind with this workload",
    "this project is sucking the life out of me",
    # Family stress
    "my in-laws will kill me if they find out",
    "mummy will kill me if she discovers the truth",
    "papa will have my head for this",
    "my mother-in-law will eat me alive",
    # Life stress
    "this tension is killing me slowly",
    "I'm dying of stress with all these responsibilities",
    "can't breathe with this pressure from all sides",
    "the EMIs are killing me every month",
    # Wedding/festival stress
    "wedding preparations will kill me before the actual wedding",
    "Diwali cleaning is killing me this year",
    "festival cooking for the whole family is going to be the end of me",
]

# Context patterns for realistic scenarios
FAMILY_CONTEXTS = [
    "Indian joint family with in-laws",
    "Nuclear family in metro city (Mumbai, Delhi, Bangalore)",
    "Middle-class Indian family",
    "South Indian family (Tamil, Telugu, Kannada)",
    "North Indian family (Punjabi, UP, Bihar)",
    "Bengali family",
    "Gujarati family",
    "Marathi family",
    "NRI family in USA/UK/Canada/Australia",
    "Newly married couple in Indian household",
    "Family with school-going children",
    "Family with teenage kids",
    "Family with elderly parents",
]

SCENARIOS = [
    "morning chaos getting kids ready for school",
    "cooking elaborate meal for guests",
    "dealing with wedding season stress",
    "festival preparations (Diwali, Holi, Navratri)",
    "PTA meetings and school drama",
    "in-law visit and house cleaning",
    "work from home with kids around",
    "relatives asking about marriage/kids",
    "traffic and commute stress",
    "monsoon and power cuts",
    "summer heat and water shortage",
    "exam season for kids",
    "family WhatsApp group drama",
    "shopping at sale (Amazon/Flipkart/local market)",
    "maid/cook didn't show up",
    "internet/electricity issues during WFH",
    "cooking when guests arrive unexpectedly",
    "husband not helping with housework",
    "dealing with noisy neighbors",
    "kids refusing to eat/study/sleep",
]

TEXT_STYLES = [
    "casual WhatsApp message to friend",
    "voice note to sister/bestie",
    "complaint to husband/spouse",
    "diary entry after long day",
    "family group chat message",
    "text to mom about daily struggles",
    "rant to colleague about work-life balance",
    "Instagram story caption",
    "tweet about daily life",
    "conversation with neighbor",
]

# =============================================================================
# System Prompt (THE KEY PROMPT FOR INDIAN HYPERBOLE)
# =============================================================================

SYSTEM_PROMPT = """You are an expert at generating INDIAN ENGLISH expressions for safety classification training.

## CRITICAL: ENGLISH ONLY - NO HINDI/HINGLISH

Generate text in ENGLISH ONLY. Indian English is English spoken by Indians - it uses English words but with Indian cultural context, expressions, and scenarios.

DO NOT use:
- Hindi words (no "yaar", "mummy", "papa", "bhai", "didi")
- Hinglish mixing (no "main", "toh", "hai", "ho gaya")
- Hindi script (no देवनागरी)
- Romanized Hindi

DO use:
- Standard English words
- Indian cultural contexts (joint family, in-laws, festivals, traffic, weather)
- Indian English expressions ("I'll die of embarrassment", "killing me", "going mad")
- References to Indian life (Diwali, wedding season, monsoon, maid not coming)

## YOUR CRITICAL TASK

Generate realistic INDIAN ENGLISH expressions that use HYPERBOLIC/FIGURATIVE language about death, dying, killing, madness - but are COMPLETELY HARMLESS everyday expressions.

The FamilyOS safety model must learn that these are NOT crisis indicators:
- "I'll die of embarrassment" → GREEN (not CRISIS)
- "This traffic is killing me" → GREEN (not CRISIS)
- "My mother-in-law will kill me" → GREEN (figurative, not literal)

## SAFETY BAND SCHEMA

| Band | ID | Use For |
|------|----|---------|
| GREEN | 0 | Harmless hyperbole, everyday exaggeration, figurative speech |
| AMBER | 1 | Stress expressions that need monitoring but aren't crisis |

⚠️ CRITICAL: For this dataset, ONLY use GREEN and AMBER labels!
- GREEN: ~70% of samples (completely harmless hyperbole)
- AMBER: ~30% of samples (stress but not crisis)
- NEVER use RED (2) or CRISIS (3) for hyperbole expressions

## INDIAN ENGLISH HYPERBOLE PATTERNS

### Death/Dying (GREEN - figurative, not literal):
- "I'll die of embarrassment if anyone sees this"
- "Almost died laughing at that joke"
- "I would die for this biryani"
- "So tired I could just die"
- "I'll die of boredom in this meeting"
- "This heat will be the death of me"

### Heart Attack (GREEN - exaggeration):
- "Almost had a heart attack seeing the electricity bill"
- "I'll have a heart attack if the maid doesn't show up"
- "Nearly had a heart attack when I saw my son's report card"

### Killing Me (GREEN/AMBER - stress expression):
- "This Mumbai traffic is killing me" (GREEN)
- "The summer heat is killing me" (GREEN)
- "Work is killing me slowly these days" (AMBER - ongoing stress)
- "These deadlines will be the death of me" (AMBER - work pressure)
- "The mosquitoes are killing me" (GREEN)

### Going Mad/Crazy (GREEN/AMBER):
- "The kids are driving me mad" (GREEN)
- "My mother-in-law is driving me crazy" (GREEN)
- "I'm going to lose my mind with this workload" (AMBER)
- "This job is making me insane" (AMBER)
- "Wedding preparations are driving everyone crazy" (GREEN)

### Head Bursting (GREEN):
- "My head is bursting with this headache"
- "Head is splitting from all this noise"
- "I'll burst if I have to attend one more wedding this month"

### Dramatic Expressions (GREEN):
- "I'm done for if my boss finds out"
- "It's all over, I burnt the dinner"
- "My life is ruined, I forgot our anniversary"
- "I'm finished if the in-laws arrive early"

## INDIAN CULTURAL CONTEXTS (Use these, but write in English)

- Joint family dynamics (mother-in-law, father-in-law, siblings)
- Festival preparations (Diwali cleaning, cooking for guests)
- Wedding season stress
- School exam pressure for children
- Traffic in Indian cities
- Summer heat and monsoon issues
- Maid/domestic help not showing up
- Power cuts and water shortage
- Relatives asking personal questions (marriage, kids, salary)
- Office work pressure
- EMI and financial stress

## OUTPUT FORMAT (STRICT JSONL)

Each line must be valid JSON:
{
  "text": "The expression in ENGLISH ONLY (10-100 words)",
  "label": 0 or 1,
  "label_name": "GREEN" or "AMBER",
  "hyperbole_type": "death" | "heart_attack" | "killing" | "madness" | "head_bursting" | "dramatic",
  "context": "work" | "family" | "daily_life" | "festival" | "traffic" | "weather" | "kids" | "in_laws" | "social"
}

## EXAMPLES (ALL IN ENGLISH)

✅ GOOD (GREEN - death hyperbole):
{"text": "I'll die of embarrassment if my mother-in-law sees my messy kitchen. She's coming over in an hour and I haven't even started cleaning!", "label": 0, "label_name": "GREEN", "hyperbole_type": "death", "context": "in_laws"}

✅ GOOD (GREEN - killing hyperbole):
{"text": "This Bangalore traffic is killing me. I've been stuck at the same signal for 20 minutes now. At this rate, I'll miss my daughter's school annual day.", "label": 0, "label_name": "GREEN", "hyperbole_type": "killing", "context": "traffic"}

✅ GOOD (AMBER - work stress):
{"text": "This project deadline will be the death of me. Haven't slept properly in three days, and my manager keeps adding more requirements.", "label": 1, "label_name": "AMBER", "hyperbole_type": "death", "context": "work"}

✅ GOOD (GREEN - madness):
{"text": "The kids are driving me absolutely mad during summer holidays. They've been fighting over the remote control since morning!", "label": 0, "label_name": "GREEN", "hyperbole_type": "madness", "context": "kids"}

✅ GOOD (GREEN - heart attack):
{"text": "Almost had a heart attack when I saw the wedding caterer's quote. Two lakhs just for food? In this economy?", "label": 0, "label_name": "GREEN", "hyperbole_type": "heart_attack", "context": "festival"}

❌ BAD (Contains Hindi/Hinglish - NEVER DO THIS):
{"text": "Yaar, this traffic is killing me, late ho jaungi...", ...}

❌ BAD (Using CRISIS for hyperbole - NEVER DO THIS):
{"text": "Traffic is killing me", "label": 3, ...}

## REQUIREMENTS

- ENGLISH ONLY - No Hindi, no Hinglish, no code-mixing
- All samples must contain hyperbolic/figurative language
- Realistic Indian scenarios (but described in English)
- Vary sentence length and tone
- Cover all hyperbole types
- 70% GREEN, 30% AMBER distribution

Now generate the requested samples. Output ONLY valid JSONL, no explanations:"""


def get_user_prompt(
    num_samples: int,
    focus_type: str,
    batch_id: int,
) -> str:
    """Generate diverse user prompts for variety."""
    context = FAMILY_CONTEXTS[batch_id % len(FAMILY_CONTEXTS)]
    scenario = SCENARIOS[batch_id % len(SCENARIOS)]
    style = TEXT_STYLES[batch_id % len(TEXT_STYLES)]

    # Rotate hyperbole patterns (now just strings, not tuples)
    green_pattern = HYPERBOLE_GREEN[batch_id % len(HYPERBOLE_GREEN)]
    amber_pattern = HYPERBOLE_AMBER[batch_id % len(HYPERBOLE_AMBER)]

    prompt = f"""Generate {num_samples} Indian English hyperbole samples for safety classification training.

⚠️ CRITICAL: ENGLISH ONLY - NO HINDI/HINGLISH ⚠️

**Focus hyperbole type**: {focus_type}
**Family context**: {context}
**Scenario**: {scenario}
**Text style**: {style}

**Pattern inspiration (use variations, not exact copies)**:
- GREEN example: "{green_pattern}"
- AMBER example: "{amber_pattern}"

**Requirements**:
- ENGLISH ONLY - No Hindi words, no Hinglish, no code-mixing
- ~70% GREEN (harmless hyperbole, exaggeration)
- ~30% AMBER (stress expressions)
- NEVER use RED or CRISIS labels!
- All samples must contain hyperbolic/figurative language
- Realistic Indian scenarios described in English
- Vary the expressions - don't repeat the same phrase

**Hyperbole types to include** (distribute across these):
- death: "I'll die of...", "will be the death of me"
- heart_attack: "almost had a heart attack"
- killing: "is killing me", "these deadlines are killing me"
- madness: "driving me mad", "losing my mind"
- head_bursting: "my head is bursting"
- dramatic: "I'm done for", "it's all over"

Output JSONL only (ENGLISH ONLY):"""

    return prompt


# =============================================================================
# Validation
# =============================================================================


def validate_hyperbole_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single hyperbole sample."""
    required_fields = ["text", "label"]

    for field in required_fields:
        if field not in sample:
            return False, f"Missing '{field}' field"

    text = sample["text"]
    label = sample["label"]

    # Text validation
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False, "Text too short (min 10 chars)"

    if len(text) > 500:
        return False, "Text too long (max 500 chars)"

    # Label validation - ONLY GREEN (0) or AMBER (1) allowed!
    if label not in [0, 1]:
        return False, f"Invalid label: {label}. Only 0 (GREEN) or 1 (AMBER) allowed for hyperbole"

    # Ensure label_name matches label
    label_name = sample.get("label_name", "")
    expected_name = "GREEN" if label == 0 else "AMBER"
    if label_name and label_name not in ["GREEN", "AMBER"]:
        sample["label_name"] = expected_name
    elif not label_name:
        sample["label_name"] = expected_name

    # Validate hyperbole_type
    valid_types = {
        "death",
        "heart_attack",
        "killing",
        "madness",
        "head_bursting",
        "dramatic",
        "stress",
    }
    if sample.get("hyperbole_type", "") not in valid_types:
        sample["hyperbole_type"] = "dramatic"  # Default

    # Remove language field - we only generate English now
    sample.pop("language", None)

    # Validate context
    valid_contexts = {
        "work",
        "family",
        "daily_life",
        "festival",
        "traffic",
        "weather",
        "kids",
        "in_laws",
        "social",
    }
    if sample.get("context", "") not in valid_contexts:
        sample["context"] = "daily_life"  # Default

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    """Compute hash for deduplication."""
    text = sample["text"].lower().strip()
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
            json_match = re.search(r'\{[^{}]*"text"[^{}]*"label"[^{}]*\}', line, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                sample = json.loads(json_str)
            else:
                sample = json.loads(line)

            is_valid, error = validate_hyperbole_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")

        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue

    return valid_samples


# =============================================================================
# OpenRouter Client (Per-Key Rate Limiting with Proxy Support)
# =============================================================================


class ProxyManager:
    """Manages proxy rotation for rate limit evasion."""

    def __init__(self, proxies: list[str] | None = None):
        self.proxies = proxies or []
        self.current_index = 0
        self.lock = threading.Lock()
        self.failed_proxies: set[str] = set()

    def add_proxy(self, proxy: str) -> None:
        with self.lock:
            if proxy not in self.proxies:
                self.proxies.append(proxy)
                logger.info(f"Added proxy: {proxy[:30]}...")

    def add_proxies_from_file(self, filepath: str) -> int:
        count = 0
        try:
            with open(filepath) as f:
                for line in f:
                    proxy = line.strip()
                    if proxy and not proxy.startswith("#"):
                        self.add_proxy(proxy)
                        count += 1
        except FileNotFoundError:
            logger.error(f"Proxy file not found: {filepath}")
        return count

    def get_next_proxy(self) -> str | None:
        with self.lock:
            if not self.proxies:
                return None

            attempts = 0
            while attempts < len(self.proxies):
                proxy = self.proxies[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.proxies)

                if proxy not in self.failed_proxies:
                    return proxy
                attempts += 1

            logger.warning("All proxies failed, resetting...")
            self.failed_proxies.clear()
            return self.proxies[0] if self.proxies else None

    def mark_failed(self, proxy: str) -> None:
        with self.lock:
            self.failed_proxies.add(proxy)
            logger.warning(f"Marked proxy as failed: {proxy[:30]}...")

    def has_proxies(self) -> bool:
        return len(self.proxies) > 0


_proxy_manager: ProxyManager | None = None


def get_proxy_manager() -> ProxyManager:
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager


class OpenRouterClient:
    """Client for OpenRouter API with per-key rate limiting and proxy support."""

    def __init__(
        self,
        api_key: str,
        key_id: int,
        base_url: str = OPENROUTER_BASE_URL,
        requests_per_minute: int = REQUESTS_PER_MINUTE_PER_KEY,
        requests_per_day: int = REQUESTS_PER_DAY_PER_KEY,
        proxy: str | None = None,
    ):
        self.api_key = api_key
        self.key_id = key_id
        self.base_url = base_url
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.current_proxy = proxy

        self.request_times: list[datetime] = []
        self.daily_count = 0
        self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)

        self.client = self._create_client()
        self.lock = threading.Lock()

    def _create_client(self) -> httpx.Client:
        if self.current_proxy:
            logger.info(f"[Key {self.key_id}] Using proxy: {self.current_proxy[:40]}...")
            return httpx.Client(timeout=180.0, proxy=self.current_proxy)
        return httpx.Client(timeout=180.0)

    def rotate_proxy(self) -> bool:
        proxy_mgr = get_proxy_manager()
        if not proxy_mgr.has_proxies():
            return False

        if self.current_proxy:
            proxy_mgr.mark_failed(self.current_proxy)

        new_proxy = proxy_mgr.get_next_proxy()
        if new_proxy:
            self.current_proxy = new_proxy
            self.client.close()
            self.client = self._create_client()
            self.daily_count = 0
            self.request_times.clear()
            logger.info(f"[Key {self.key_id}] Rotated to new proxy")
            return True
        return False

    def _wait_for_rate_limit(self) -> None:
        with self.lock:
            now = datetime.now()

            if now >= self.daily_reset:
                self.daily_count = 0
                self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                logger.info(f"[Key {self.key_id}] Daily rate limit reset")

            if self.daily_count >= self.requests_per_day:
                raise RuntimeError(f"Key {self.key_id}: Daily rate limit reached")

            minute_ago = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > minute_ago]

            if len(self.request_times) >= self.requests_per_minute:
                oldest = min(self.request_times)
                wait_seconds = (oldest + timedelta(minutes=1) - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"[Key {self.key_id}] Rate limiting: waiting {wait_seconds:.1f}s")
                    time.sleep(wait_seconds)

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.95,
        max_tokens: int = 16000,
        retry_with_proxy: bool = True,
    ) -> str:
        self._wait_for_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/familyos",
            "X-Title": "FamilyOS Indian Hyperbole Generator",
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

                with self.lock:
                    self.request_times.append(datetime.now())
                    self.daily_count += 1

                data = response.json()
                content = data["choices"][0]["message"]["content"]

                logger.info(
                    f"[Key {self.key_id}] API call successful (daily: {self.daily_count}/{self.requests_per_day})"
                )
                return content

            except httpx.HTTPStatusError as e:
                logger.error(f"[Key {self.key_id}] HTTP error: {e.response.status_code}")
                if e.response.status_code == 429:
                    if retry_with_proxy and self.rotate_proxy():
                        logger.info(
                            f"[Key {self.key_id}] Retrying with new proxy (attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    else:
                        logger.warning(
                            f"[Key {self.key_id}] Rate limited, no proxy available. Waiting 60s..."
                        )
                        time.sleep(60)
                raise
            except httpx.ProxyError as e:
                logger.error(f"[Key {self.key_id}] Proxy error: {e}")
                if self.rotate_proxy():
                    logger.info(
                        f"[Key {self.key_id}] Retrying with new proxy (attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                raise

        raise RuntimeError(f"[Key {self.key_id}] Max retries exceeded")

    def close(self):
        self.client.close()


# =============================================================================
# Thread-Safe Silver Data Manager
# =============================================================================


class SilverDataManager:
    """Thread-safe manager for sharded silver data storage."""

    def __init__(self, silver_dir: Path = SILVER_DIR, shard_size: int = SHARD_SIZE):
        self.silver_dir = silver_dir
        self.shard_size = shard_size
        self.silver_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()
        self._load_existing_hashes()

        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

        # Track label and hyperbole type coverage
        self.label_counts: dict[str, int] = defaultdict(int)
        self.type_counts: dict[str, int] = defaultdict(int)
        self._load_counts()

    def _load_existing_hashes(self) -> None:
        for shard_file in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.seen_hashes.add(compute_sample_hash(sample))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(self.seen_hashes)} existing hashes")

    def _load_counts(self) -> None:
        for shard_file in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        label_name = sample.get("label_name", "GREEN")
                        self.label_counts[label_name] += 1
                        hyperbole_type = sample.get("hyperbole_type", "dramatic")
                        self.type_counts[hyperbole_type] += 1
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Label counts: {dict(self.label_counts)}")
        logger.info(f"Type counts: {dict(self.type_counts)}")

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
        added = 0

        with self.lock:
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

                label_name = sample.get("label_name", "GREEN")
                self.label_counts[label_name] += 1
                hyperbole_type = sample.get("hyperbole_type", "dramatic")
                self.type_counts[hyperbole_type] += 1

                added += 1

        return added

    def get_total_samples(self) -> int:
        with self.lock:
            return len(self.seen_hashes)

    def get_underrepresented_type(self) -> str:
        with self.lock:
            all_types = ["death", "heart_attack", "killing", "madness", "head_bursting", "dramatic"]
            sorted_types = sorted(all_types, key=lambda t: self.type_counts.get(t, 0))
            return sorted_types[0]

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            return {
                "total_samples": len(self.seen_hashes),
                "num_shards": len(list(self.silver_dir.glob("shard_*.jsonl"))),
                "label_counts": dict(self.label_counts),
                "type_counts": dict(self.type_counts),
            }


# =============================================================================
# Multi-Key Parallel Generator
# =============================================================================


class IndianHyperboleGeneratorAgent:
    """Multi-key parallel Indian hyperbole data generator."""

    def __init__(
        self,
        api_keys: list[str] = OPENROUTER_API_KEYS,
        samples_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
    ):
        self.api_keys = [k for k in api_keys if k and "REPLACE" not in k]
        if not self.api_keys:
            raise ValueError("No valid API keys provided!")

        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests

        self.clients = [
            OpenRouterClient(api_key=key, key_id=i) for i, key in enumerate(self.api_keys)
        ]

        self.silver_manager = SilverDataManager()
        self.batch_counter = 0
        self.batch_lock = threading.Lock()

        logger.info(f"Initialized with {len(self.clients)} API keys")

    def _get_next_batch_id(self) -> int:
        with self.batch_lock:
            batch_id = self.batch_counter
            self.batch_counter += 1
            return batch_id

    def _generate_batch(self, client: OpenRouterClient) -> int:
        batch_id = self._get_next_batch_id()

        focus_type = self.silver_manager.get_underrepresented_type()

        user_prompt = get_user_prompt(
            num_samples=self.samples_per_request,
            focus_type=focus_type,
            batch_id=batch_id,
        )

        try:
            response = client.generate(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)

            logger.info(
                f"[Key {client.key_id}] Generated {len(samples)} valid, added {added} new. "
                f"Focus: {focus_type}. Total: {self.silver_manager.get_total_samples()}"
            )

            return added

        except Exception as e:
            logger.error(f"[Key {client.key_id}] Batch failed: {e}")
            return 0

    def _worker(
        self,
        client: OpenRouterClient,
        stop_event: threading.Event,
        stats_queue: Queue,
    ) -> None:
        while not stop_event.is_set():
            try:
                added = self._generate_batch(client)
                stats_queue.put({"added": added, "requests": 1, "errors": 0 if added > 0 else 1})
                time.sleep(self.delay_between_requests)
            except RuntimeError as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"[Key {client.key_id}] Daily limit reached, stopping worker")
                    break
                raise
            except Exception as e:
                logger.error(f"[Key {client.key_id}] Worker error: {e}")
                stats_queue.put({"added": 0, "requests": 1, "errors": 1})
                time.sleep(self.delay_between_requests)

    def run(
        self,
        target_samples: int | None = None,
        max_requests: int | None = None,
        run_time_minutes: int | None = None,
    ) -> dict[str, Any]:
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=run_time_minutes) if run_time_minutes else None

        existing_count = self.silver_manager.get_total_samples()

        stats = {
            "start_time": start_time.isoformat(),
            "num_keys": len(self.clients),
            "existing_samples": existing_count,
            "new_samples": 0,
            "requests": 0,
            "errors": 0,
        }

        logger.info(f"Starting parallel generation with {len(self.clients)} keys")
        logger.info(f"Existing samples: {existing_count}")

        stop_event = threading.Event()
        stats_queue: Queue = Queue()

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [
                executor.submit(self._worker, client, stop_event, stats_queue)
                for client in self.clients
            ]

            try:
                while True:
                    if target_samples and stats["new_samples"] >= target_samples:
                        logger.info(f"Reached target: {target_samples} samples")
                        break

                    if max_requests and stats["requests"] >= max_requests:
                        logger.info(f"Reached max requests: {max_requests}")
                        break

                    if end_time and datetime.now() >= end_time:
                        logger.info(f"Reached time limit: {run_time_minutes} minutes")
                        break

                    while not stats_queue.empty():
                        batch_stats = stats_queue.get_nowait()
                        stats["new_samples"] += batch_stats["added"]
                        stats["requests"] += batch_stats["requests"]
                        stats["errors"] += batch_stats["errors"]

                    if all(f.done() for f in futures):
                        logger.info("All workers finished")
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                stop_event.set()

        for client in self.clients:
            client.close()

        stats["end_time"] = datetime.now().isoformat()
        stats["total_samples"] = self.silver_manager.get_total_samples()
        stats["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60
        stats["samples_per_minute"] = stats["new_samples"] / max(stats["duration_minutes"], 0.1)
        stats["coverage"] = self.silver_manager.get_stats()

        logger.info(f"\n{'='*60}")
        logger.info("Generation complete!")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"New samples: {stats['new_samples']}")
        logger.info(f"Duration: {stats['duration_minutes']:.1f} minutes")
        logger.info(f"Rate: {stats['samples_per_minute']:.1f} samples/minute")
        logger.info(f"{'='*60}")

        return stats


# =============================================================================
# Export Functions
# =============================================================================


def export_for_training(
    output_file: Path,
    max_samples: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> int:
    """Export silver data to training format for safety_familyos."""
    if shuffle:
        random.seed(seed)

    samples = []

    for shard_path in sorted(SILVER_DIR.glob("shard_*.jsonl")):
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Convert to safety_familyos training format
                    train_sample = {
                        "text": sample["text"],
                        "label": sample["label"],  # 0=GREEN, 1=AMBER
                        "subcategories": [sample.get("hyperbole_type", "dramatic")],
                    }
                    samples.append(train_sample)

                    if max_samples and len(samples) >= max_samples:
                        break
                except json.JSONDecodeError:
                    continue

        if max_samples and len(samples) >= max_samples:
            break

    if shuffle:
        random.shuffle(samples)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(samples)} samples to {output_file}")
    return len(samples)


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Indian Hyperbole Data Generator for Cultural Robustness"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate hyperbole data")
    gen_parser.add_argument("--target", type=int, default=500, help="Target samples (default: 500)")
    gen_parser.add_argument("--max-requests", type=int, default=None, help="Max requests")
    gen_parser.add_argument("--run-time", type=int, default=None, help="Run time (minutes)")
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=50, help="Samples per API call"
    )
    gen_parser.add_argument("--delay", type=float, default=6.0, help="Delay between requests")
    gen_parser.add_argument(
        "--keys", type=str, nargs="+", default=None, help="API keys (space-separated)"
    )
    gen_parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Single proxy URL",
    )
    gen_parser.add_argument(
        "--proxy-file",
        type=str,
        default=None,
        help="File with proxy URLs",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to training format")
    export_parser.add_argument("--output", type=str, required=True, help="Output file")
    export_parser.add_argument("--max", type=int, default=None, help="Max samples")
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "generate":
        keys = args.keys or OPENROUTER_API_KEYS

        valid_keys = [k for k in keys if k and "REPLACE" not in k and len(k) > 20]

        if not valid_keys:
            print("ERROR: No valid API keys provided!")
            print("Set OPENROUTER_API_KEY_1, etc. in .env, or use --keys key1 key2")
            return

        proxy_mgr = get_proxy_manager()

        if args.proxy:
            proxy_mgr.add_proxy(args.proxy)
            print(f"Using single proxy: {args.proxy[:40]}...")

        if args.proxy_file:
            count = proxy_mgr.add_proxies_from_file(args.proxy_file)
            print(f"Loaded {count} proxies from {args.proxy_file}")

        print(f"Using {len(valid_keys)} API keys")
        print(f"Target: {args.target} samples")
        print(f"Output: {SILVER_DIR}")

        agent = IndianHyperboleGeneratorAgent(
            api_keys=valid_keys,
            samples_per_request=args.samples_per_request,
            delay_between_requests=args.delay,
        )

        stats = agent.run(
            target_samples=args.target,
            max_requests=args.max_requests,
            run_time_minutes=args.run_time,
        )

        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "stats":
        manager = SilverDataManager()
        stats = manager.get_stats()

        print("\n=== Indian Hyperbole Data Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of shards: {stats['num_shards']}")
        print("\nLabel distribution:")
        for label, count in stats["label_counts"].items():
            pct = count * 100 / max(stats["total_samples"], 1)
            print(f"  {label}: {count} ({pct:.1f}%)")
        print("\nHyperbole type distribution:")
        for type_name, count in sorted(stats["type_counts"].items(), key=lambda x: -x[1]):
            print(f"  {type_name}: {count}")

    elif args.command == "export":
        output = Path(args.output)
        count = export_for_training(output, max_samples=args.max, seed=args.seed)
        print(f"Exported {count} samples to {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
