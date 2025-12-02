"""
Emotion Data Generator Agent (Multi-Key Parallel)

Uses OpenRouter API with multiple accounts to generate synthetic emotion training data
for the FamilyOS Emotions dataset at scale.

Features:
- 6 API keys for parallel generation (6x throughput)
- Thread-safe shared storage with deduplication
- Balanced emotion coverage across all 44 classes
- Indian + Western family contexts
- Multi-label emotion annotation
- Proxy/VPN support for IP rotation
- Automatic proxy rotation on rate limits

Target: 200,000 samples (~6 hours with 6 keys)

Usage:
    python emotion_data_generator.py generate --target 200000
    python emotion_data_generator.py generate --target 200000 --proxy socks5://127.0.0.1:1080
    python emotion_data_generator.py generate --target 200000 --proxy-file proxies.txt
    python emotion_data_generator.py stats
    python emotion_data_generator.py export --output train.jsonl
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
DATA_DIR = Path("D:/Modeling_studio/data/familyos/emotions")
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
SHARD_SIZE = 10000  # Samples per shard

SAMPLES_PER_REQUEST = 150  # Emotion samples per API call

# =============================================================================
# 44 Emotion Schema (from README.md)
# =============================================================================

EMOTIONS = {
    # Core Emotions (8)
    "neutral": {"id": 0, "category": "core", "description": "No strong emotion"},
    "joy": {"id": 1, "category": "core", "description": "Happiness, delight"},
    "sadness": {"id": 2, "category": "core", "description": "Sorrow, unhappiness"},
    "anger": {"id": 3, "category": "core", "description": "Strong displeasure"},
    "fear": {"id": 4, "category": "core", "description": "Anxiety, worry"},
    "surprise": {"id": 5, "category": "core", "description": "Unexpected reaction"},
    "love": {"id": 6, "category": "core", "description": "Deep affection"},
    "disgust": {"id": 7, "category": "core", "description": "Strong aversion"},
    # Positive Emotions (12)
    "admiration": {"id": 8, "category": "positive", "description": "Respect, appreciation"},
    "amusement": {"id": 9, "category": "positive", "description": "Finding something funny"},
    "approval": {"id": 10, "category": "positive", "description": "Agreeing, endorsing"},
    "caring": {"id": 11, "category": "positive", "description": "Showing concern"},
    "excitement": {"id": 12, "category": "positive", "description": "Eager anticipation"},
    "gratitude": {"id": 13, "category": "positive", "description": "Thankfulness"},
    "optimism": {"id": 14, "category": "positive", "description": "Hopeful outlook"},
    "pride": {"id": 15, "category": "positive", "description": "Satisfaction in achievement"},
    "relief": {"id": 16, "category": "positive", "description": "Ease after worry"},
    "contentment": {"id": 17, "category": "positive", "description": "Peaceful satisfaction"},
    "hope": {"id": 18, "category": "positive", "description": "Wish for positive outcome"},
    "tenderness": {"id": 19, "category": "positive", "description": "Gentle affection"},
    # Negative Emotions (10)
    "annoyance": {"id": 20, "category": "negative", "description": "Mild irritation"},
    "disappointment": {"id": 21, "category": "negative", "description": "Unmet expectations"},
    "disapproval": {"id": 22, "category": "negative", "description": "Disagreement"},
    "embarrassment": {"id": 23, "category": "negative", "description": "Self-consciousness"},
    "grief": {"id": 24, "category": "negative", "description": "Deep sorrow from loss"},
    "nervousness": {"id": 25, "category": "negative", "description": "Anxious anticipation"},
    "remorse": {"id": 26, "category": "negative", "description": "Regret, guilt"},
    "frustration": {"id": 27, "category": "negative", "description": "Blocked goals"},
    "overwhelmed": {"id": 28, "category": "negative", "description": "Too much to handle"},
    "emptiness": {"id": 29, "category": "negative", "description": "Feeling void"},
    # Family-Specific Emotions (14)
    "nostalgia": {"id": 30, "category": "family", "description": "Fond memories of past"},
    "protectiveness": {"id": 31, "category": "family", "description": "Urge to keep safe"},
    "togetherness": {"id": 32, "category": "family", "description": "Family bonding"},
    "longing": {"id": 33, "category": "family", "description": "Missing someone/something"},
    "warmth": {"id": 34, "category": "family", "description": "Comfortable affection"},
    "playfulness": {"id": 35, "category": "family", "description": "Lighthearted fun"},
    "celebration": {"id": 36, "category": "family", "description": "Marking achievements"},
    "belonging": {"id": 37, "category": "family", "description": "Feeling part of family"},
    "parental_pride": {"id": 38, "category": "family", "description": "Pride in children"},
    "parental_guilt": {"id": 39, "category": "family", "description": "Guilt about parenting"},
    "patience": {"id": 40, "category": "family", "description": "Calm endurance"},
    "worry": {"id": 41, "category": "family", "description": "Concern for loved ones"},
    "bittersweet": {"id": 42, "category": "family", "description": "Mixed happy-sad"},
    "homesickness": {"id": 43, "category": "family", "description": "Missing home/family"},
}

EMOTION_NAMES = list(EMOTIONS.keys())
EMOTION_BY_CATEGORY = {
    "core": [e for e, v in EMOTIONS.items() if v["category"] == "core"],
    "positive": [e for e, v in EMOTIONS.items() if v["category"] == "positive"],
    "negative": [e for e, v in EMOTIONS.items() if v["category"] == "negative"],
    "family": [e for e, v in EMOTIONS.items() if v["category"] == "family"],
}

# =============================================================================
# Diverse Prompt Templates (Inspired by NER Generator)
# =============================================================================

FAMILY_CONTEXTS = [
    "a Western nuclear family (mom, dad, two kids Emma and Jack)",
    "an Indian joint family (parents, grandparents, uncle, aunt, cousins living together)",
    "a mixed Western-Indian family celebrating both cultures",
    "a single-parent family with grandparents helping raise the kids",
    "a family with multiple generations under one roof",
    "a newly married couple (wife and husband Mike) adjusting to family life",
    "parents with teenagers navigating adolescence",
    "parents with young children (toddlers, babies)",
    "empty nesters whose children have moved out",
    "a family dealing with a long-distance relationship (spouse Mike working abroad)",
    "a family caring for elderly parents",
    "siblings who are very close despite living apart",
    "a couple (wife and husband Mike) dealing with parenting challenges together",
    "a family recovering from loss of a loved one (grandparent, parent)",
    "a family preparing for or recovering from a funeral",
    "parents dealing with the grief of a miscarriage or child loss",
]

SCENARIOS = [
    "morning routines and getting ready for the day",
    "dinner time conversations and family meals",
    "weekend family activities and outings",
    "birthday celebrations and party planning",
    "holiday gatherings (Diwali, Christmas, Eid, Thanksgiving)",
    "bedtime routines and tucking kids in",
    "school-related moments (homework, report cards, PTM)",
    "family game nights and movie nights",
    "visiting grandparents or relatives",
    "dealing with a family member's illness or health scare",
    "celebrating achievements (graduation, promotion, first steps)",
    "family vacations and travel memories",
    "arguments and reconciliations between family members",
    "video calls with distant family members",
    "cooking and baking together",
    "helping children through difficult times",
    "reminiscing about old times and looking at photos",
    "welcoming a new baby or pet to the family",
    "moving to a new house or city",
    "planning for the future (college, retirement, weddings)",
    # LOSS/GRIEF scenarios (underrepresented - need 10x more)
    "the days after losing a grandparent",
    "grieving the loss of a parent",
    "coping with the death of a beloved pet",
    "anniversary of a loved one's passing",
    "going through belongings of someone who passed away",
    "first holiday season after a family loss",
    "supporting a spouse through grief",
    "explaining death to young children",
    "visiting a grave or memorial",
    "remembering a lost family member on their birthday",
]

TEXT_STYLES = [
    "casual diary entry (personal, reflective)",
    "text message to family (short, informal, with emojis implied)",
    "voice note transcript (conversational, natural pauses)",
    "photo caption for a family moment",
    "calendar reminder or note to self",
    "letter or message to a family member",
    "social media post about family",
    "journal reflection at end of day",
    "conversation snippet between family members",
    "internal monologue/thoughts about family",
]

# Normalized intensity levels (analysis showed 'moderate' vs 'medium' inconsistency)
INTENSITY_LEVELS = ["low", "medium", "high"]

# Co-occurrence patterns based on ACTUAL DATA ANALYSIS (top pairs from 88K samples)
# These are the real patterns found in the dataset
EMOTION_COOCCURRENCES = [
    # TOP PAIRS FROM ANALYSIS (count > 2000)
    ("joy", "togetherness"),  # 3,112 occurrences
    ("excitement", "joy"),  # 3,068 occurrences
    ("celebration", "joy"),  # 3,006 occurrences
    ("joy", "pride"),  # 2,874 occurrences
    ("joy", "playfulness"),  # 2,780 occurrences
    ("joy", "warmth"),  # 2,695 occurrences
    ("fear", "worry"),  # 2,635 occurrences
    ("longing", "sadness"),  # 2,519 occurrences
    ("annoyance", "frustration"),  # 2,467 occurrences
    ("amusement", "joy"),  # 2,381 occurrences
    ("bittersweet", "nostalgia"),  # 2,365 occurrences
    ("love", "tenderness"),  # 2,338 occurrences
    ("frustration", "overwhelmed"),  # 2,236 occurrences
    ("joy", "love"),  # 2,217 occurrences
    ("joy", "parental_pride"),  # 2,203 occurrences
    ("nervousness", "worry"),  # 2,151 occurrences
    ("longing", "nostalgia"),  # 2,120 occurrences
    ("homesickness", "longing"),  # 2,116 occurrences
    ("gratitude", "joy"),  # 2,088 occurrences
    ("togetherness", "warmth"),  # 2,070 occurrences
    # EXTENDED PATTERNS (3-4 emotions for richer annotations)
    ("joy", "pride", "parental_pride", "love"),  # Proud parent moment
    ("grief", "sadness", "emptiness", "longing"),  # Loss/bereavement
    ("joy", "excitement", "celebration"),  # Milestone/party
    ("worry", "fear", "nervousness", "protectiveness"),  # Health concern
    ("frustration", "annoyance", "overwhelmed", "patience"),  # Parenting stress
    ("nostalgia", "bittersweet", "longing", "warmth"),  # Fond memories
    ("love", "tenderness", "warmth", "caring"),  # Intimate moment
    ("relief", "gratitude", "joy"),  # Good news after worry
    ("homesickness", "longing", "sadness", "nostalgia"),  # Missing family
    ("togetherness", "warmth", "belonging", "joy"),  # Family gathering
]

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert emotion annotation specialist for FamilyOS, a family-focused AI assistant.
Your task is to generate realistic family diary entries, messages, and reflections with accurate multi-label emotion annotations.

## THE 44 EMOTION SCHEMA (USE EXACTLY THESE)

### Core Emotions (8)
neutral, joy, sadness, anger, fear, surprise, love, disgust

### Positive Emotions (12)
admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness

### Negative Emotions (10)
annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness

### Family-Specific Emotions (14)
nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

## ANNOTATION RULES (CRITICAL)

1. **Multi-label is the norm**: Most texts express 2-4 emotions simultaneously
   - "Emma took her first steps!" → ["joy", "pride", "parental_pride", "excitement"]
   - NOT just ["joy"]

2. **Primary emotion**: The dominant feeling that drives the text

3. **Intensity matters**:
   - "I'm a bit annoyed" → low intensity annoyance
   - "I'm absolutely furious!" → high intensity anger

4. **Context is key**: Same words, different emotions based on context
   - "I can't believe it" + good news → surprise, joy
   - "I can't believe it" + bad news → surprise, disappointment

5. **Family-specific emotions are important**:
   - Use nostalgia, togetherness, parental_pride, bittersweet, homesickness liberally
   - These are what make FamilyOS unique

6. **Cultural authenticity**:
   - Include Indian family dynamics: joint family, festivals, kinship terms
   - Use: mummy, papa, didi, bhai, nana, nani, dada, dadi, chacha, chachi, maasi, etc.
   - Include: Diwali, Holi, Rakhi, Karwa Chauth, etc.

## REALISTIC CO-OCCURRENCES (USE THESE PATTERNS)

- Proud parent: joy + pride + parental_pride + love
- Missing family: sadness + nostalgia + longing + homesickness
- Kids growing up: bittersweet + nostalgia + pride + sadness
- Family gathering: togetherness + warmth + joy + belonging
- Health worry: worry + fear + nervousness + love
- Parenting stress: overwhelmed + frustration + patience + love
- Good news: relief + joy + gratitude + excitement
- Loss/grief: grief + sadness + emptiness + longing

## OUTPUT FORMAT (STRICT JSONL)

Each line must be valid JSON with these fields:
{
  "text": "The actual diary entry or message (10-100 words)",
  "emotions": ["emotion1", "emotion2", "emotion3"],  // 1-5 emotions, most have 2-4
  "primary_emotion": "emotion1",  // The dominant one
  "intensity": "low" | "medium" | "high",
  "context": "milestone" | "daily_life" | "health" | "conflict" | "celebration" | "memory" | "loss" | "relationship" | "parenting" | "self_reflection"
}

## QUALITY REQUIREMENTS

✅ GOOD (Multi-label, nuanced, realistic):
{"text": "Mummy called today. Hearing her voice made me so happy but also made me miss home terribly. The kids were fighting in the background and I could hear Papa scolding them - reminded me of my childhood.", "emotions": ["joy", "homesickness", "nostalgia", "longing", "warmth"], "primary_emotion": "homesickness", "intensity": "high", "context": "relationship"}

✅ GOOD (Family-specific emotion):
{"text": "Watched Emma sleep tonight. She's growing up so fast. Soon she won't want bedtime stories anymore.", "emotions": ["love", "bittersweet", "tenderness", "nostalgia"], "primary_emotion": "bittersweet", "intensity": "medium", "context": "parenting"}

✅ GOOD (Spouse/partner mention - NEED MORE OF THESE):
{"text": "Mike stayed up late with me while I worried about Papa's surgery tomorrow. He just held my hand and said nothing. That's all I needed.", "emotions": ["worry", "love", "gratitude", "fear", "warmth"], "primary_emotion": "worry", "intensity": "high", "context": "health"}

✅ GOOD (Loss/grief context - NEED MORE OF THESE):
{"text": "Found Dadi's recipe book while cleaning today. Her handwriting on the margins, notes about how Dada liked extra ghee. Cried for an hour.", "emotions": ["grief", "nostalgia", "longing", "sadness", "love"], "primary_emotion": "grief", "intensity": "high", "context": "loss"}

✅ GOOD (Loss/grief with hope):
{"text": "First Diwali without Nani. We lit her favorite diya and told the kids stories about her. Bittersweet, but she would have loved seeing them in new clothes.", "emotions": ["grief", "bittersweet", "nostalgia", "love", "warmth"], "primary_emotion": "bittersweet", "intensity": "medium", "context": "loss"}

❌ BAD (Too simple, single emotion):
{"text": "Happy today", "emotions": ["joy"], "primary_emotion": "joy", "intensity": "medium", "context": "daily_life"}

❌ BAD (Emotions don't match text):
{"text": "The funeral was today", "emotions": ["joy", "excitement"], ...}

## DIVERSITY REQUIREMENTS

Across your samples:
- Cover ALL 44 emotions (especially family-specific ones)
- Mix of Indian and Western family contexts
- Include spouse/partner (Mike/husband/wife) in ~10% of samples
- Include loss/grief contexts in ~5-10% of samples
- Vary text length (short notes to longer reflections)
- Include different intensity levels (low/medium/high ONLY)
- Use all 10 context categories
- Mix of positive, negative, and mixed emotional states

Now generate the requested samples. Output ONLY valid JSONL, no explanations:"""


def get_user_prompt(
    num_samples: int,
    focus_emotions: list[str],
    focus_category: str,
    batch_id: int,
) -> str:
    """Generate diverse user prompts for variety."""
    context = FAMILY_CONTEXTS[batch_id % len(FAMILY_CONTEXTS)]
    scenario = SCENARIOS[batch_id % len(SCENARIOS)]
    style = TEXT_STYLES[batch_id % len(TEXT_STYLES)]
    intensity = INTENSITY_LEVELS[batch_id % len(INTENSITY_LEVELS)]

    # Get a co-occurrence pattern to encourage
    cooccur = EMOTION_COOCCURRENCES[batch_id % len(EMOTION_COOCCURRENCES)]

    prompt = f"""Generate {num_samples} emotion-annotated samples.

**Context**: {context}
**Scenario**: {scenario}
**Text Style**: {style}
**Intensity Focus**: {intensity}

**Priority emotions to include** (ensure good coverage):
{', '.join(focus_emotions)}

**Encourage this emotion combination** (natural co-occurrence):
{' + '.join(cooccur)}

**Category focus**: {focus_category} emotions
- Core: neutral, joy, sadness, anger, fear, surprise, love, disgust
- Positive: admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness
- Negative: annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness
- Family: nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

**Requirements**:
- Most samples should have 2-4 emotions (multi-label)
- ~30% should include Indian family context (didi, bhai, Diwali, etc.)
- ~10% should include spouse mentions (husband Mike, wife)
- ~5% should be loss/grief contexts (bereavement, mourning, missing deceased)
- Vary text length from short notes to longer reflections
- Make emotions match the text realistically
- Use ONLY these intensity values: "low", "medium", "high"
- Use ONLY these context values: "milestone", "daily_life", "health", "conflict", "celebration", "memory", "loss", "relationship", "parenting", "self_reflection"

Output JSONL only:"""

    return prompt


# =============================================================================
# Validation
# =============================================================================


def validate_emotion_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single emotion sample."""
    required_fields = ["text", "emotions", "primary_emotion"]

    for field in required_fields:
        if field not in sample:
            return False, f"Missing '{field}' field"

    text = sample["text"]
    emotions = sample["emotions"]
    primary = sample["primary_emotion"]

    # Text validation
    if not isinstance(text, str) or len(text.strip()) < 10:
        return False, "Text too short (min 10 chars)"

    if len(text) > 1000:
        return False, "Text too long (max 1000 chars)"

    # Emotions validation
    if not isinstance(emotions, list) or len(emotions) == 0:
        return False, "Emotions must be a non-empty list"

    if len(emotions) > 6:
        return False, "Too many emotions (max 6)"

    # Check all emotions are valid
    for emotion in emotions:
        if emotion not in EMOTION_NAMES:
            return False, f"Invalid emotion: {emotion}"

    # Primary must be in emotions list
    if primary not in emotions:
        return False, f"Primary emotion '{primary}' not in emotions list"

    # Check primary is valid
    if primary not in EMOTION_NAMES:
        return False, f"Invalid primary emotion: {primary}"

    # Validate and normalize intensity (analysis showed 'moderate' inconsistency)
    valid_intensities = {"low", "medium", "high"}
    intensity = sample.get("intensity", "medium")
    if intensity not in valid_intensities:
        # Auto-fix common issues
        if intensity in {"moderate", "mid"}:
            sample["intensity"] = "medium"
        elif intensity in {"subtle", "mild", "slight"}:
            sample["intensity"] = "low"
        elif intensity in {"strong", "intense", "extreme"}:
            sample["intensity"] = "high"
        else:
            sample["intensity"] = "medium"  # Default fallback

    # Validate and normalize context (consolidate 35+ micro-contexts to 10 core)
    valid_contexts = {
        "milestone",
        "daily_life",
        "health",
        "conflict",
        "celebration",
        "memory",
        "loss",
        "relationship",
        "parenting",
        "self_reflection",
    }
    context = sample.get("context", "daily_life")
    if context not in valid_contexts:
        # Map micro-contexts to core contexts
        context_map = {
            "family": "relationship",
            "togetherness": "relationship",
            "planning": "daily_life",
            "reconciliation": "conflict",
            "travel": "daily_life",
            "outing": "daily_life",
            "cooking": "daily_life",
            "playfulness": "parenting",
            "party_planning": "celebration",
            "fear": "health",
            "sibling": "relationship",
            "adventure": "daily_life",
            "family_gathering": "celebration",
            "nostalgia": "memory",
            "creative": "daily_life",
            "amusement": "daily_life",
            "parental_pride": "parenting",
            "surprise": "celebration",
            "identity": "self_reflection",
            "nightmare": "health",
            "school": "parenting",
            "security": "self_reflection",
            "grief": "loss",
            "bereavement": "loss",
            "funeral": "loss",
            "mourning": "loss",
        }
        sample["context"] = context_map.get(context, "daily_life")

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    """Compute hash for deduplication."""
    text = sample["text"].lower().strip()
    return hashlib.md5(text.encode()).hexdigest()


def get_emotion_coverage(sample: dict[str, Any]) -> dict[str, int]:
    """Count emotions in a sample."""
    counts: dict[str, int] = defaultdict(int)
    for emotion in sample.get("emotions", []):
        counts[emotion] += 1
    return dict(counts)


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
            json_match = re.search(r'\{[^{}]*"text"[^{}]*"emotions"[^{}]*\}', line, re.DOTALL)
            if json_match:
                # Handle nested quotes in text
                json_str = json_match.group()
                sample = json.loads(json_str)
            else:
                sample = json.loads(line)

            is_valid, error = validate_emotion_sample(sample)
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


# Global proxy manager for IP rotation
class ProxyManager:
    """Manages proxy rotation for rate limit evasion."""

    def __init__(self, proxies: list[str] | None = None):
        self.proxies = proxies or []
        self.current_index = 0
        self.lock = threading.Lock()
        self.failed_proxies: set[str] = set()

    def add_proxy(self, proxy: str) -> None:
        """Add a proxy to the pool."""
        with self.lock:
            if proxy not in self.proxies:
                self.proxies.append(proxy)
                logger.info(f"Added proxy: {proxy[:30]}...")

    def add_proxies_from_file(self, filepath: str) -> int:
        """Load proxies from a file (one per line)."""
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
        """Get the next available proxy (round-robin)."""
        with self.lock:
            if not self.proxies:
                return None

            # Find next working proxy
            attempts = 0
            while attempts < len(self.proxies):
                proxy = self.proxies[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.proxies)

                if proxy not in self.failed_proxies:
                    return proxy
                attempts += 1

            # All proxies failed, reset and try again
            logger.warning("All proxies failed, resetting...")
            self.failed_proxies.clear()
            return self.proxies[0] if self.proxies else None

    def mark_failed(self, proxy: str) -> None:
        """Mark a proxy as failed."""
        with self.lock:
            self.failed_proxies.add(proxy)
            logger.warning(f"Marked proxy as failed: {proxy[:30]}...")

    def has_proxies(self) -> bool:
        return len(self.proxies) > 0


# Global proxy manager instance
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
        """Create HTTP client with optional proxy."""
        if self.current_proxy:
            logger.info(f"[Key {self.key_id}] Using proxy: {self.current_proxy[:40]}...")
            return httpx.Client(timeout=180.0, proxy=self.current_proxy)
        return httpx.Client(timeout=180.0)

    def rotate_proxy(self) -> bool:
        """Rotate to next available proxy. Returns True if successful."""
        proxy_mgr = get_proxy_manager()
        if not proxy_mgr.has_proxies():
            return False

        # Mark current proxy as failed if we have one
        if self.current_proxy:
            proxy_mgr.mark_failed(self.current_proxy)

        # Get next proxy
        new_proxy = proxy_mgr.get_next_proxy()
        if new_proxy:
            self.current_proxy = new_proxy
            self.client.close()
            self.client = self._create_client()
            # Reset rate limits for new IP
            self.daily_count = 0
            self.request_times.clear()
            logger.info(f"[Key {self.key_id}] Rotated to new proxy")
            return True
        return False

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = datetime.now()

            # Reset daily counter if needed
            if now >= self.daily_reset:
                self.daily_count = 0
                self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
                logger.info(f"[Key {self.key_id}] Daily rate limit reset")

            # Check daily limit
            if self.daily_count >= self.requests_per_day:
                raise RuntimeError(f"Key {self.key_id}: Daily rate limit reached")

            # Check per-minute limit
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
        max_tokens: int = 32000,
        retry_with_proxy: bool = True,
    ) -> str:
        """Generate text using OpenRouter API with automatic proxy rotation on 429."""
        self._wait_for_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/familyos",
            "X-Title": "FamilyOS Emotion Data Generator",
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
                    # Try rotating proxy first
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

        # Track emotion coverage
        self.emotion_counts: dict[str, int] = defaultdict(int)
        self._load_emotion_counts()

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

    def _load_emotion_counts(self) -> None:
        """Load emotion counts from existing data."""
        for shard_file in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        for emotion in sample.get("emotions", []):
                            self.emotion_counts[emotion] += 1
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded emotion counts: {dict(self.emotion_counts)}")

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
        """Thread-safe: Add samples to storage."""
        added = 0

        with self.lock:
            for sample in samples:
                sample_hash = compute_sample_hash(sample)
                if sample_hash in self.seen_hashes:
                    continue

                # Check shard size
                if self.current_shard_count >= self.shard_size:
                    self.current_shard_id += 1
                    self.current_shard_count = 0
                    logger.info(f"Started new shard: shard_{self.current_shard_id:04d}")

                # Write to shard
                shard_path = self._get_shard_path(self.current_shard_id)
                with open(shard_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                # Update tracking
                self.seen_hashes.add(sample_hash)
                self.current_shard_count += 1
                for emotion in sample.get("emotions", []):
                    self.emotion_counts[emotion] += 1
                added += 1

        return added

    def get_total_samples(self) -> int:
        with self.lock:
            return len(self.seen_hashes)

    def get_underrepresented_emotions(self, n: int = 5) -> list[str]:
        """Get the n most underrepresented emotions."""
        with self.lock:
            sorted_emotions = sorted(EMOTION_NAMES, key=lambda e: self.emotion_counts.get(e, 0))
            return sorted_emotions[:n]

    def get_underrepresented_category(self) -> str:
        """Get the category with lowest coverage."""
        with self.lock:
            category_totals = {}
            for cat, emotions in EMOTION_BY_CATEGORY.items():
                total = sum(self.emotion_counts.get(e, 0) for e in emotions)
                category_totals[cat] = total / len(emotions)  # Average per emotion
            return min(category_totals, key=category_totals.get)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the data."""
        with self.lock:
            return {
                "total_samples": len(self.seen_hashes),
                "num_shards": len(list(self.silver_dir.glob("shard_*.jsonl"))),
                "emotion_counts": dict(self.emotion_counts),
                "category_coverage": {
                    cat: sum(self.emotion_counts.get(e, 0) for e in emotions)
                    for cat, emotions in EMOTION_BY_CATEGORY.items()
                },
            }


# =============================================================================
# Multi-Key Parallel Generator
# =============================================================================


class EmotionDataGeneratorAgent:
    """Multi-key parallel emotion data generator."""

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

        # Create clients for each key
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
        """Generate a single batch using one client."""
        batch_id = self._get_next_batch_id()

        # Get underrepresented emotions for balanced coverage
        focus_emotions = self.silver_manager.get_underrepresented_emotions(5)
        focus_category = self.silver_manager.get_underrepresented_category()

        user_prompt = get_user_prompt(
            num_samples=self.samples_per_request,
            focus_emotions=focus_emotions,
            focus_category=focus_category,
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
                f"Focus: {focus_emotions[:3]}. Total: {self.silver_manager.get_total_samples()}"
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
        """Worker thread for one API key."""
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
        """Run parallel generation with all API keys."""
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

        # Start worker threads
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [
                executor.submit(self._worker, client, stop_event, stats_queue)
                for client in self.clients
            ]

            try:
                while True:
                    # Check stopping conditions
                    if target_samples and stats["new_samples"] >= target_samples:
                        logger.info(f"Reached target: {target_samples} samples")
                        break

                    if max_requests and stats["requests"] >= max_requests:
                        logger.info(f"Reached max requests: {max_requests}")
                        break

                    if end_time and datetime.now() >= end_time:
                        logger.info(f"Reached time limit: {run_time_minutes} minutes")
                        break

                    # Collect stats from queue
                    while not stats_queue.empty():
                        batch_stats = stats_queue.get_nowait()
                        stats["new_samples"] += batch_stats["added"]
                        stats["requests"] += batch_stats["requests"]
                        stats["errors"] += batch_stats["errors"]

                    # Check if all workers are done
                    if all(f.done() for f in futures):
                        logger.info("All workers finished")
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                stop_event.set()

        # Close clients
        for client in self.clients:
            client.close()

        # Final stats
        stats["end_time"] = datetime.now().isoformat()
        stats["total_samples"] = self.silver_manager.get_total_samples()
        stats["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60
        stats["samples_per_minute"] = stats["new_samples"] / max(stats["duration_minutes"], 0.1)
        stats["emotion_coverage"] = self.silver_manager.get_stats()["emotion_counts"]

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
    """Export silver data to training format."""
    if shuffle:
        random.seed(seed)

    samples = []

    for shard_path in sorted(SILVER_DIR.glob("shard_*.jsonl")):
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Convert to training format
                    train_sample = {
                        "text": sample["text"],
                        "labels": sample["emotions"],  # Multi-label
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

    parser = argparse.ArgumentParser(description="Multi-key Emotion Data Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate emotion data")
    gen_parser.add_argument("--target", type=int, default=None, help="Target samples")
    gen_parser.add_argument("--max-requests", type=int, default=None, help="Max requests")
    gen_parser.add_argument("--run-time", type=int, default=None, help="Run time (minutes)")
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=150, help="Samples per API call"
    )
    gen_parser.add_argument("--delay", type=float, default=6.0, help="Delay between requests")
    gen_parser.add_argument(
        "--keys", type=str, nargs="+", default=None, help="API keys (space-separated)"
    )
    # Proxy support
    gen_parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Single proxy URL (e.g., socks5://127.0.0.1:1080, http://user:pass@proxy:8080)",
    )
    gen_parser.add_argument(
        "--proxy-file",
        type=str,
        default=None,
        help="File with proxy URLs (one per line) for rotation",
    )
    gen_parser.add_argument(
        "--proxy-list", type=str, nargs="+", default=None, help="List of proxy URLs for rotation"
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

        # Filter out placeholder keys
        valid_keys = [k for k in keys if k and "REPLACE" not in k and len(k) > 20]

        if not valid_keys:
            print("ERROR: No valid API keys provided!")
            print("Edit the script to add your keys, or use --keys key1 key2 key3 key4")
            return

        # Setup proxies
        proxy_mgr = get_proxy_manager()

        if args.proxy:
            proxy_mgr.add_proxy(args.proxy)
            print(f"Using single proxy: {args.proxy[:40]}...")

        if args.proxy_file:
            count = proxy_mgr.add_proxies_from_file(args.proxy_file)
            print(f"Loaded {count} proxies from {args.proxy_file}")

        if args.proxy_list:
            for proxy in args.proxy_list:
                proxy_mgr.add_proxy(proxy)
            print(f"Added {len(args.proxy_list)} proxies from command line")

        if proxy_mgr.has_proxies():
            print(f"Proxy rotation enabled with {len(proxy_mgr.proxies)} proxies")

        print(f"Using {len(valid_keys)} API keys")

        agent = EmotionDataGeneratorAgent(
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

        print("\n=== Emotion Data Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Number of shards: {stats['num_shards']}")
        print("\nCategory coverage:")
        for cat, count in stats["category_coverage"].items():
            print(f"  {cat}: {count}")
        print("\nEmotion counts:")
        sorted_emotions = sorted(stats["emotion_counts"].items(), key=lambda x: -x[1])
        for emotion, count in sorted_emotions:
            print(f"  {emotion}: {count}")

    elif args.command == "export":
        output = Path(args.output)
        count = export_for_training(output, max_samples=args.max, seed=args.seed)
        print(f"Exported {count} samples to {output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
