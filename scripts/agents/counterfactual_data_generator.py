"""
Counterfactual Data Generator for FamilyOS UltraBERT Decoder

Generates high-quality counterfactual pairs for training the UltraBERT decoder head.
Each pair consists of a life scenario (input) and a helpful counterfactual response
that suggests what could have been done differently for a better outcome.

UPDATES (Dec 17, 2025):
- PRIORITIZED critical domain gaps: health_mental (10→5000), relationship_spouse (9→5000),
  relationship_inlaws (10→2000), emotions_grief infant loss (0→500), routine_evening (6→1000)
- ADDED 5 new high-quality examples for critical domains (health_mental, spouse, inlaws, infant loss)
- ADDED critical safety validation to prevent dangerous outputs:
  * Infant loss scenarios CANNOT mention childcare/babysitting
  * Mental health crises MUST include professional help references
  * Physical symptoms MUST recommend medical evaluation
- UPDATED subdomain distribution weights to 20x for health_mental, relationship_spouse

Focus Areas (85 Family Life Subdomains across 15 Domains):
- Parenting: discipline, education, bonding, milestones, screen_time, homework
- Relationships: spouse, siblings, in_laws, extended_family, friends, conflicts
- Health & Wellness: sleep, nutrition, exercise, mental_health, chronic_conditions
- Daily Routines: morning, evening, meals, chores, commute, self_care
- Work-Life Balance: remote_work, overtime, career, boundaries, burnout
- Finances: budgeting, savings, debt, education_fund, retirement, expenses
- Home & Living: organization, maintenance, moves, decoration, safety
- Life Events: weddings, births, deaths, graduations, promotions, relocations
- Cultural & Traditions: festivals, rituals, religious, family_customs, heritage
- Emotional: stress, anxiety, anger, joy, grief, loneliness, overwhelm
- Communication: arguments, difficult_conversations, listening, boundaries
- Caregiving: elderly_care, special_needs, pet_care, babysitting
- Time Management: prioritization, scheduling, procrastination, delegation
- Technology: screen_addiction, digital_boundaries, online_safety, social_media
- Social: isolation, friendships, community, support_networks

Target: Generate 100,000+ high-quality counterfactual pairs for decoder training

Usage:
    python counterfactual_data_generator.py generate --count 50000 --vertex-ai --gcp-project <project>
    python counterfactual_data_generator.py generate --count 5000  # OpenRouter (slower)
    python counterfactual_data_generator.py stats
    python counterfactual_data_generator.py export --format jsonl
"""

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue

import httpx
from dotenv import load_dotenv

# Optional: Google Cloud Vertex AI
try:
    from google import genai
    from google.genai import types as genai_types  # type: ignore

    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False
    genai = None
    genai_types = None

# Load environment variables
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


OPENROUTER_API_KEYS = _load_api_keys_from_env()

if not OPENROUTER_API_KEYS:
    logger.warning("No API keys found in .env, using fallback")
    fallback_key = os.environ.get("OPENROUTER_API_KEY", "")
    if fallback_key:
        OPENROUTER_API_KEYS = [fallback_key]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "amazon/nova-2-lite-v1:free"

# GCP Vertex AI Configuration
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "false").lower() == "true"

# Rate limiting configuration
# Speed presets: "slow", "normal", "fast", "burst"
RATE_LIMIT_PRESETS = {
    "slow": {"requests_per_minute": 5, "delay_between_requests": 12.0, "requests_per_day": 500},
    "normal": {"requests_per_minute": 10, "delay_between_requests": 6.0, "requests_per_day": 900},
    "fast": {"requests_per_minute": 20, "delay_between_requests": 3.0, "requests_per_day": 2000},
    "burst": {"requests_per_minute": 30, "delay_between_requests": 2.0, "requests_per_day": 5000},
}

# Default rate limiting (can be overridden via CLI or env)
RATE_LIMIT_PRESET = os.environ.get("RATE_LIMIT_PRESET", "normal")
_preset = RATE_LIMIT_PRESETS.get(RATE_LIMIT_PRESET, RATE_LIMIT_PRESETS["normal"])
REQUESTS_PER_MINUTE_PER_KEY = int(
    os.environ.get("REQUESTS_PER_MINUTE", _preset["requests_per_minute"])
)
REQUESTS_PER_DAY_PER_KEY = int(os.environ.get("REQUESTS_PER_DAY", _preset["requests_per_day"]))
DELAY_BETWEEN_REQUESTS = float(
    os.environ.get("DELAY_BETWEEN_REQUESTS", _preset["delay_between_requests"])
)

# Paths - use environment variable with fallback to script location
BASE_DIR = Path(os.environ.get("FAMILYOS_BASE_DIR", Path(__file__).resolve().parents[2]))
OUTPUT_DIR = BASE_DIR / "data" / "counterfactual" / "synthetic"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
HASH_INDEX_FILE = OUTPUT_DIR / "hash_index.jsonl"

# Processing settings
SAMPLES_PER_REQUEST = 10  # Generate 10 counterfactual pairs per API call (quality over quantity)


# =============================================================================
# System Prompt - Counterfactual Generation
# =============================================================================

SYSTEM_PROMPT = """You are an expert counterfactual reasoning generator for FamilyOS, a family wellness AI assistant.

## MISSION
Generate high-quality counterfactual pairs that help families learn from life experiences.
Scenarios can have THREE types of outcome valence:
- NEGATIVE: Things went wrong → suggest what could have been done better
- NEUTRAL: Decision point → explore alternative paths that could work
- POSITIVE: Things went well → reinforce what worked and why

## OUTPUT FORMAT (JSONL - one JSON object per line)
{"id": "cf_XXXXX", "domain": "<domain>", "subdomain": "<subdomain>", "input": {"text": "<2-4 sentence scenario>", "outcome_valence": "negative|neutral|positive", "severity": "minor|moderate|significant", "family_members": ["<who>"]}, "counterfactual": {"alternative_action": "<specific action>", "predicted_outcome": "<result>", "causal_mechanism": "<why it works>", "full_text": "<complete response>"}, "metadata": {"emotions_before": ["<emotions>"], "emotions_after": ["<emotions>"], "actionability": "immediate|short_term|long_term", "cultural_context": "universal|indian|western|asian"}}

## DOMAINS (15 categories, 85 subdomains)
- parenting: discipline, education, bonding, milestones, screen_time, siblings, teens, toddlers
- relationship: spouse, inlaws, extended, friends, conflicts, communication, trust, grandparents
- health: sleep, nutrition, exercise, mental, chronic, preventive, children, elderly
- routine: morning, evening, meals, chores, commute, self_care
- work: boundaries, remote, burnout, career, childcare
- finance: budgeting, savings, debt, education, family_expenses
- emotions: stress, anxiety, anger, grief, loneliness, overwhelm
- communication: arguments, difficult_conversations, listening, boundaries, family_meetings
- caregiving: elderly, special_needs, babysitting, respite, coordination
- time_management: prioritization, scheduling, procrastination, delegation, quality_time
- technology: screen_addiction, digital_boundaries, online_safety, social_media, family_apps
- social: isolation, friendships, community, support_networks, neighborhood
- home: organization, maintenance, moves, decoration, safety
- life_events: weddings, births, deaths, graduations, relocations
- cultural: festivals, rituals, religious, traditions, heritage

## CRITICAL QUALITY REQUIREMENTS

### 1. SPECIFICITY (Not vague advice)
- BAD: "If you had been more patient..."
- GOOD: "If you had taken three deep breaths and counted to ten before responding..."

### 2. ACTIONABILITY (Practical steps)
- BAD: "If you had communicated better..."
- GOOD: "If you had used 'I feel' statements instead of 'You always' accusations..."

### 3. EMPATHY (Non-judgmental tone)
- BAD: "If you hadn't made the mistake of..."
- GOOD: "If the situation had been approached with a brief pause to gather thoughts..."

### 4. CAUSAL CLARITY (Explain the mechanism)
- BAD: "Things would have been better."
- GOOD: "Children respond better to calm guidance, as it models emotional regulation."

### 5. REALISM (Achievable alternatives)
- BAD: "If you had hired a full-time nanny..."
- GOOD: "If you had asked your neighbor for 30 minutes of help..."

## CULTURAL DISTRIBUTION
- 40% Indian contexts (Papa, Mummy, Dadi, Nani, joint family, in-laws, festivals like Diwali/Holi)
- 35% Western contexts (Mom, Dad, nuclear family, Thanksgiving, therapy normalization)
- 20% Universal (sleep, health, work stress, parenting basics)
- 5% Asian (Japanese, Chinese, Korean family dynamics)

## SEVERITY DISTRIBUTION
- 40% minor: Daily inconveniences, small misunderstandings
- 45% moderate: Relationship strain, missed opportunities, health impacts
- 15% significant: Major life events, relationship ruptures, health crises

## ACTIONABILITY DISTRIBUTION
- 50% immediate: Can implement today/this week
- 35% short_term: Requires 1-4 weeks of effort
- 15% long_term: Lifestyle changes over months

## VALID EMOTIONS
Negative emotions: frustration, anger, sadness, worry, fear, anxiety, overwhelmed, disappointment, embarrassment, remorse, parental_guilt, grief, loneliness, emptiness, annoyance, nervousness
Positive emotions: joy, relief, hope, pride, contentment, gratitude, love, warmth, togetherness, parental_pride, patience, optimism, celebration, belonging
Neutral emotions: curiosity, contemplation, anticipation, uncertainty, acceptance

## VALENCE-SPECIFIC RESPONSE FORMATS

### NEGATIVE scenarios (outcome_valence: "negative")
- Input: Scenario where something went wrong
- emotions_before: negative emotions (frustration, anger, worry, etc.)
- emotions_after: positive emotions (relief, hope, joy, etc.)
- full_text starts with: "If you had..." (suggesting what could have been done differently)
- Example: "If you had taken a deep breath before responding to your teenager's outburst, you would have modeled emotional regulation..."

### NEUTRAL scenarios (outcome_valence: "neutral")
- Input: Decision point or uncertain situation (neither clearly good nor bad)
- emotions_before: neutral/mixed emotions (curiosity, uncertainty, anticipation)
- emotions_after: positive emotions (clarity, confidence, relief)
- full_text starts with: "In this situation, you might consider..." or "One approach would be to..."
- Example: "We're trying to decide whether to put our son in private school or keep him with his neighborhood friends..."

### POSITIVE scenarios (outcome_valence: "positive")
- Input: Scenario where things went WELL - the person made good choices
- emotions_before: could be mixed (some stress but managed well)
- emotions_after: positive emotions (pride, joy, satisfaction)
- full_text starts with: "What you did worked well because..." or "Your approach was effective because..."
- This REINFORCES good behavior and explains WHY it worked
- Example: "When my daughter failed her driving test, I stayed calm and offered to practice more with her. She passed the next time."
- Response: "What you did worked well because staying calm modeled emotional resilience. By offering support instead of criticism, you maintained trust..."

## GENERATION RULES
1. Input scenarios: 2-4 sentences with rich context
2. Counterfactuals: Name exact actions, times, words (be specific)
3. Include multiple family members when appropriate
4. Vary severity - not everything is a crisis
5. Never blame - always understand and empathize
6. Explain WHY the alternative works (causal mechanism)
7. Be culturally authentic - Indian contexts should feel genuinely Indian
8. Suggest realistic alternatives - don't assume unlimited resources
9. Full text must read naturally as standalone advice
10. For POSITIVE scenarios: Celebrate what went right, explain the psychology behind why it worked

## OUTPUT
Generate diverse, realistic counterfactual pairs in JSONL format.
One complete JSON object per line. No markdown, no explanations.
Start output immediately:"""


# =============================================================================
# 40 Family Domains Configuration
# =============================================================================

# All 40 subdomains for counterfactual generation
COUNTERFACTUAL_DOMAINS = {
    "parenting": [
        "parenting_discipline",
        "parenting_education",
        "parenting_bonding",
        "parenting_milestones",
        "parenting_screen_time",
        "parenting_siblings",
        "parenting_teens",
        "parenting_toddlers",
    ],
    "relationship": [
        "relationship_spouse",
        "relationship_inlaws",
        "relationship_extended",
        "relationship_friends",
        "relationship_conflicts",
        "relationship_communication",
        "relationship_trust",
        "relationship_grandparents",
    ],
    "health": [
        "health_sleep",
        "health_nutrition",
        "health_exercise",
        "health_mental",
        "health_chronic",
        "health_preventive",
        "health_children",
        "health_elderly",
    ],
    "routine": [
        "routine_morning",
        "routine_evening",
        "routine_meals",
        "routine_chores",
        "routine_commute",
        "routine_self_care",
    ],
    "work": [
        "work_boundaries",
        "work_remote",
        "work_burnout",
        "work_career",
        "work_childcare",
    ],
    "finance": [
        "finance_budgeting",
        "finance_savings",
        "finance_debt",
        "finance_education",
        "finance_family_expenses",
    ],
    # NEW DOMAINS - These were missing entirely!
    "emotions": [
        "emotions_stress",
        "emotions_anxiety",
        "emotions_anger",
        "emotions_grief",
        "emotions_loneliness",
        "emotions_overwhelm",
    ],
    "communication": [
        "communication_arguments",
        "communication_difficult_conversations",
        "communication_listening",
        "communication_boundaries",
        "communication_family_meetings",
    ],
    "caregiving": [
        "caregiving_elderly",
        "caregiving_special_needs",
        "caregiving_babysitting",
        "caregiving_respite",
        "caregiving_coordination",
    ],
    "time_management": [
        "time_prioritization",
        "time_scheduling",
        "time_procrastination",
        "time_delegation",
        "time_quality_time",
    ],
    "technology": [
        "tech_screen_addiction",
        "tech_digital_boundaries",
        "tech_online_safety",
        "tech_social_media",
        "tech_family_apps",
    ],
    "social": [
        "social_isolation",
        "social_friendships",
        "social_community",
        "social_support_networks",
        "social_neighborhood",
    ],
    "home": [
        "home_organization",
        "home_maintenance",
        "home_moves",
        "home_decoration",
        "home_safety",
    ],
    "life_events": [
        "life_weddings",
        "life_births",
        "life_deaths",
        "life_graduations",
        "life_relocations",
    ],
    "cultural": [
        "cultural_festivals",
        "cultural_rituals",
        "cultural_religious",
        "cultural_traditions",
        "cultural_heritage",
    ],
}

# Flatten for easy access
ALL_SUBDOMAINS = []
for domain, subdomains in COUNTERFACTUAL_DOMAINS.items():
    ALL_SUBDOMAINS.extend(subdomains)

# Valid values for validation
VALID_DOMAINS = set(COUNTERFACTUAL_DOMAINS.keys())
VALID_SUBDOMAINS = set(ALL_SUBDOMAINS)
VALID_OUTCOME_VALENCE = {"negative", "neutral", "positive"}
VALID_SEVERITY = {"minor", "moderate", "significant"}
VALID_ACTIONABILITY = {"immediate", "short_term", "long_term"}
VALID_CULTURAL_CONTEXT = {"universal", "indian", "western", "asian"}

# Emotions that can appear in counterfactual scenarios
VALID_EMOTIONS = {
    # Negative (before counterfactual for negative scenarios)
    "frustration", "anger", "sadness", "worry", "fear", "anxiety", "overwhelmed",
    "disappointment", "embarrassment", "remorse", "parental_guilt", "grief",
    "loneliness", "emptiness", "annoyance", "nervousness", "bittersweet",
    # Positive (after counterfactual, or before/after for positive scenarios)
    "joy", "relief", "hope", "pride", "contentment", "gratitude", "love",
    "warmth", "togetherness", "parental_pride", "patience", "optimism",
    "celebration", "belonging", "admiration", "caring", "tenderness",
    "playfulness", "excitement", "amusement", "approval", "satisfaction",
    "confidence", "empowerment", "clarity", "peace",
    # Neutral (for neutral scenarios - decision points)
    "neutral", "curiosity", "contemplation", "anticipation", "uncertainty",
    "acceptance", "thoughtfulness", "openness",
}

# =============================================================================
# Target Distributions for Balanced Generation
# =============================================================================

TARGET_DISTRIBUTIONS = {
    # Domain distribution (15 parent categories - EQUAL WEIGHT)
    "domain": {
        "parenting": 10,       # 10% each = balanced
        "relationship": 10,
        "health": 10,
        "routine": 8,
        "work": 8,
        "finance": 7,
        "emotions": 8,         # NEW
        "communication": 7,    # NEW
        "caregiving": 6,       # NEW
        "time_management": 6,  # NEW
        "technology": 5,       # NEW
        "social": 5,           # NEW
        "home": 4,             # NEW
        "life_events": 3,      # NEW
        "cultural": 3,         # NEW
    },
    # Subdomain distribution (PRIORITIZE CRITICAL GAPS)
    "subdomain": {
        # CRITICAL GAPS - HEAVILY PRIORITIZE (Current: <10 examples each)
        "health_mental": 20.0,              # Only 10 examples! Depression, anxiety, therapy
        "relationship_spouse": 20.0,         # Only 9 examples! Intimacy, marriage, communication
        "relationship_inlaws": 15.0,         # Only 10 examples! Boundaries, criticism, conflicts
        "routine_evening": 10.0,             # Only 6 examples! Bedtime, wind down, family time

        # MEDIUM GAPS - HIGH PRIORITY (Current: <1000 examples)
        "emotions_grief": 10.0,              # 1159 examples but POOR quality on infant loss
        "parenting_toddlers": 5.0,           # 824 examples
        "relationship_communication": 5.0,   # 632 examples

        # All other subdomains - baseline coverage
        **{subdomain: 1.0 for subdomain in ALL_SUBDOMAINS
           if subdomain not in [
               "health_mental", "relationship_spouse", "relationship_inlaws",
               "routine_evening", "emotions_grief", "parenting_toddlers",
               "relationship_communication"
           ]}
    },

    # Outcome valence (balanced for 300K dataset)
    # Positive scenarios: what you did right, keep doing it
    # Neutral scenarios: decision points with multiple valid paths
    # Negative scenarios: learn from mistakes
    "outcome_valence": {
        "negative": 40,       # Learn from suboptimal outcomes
        "neutral": 30,        # Decision points, neither good nor bad yet
        "positive": 30,       # Reinforce what went well
    },
    # Severity distribution
    "severity": {
        "minor": 40,
        "moderate": 45,
        "significant": 15,
    },
    # Actionability distribution
    "actionability": {
        "immediate": 50,
        "short_term": 35,
        "long_term": 15,
    },
    # Cultural context distribution
    "cultural_context": {
        "indian": 40,
        "western": 35,
        "universal": 20,
        "asian": 5,           # Japanese, Chinese, Korean contexts
    },
    # Emotions before (negative scenarios)
    "emotions_before": {
        "frustration": 15, "worry": 12, "overwhelmed": 10, "anger": 8,
        "disappointment": 8, "parental_guilt": 8, "sadness": 7, "anxiety": 6,
        "embarrassment": 5, "remorse": 5, "annoyance": 5, "nervousness": 4,
        "fear": 3, "grief": 2, "loneliness": 2,
    },
    # Emotions after (positive outcomes)
    "emotions_after": {
        "relief": 15, "hope": 12, "togetherness": 10, "contentment": 10,
        "warmth": 8, "pride": 8, "joy": 7, "parental_pride": 6,
        "love": 5, "patience": 5, "gratitude": 5, "optimism": 4,
        "caring": 3, "belonging": 2,
    },
}


def load_current_distribution() -> dict:
    """Load current distribution from existing counterfactual data."""
    from collections import Counter

    output_dirs = [
        OUTPUT_DIR,
        Path("D:/Modeling_studio/data/counterfactual/synthetic"),
        Path("D:/Modeling_studio/data/counterfactual/merged"),  # Main training data (217K examples)
    ]

    stats = {
        "total": 0,
        "domain": Counter(),
        "subdomain": Counter(),
        "outcome_valence": Counter(),
        "severity": Counter(),
        "actionability": Counter(),
        "cultural_context": Counter(),
        "emotions_before": Counter(),
        "emotions_after": Counter(),
    }

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for shard in output_dir.glob("shard_*.jsonl"):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        stats["total"] += 1

                        # Domain and subdomain
                        stats["domain"][sample.get("domain", "parenting")] += 1
                        stats["subdomain"][sample.get("subdomain", "")] += 1

                        # Input metadata
                        input_data = sample.get("input", {})
                        stats["outcome_valence"][input_data.get("outcome_valence", "negative")] += 1
                        stats["severity"][input_data.get("severity", "moderate")] += 1

                        # Metadata
                        metadata = sample.get("metadata", {})
                        stats["actionability"][metadata.get("actionability", "immediate")] += 1
                        stats["cultural_context"][metadata.get("cultural_context", "universal")] += 1

                        # Emotions
                        for e in metadata.get("emotions_before", []):
                            stats["emotions_before"][e] += 1
                        for e in metadata.get("emotions_after", []):
                            stats["emotions_after"][e] += 1

                    except (json.JSONDecodeError, KeyError):
                        pass

    return stats


def calculate_gaps(current_stats: dict) -> dict:
    """Calculate what's underrepresented vs target distribution for counterfactuals."""
    gaps = {}
    total = max(current_stats["total"], 1)

    # Domain gaps
    gaps["domain"] = []
    for domain, target_pct in TARGET_DISTRIBUTIONS["domain"].items():
        current_pct = (current_stats["domain"].get(domain, 0) / total) * 100
        if current_pct < target_pct - 3:
            gaps["domain"].append((domain, current_pct, target_pct))
    gaps["domain"].sort(key=lambda x: x[1])

    # Subdomain gaps
    gaps["subdomain"] = []
    for subdomain, target_pct in TARGET_DISTRIBUTIONS["subdomain"].items():
        current_pct = (current_stats["subdomain"].get(subdomain, 0) / total) * 100
        if current_pct < target_pct * 0.5:  # Less than 50% of target
            gaps["subdomain"].append((subdomain, current_pct, target_pct))
    gaps["subdomain"].sort(key=lambda x: x[1])

    # Outcome valence gaps
    gaps["outcome_valence"] = []
    for valence, target_pct in TARGET_DISTRIBUTIONS["outcome_valence"].items():
        current_pct = (current_stats["outcome_valence"].get(valence, 0) / total) * 100
        if current_pct < target_pct - 5:
            gaps["outcome_valence"].append((valence, current_pct, target_pct))

    # Severity gaps
    gaps["severity"] = []
    for severity, target_pct in TARGET_DISTRIBUTIONS["severity"].items():
        current_pct = (current_stats["severity"].get(severity, 0) / total) * 100
        if current_pct < target_pct - 5:
            gaps["severity"].append((severity, current_pct, target_pct))
    gaps["severity"].sort(key=lambda x: x[1])

    # Actionability gaps
    gaps["actionability"] = []
    for action, target_pct in TARGET_DISTRIBUTIONS["actionability"].items():
        current_pct = (current_stats["actionability"].get(action, 0) / total) * 100
        if current_pct < target_pct - 5:
            gaps["actionability"].append((action, current_pct, target_pct))
    gaps["actionability"].sort(key=lambda x: x[1])

    # Cultural context gaps
    gaps["cultural_context"] = []
    for culture, target_pct in TARGET_DISTRIBUTIONS["cultural_context"].items():
        current_pct = (current_stats["cultural_context"].get(culture, 0) / total) * 100
        if current_pct < target_pct - 3:
            gaps["cultural_context"].append((culture, current_pct, target_pct))
    gaps["cultural_context"].sort(key=lambda x: x[1])

    # Emotions before gaps
    emo_before_total = sum(current_stats["emotions_before"].values()) or 1
    gaps["emotions_before"] = []
    for emo, target_pct in TARGET_DISTRIBUTIONS["emotions_before"].items():
        current_pct = (current_stats["emotions_before"].get(emo, 0) / emo_before_total) * 100
        if current_pct < target_pct - 2:
            gaps["emotions_before"].append((emo, current_pct, target_pct))
    gaps["emotions_before"].sort(key=lambda x: x[1])

    # Emotions after gaps
    emo_after_total = sum(current_stats["emotions_after"].values()) or 1
    gaps["emotions_after"] = []
    for emo, target_pct in TARGET_DISTRIBUTIONS["emotions_after"].items():
        current_pct = (current_stats["emotions_after"].get(emo, 0) / emo_after_total) * 100
        if current_pct < target_pct - 2:
            gaps["emotions_after"].append((emo, current_pct, target_pct))
    gaps["emotions_after"].sort(key=lambda x: x[1])

    return gaps


# Global rebalance mode tracking
_REBALANCE_MODE_DOMAINS: dict[str, int] | None = None
_REBALANCE_MODE_SUBDOMAINS: dict[str, int] | None = None
_TARGET_VALENCE: str | None = None  # Global valence filter: "positive", "neutral", "negative", or None


def set_target_valence(valence: str | None) -> None:
    """Set global target valence for generation."""
    global _TARGET_VALENCE
    _TARGET_VALENCE = valence
    if valence:
        logger.info(f"VALENCE MODE: Will generate ONLY '{valence}' scenarios")


def get_target_valence() -> str | None:
    """Get global target valence."""
    return _TARGET_VALENCE


def set_rebalance_mode(domains: dict[str, int] | None) -> None:
    """Set rebalance mode with specific domain targets."""
    global _REBALANCE_MODE_DOMAINS
    _REBALANCE_MODE_DOMAINS = domains
    if domains:
        logger.info(f"REBALANCE MODE: Targeting {len(domains)} domains: {list(domains.keys())}")


def set_subdomain_rebalance_mode(subdomains: dict[str, int] | None) -> None:
    """Set rebalance mode with specific subdomain targets."""
    global _REBALANCE_MODE_SUBDOMAINS
    _REBALANCE_MODE_SUBDOMAINS = subdomains
    if subdomains:
        logger.info(f"SUBDOMAIN REBALANCE MODE: Targeting {len(subdomains)} subdomains: {list(subdomains.keys())}")


def get_domain_for_subdomain(subdomain: str) -> str | None:
    """Find the parent domain for a given subdomain."""
    for domain, subdomains in COUNTERFACTUAL_DOMAINS.items():
        if subdomain in subdomains:
            return domain
    return None


def generate_subdomain_rebalance_prompts(subdomains: dict[str, int], num_workers: int = 20) -> list[str]:
    """Generate worker prompts that STRICTLY target specific subdomains only.

    In subdomain rebalance mode, each worker is assigned 1-2 subdomains and ONLY generates for those.
    This ensures we fill the critical gaps without adding more to over-represented subdomains.
    """
    logger.info(f"SUBDOMAIN REBALANCE MODE: Generating prompts for {len(subdomains)} subdomains across {num_workers} workers")

    # Sort subdomains by target count (most needed first)
    sorted_subdomains = sorted(subdomains.items(), key=lambda x: x[1], reverse=True)

    worker_prompts = []

    for worker_id in range(num_workers):
        # Round-robin assign subdomains to workers
        subdomain_idx = worker_id % len(sorted_subdomains)
        target_subdomain, target_count = sorted_subdomains[subdomain_idx]
        parent_domain = get_domain_for_subdomain(target_subdomain)

        if not parent_domain:
            logger.warning(f"Unknown subdomain: {target_subdomain}, skipping")
            parent_domain = "parenting"  # Fallback

        # Get examples for this specific subdomain
        subdomain_examples = get_subdomain_examples(target_subdomain)

        prompt = f"""## CRITICAL SUBDOMAIN REBALANCE - TARGET: {target_subdomain.upper()}

You MUST generate samples ONLY for the subdomain "{target_subdomain}" within the "{parent_domain}" domain.
This is a critical rebalancing operation - we have SEVERE gaps in this subdomain.

### MANDATORY REQUIREMENTS:
1. EVERY sample MUST have "domain": "{parent_domain}"
2. EVERY sample MUST have "subdomain": "{target_subdomain}"
3. Samples with ANY OTHER subdomain will be REJECTED

### Subdomain: {target_subdomain}
Parent domain: {parent_domain}

### High-Quality Examples for {target_subdomain}:
{subdomain_examples}

### Cultural Distribution:
- 40% Indian contexts (Papa, Mummy, Dadi, joint family dynamics)
- 35% Western contexts (Mom, Dad, nuclear family)
- 20% Universal (applicable everywhere)
- 5% Asian (Japanese, Chinese, Korean contexts)

### Quality Requirements:
- 2-4 sentence realistic scenarios
- SPECIFIC alternative actions (not vague advice)
- Clear causal mechanisms
- Empathetic, non-judgmental tone
- Show understanding and practical solutions

Generate diverse, high-quality counterfactual pairs for "{target_subdomain}" ONLY.
Remember: We need {target_count} samples for this subdomain - quality is critical."""

        worker_prompts.append(prompt)

    return worker_prompts


def get_subdomain_examples(subdomain: str) -> str:
    """Get high-quality examples for a specific subdomain."""
    examples = {
        "health_mental": '''Example 1:
{
  "scenario": "I've been feeling so overwhelmed and anxious about everything lately. Even small tasks feel impossible.",
  "counterfactual": "Those feelings are valid and more common than you might think. A helpful first step could be speaking with a mental health professional who can provide personalized support. In the meantime, try breaking tasks into tiny steps and celebrate each small win.",
  "domain": "health",
  "subdomain": "health_mental",
  "emotion": "overwhelmed"
}

Example 2:
{
  "scenario": "My teenager has been isolating themselves and losing interest in things they used to love. I'm worried.",
  "counterfactual": "Thank you for noticing these changes - your concern shows how much you care. These could be signs of depression, which is treatable. Consider scheduling an appointment with a mental health professional who specializes in adolescents. Let your teen know you're there for them without judgment.",
  "domain": "health",
  "subdomain": "health_mental",
  "emotion": "worried"
}''',
        "relationship_spouse": '''Example 1:
{
  "scenario": "My husband and I keep having the same argument about household chores. It feels like we're going in circles.",
  "counterfactual": "Recurring conflicts often signal deeper needs not being expressed. Try having a conversation when you're both calm, focusing on feelings rather than blame - 'I feel exhausted when...' instead of 'You never...'. Consider creating a shared chore system together where both partners have input.",
  "domain": "relationship",
  "subdomain": "relationship_spouse",
  "emotion": "frustrated"
}

Example 2:
{
  "scenario": "Ever since the baby arrived, my wife and I barely talk. We're like roommates managing a project.",
  "counterfactual": "This transition is incredibly common and doesn't mean your connection is broken. Even 10 minutes of intentional connection daily - a check-in without baby talk, holding hands, or a genuine 'how are you really doing?' - can rebuild intimacy. Consider scheduling regular 'couple time' even if brief.",
  "domain": "relationship",
  "subdomain": "relationship_spouse",
  "emotion": "disconnected"
}''',
        "relationship_inlaws": '''Example 1:
{
  "scenario": "My mother-in-law keeps giving unsolicited advice about how to raise my children. It makes me feel like I'm not a good parent.",
  "counterfactual": "Her advice likely comes from love, but boundaries are healthy. Consider thanking her for her perspective while gently asserting: 'We've thought about this and decided to try our approach first.' Present a united front with your spouse when discussing boundaries.",
  "domain": "relationship",
  "subdomain": "relationship_inlaws",
  "emotion": "undermined"
}

Example 2:
{
  "scenario": "Every time we visit my in-laws, they compare my cooking to their daughter's. I feel like I'll never be good enough.",
  "counterfactual": "Those comparisons hurt, and your feelings are valid. Remember that their comments reflect their adjustment, not your worth. Try sharing a dish that represents your own family traditions, and let your spouse know how these comparisons affect you so they can advocate for you.",
  "domain": "relationship",
  "subdomain": "relationship_inlaws",
  "emotion": "inadequate"
}''',
        "emotions_grief": '''Example 1:
{
  "scenario": "It's been a year since my father passed, but some days the grief hits me just as hard as day one.",
  "counterfactual": "Grief doesn't follow a timeline, and waves of intense emotion are completely normal even years later. Your father's importance in your life means this loss will always be significant. Consider creating rituals to honor his memory, and reach out to grief support groups where others understand this journey.",
  "domain": "emotions",
  "subdomain": "emotions_grief",
  "emotion": "grieving"
}

Example 2:
{
  "scenario": "I had a miscarriage last month and everyone expects me to be 'over it' already. They don't understand.",
  "counterfactual": "Your loss is real and deserves to be grieved fully, regardless of what others expect. Pregnancy loss can be isolating because it's often invisible to others. Consider connecting with miscarriage support communities where your experience is understood. Give yourself permission to grieve at your own pace.",
  "domain": "emotions",
  "subdomain": "emotions_grief",
  "emotion": "grieving"
}''',
        "routine_evening": '''Example 1:
{
  "scenario": "Every evening is chaos. The kids won't settle down for bed and I end up yelling, then feeling guilty.",
  "counterfactual": "Evening chaos is exhausting. Consider creating a predictable wind-down routine: dim lights an hour before bed, quiet activities like reading, and clear expectations with visual timers. When you feel yelling building, try a 30-second pause - even stepping briefly into another room can reset your nervous system.",
  "domain": "routine",
  "subdomain": "routine_evening",
  "emotion": "overwhelmed"
}

Example 2:
{
  "scenario": "I want to have quality time with my toddler after work but I'm so tired I just put on the TV until bedtime.",
  "counterfactual": "Working parent guilt is real, but quantity isn't everything. Even 15 minutes of truly present time - floor play, bath time fun, or reading together - can be deeply connecting. On exhausted days, gentle parallel activities like you resting while they play nearby is also valid and bonding.",
  "domain": "routine",
  "subdomain": "routine_evening",
  "emotion": "guilty"
}'''
    }
    return examples.get(subdomain, f"Generate high-quality examples for {subdomain} with empathetic, practical guidance.")


def generate_rebalance_prompts(domains: dict[str, int], num_workers: int = 20) -> list[str]:
    """Generate worker prompts that STRICTLY target specific domains only.

    In rebalance mode, each worker is assigned 1-2 domains and ONLY generates for those.
    This ensures we fill the gaps without adding more to over-represented domains.
    """
    logger.info(f"REBALANCE MODE: Generating prompts for {len(domains)} domains across {num_workers} workers")

    # Sort domains by target count (most needed first)
    sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)

    worker_prompts = []

    for worker_id in range(num_workers):
        # Round-robin assign domains to workers
        domain_idx = worker_id % len(sorted_domains)
        primary_domain, target_count = sorted_domains[domain_idx]
        subdomains = COUNTERFACTUAL_DOMAINS[primary_domain]

        prompt = f"""## STRICT REBALANCE MODE - DOMAIN: {primary_domain.upper()}

You MUST generate samples ONLY for the "{primary_domain}" domain.
This is a rebalancing operation - we are filling gaps in the dataset.

### CRITICAL RULES:
1. EVERY sample MUST have "domain": "{primary_domain}"
2. EVERY sample MUST have "subdomain" from this list: {', '.join(subdomains)}
3. Samples with ANY OTHER domain will be REJECTED and wasted

### Domain Description: {primary_domain}
Subdomains to cover:
{chr(10).join(f'- {sub}' for sub in subdomains)}

### Domain Example:
{get_domain_examples(primary_domain)}

### Cultural Distribution:
- 40% Indian contexts (Papa, Mummy, Dadi, joint family dynamics)
- 35% Western contexts (Mom, Dad, nuclear family)
- 20% Universal (applicable everywhere)
- 5% Asian (Japanese, Chinese, Korean contexts)

### Quality Requirements:
- 2-4 sentence realistic scenarios
- SPECIFIC alternative actions (not vague advice)
- Clear causal mechanisms
- Empathetic, non-judgmental tone

Generate diverse, high-quality counterfactual pairs for {primary_domain} ONLY."""

        worker_prompts.append(prompt)

    return worker_prompts

def generate_dynamic_worker_prompts(num_workers: int = 20) -> list[str]:
    """Generate worker prompts that STRICTLY enforce domain balance.

    Each worker is assigned specific domains to generate.
    This ensures balanced coverage regardless of LLM tendencies.
    """

    logger.debug("=" * 60)
    logger.debug("GENERATING DOMAIN-STRICT WORKER PROMPTS")
    logger.debug("=" * 60)

    # Get all domains
    all_domains = list(COUNTERFACTUAL_DOMAINS.keys())

    # Load current stats to find underrepresented domains
    current_stats = get_cached_stats()
    total = max(current_stats["total"], 1)

    # Calculate domain coverage and find gaps
    domain_coverage = {}
    for domain in all_domains:
        count = current_stats["domain"].get(domain, 0)
        pct = (count / total) * 100 if total > 0 else 0
        target = TARGET_DISTRIBUTIONS["domain"].get(domain, 6.67)
        gap = target - pct
        domain_coverage[domain] = {
            "count": count,
            "pct": pct,
            "target": target,
            "gap": gap,
            "needs_more": gap > 1.0,  # More than 1% under target
        }

    # Sort domains by gap (most underrepresented first)
    sorted_domains = sorted(all_domains, key=lambda d: domain_coverage[d]["gap"], reverse=True)

    logger.info("Domain gaps (sorted by need):")
    for domain in sorted_domains[:8]:
        info = domain_coverage[domain]
        logger.info(f"  {domain}: {info['pct']:.1f}% (target: {info['target']}%, gap: {info['gap']:+.1f}%)")

    # Create domain assignments for workers
    # Priority: Most underrepresented domains get more workers
    worker_prompts = []

    for worker_id in range(num_workers):
        # Assign domains based on gaps - rotate through underrepresented ones
        # First 60% of workers focus on underrepresented domains
        # Last 40% do balanced generation

        if worker_id < num_workers * 0.6:
            # Focus worker - assigned to specific underrepresented domain
            domain_idx = worker_id % len(sorted_domains)
            primary_domain = sorted_domains[domain_idx]
            subdomains = COUNTERFACTUAL_DOMAINS[primary_domain]

            prompt = f"""## STRICT DOMAIN ASSIGNMENT: {primary_domain.upper()}

You MUST generate samples ONLY for the "{primary_domain}" domain.
DO NOT generate samples for any other domain.

### Required Subdomains (generate 1-2 samples each):
{chr(10).join(f'- {sub}' for sub in subdomains)}

### Domain-Specific Examples:
"""
            # Add domain-specific examples
            prompt += get_domain_examples(primary_domain)

            prompt += f"""

### VALIDATION RULES (STRICT):
1. Every sample MUST have "domain": "{primary_domain}"
2. Every sample MUST have "subdomain" from the list above
3. Samples with wrong domain will be REJECTED

### Cultural Mix for this batch:
- 40% Indian contexts (Papa, Mummy, Dadi, joint family)
- 35% Western contexts (Mom, Dad, nuclear family)
- 20% Universal (applicable everywhere)
- 5% Asian (Japanese, Chinese, Korean)

Generate diverse, high-quality counterfactual pairs for {primary_domain}."""

        else:
            # Balanced worker - generates across all domains with focus on gaps
            underrepresented = [d for d in sorted_domains if domain_coverage[d]["gap"] > 0][:6]

            prompt = f"""## BALANCED GENERATION (Gap-Filling Focus)

Generate samples across these UNDERREPRESENTED domains:
{chr(10).join(f'- {d}: {domain_coverage[d]["pct"]:.1f}% (need +{domain_coverage[d]["gap"]:.1f}%)' for d in underrepresented)}

### Required Distribution for this batch:
Generate 2 samples for each of these domains: {', '.join(underrepresented[:5])}

### AVOID these over-represented domains:
"""
            overrepresented = [d for d in sorted_domains if domain_coverage[d]["gap"] < -2]
            if overrepresented:
                prompt += f"- {', '.join(overrepresented)} (already have enough)\n"
            else:
                prompt += "- (none - dataset is fairly balanced)\n"

            prompt += """
### Cultural Mix:
- 40% Indian, 35% Western, 20% Universal, 5% Asian

Generate diverse counterfactual pairs focusing on the underrepresented domains."""

        worker_prompts.append(prompt)

    logger.info(f"Generated {len(worker_prompts)} domain-strict worker prompts")
    return worker_prompts


def get_domain_examples(domain: str) -> str:
    """Get domain-specific example to guide generation."""
    examples = {
        "parenting": '''
**Parenting Example:**
{"domain": "parenting", "subdomain": "parenting_discipline", "input": {"text": "My son got a bad grade. I yelled at him and took away his phone for a month. He became withdrawn.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Sit with him to understand the struggle, create a study plan together", "predicted_outcome": "He would feel supported, grades improve while preserving bond", "full_text": "If you had sat down to understand his struggles..."}}''',

        "relationship": '''
**Relationship Example:**
{"domain": "relationship", "subdomain": "relationship_spouse", "input": {"text": "My wife mentioned we overspend on groceries. I got defensive and listed my contributions. We didn't speak for 2 days.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Pause, acknowledge her concern, suggest reviewing budget together", "predicted_outcome": "Conversation becomes collaborative, discover root cause", "full_text": "If you had paused and acknowledged her concern..."}}''',

        "health": '''
**Health Example:**
{"domain": "health", "subdomain": "health_sleep", "input": {"text": "I stayed up until 2am working. Next day I was irritable with kids and made mistakes at work.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Set firm 11pm cutoff, finish 80% and wake early for rest", "predicted_outcome": "Wake refreshed, patient with kids, better focus at work", "full_text": "If you had set a firm 11pm cutoff..."}}''',

        "emotions": '''
**Emotions Example:**
{"domain": "emotions", "subdomain": "emotions_anxiety", "input": {"text": "I kept worrying about my daughter's college applications for weeks. I couldn't sleep and snapped at everyone. My constant anxiety affected the whole family.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Schedule specific worry time (20 min/day), practice box breathing, share concerns with spouse", "predicted_outcome": "Contained anxiety allows rest, family feels less tense, daughter feels supported not pressured", "full_text": "If you had scheduled specific worry time and shared concerns..."}}''',

        "communication": '''
**Communication Example:**
{"domain": "communication", "subdomain": "communication_difficult_conversations", "input": {"text": "I needed to tell my aging father he shouldn't drive anymore. I blurted it out at dinner. He felt humiliated and refused to discuss it for months.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Choose private moment, start by acknowledging his independence matters, present safety concerns gently", "predicted_outcome": "Father feels respected, open to compromise like driving only daytime", "full_text": "If you had chosen a private moment and acknowledged his independence..."}}''',

        "caregiving": '''
**Caregiving Example:**
{"domain": "caregiving", "subdomain": "caregiving_elderly", "input": {"text": "I've been caring for my mother-in-law alone for 6 months. I'm exhausted, resentful toward my spouse, and snapping at the kids. Haven't had a day off.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Create caregiving schedule with spouse and siblings, arrange respite care twice monthly", "predicted_outcome": "Sustainable caregiving, preserved relationships, better care for mother-in-law", "full_text": "If you had created a caregiving schedule with family..."}}''',

        "time_management": '''
**Time Management Example:**
{"domain": "time_management", "subdomain": "time_prioritization", "input": {"text": "I said yes to every school committee request. Now I'm overwhelmed, missing my kids' actual events while organizing others' events.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Limit to one committee per semester, say 'let me check my calendar' before committing", "predicted_outcome": "Present at kids' events, manageable volunteer load, model healthy boundaries", "full_text": "If you had limited committees and paused before committing..."}}''',

        "technology": '''
**Technology Example:**
{"domain": "technology", "subdomain": "tech_screen_addiction", "input": {"text": "My 14-year-old is on TikTok until 1am. When I took his phone, he had a meltdown and punched a hole in the wall.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Implement gradual screen time limits together, create phone-free zones not sudden removal", "predicted_outcome": "Teen learns self-regulation, avoids power struggle, sleep improves gradually", "full_text": "If you had implemented gradual limits and phone-free zones together..."}}''',

        "social": '''
**Social Example:**
{"domain": "social", "subdomain": "social_isolation", "input": {"text": "After moving cities for my job, my wife has no friends here. She's home alone all day with toddler and becoming depressed.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Research mom groups before move, spouse takes one evening/week for wife's social activities", "predicted_outcome": "Wife builds local friendships, feels less isolated, mental health stabilizes", "full_text": "If you had researched mom groups and ensured wife has social time..."}}''',

        "home": '''
**Home Example:**
{"domain": "home", "subdomain": "home_organization", "input": {"text": "Our house is so cluttered we can't find anything. Kids are late to school because we can't find shoes. Mornings are chaotic screaming matches.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Create launch pad near door for essentials, 15-min family tidy time before bed", "predicted_outcome": "Morning items always findable, calm starts to day, family teamwork on tidying", "full_text": "If you had created a launch pad and evening tidy routine..."}}''',

        "life_events": '''
**Life Events Example:**
{"domain": "life_events", "subdomain": "life_deaths", "input": {"text": "When grandpa died, I tried to shield the kids completely. They found out at school and felt betrayed that we hadn't told them.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Share news age-appropriately at home, answer questions honestly, include in mourning rituals", "predicted_outcome": "Children process grief healthily, feel trusted, learn death is part of life", "full_text": "If you had shared the news at home age-appropriately..."}}''',

        "cultural": '''
**Cultural Example:**
{"domain": "cultural", "subdomain": "cultural_festivals", "input": {"text": "I skipped Diwali celebrations because of work deadline. My kids missed the puja and were sad seeing friends' photos. They said they feel less Indian.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Block festival days in calendar months ahead, do simple celebration even if short", "predicted_outcome": "Kids feel connected to heritage, create family memories, cultural identity strengthened", "full_text": "If you had blocked festival days in advance and celebrated even briefly..."}}''',

        "routine": '''
**Routine Example:**
{"domain": "routine", "subdomain": "routine_morning", "input": {"text": "Every morning with my toddler is a battle. Today she refused uniform, threw breakfast, we arrived late in tears.", "outcome_valence": "negative", "severity": "moderate"}, "counterfactual": {"alternative_action": "Wake 20 min earlier, offer 2 clothing choices, make breakfast a fun activity together", "predicted_outcome": "Extra buffer reduces stress, choices give autonomy, arrive calm and connected", "full_text": "If you had woken earlier and offered clothing choices..."}}''',

        "work": '''
**Work Example:**
{"domain": "work", "subdomain": "work_boundaries", "input": {"text": "I took a work call during daughter's dance recital. She saw me on phone and her face fell. Wife was furious.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Set out-of-office 2 hours before, silence phone completely, address work after", "predicted_outcome": "Daughter sees you fully present, runs to you excitedly, wife feels family prioritized", "full_text": "If you had silenced phone and been fully present..."}}''',

        "finance": '''
**Finance Example:**
{"domain": "finance", "subdomain": "finance_budgeting", "input": {"text": "Papa's medical bills arrived and we didn't have enough saved. Had to borrow from brother-in-law causing family awkwardness.", "outcome_valence": "negative", "severity": "significant"}, "counterfactual": {"alternative_action": "Start parents healthcare fund with 2000 rupees monthly, research senior insurance", "predicted_outcome": "Emergency fund covers bills, no family loans needed, parents' dignity preserved", "full_text": "If you had started a healthcare fund and researched insurance..."}}''',
    }
    return examples.get(domain, examples["parenting"])

    # Log summary
    logger.info(
        f"Gaps: domains={len(domain_gaps)}, subdomains={len(subdomain_gaps)}, "
        f"cultural={len(cultural_gaps)}, severity={len(severity_gaps)}, "
        f"emo_before={len(emo_before_gaps)}, emo_after={len(emo_after_gaps)}"
    )

    # ALL workers get the SAME prompt for balanced generation
    return [combined_prompt] * num_workers

    # === TEMPORAL SECTION ===
    temp_gaps = gaps.get("temporal", [])
    if temp_gaps:
        temp_lines = [f"  - {t[0]}: {t[1]:.1f}%" for t in temp_gaps[:4]]
        prompt_sections.append(
            f"""🎯 TEMPORAL (priority: LOW)
{chr(10).join(temp_lines)}
Examples: AGE ("when she was 5"), DURATION ("for 3 hours"), DATE_ABS ("March 15")"""
        )

    # === SAFETY SECTION ===
    safety_gaps = gaps.get("safety", [])
    if safety_gaps:
        safe_lines = [f"  - {s[0]}: {s[1]:.1f}%" for s in safety_gaps]
        prompt_sections.append(
            f"""🎯 SAFETY (priority: LOW)
{chr(10).join(safe_lines)}
Add some AMBER (stress, worry) and RED (serious concerns) samples"""
        )

    # Combine all sections into one comprehensive prompt
    if prompt_sections:
        combined_prompt = """📊 DISTRIBUTION GAPS TO FILL

The following labels are UNDERREPRESENTED. Generate samples that address these gaps proportionally.

""" + "\n\n".join(
            prompt_sections
        )
    else:
        combined_prompt = """📊 BALANCED GENERATION

No major gaps detected. Generate diverse, balanced samples across all categories:
- Mix all 5 sentiments equally
- Include variety of emotions, relations, intents
- Use different NER entities and temporal expressions"""

    # Log summary
    logger.info(
        f"Gaps: sentiment={len(sentiment_gaps)}, emotions={len(critical_emotions)}+{len(low_emotions)}, "
        f"rel={len(rel_gaps)}, int={len(intent_gaps)}, ing={len(ingress_gaps)}, "
        f"ner={len(ner_gaps)}, temp={len(temp_gaps)}, safety={len(safety_gaps)}"
    )

    # ALL workers get the SAME prompt for balanced generation
    return [combined_prompt] * num_workers


# Global cache for dynamic prompts (loaded once at startup)
_PROMPTS_LOCK = threading.Lock()
_DYNAMIC_PROMPTS_CACHE: list[str] | None = None
_CURRENT_STATS_CACHE: dict | None = None
_STATS_LOCK = threading.Lock()


def get_cached_stats() -> dict:
    """Get cached stats, computing once if needed."""
    global _CURRENT_STATS_CACHE
    with _STATS_LOCK:
        if _CURRENT_STATS_CACHE is None:
            _CURRENT_STATS_CACHE = load_current_distribution()
            logger.info(f"Loaded {_CURRENT_STATS_CACHE['total']:,} existing samples (cached)")
        return _CURRENT_STATS_CACHE


def update_cached_stats(manager_stats: dict) -> None:
    """Incrementally update cached stats from SyntheticDataManager."""
    global _CURRENT_STATS_CACHE
    with _STATS_LOCK:
        if _CURRENT_STATS_CACHE is not None:
            # Merge manager stats into cached stats
            _CURRENT_STATS_CACHE["total"] += manager_stats.get("total_samples", 0)
            for key in [
                "emotions",
                "sentiment",
                "safety",
                "intent",
                "ingress",
                "relations",
                "ner",
                "temporal",
            ]:
                if key in manager_stats:
                    for k, v in manager_stats[key].items():
                        _CURRENT_STATS_CACHE[key][k] = _CURRENT_STATS_CACHE[key].get(k, 0) + v


def refresh_prompts_cache(num_workers: int = 20, force_reload_stats: bool = False) -> None:
    """Refresh the prompts cache (called by worker 0 during periodic refresh)."""
    global _DYNAMIC_PROMPTS_CACHE, _CURRENT_STATS_CACHE
    with _PROMPTS_LOCK:
        if force_reload_stats:
            with _STATS_LOCK:
                _CURRENT_STATS_CACHE = load_current_distribution()
                logger.info(f"Reloaded stats: {_CURRENT_STATS_CACHE['total']:,} samples")

        # Priority: subdomain rebalance > domain rebalance > dynamic
        if _REBALANCE_MODE_SUBDOMAINS:
            _DYNAMIC_PROMPTS_CACHE = generate_subdomain_rebalance_prompts(_REBALANCE_MODE_SUBDOMAINS, num_workers)
            logger.info(f"Generated {num_workers} SUBDOMAIN REBALANCE prompts for {len(_REBALANCE_MODE_SUBDOMAINS)} subdomains")
        elif _REBALANCE_MODE_DOMAINS:
            _DYNAMIC_PROMPTS_CACHE = generate_rebalance_prompts(_REBALANCE_MODE_DOMAINS, num_workers)
            logger.info(f"Generated {num_workers} REBALANCE prompts for {len(_REBALANCE_MODE_DOMAINS)} domains")
        else:
            _DYNAMIC_PROMPTS_CACHE = generate_dynamic_worker_prompts(num_workers)
            logger.info(f"Refreshed {num_workers} worker prompts based on current gaps")


def get_worker_user_prompt(worker_id: int, num_samples: int = 20) -> str:
    """Get the dynamically generated user prompt for a specific worker.

    The num_samples parameter is the ONLY place that specifies how many samples to generate.
    Dynamic prompts focus on WHAT to generate, this function specifies HOW MANY.
    """
    global _DYNAMIC_PROMPTS_CACHE

    with _PROMPTS_LOCK:
        if _DYNAMIC_PROMPTS_CACHE is None:
            # Priority: subdomain rebalance > domain rebalance > dynamic
            if _REBALANCE_MODE_SUBDOMAINS:
                _DYNAMIC_PROMPTS_CACHE = generate_subdomain_rebalance_prompts(_REBALANCE_MODE_SUBDOMAINS, 20)
            elif _REBALANCE_MODE_DOMAINS:
                _DYNAMIC_PROMPTS_CACHE = generate_rebalance_prompts(_REBALANCE_MODE_DOMAINS, 20)
            else:
                _DYNAMIC_PROMPTS_CACHE = generate_dynamic_worker_prompts(20)

    prompt = _DYNAMIC_PROMPTS_CACHE[worker_id % len(_DYNAMIC_PROMPTS_CACHE)]
    base_prompt = f"Generate exactly {num_samples} samples with this focus:\n\n{prompt}"

    # CRITICAL: Add valence instruction if target valence is set
    target_valence = get_target_valence()
    if target_valence:
        valence_instruction = f"""

## CRITICAL VALENCE REQUIREMENT
You MUST generate ONLY '{target_valence}' outcome_valence scenarios.
EVERY sample MUST have: "outcome_valence": "{target_valence}"
Samples with any other outcome_valence will be REJECTED.
"""
        if target_valence == "positive":
            valence_instruction += """
### POSITIVE scenario format:
- Input: Person made GOOD choices, things went WELL
- emotions_before: can include initial stress but MANAGED well
- emotions_after: MUST be positive (pride, joy, satisfaction, warmth)
- full_text: MUST start with "What you did worked well because..." or "Your approach was effective because..."
"""
        elif target_valence == "neutral":
            valence_instruction += """
### NEUTRAL scenario format:
- Input: Decision point, crossroads, weighing options (neither good nor bad YET)
- emotions_before: neutral/mixed (curiosity, uncertainty, anticipation, contemplation)
- emotions_after: positive (clarity, confidence, relief)
- full_text: MUST start with "In this situation, you might consider..." or "One approach would be to..."
"""
        elif target_valence == "negative":
            valence_instruction += """
### NEGATIVE scenario format:
- Input: Something went WRONG, made a mistake, regret
- emotions_before: MUST be negative (frustration, anger, worry, disappointment)
- emotions_after: positive (showing improvement from counterfactual)
- full_text: MUST start with "If you had..." (suggesting what could have been done differently)
"""
        base_prompt = base_prompt + valence_instruction

    return base_prompt


# =============================================================================
# OpenRouter Client (Copied from unified_data_generator.py)
# =============================================================================


class OpenRouterClient:
    """Client for OpenRouter API with rate limiting."""

    def __init__(
        self,
        api_key: str,
        key_id: int,
        base_url: str = OPENROUTER_BASE_URL,
        requests_per_minute: int = REQUESTS_PER_MINUTE_PER_KEY,
        requests_per_day: int = REQUESTS_PER_DAY_PER_KEY,
    ):
        self.api_key = api_key
        self.key_id = key_id
        self.base_url = base_url
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day

        self.request_times: list[datetime] = []
        self.daily_count = 0
        self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)

        self.client = httpx.Client(timeout=180.0)
        self.lock = threading.Lock()

    def _wait_for_rate_limit(self) -> None:
        with self.lock:
            now = datetime.now()
            if now >= self.daily_reset:
                self.daily_count = 0
                self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)

            if self.daily_count >= self.requests_per_day:
                raise RuntimeError(
                    f"[Key {self.key_id}] Daily limit reached ({self.requests_per_day})"
                )

            minute_ago = now - timedelta(minutes=1)
            self.request_times = [t for t in self.request_times if t > minute_ago]

            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 16000,
    ) -> str:
        self._wait_for_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/familyos",
            "X-Title": "FamilyOS Synthetic Data Generator",
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
        for _attempt in range(max_retries):
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
                    f"[Key {self.key_id}] API call successful "
                    f"(daily: {self.daily_count}/{self.requests_per_day})"
                )
                return content

            except httpx.HTTPStatusError as e:
                logger.error(f"[Key {self.key_id}] HTTP error: {e.response.status_code}")
                if e.response.status_code == 429:
                    time.sleep(60)
                raise

        raise RuntimeError(f"[Key {self.key_id}] Max retries exceeded")

    def close(self):
        self.client.close()


# =============================================================================
# Vertex AI Client (Copied from unified_data_generator.py)
# =============================================================================


class VertexAIClient:
    """Client for Google Cloud Vertex AI (Gemini models)."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        key_id: int = 0,
        api_key: str | None = None,
        system_prompt: str | None = None,
        cache_ttl: str = "86400s",
    ):
        if not HAS_VERTEX_AI:
            raise ImportError("Google GenAI SDK not installed. Run: pip install google-genai")

        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.key_id = key_id
        self.lock = threading.Lock()
        self.request_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        self.api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")

        if self.api_key:
            self.client = genai.Client(  # type: ignore
                vertexai=True,
                project=project_id,
                location=location,
                api_key=self.api_key,
            )
        else:
            self.client = genai.Client(  # type: ignore
                vertexai=True,
                project=project_id,
                location=location,
            )

        logger.info("[Vertex AI] Initialized for synthetic generation")

        # Create cache for system prompt
        self.cached_content_name = None
        if system_prompt:
            self._create_cache(system_prompt, cache_ttl)

    def _create_cache(self, system_prompt: str, ttl: str = "86400s") -> None:
        """Create explicit cache for system prompt."""
        import concurrent.futures

        def create_cache_sync():
            return self.client.caches.create(
                model=self.model_name,
                config=genai_types.CreateCachedContentConfig(  # type: ignore
                    system_instruction=system_prompt,
                    ttl=ttl,
                ),
            )

        try:
            logger.info("[Vertex AI] Creating cache (timeout: 60s)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(create_cache_sync)
                cached_content = future.result(timeout=60)  # 60 second timeout
            self.cached_content_name = cached_content.name
            logger.info(f"[Vertex AI] Created cache: {cached_content.name}")
        except concurrent.futures.TimeoutError:
            logger.warning(
                "[Vertex AI] Cache creation timed out after 60s, proceeding without cache"
            )
            self.cached_content_name = None
        except Exception as e:
            logger.warning(f"[Vertex AI] Failed to create cache: {e}")
            self.cached_content_name = None

    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 8192,
    ) -> str:
        """Generate response using Vertex AI."""
        safety_settings = [
            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),  # type: ignore
            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),  # type: ignore
            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),  # type: ignore
            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),  # type: ignore
        ]

        if self.cached_content_name:
            config = genai_types.GenerateContentConfig(  # type: ignore
                cached_content=self.cached_content_name,
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                safety_settings=safety_settings,
            )
            contents = user_prompt
        else:
            config = genai_types.GenerateContentConfig(  # type: ignore
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                system_instruction=system_prompt,
                safety_settings=safety_settings,
            )
            contents = user_prompt

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

                with self.lock:
                    self.request_count += 1
                    if hasattr(response, "usage_metadata"):
                        self.total_input_tokens += getattr(
                            response.usage_metadata, "prompt_token_count", 0
                        )
                        self.total_output_tokens += getattr(
                            response.usage_metadata, "candidates_token_count", 0
                        )

                logger.info(f"[Vertex AI] Request {self.request_count} successful")
                return response.text

            except Exception as e:
                logger.error(f"[Vertex AI] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    raise

        raise RuntimeError("[Vertex AI] Max retries exceeded")

    def close(self):
        cost = self.get_cost_estimate()
        logger.info(f"[Vertex AI] Session complete. Estimated cost: ${cost['total_cost_usd']:.4f}")

    def get_cost_estimate(self) -> dict:
        """Estimate cost based on token usage."""
        input_price = 0.10
        output_price = 0.40
        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(input_cost + output_cost, 4),
        }


# =============================================================================
# Data Manager
# =============================================================================


class CounterfactualDataManager:
    """Thread-safe manager for counterfactual output data with cross-run deduplication."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, shard_size: int = 5000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()
        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

        # Load existing hashes for cross-run deduplication
        self._load_hash_index()

        # Counterfactual-specific stats
        self.stats = {
            "total_samples": 0,
            "domain_distribution": Counter(),
            "subdomain_distribution": Counter(),
            "outcome_valence_distribution": Counter(),
            "severity_distribution": Counter(),
            "actionability_distribution": Counter(),
            "cultural_context_distribution": Counter(),
            "emotions_before_distribution": Counter(),
            "emotions_after_distribution": Counter(),
        }

        # Track new hashes added this session for saving
        self._new_hashes: list[str] = []
        self._hash_save_threshold = 1000  # Save hashes every N new entries

    def _load_hash_index(self) -> None:
        """Load existing hashes from hash index file for cross-run deduplication."""
        hash_file = self.output_dir / "hash_index.jsonl"
        if hash_file.exists():
            try:
                with open(hash_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.seen_hashes.add(line)
                logger.info(f"Loaded {len(self.seen_hashes):,} hashes for cross-run deduplication")
            except Exception as e:
                logger.warning(f"Failed to load hash index: {e}")
        else:
            # Build hash index from existing shards
            logger.info("Building hash index from existing shards...")
            for shard in self.output_dir.glob("shard_*.jsonl"):
                try:
                    with open(shard, encoding="utf-8") as f:
                        for line in f:
                            try:
                                sample = json.loads(line.strip())
                                # Hash the input text for counterfactuals
                                input_text = sample.get("input", {}).get("text", "").lower().strip()
                                sample_hash = hashlib.md5(input_text.encode()).hexdigest()
                                self.seen_hashes.add(sample_hash)
                            except (json.JSONDecodeError, KeyError):
                                pass
                except Exception as e:
                    logger.warning(f"Failed to read shard {shard}: {e}")

            if self.seen_hashes:
                self._save_hash_index(list(self.seen_hashes))
                logger.info(f"Built hash index with {len(self.seen_hashes):,} entries")

    def _save_hash_index(self, hashes: list[str]) -> None:
        """Append new hashes to the hash index file."""
        hash_file = self.output_dir / "hash_index.jsonl"
        try:
            with open(hash_file, "a", encoding="utf-8") as f:
                for h in hashes:
                    f.write(h + "\n")
        except Exception as e:
            logger.warning(f"Failed to save hash index: {e}")

    def _get_shard_path(self, shard_id: int) -> Path:
        return self.output_dir / f"shard_{shard_id:04d}.jsonl"

    def _count_shard_samples(self, shard_id: int) -> int:
        shard_path = self._get_shard_path(shard_id)
        if not shard_path.exists():
            return 0
        with open(shard_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _get_next_shard_id(self) -> int:
        existing = list(self.output_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
        if self._count_shard_samples(max_id) >= self.shard_size:
            return max_id + 1
        return max_id

    def add_samples(self, samples: list[dict]) -> int:
        added = 0

        with self.lock:
            for sample in samples:
                # Hash the input text for deduplication
                input_text = sample.get("input", {}).get("text", "").lower().strip()
                sample_hash = hashlib.md5(input_text.encode()).hexdigest()

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
                self._new_hashes.append(sample_hash)
                self.current_shard_count += 1

                # Track counterfactual-specific stats
                self.stats["total_samples"] += 1

                # Domain and subdomain
                domain = sample.get("domain", "")
                subdomain = sample.get("subdomain", "")
                self.stats["domain_distribution"][domain] += 1
                self.stats["subdomain_distribution"][subdomain] += 1

                # Input metadata
                input_data = sample.get("input", {})
                self.stats["outcome_valence_distribution"][input_data.get("outcome_valence", "")] += 1
                self.stats["severity_distribution"][input_data.get("severity", "")] += 1

                # Metadata
                metadata = sample.get("metadata", {})
                self.stats["actionability_distribution"][metadata.get("actionability", "")] += 1
                self.stats["cultural_context_distribution"][metadata.get("cultural_context", "")] += 1

                # Emotions
                for emotion in metadata.get("emotions_before", []):
                    self.stats["emotions_before_distribution"][emotion] += 1
                for emotion in metadata.get("emotions_after", []):
                    self.stats["emotions_after_distribution"][emotion] += 1

                added += 1

            # Periodically save hash index
            if len(self._new_hashes) >= self._hash_save_threshold:
                self._save_hash_index(self._new_hashes)
                self._new_hashes = []

        return added

    def flush_hash_index(self) -> None:
        """Flush any remaining hashes to the index file."""
        with self.lock:
            if self._new_hashes:
                self._save_hash_index(self._new_hashes)
                self._new_hashes = []

    def get_stats(self) -> dict:
        with self.lock:
            return dict(self.stats)


# Alias for backward compatibility
SyntheticDataManager = CounterfactualDataManager


# =============================================================================
# Validation & Parsing for Counterfactual Data
# =============================================================================


def clean_and_validate_counterfactual(sample: dict) -> tuple[bool, str]:
    """Clean and validate a counterfactual sample."""

    # Required fields
    if "id" not in sample:
        return False, "Missing id"

    if "domain" not in sample or sample["domain"] not in VALID_DOMAINS:
        return False, f"Invalid domain: {sample.get('domain')}"

    if "subdomain" not in sample or sample["subdomain"] not in VALID_SUBDOMAINS:
        # Try to fix subdomain from domain
        domain = sample.get("domain", "")
        if domain in COUNTERFACTUAL_DOMAINS:
            sample["subdomain"] = COUNTERFACTUAL_DOMAINS[domain][0]  # Default to first
        else:
            return False, f"Invalid subdomain: {sample.get('subdomain')}"

    # Validate input section
    if "input" not in sample:
        return False, "Missing input section"

    input_data = sample["input"]
    if "text" not in input_data or not input_data["text"] or len(input_data["text"]) < 20:
        return False, "Input text too short or missing"

    # CRITICAL: Validate sensitive topics for dangerous content mismatches
    input_text_lower = input_data["text"].lower()
    cf_full_text_lower = sample.get("counterfactual", {}).get("full_text", "").lower()

    # Check for infant loss scenario - MUST NOT mention childcare/babysitting
    if any(phrase in input_text_lower for phrase in [
        "lost our baby", "lost the baby", "baby died", "miscarriage",
        "stillborn", "pregnancy loss", "infant loss"
    ]):
        # These words are FORBIDDEN in response for infant loss
        forbidden_in_response = [
            "childcare", "babysit", "watch the baby", "watching the baby",
            "daycare", "nap time", "feeding time", "diaper"
        ]
        if any(forbidden in cf_full_text_lower for forbidden in forbidden_in_response):
            return False, "CRITICAL: Infant loss scenario contains childcare references - context mismatch"

    # Check for depression/mental health - MUST mention professional help or therapy
    if any(phrase in input_text_lower for phrase in [
        "depressed", "depression", "suicidal", "want to die",
        "panic attack", "severe anxiety", "postpartum depression"
    ]):
        # Response MUST contain mental health professional guidance
        required_terms = ["therapist", "therapy", "counseling", "mental health professional",
                         "psychiatrist", "psychologist", "doctor", "medical"]
        if not any(term in cf_full_text_lower for term in required_terms):
            return False, "CRITICAL: Mental health crisis missing professional help guidance"

    # Check for physical health symptoms - MUST NOT suggest exercise when symptoms described
    if any(phrase in input_text_lower for phrase in [
        "chest pain", "heart pain", "headache after gym", "dizzy after diet",
        "severe pain", "can't breathe", "blood"
    ]):
        # Should mention doctor/medical, NOT just exercise
        if ("exercise" in cf_full_text_lower or "workout" in cf_full_text_lower) and \
           "doctor" not in cf_full_text_lower and "medical" not in cf_full_text_lower:
            return False, "CRITICAL: Physical symptoms require medical attention, not exercise advice"

    # Validate outcome_valence
    if input_data.get("outcome_valence") not in VALID_OUTCOME_VALENCE:
        input_data["outcome_valence"] = "negative"  # Default

    # CRITICAL: Filter by target valence if set (reject samples with wrong valence)
    target_valence = get_target_valence()
    if target_valence and input_data.get("outcome_valence") != target_valence:
        return False, f"VALENCE FILTER: Expected '{target_valence}', got '{input_data.get('outcome_valence')}'"

    # Validate severity
    if input_data.get("severity") not in VALID_SEVERITY:
        input_data["severity"] = "moderate"  # Default

    # Validate counterfactual section
    if "counterfactual" not in sample:
        return False, "Missing counterfactual section"

    cf_data = sample["counterfactual"]

    # Check required fields (alternative_action and predicted_outcome are essential)
    if "alternative_action" not in cf_data or not cf_data["alternative_action"]:
        return False, "Missing counterfactual.alternative_action"
    if "predicted_outcome" not in cf_data or not cf_data["predicted_outcome"]:
        return False, "Missing counterfactual.predicted_outcome"

    # Auto-generate full_text if missing (combine action + outcome + mechanism)
    if "full_text" not in cf_data or not cf_data["full_text"]:
        action = cf_data["alternative_action"]
        outcome = cf_data["predicted_outcome"]
        mechanism = cf_data.get("causal_mechanism", "")
        if mechanism:
            cf_data["full_text"] = f"If you had {action.lower()}, {outcome.lower()} {mechanism}"
        else:
            cf_data["full_text"] = f"If you had {action.lower()}, {outcome.lower()}"

    # Validate full_text is substantial
    if len(cf_data["full_text"]) < 50:
        return False, "Counterfactual full_text too short"

    # Validate metadata section
    if "metadata" not in sample:
        sample["metadata"] = {}

    metadata = sample["metadata"]

    # Validate actionability
    if metadata.get("actionability") not in VALID_ACTIONABILITY:
        metadata["actionability"] = "immediate"  # Default

    # Validate cultural_context
    if metadata.get("cultural_context") not in VALID_CULTURAL_CONTEXT:
        metadata["cultural_context"] = "universal"  # Default

    # Clean emotions_before - remove invalid
    if "emotions_before" in metadata:
        valid_emo = [e for e in metadata["emotions_before"] if e in VALID_EMOTIONS]
        metadata["emotions_before"] = valid_emo if valid_emo else ["frustration"]
    else:
        metadata["emotions_before"] = ["frustration"]

    # Clean emotions_after - remove invalid
    if "emotions_after" in metadata:
        valid_emo = [e for e in metadata["emotions_after"] if e in VALID_EMOTIONS]
        metadata["emotions_after"] = valid_emo if valid_emo else ["relief"]
    else:
        metadata["emotions_after"] = ["relief"]

    return True, ""


def clean_and_validate_sample(sample: dict) -> tuple[bool, str]:
    """Clean and validate a counterfactual sample (wrapper for compatibility)."""
    return clean_and_validate_counterfactual(sample)


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate a sample (legacy compatibility)."""
    return clean_and_validate_counterfactual(sample)


def parse_counterfactual_response(response_text: str) -> list[dict]:
    """Parse JSONL from LLM response for counterfactual data.

    Handles both single-line JSONL and multi-line formatted JSON.
    """
    valid_samples = []
    invalid_count = 0
    parse_errors = 0

    # Step 1: Try parsing as complete JSON array first
    text = response_text.strip()
    if text.startswith("```"):
        # Extract content from markdown code block
        lines = text.split("\n")
        filtered = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(filtered).strip()

    # Try parsing as JSON array first
    if text.startswith("["):
        try:
            samples = json.loads(text)
            if isinstance(samples, list):
                for sample in samples:
                    is_valid, error = clean_and_validate_counterfactual(sample)
                    if is_valid:
                        valid_samples.append(sample)
                    else:
                        invalid_count += 1
                        logger.debug(f"Validation failed: {error}")
                if valid_samples or samples:
                    if invalid_count > 0:
                        logger.warning(f"Parsed JSON array: {len(valid_samples)} valid, {invalid_count} invalid")
                    return valid_samples
        except json.JSONDecodeError:
            pass  # Fall through to line-by-line parsing

    # Step 2: Try parsing as multi-line JSON objects (each object on multiple lines)
    # Find all JSON objects using brace matching
    json_objects = []
    depth = 0
    start_idx = None

    for i, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                json_objects.append(text[start_idx:i+1])
                start_idx = None

    # Try to parse each extracted JSON object
    for obj_text in json_objects:
        try:
            sample = json.loads(obj_text)
            is_valid, error = clean_and_validate_counterfactual(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                invalid_count += 1
                logger.debug(f"Validation failed: {error}")
        except json.JSONDecodeError as e:
            parse_errors += 1
            logger.debug(f"JSON parse error: {e}")

    # Log warning if high failure rate
    total_attempted = len(json_objects)
    if total_attempted > 0 and invalid_count + parse_errors > len(valid_samples):
        logger.warning(
            f"High parse failure rate: {len(valid_samples)}/{total_attempted} valid "
            f"({invalid_count} validation errors, {parse_errors} parse errors)"
        )
        if valid_samples == 0 and total_attempted > 0:
            # Log a sample of the response for debugging
            logger.warning(f"Sample response (first 500 chars): {text[:500]}")

    return valid_samples


# Alias for backward compatibility
parse_synthetic_response = parse_counterfactual_response


# =============================================================================
# Counterfactual Generator Agent
# =============================================================================


class CounterfactualGenerator:
    """Generate synthetic counterfactual training samples for decoder."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        samples_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
        use_vertex_ai: bool = False,
        gcp_project_id: str | None = None,
        gcp_location: str = "us-central1",
        vertex_model: str = "gemini-2.5-flash",
        num_parallel: int = 1,
        output_dir: Path | str | None = None,
        rebalance_domains: dict[str, int] | None = None,
        target_subdomains: dict[str, int] | None = None,
        target_valence: str | None = None,
        valence_distribution: dict[str, int] | None = None,
    ):
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.rebalance_domains = rebalance_domains  # {domain: target_count}
        self.target_subdomains = target_subdomains  # {subdomain: target_count}
        self.target_valence = target_valence  # "positive", "neutral", "negative", or None for all
        self.valence_distribution = valence_distribution  # Custom {neg: x, neu: y, pos: z}

        # CRITICAL: Set global valence filter for prompt generation and validation
        set_target_valence(target_valence)

        # Build valence-specific system prompt if targeting specific valence
        system_prompt = self._build_system_prompt()

        if use_vertex_ai or USE_VERTEX_AI:
            project_id = gcp_project_id or GCP_PROJECT_ID
            if not project_id:
                raise ValueError("GCP_PROJECT_ID not set")

            # Create first client with cache attempt, others without
            first_client = VertexAIClient(
                project_id=project_id,
                location=gcp_location or GCP_LOCATION,
                model_name=vertex_model or VERTEX_MODEL,
                key_id=0,
                system_prompt=system_prompt,
                cache_ttl="86400s",
            )

            # Share cache name with other workers (or None if cache failed)
            shared_cache_name = first_client.cached_content_name

            self.clients = [first_client]
            for i in range(1, num_parallel):
                client = VertexAIClient(
                    project_id=project_id,
                    location=gcp_location or GCP_LOCATION,
                    model_name=vertex_model or VERTEX_MODEL,
                    key_id=i,
                    system_prompt=None,  # Don't try to create cache
                    cache_ttl="86400s",
                )
                client.cached_content_name = shared_cache_name  # Share cache
                self.clients.append(client)

            logger.info(f"Using Vertex AI with {num_parallel} parallel worker(s)")
        else:
            api_keys = api_keys or OPENROUTER_API_KEYS
            self.api_keys = [k for k in api_keys if k and "REPLACE" not in k]
            if not self.api_keys:
                raise ValueError("No valid API keys provided!")
            self.clients = [
                OpenRouterClient(api_key=key, key_id=i) for i, key in enumerate(self.api_keys)
            ]
            logger.info(f"Using OpenRouter with {len(self.clients)} API keys")

        self.output_manager = CounterfactualDataManager(output_dir=self.output_dir)
        self.batch_counter = 0
        self.batch_lock = threading.Lock()

        # Activate rebalance mode if specified
        if rebalance_domains:
            set_rebalance_mode(rebalance_domains)
            logger.info(f"REBALANCE MODE ACTIVE: Targeting {len(rebalance_domains)} specific domains")

        # Activate subdomain rebalance mode if specified
        if target_subdomains:
            set_subdomain_rebalance_mode(target_subdomains)
            logger.info(f"SUBDOMAIN REBALANCE MODE ACTIVE: Targeting {len(target_subdomains)} specific subdomains")

        # Log valence mode
        if target_valence:
            logger.info(f"VALENCE MODE: Generating ONLY {target_valence} scenarios")
        elif valence_distribution:
            logger.info(f"CUSTOM VALENCE DISTRIBUTION: {valence_distribution}")

    def _build_system_prompt(self) -> str:
        """Build system prompt, optionally customized for specific valence generation."""
        if not self.target_valence:
            return SYSTEM_PROMPT

        # Create valence-specific prompts
        valence_instructions = {
            "positive": '''
## SPECIAL MODE: POSITIVE SCENARIOS ONLY
You are generating ONLY positive outcome scenarios where the person made GOOD choices.

For POSITIVE scenarios:
- Input: Describe a situation where the person handled things WELL
- The scenario should show good parenting, communication, or decision-making
- emotions_before: Can include initial stress/uncertainty that was managed well
- emotions_after: MUST be positive (pride, joy, satisfaction, warmth)
- full_text starts with: "What you did worked well because..." or "Your approach was effective because..."
- Explain WHY their approach worked (the psychology/mechanism behind it)

EXAMPLES OF POSITIVE SCENARIOS:
1. "When my daughter failed her test, I stayed calm and offered to help her study. She passed the next time and thanked me for believing in her."
   Response: "What you did worked well because staying calm during setbacks models emotional resilience for your child..."

2. "My teenager was upset about a friendship conflict. Instead of giving advice, I just listened. She later told me she felt really understood."
   Response: "Your approach was effective because active listening without judgment creates psychological safety..."

3. "We made a family decision about moving by involving our children in the discussion. Everyone felt heard and the transition went smoothly."
   Response: "What you did worked well because including children in family decisions gives them a sense of agency..."

Generate ONLY scenarios with outcome_valence: "positive"
''',
            "neutral": '''
## SPECIAL MODE: NEUTRAL SCENARIOS ONLY
You are generating ONLY neutral decision-point scenarios.

For NEUTRAL scenarios:
- Input: Describe a decision point or crossroads (no clear right/wrong yet)
- The person is weighing options, considering choices
- emotions_before: neutral/mixed (curiosity, uncertainty, anticipation, contemplation)
- emotions_after: positive emotions (clarity, confidence, relief)
- full_text starts with: "In this situation, you might consider..." or "One approach would be to..."
- Explore different valid paths, not just one "right" answer

EXAMPLES OF NEUTRAL SCENARIOS:
1. "We're trying to decide whether to put our son in private school or keep him with his neighborhood friends at the local school."
   Response: "In this situation, you might consider both the academic opportunities and social connections..."

2. "My mother-in-law wants to move in with us. We're not sure if this is the right decision for our family."
   Response: "One approach would be to have a family meeting to discuss expectations, boundaries, and trial periods..."

3. "My teenager wants to take a gap year before college. We're weighing the pros and cons."
   Response: "In this situation, consider both the potential benefits of maturity and self-discovery..."

Generate ONLY scenarios with outcome_valence: "neutral"
''',
            "negative": '''
## SPECIAL MODE: NEGATIVE SCENARIOS ONLY
You are generating ONLY negative outcome scenarios (traditional counterfactuals).

For NEGATIVE scenarios:
- Input: Describe a situation where something went wrong
- emotions_before: MUST be negative (frustration, anger, worry, disappointment)
- emotions_after: MUST be positive (showing improvement)
- full_text starts with: "If you had..." (suggesting what could have been done differently)
- This is the traditional counterfactual format

Generate ONLY scenarios with outcome_valence: "negative"
'''
        }

        # Append valence-specific instructions to base prompt
        base_prompt = SYSTEM_PROMPT
        if self.target_valence in valence_instructions:
            base_prompt = base_prompt + "\n" + valence_instructions[self.target_valence]

        return base_prompt

    def _get_next_batch_id(self) -> int:
        with self.batch_lock:
            batch_id = self.batch_counter
            self.batch_counter += 1
            return batch_id

    def _generate_batch(self, client, user_prompt: str) -> int:
        """Generate one batch of synthetic samples using worker-specific prompt."""
        batch_id = self._get_next_batch_id()

        try:
            # Use instance's system prompt (may be valence-specific)
            system_prompt = self._build_system_prompt()
            response = client.generate(
                model=MODEL if hasattr(client, "api_key") else client.model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.85,
            )

            samples = parse_synthetic_response(response)
            added = self.output_manager.add_samples(samples)

            logger.info(
                f"[Worker {client.key_id}] Batch {batch_id}: "
                f"Generated {len(samples)}, Added {added}. "
                f"Total: {self.output_manager.stats['total_samples']}"
            )

            return added

        except Exception as e:
            logger.error(f"[Worker {client.key_id}] Batch {batch_id} failed: {e}")
            return 0

    def _worker(
        self,
        client,
        target_samples: int,
        stop_event: threading.Event,
        stats_queue: Queue,
        refresh_interval: int = 333,
    ) -> None:
        """Worker thread with dedicated focus area based on worker_id.

        Args:
            refresh_interval: Re-calculate distribution gaps every N batches (default: 333 = ~5000 samples)
        """
        worker_id = client.key_id
        samples_generated = 0
        batch_count = 0

        # Get worker-specific prompt (dynamically generated based on gaps)
        user_prompt = get_worker_user_prompt(worker_id, self.samples_per_request)

        logger.debug(f"[Worker {worker_id}] Starting with focus: {user_prompt[:80]}...")

        while not stop_event.is_set() and samples_generated < target_samples:
            try:
                # Batch progress tracking every 100 batches
                if batch_count > 0 and batch_count % 100 == 0:
                    progress_pct = (samples_generated / target_samples) * 100
                    logger.info(
                        f"[Worker {worker_id}] PROGRESS: batch {batch_count}, "
                        f"{samples_generated:,}/{target_samples:,} samples ({progress_pct:.1f}%)"
                    )

                # Periodic refresh: Worker 0 triggers re-calculation for all workers
                if worker_id == 0 and batch_count > 0 and batch_count % refresh_interval == 0:
                    logger.info(
                        f"PERIODIC REFRESH at batch {batch_count} - Re-analyzing distribution gaps..."
                    )
                    refresh_prompts_cache(20, force_reload_stats=True)

                # Get potentially updated prompt
                user_prompt = get_worker_user_prompt(worker_id, self.samples_per_request)

                added = self._generate_batch(client, user_prompt)
                samples_generated += added
                batch_count += 1

                stats_queue.put(
                    {
                        "key_id": client.key_id,
                        "added": added,
                        "generated": added,
                    }
                )

                time.sleep(self.delay_between_requests)

            except RuntimeError as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"[Key {client.key_id}] Rate limit reached")
                    break
                raise
            except Exception as e:
                logger.error(f"[Key {client.key_id}] Worker error: {e}")
                time.sleep(self.delay_between_requests)

        logger.info(f"[Key {client.key_id}] Finished. Generated {samples_generated} samples")

    def run(self, target_samples: int = 10000) -> dict:
        """Run parallel counterfactual generation."""
        start_time = datetime.now()
        samples_per_worker = target_samples // len(self.clients)

        logger.info("=" * 60)
        logger.info("COUNTERFACTUAL DATA GENERATION")
        logger.info("=" * 60)
        logger.info(f"Target: {target_samples:,} samples")
        logger.info(f"Workers: {len(self.clients)}")
        logger.info(f"Per worker: {samples_per_worker:,} samples")
        logger.info("=" * 60)

        stats = {
            "start_time": start_time.isoformat(),
            "target_samples": target_samples,
            "generated_samples": 0,
        }

        stop_event = threading.Event()
        stats_queue: Queue = Queue()

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [
                executor.submit(
                    self._worker,
                    client,
                    samples_per_worker,
                    stop_event,
                    stats_queue,
                )
                for client in self.clients
            ]

            try:
                while not all(f.done() for f in futures):
                    while not stats_queue.empty():
                        batch_stats = stats_queue.get_nowait()
                        stats["generated_samples"] += batch_stats["added"]
                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("INTERRUPTED BY USER")
                logger.info("=" * 60)
                stop_event.set()

        # Final stats and cleanup
        self.output_manager.flush_hash_index()  # Ensure all hashes are saved
        final_stats = self.output_manager.get_stats()
        stats.update(final_stats)
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60

        logger.info(f"\n{'='*60}")
        logger.info("COUNTERFACTUAL GENERATION COMPLETE")
        logger.info(f"Total samples: {final_stats['total_samples']:,}")
        logger.info(f"Domains covered: {len(final_stats['domain_distribution'])}")
        logger.info(f"Duration: {stats['duration_minutes']:.1f} minutes")
        logger.info(f"{'='*60}")

        for client in self.clients:
            client.close()

        return stats


# =============================================================================
# CLI
# =============================================================================


def show_stats(output_dir: Path = OUTPUT_DIR):
    """Show counterfactual generation statistics."""
    if not output_dir.exists():
        print(f"No counterfactual data found at {output_dir}")
        return

    shards = list(output_dir.glob("shard_*.jsonl"))

    if not shards:
        print(f"No counterfactual shards found in {output_dir}")
        return

    # Load all samples for stats
    all_samples = []
    for shard in sorted(shards):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    all_samples.append(json.loads(line.strip()))
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(all_samples)

    # Counterfactual-specific distributions
    domain_dist = Counter(s.get("domain", "") for s in all_samples)
    subdomain_dist = Counter(s.get("subdomain", "") for s in all_samples)
    valence_dist = Counter(s.get("input", {}).get("outcome_valence", "") for s in all_samples)
    severity_dist = Counter(s.get("input", {}).get("severity", "") for s in all_samples)
    actionability_dist = Counter(s.get("metadata", {}).get("actionability", "") for s in all_samples)
    cultural_dist = Counter(s.get("metadata", {}).get("cultural_context", "") for s in all_samples)

    # Emotion distributions
    emotions_before = Counter()
    emotions_after = Counter()
    for s in all_samples:
        for e in s.get("metadata", {}).get("emotions_before", []):
            emotions_before[e] += 1
        for e in s.get("metadata", {}).get("emotions_after", []):
            emotions_after[e] += 1

    print("\n" + "=" * 70)
    print(f"COUNTERFACTUAL DATA STATISTICS: {output_dir}")
    print("=" * 70)
    print(f"\nTotal samples: {total:,}")
    print(f"Number of shards: {len(shards)}")

    print("\n--- Domain Distribution ---")
    for domain, count in domain_dist.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {domain:20s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Subdomain Distribution (Top 15) ---")
    for subdomain, count in subdomain_dist.most_common(15):
        pct = 100 * count / total if total > 0 else 0
        print(f"  {subdomain:25s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Outcome Valence Distribution ---")
    for valence, count in valence_dist.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {valence:20s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Severity Distribution ---")
    for severity, count in severity_dist.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {severity:20s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Actionability Distribution ---")
    for action, count in actionability_dist.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {action:20s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Cultural Context Distribution ---")
    for culture, count in cultural_dist.most_common():
        pct = 100 * count / total if total > 0 else 0
        print(f"  {culture:20s} {count:6,} ({pct:5.1f}%)")

    print("\n--- Top 10 Emotions Before (Negative State) ---")
    for emotion, count in emotions_before.most_common(10):
        print(f"  {emotion:20s} {count:6,}")

    print("\n--- Top 10 Emotions After (Positive State) ---")
    for emotion, count in emotions_after.most_common(10):
        print(f"  {emotion:20s} {count:6,}")

    print("=" * 70)


def merge_directories(input_dirs: list[str], output_dir: str | None = None):
    """
    Merge and deduplicate counterfactual data from multiple parallel runs.

    Args:
        input_dirs: List of input directories (e.g., ["synthetic_p1", "synthetic_p2"])
        output_dir: Output directory for merged data (default: data/counterfactual/synthetic)
    """
    output_path = Path(output_dir) if output_dir else OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MERGING COUNTERFACTUAL DATA FROM MULTIPLE RUNS")
    print("=" * 70)

    seen_hashes: set[str] = set()
    all_samples: list[dict] = []
    stats = {
        "input_dirs": [],
        "samples_per_dir": {},
        "duplicates_removed": 0,
        "total_merged": 0,
    }

    # Load existing hashes from output dir if it exists
    existing_hash_file = output_path / "hash_index.jsonl"
    if existing_hash_file.exists():
        with open(existing_hash_file, encoding="utf-8") as f:
            for line in f:
                seen_hashes.add(line.strip())
        print(f"Loaded {len(seen_hashes):,} existing hashes from output dir")

    # Process each input directory
    for input_dir in input_dirs:
        input_path = BASE_DIR / "data" / "counterfactual" / input_dir
        if not input_path.exists():
            # Try as absolute path
            input_path = Path(input_dir)

        if not input_path.exists():
            print(f"  [SKIP] Directory not found: {input_dir}")
            continue

        print(f"\nProcessing: {input_path}")
        stats["input_dirs"].append(str(input_path))

        dir_samples = 0
        dir_duplicates = 0

        shards = sorted(input_path.glob("shard_*.jsonl"))
        for shard in shards:
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        input_text = sample.get("input", {}).get("text", "").lower().strip()
                        sample_hash = hashlib.md5(input_text.encode()).hexdigest()

                        if sample_hash in seen_hashes:
                            dir_duplicates += 1
                            continue

                        seen_hashes.add(sample_hash)
                        all_samples.append(sample)
                        dir_samples += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse line in {shard}: {e}")

        print(f"  Samples: {dir_samples:,} (duplicates removed: {dir_duplicates:,})")
        stats["samples_per_dir"][str(input_path)] = dir_samples
        stats["duplicates_removed"] += dir_duplicates

    if not all_samples:
        print("\n[ERROR] No samples found to merge!")
        return

    # Write merged data in shards
    SHARD_SIZE = 5000
    print(f"\nWriting {len(all_samples):,} samples to {output_path}...")

    for i, start_idx in enumerate(range(0, len(all_samples), SHARD_SIZE)):
        shard_samples = all_samples[start_idx:start_idx + SHARD_SIZE]
        shard_path = output_path / f"shard_{i:04d}.jsonl"

        with open(shard_path, "w", encoding="utf-8") as f:
            for sample in shard_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"  Written: {shard_path.name} ({len(shard_samples):,} samples)")

    # Write hash index
    hash_file = output_path / "hash_index.jsonl"
    with open(hash_file, "w", encoding="utf-8") as f:
        for h in seen_hashes:
            f.write(h + "\n")
    print(f"  Written: hash_index.jsonl ({len(seen_hashes):,} hashes)")

    stats["total_merged"] = len(all_samples)

    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    print(f"Total merged samples: {stats['total_merged']:,}")
    print(f"Duplicates removed: {stats['duplicates_removed']:,}")
    print(f"Output directory: {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Counterfactual Data Generator for Decoder Training")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate counterfactual samples")
    gen_parser.add_argument(
        "--count", type=int, default=10000, help="Number of samples to generate"
    )
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=10, help="Samples per API call (lower = higher quality)"
    )
    gen_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for generated data (default: data/counterfactual/synthetic). "
             "Use different dirs for parallel runs with different GCP projects."
    )

    # Speed control options
    speed_group = gen_parser.add_argument_group("Speed Control")
    speed_group.add_argument(
        "--speed",
        type=str,
        choices=["slow", "normal", "fast", "burst"],
        default="normal",
        help="Request speed preset: slow (5 rpm), normal (10 rpm), fast (20 rpm), burst (30 rpm)",
    )
    speed_group.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Override delay between requests (seconds). Overrides --speed preset.",
    )
    speed_group.add_argument(
        "--requests-per-minute",
        type=int,
        default=None,
        help="Override requests per minute limit. Overrides --speed preset.",
    )

    # Vertex AI options
    gen_parser.add_argument("--vertex-ai", action="store_true", help="Use GCP Vertex AI")
    gen_parser.add_argument("--gcp-project", type=str, help="GCP Project ID")
    gen_parser.add_argument("--gcp-location", type=str, default="us-central1", help="GCP region")
    gen_parser.add_argument(
        "--vertex-model", type=str, default="gemini-2.5-flash", help="Vertex AI model"
    )
    gen_parser.add_argument(
        "--num-parallel", type=int, default=4, help="Parallel workers for Vertex AI"
    )

    # Refresh interval
    gen_parser.add_argument(
        "--refresh-interval",
        type=int,
        default=100,
        help="Re-analyze distribution gaps every N batches (default: 100)",
    )

    # Valence control options
    valence_group = gen_parser.add_argument_group("Valence Control")
    valence_group.add_argument(
        "--valence",
        type=str,
        choices=["all", "positive", "neutral", "negative"],
        default="all",
        help="Generate only specific valence type: all (balanced), positive, neutral, or negative",
    )
    valence_group.add_argument(
        "--valence-distribution",
        type=str,
        default=None,
        help="Custom valence distribution as 'negative:neutral:positive' (e.g., '40:30:30')",
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to show stats for (default: data/counterfactual/synthetic)"
    )

    # Merge command for combining multiple parallel runs
    merge_parser = subparsers.add_parser("merge", help="Merge and deduplicate data from multiple runs")
    merge_parser.add_argument(
        "--input-dirs", type=str, nargs="+", required=True,
        help="Input directories to merge (e.g., synthetic_p1 synthetic_p2)"
    )
    merge_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for merged data (default: data/counterfactual/synthetic)"
    )

    # Rebalance command - generate ONLY for missing/underrepresented domains
    rebalance_parser = subparsers.add_parser(
        "rebalance",
        help="Generate samples ONLY for underrepresented domains (ignores over-represented ones)"
    )
    rebalance_parser.add_argument(
        "--count", type=int, default=50000,
        help="Total samples to generate for missing domains"
    )
    rebalance_parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="Generate for domains with gap > threshold%% (default: 3.0)"
    )
    rebalance_parser.add_argument(
        "--subdomains", type=str, default=None,
        help="Comma-separated subdomain:count pairs to generate (e.g., health_mental:5000,relationship_spouse:5000)"
    )
    rebalance_parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/counterfactual/synthetic)"
    )
    # Vertex AI options for rebalance
    rebalance_parser.add_argument("--vertex-ai", action="store_true", help="Use GCP Vertex AI")
    rebalance_parser.add_argument("--gcp-project", type=str, help="GCP Project ID")
    rebalance_parser.add_argument("--gcp-location", type=str, default="us-central1", help="GCP region")
    rebalance_parser.add_argument(
        "--vertex-model", type=str, default="gemini-2.5-flash", help="Vertex AI model"
    )
    rebalance_parser.add_argument(
        "--num-parallel", type=int, default=10, help="Parallel workers for Vertex AI"
    )

    args = parser.parse_args()

    if args.command == "generate":
        # Resolve speed settings
        preset = RATE_LIMIT_PRESETS.get(args.speed, RATE_LIMIT_PRESETS["normal"])
        delay = args.delay if args.delay is not None else preset["delay_between_requests"]
        rpm = (
            args.requests_per_minute
            if args.requests_per_minute is not None
            else preset["requests_per_minute"]
        )

        # Resolve output directory
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Speed settings: {args.speed} preset, delay={delay}s, rpm={rpm}")

        # Resolve valence settings
        target_valence = args.valence if args.valence != "all" else None
        valence_distribution = None
        if args.valence_distribution:
            parts = args.valence_distribution.split(":")
            if len(parts) == 3:
                valence_distribution = {
                    "negative": int(parts[0]),
                    "neutral": int(parts[1]),
                    "positive": int(parts[2]),
                }
                logger.info(f"Custom valence distribution: {valence_distribution}")
        if target_valence:
            logger.info(f"Generating ONLY {target_valence} valence scenarios")

        generator = CounterfactualGenerator(
            samples_per_request=args.samples_per_request,
            delay_between_requests=delay,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
            output_dir=output_dir,
            target_valence=target_valence,
            valence_distribution=valence_distribution,
        )

        stats = generator.run(target_samples=args.count)
        print("\n=== Counterfactual Generation Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "stats":
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
        show_stats(output_dir)

    elif args.command == "merge":
        merge_directories(args.input_dirs, args.output_dir)

    elif args.command == "rebalance":
        # Rebalance mode - generate ONLY for underrepresented domains/subdomains
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

        # Get current stats
        current_stats = get_cached_stats()
        total = max(current_stats["total"], 1)

        # Check if specific subdomains are requested
        if args.subdomains:
            # Parse subdomain:count pairs (e.g., "health_mental:5000,relationship_spouse:5000")
            subdomain_targets = {}
            for pair in args.subdomains.split(","):
                parts = pair.strip().split(":")
                if len(parts) == 2:
                    subdomain = parts[0].strip()
                    count = int(parts[1].strip())
                    subdomain_targets[subdomain] = count

            print("\n" + "=" * 70)
            print("SUBDOMAIN REBALANCE - Targeting specific subdomains")
            print("=" * 70)

            total_to_generate = sum(subdomain_targets.values())
            for subdomain, target_count in subdomain_targets.items():
                current_count = current_stats["subdomain"].get(subdomain, 0)
                print(f"  {subdomain:40s}: {current_count:6,} -> {current_count + target_count:6,} (+{target_count})")

            print(f"\nWill generate {total_to_generate:,} samples across {len(subdomain_targets)} subdomains")

            # Map subdomains to their parent domains for generation
            samples_per_domain = {}
            for subdomain, count in subdomain_targets.items():
                # Find parent domain
                parent_domain = None
                for domain, subs in COUNTERFACTUAL_DOMAINS.items():
                    if subdomain in subs:
                        parent_domain = domain
                        break
                if parent_domain:
                    samples_per_domain[parent_domain] = samples_per_domain.get(parent_domain, 0) + count

            # Run generation with subdomain focus
            generator = CounterfactualGenerator(
                samples_per_request=10,
                delay_between_requests=3.0,
                use_vertex_ai=args.vertex_ai,
                gcp_project_id=args.gcp_project,
                gcp_location=args.gcp_location,
                vertex_model=args.vertex_model,
                num_parallel=args.num_parallel,
                output_dir=output_dir,
                rebalance_domains=samples_per_domain,
                target_subdomains=subdomain_targets,  # Pass specific subdomain targets
            )

            stats = generator.run(target_samples=total_to_generate)
            print("\n=== Subdomain Rebalance Statistics ===")
            print(json.dumps(stats, indent=2, default=str))
            return

        # Find underrepresented domains
        all_domains = list(COUNTERFACTUAL_DOMAINS.keys())
        missing_domains = []

        print("\n" + "=" * 70)
        print("REBALANCE ANALYSIS - Finding underrepresented domains")
        print("=" * 70)

        for domain in all_domains:
            count = current_stats["domain"].get(domain, 0)
            pct = (count / total) * 100 if total > 0 else 0
            target = TARGET_DISTRIBUTIONS["domain"].get(domain, 6.67)
            gap = target - pct

            status = "OK" if gap <= args.threshold else "NEEDS MORE"
            print(f"  {domain:20s}: {pct:5.1f}% (target: {target}%, gap: {gap:+5.1f}%) [{status}]")

            if gap > args.threshold:
                missing_domains.append((domain, gap))

        if not missing_domains:
            print("\nDataset is balanced! No rebalancing needed.")
            return

        # Sort by gap (most needed first)
        missing_domains.sort(key=lambda x: x[1], reverse=True)

        print(f"\n{len(missing_domains)} domains need rebalancing:")
        for domain, gap in missing_domains:
            print(f"  - {domain}: +{gap:.1f}% needed")

        # Calculate samples per domain
        total_gap = sum(gap for _, gap in missing_domains)
        samples_per_domain = {}

        for domain, gap in missing_domains:
            proportion = gap / total_gap
            samples = int(args.count * proportion)
            samples_per_domain[domain] = max(samples, 1000)  # Minimum 1000 per domain

        print(f"\nWill generate ~{sum(samples_per_domain.values())} samples across {len(missing_domains)} domains")
        for domain, count in samples_per_domain.items():
            print(f"  - {domain}: ~{count} samples")

        # Run rebalanced generation
        generator = CounterfactualGenerator(
            samples_per_request=10,
            delay_between_requests=3.0,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
            output_dir=output_dir,
            rebalance_domains=samples_per_domain,  # Pass specific domain targets
        )

        stats = generator.run(target_samples=args.count)
        print("\n=== Rebalance Generation Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
