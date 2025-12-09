"""
Golden Test Set Generator

Generates CHALLENGING, LONG-FORM, MULTI-SENTENCE test samples covering DIVERSE WORLD CULTURES.
This creates a golden test set for rigorous model evaluation.

Key Differences from synthetic_task_generator.py:
- LONG TEXTS: 2-5 sentences (50-150 words) per sample
- MULTI-CULTURAL: India, China, Japan, Korea, Middle East, Africa, Latin America, Europe
- COMPLEX: Multiple entities, nuanced emotions, mixed feelings
- CHALLENGING: Edge cases, ambiguous scenarios, complex family dynamics

Target: Generate 500 high-quality challenging test samples

Usage:
    python golden_set_generator.py generate --count 500 --vertex-ai --gcp-project <project>
    python golden_set_generator.py generate --count 500  # OpenRouter
    python golden_set_generator.py stats
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
OUTPUT_DIR = BASE_DIR / "data" / "familyos" / "unified" / "golden_set"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
HASH_INDEX_FILE = OUTPUT_DIR / "hash_index.jsonl"

# Processing settings
SAMPLES_PER_REQUEST = 15  # Generate 15 new samples per API call


# =============================================================================
# System Prompt - Synthetic Generation
# =============================================================================

SYSTEM_PROMPT = """You are an expert synthetic data generator for FamilyOS, a family-focused AI assistant. Your task is to generate CHALLENGING, LONG-FORM, MULTI-SENTENCE test samples that cover DIVERSE WORLD CULTURES.

## TASK: Generate GOLDEN TEST SET - Complex Multi-Sentence Samples

**CRITICAL REQUIREMENTS:**
1. **LONG TEXTS**: Each sample MUST be 2-5 sentences (50-150 words). NO short one-liners.
2. **MULTI-CULTURAL**: Cover families from India, China, Japan, Korea, Middle East, Africa, Latin America, Europe, Southeast Asia
3. **COMPLEX SCENARIOS**: Include nuanced emotions, multiple family members, mixed feelings
4. **CHALLENGING NER**: Multiple entities per sample, overlapping contexts
5. **REALISTIC DIALOGUE**: Natural speech patterns, cultural expressions, code-switching

Generate conversations where users interact with a family assistant for:
- Setting reminders (appointments, calls, tasks)
- Querying past events (memories, schedules)
- Planning activities (trips, events, routines)
- Seeking advice (parenting, relationships, health)
- Health tracking and wellness
- Work-life balance and productivity
- Financial planning and budgets

## OUTPUT FORMAT

Generate complete multi-task annotations in this exact JSON format:

```json
{
  "id": "syn_XXXXX",
  "text": "<realistic user message>",
  "tasks": {
    "emotions": ["emotion1", "emotion2"],
    "sentiment": "very_positive" | "positive" | "neutral" | "negative" | "very_negative",
    "ner_family": [
      {"start": <char_start>, "end": <char_end>, "label": "<LABEL>", "token": "<text>"}
    ],
    "safety_familyos": "GREEN" | "AMBER" | "RED" | "CRISIS",
    "intent": "<intent_label>",
    "ingress": "<domain_label>",
    "relations": [
      {"subject": "<entity1>", "predicate": "<relation>", "object": "<entity2>"}
    ],
    "temporal": [
      {"start": <char_start>, "end": <char_end>, "label": "<TEMPORAL_TYPE>", "token": "<text>"}
    ]
  },
  "hub_routing": {
    "EMO": true/false,
    "REL": true/false,
    "MEM": true/false,
    "TASK": true/false
  }
}
```

## TASK SCHEMAS (USE EXACTLY THESE VALUES)

### 1. EMOTIONS (44 classes ONLY, multi-label)
🚨 **CRITICAL: Use ONLY these 44 emotions. DO NOT create new emotions like "responsibility", "planning", "curiosity", "concern", etc.**

**Core (8):** neutral, joy, sadness, anger, fear, surprise, love, disgust
**Positive (12):** admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness
**Negative (10):** annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness
**Family (14):** nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

**DO NOT USE:** responsibility, planning, curiosity, anticipation, fondness, concern, stress, discipline, health, routine, organization, memory, reflection, determination, guilt, uncertainty, care, anxiety, supportive, friendship, accomplishment, wellness, support, contemplation, indecision, professionalism, appreciation, prudence, etc.

### 2. SENTIMENT (5 ONLY)
very_positive, positive, neutral, negative, very_negative

### 3. NER_FAMILY (10 entity types ONLY)
🚨 **CRITICAL: Use ONLY these 10 NER types. DO NOT use AGE, DATE_REL, TIME, FREQUENCY, DURATION, colleague_of, friend_of, WORK, RELATIONSHIP, FINANCE, etc.**

**VALID:** PERSON, KINSHIP, NICKNAME, PET, HOME_LOC, FAMILY_EVENT, ROUTINE, TRADITION, MILESTONE, HEIRLOOM

**DO NOT USE in NER:** AGE (→ use in TEMPORAL), DATE_REL (→ TEMPORAL), TIME (→ TEMPORAL), FREQUENCY (→ TEMPORAL), DURATION (→ TEMPORAL), COLLEAGUE (→ use PERSON), colleague_of (→ use in RELATIONS), friend_of (→ RELATIONS), WORK, RELATIONSHIP, FINANCE, DATE_ABS, TEMPORAL, etc.

### 4. SAFETY_FAMILYOS (4 ONLY)
GREEN, AMBER, RED, CRISIS

### 5. INTENT (8 classes ONLY)
log_memory, query_memory, set_reminder, express_feeling, seek_advice, share_news, reflect, other

### 6. INGRESS (12 domains ONLY)
DIARY, TASK, HEALTH, FINANCE, RELATIONSHIP, WORK, META, MEMORY, PLANNING, CELEBRATION, CONCERN, GRATITUDE

### 7. RELATIONS (15 types ONLY)
🚨 **CRITICAL: Use ONLY these 15 relation types. DO NOT use family_of, owner_of, nephew_niece_of, etc.**

**VALID:** no_relation, parent_of, child_of, spouse_of, sibling_of, grandparent_of, grandchild_of, aunt_uncle_of, niece_nephew_of, cousin_of, pet_of, friend_of, colleague_of, lives_at, owns

**DO NOT USE:** family_of (too generic), owner_of (→ use "owns"), nephew_niece_of (→ use "niece_nephew_of"), etc.

### 8. TEMPORAL (6 types ONLY)
🚨 **CRITICAL: Use ONLY these 6 temporal types. DO NOT use MILESTONE, ROUTINE, HOME_LOC, FAMILY_EVENT, WORK, TRADITION, etc.**

**VALID:** DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE

**DO NOT USE in TEMPORAL:** MILESTONE (→ use in NER), ROUTINE (→ NER), HOME_LOC (→ NER), FAMILY_EVENT (→ NER), TEMPORAL_TYPE, WORK, TRADITION (→ NER), etc.

---

## HUB ROUTING RULES

| Hub | Set TRUE when... |
|-----|------------------|
| EMO | Text expresses emotions, feelings, sentiment |
| REL | Text contains family relationships, entity pairs |
| MEM | Text is about memories, past events, nostalgia |
| TASK | Text is about tasks, reminders, queries, commands |

---

## EXAMPLES OF LONG MULTI-SENTENCE GOLDEN TEST SAMPLES

### Example 1: Indian family - Multi-generational planning (3 sentences)
```json
{"id": "gold_00001", "text": "My dadi has been feeling lonely since dada passed away last year, and I've been meaning to visit her more often. Can you remind me every Sunday at 10am to video call her? Also, my cousin Riya said she wants to join sometimes, so maybe we can make it a family thing.", "tasks": {"emotions": ["caring", "grief", "love", "longing", "togetherness"], "sentiment": "neutral", "ner_family": [{"start": 3, "end": 7, "label": "KINSHIP", "token": "dadi"}, {"start": 38, "end": 42, "label": "KINSHIP", "token": "dada"}, {"start": 188, "end": 194, "label": "KINSHIP", "token": "cousin"}, {"start": 195, "end": 199, "label": "PERSON", "token": "Riya"}], "safety_familyos": "AMBER", "intent": "set_reminder", "ingress": "RELATIONSHIP", "relations": [{"subject": "dadi", "predicate": "spouse_of", "object": "dada"}, {"subject": "user", "predicate": "grandchild_of", "object": "dadi"}, {"subject": "user", "predicate": "cousin_of", "object": "Riya"}], "temporal": [{"start": 55, "end": 64, "label": "DATE_REL", "token": "last year"}, {"start": 138, "end": 150, "label": "FREQUENCY", "token": "every Sunday"}, {"start": 154, "end": 158, "label": "TIME", "token": "10am"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": true, "TASK": true}}
```

### Example 2: Chinese family - Health concern with cultural context (4 sentences)
```json
{"id": "gold_00002", "text": "Mama has been complaining about her knee pain again, especially after doing tai chi in the park every morning. Baba thinks she should see Dr. Chen, but you know how stubborn she can be about Western medicine. I'm worried because the Mid-Autumn Festival is coming up next month, and she always insists on making mooncakes herself. Maybe remind me to call my sister Mei-Lin tomorrow to discuss how we can help.", "tasks": {"emotions": ["worry", "caring", "frustration", "love"], "sentiment": "negative", "ner_family": [{"start": 0, "end": 4, "label": "KINSHIP", "token": "Mama"}, {"start": 77, "end": 83, "label": "ROUTINE", "token": "tai chi"}, {"start": 111, "end": 115, "label": "KINSHIP", "token": "Baba"}, {"start": 139, "end": 147, "label": "PERSON", "token": "Dr. Chen"}, {"start": 217, "end": 237, "label": "FAMILY_EVENT", "token": "Mid-Autumn Festival"}, {"start": 296, "end": 305, "label": "TRADITION", "token": "mooncakes"}, {"start": 362, "end": 368, "label": "KINSHIP", "token": "sister"}, {"start": 369, "end": 376, "label": "PERSON", "token": "Mei-Lin"}], "safety_familyos": "AMBER", "intent": "set_reminder", "ingress": "HEALTH", "relations": [{"subject": "Mama", "predicate": "spouse_of", "object": "Baba"}, {"subject": "user", "predicate": "sibling_of", "object": "Mei-Lin"}], "temporal": [{"start": 100, "end": 113, "label": "FREQUENCY", "token": "every morning"}, {"start": 250, "end": 260, "label": "DATE_REL", "token": "next month"}, {"start": 391, "end": 399, "label": "DATE_REL", "token": "tomorrow"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": false, "TASK": true}}
```

### Example 3: Nigerian family - Celebration with extended family (3 sentences)
```json
{"id": "gold_00003", "text": "Uncle Chukwu is turning 60 next Saturday, and the whole family is planning a huge celebration at Grandma Adaeze's compound in Lagos. My wife Amara and I are flying in from London with the kids on Thursday evening. Can you help me remember to buy the traditional wrapper fabric for him as a gift before we leave?", "tasks": {"emotions": ["excitement", "joy", "love", "celebration", "togetherness"], "sentiment": "very_positive", "ner_family": [{"start": 0, "end": 12, "label": "KINSHIP", "token": "Uncle Chukwu"}, {"start": 98, "end": 112, "label": "KINSHIP", "token": "Grandma Adaeze"}, {"start": 115, "end": 123, "label": "HOME_LOC", "token": "compound"}, {"start": 140, "end": 144, "label": "KINSHIP", "token": "wife"}, {"start": 145, "end": 150, "label": "PERSON", "token": "Amara"}, {"start": 184, "end": 188, "label": "KINSHIP", "token": "kids"}, {"start": 256, "end": 280, "label": "TRADITION", "token": "traditional wrapper fabric"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "CELEBRATION", "relations": [{"subject": "user", "predicate": "niece_nephew_of", "object": "Uncle Chukwu"}, {"subject": "user", "predicate": "grandchild_of", "object": "Grandma Adaeze"}, {"subject": "user", "predicate": "spouse_of", "object": "Amara"}, {"subject": "user", "predicate": "parent_of", "object": "kids"}], "temporal": [{"start": 24, "end": 26, "label": "AGE", "token": "60"}, {"start": 27, "end": 40, "label": "DATE_REL", "token": "next Saturday"}, {"start": 192, "end": 208, "label": "DATE_REL", "token": "Thursday evening"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": false, "TASK": true}}
```

### Example 4: Japanese family - Work-life balance struggle (4 sentences)
```json
{"id": "gold_00004", "text": "I've been working overtime every day for the past three weeks, and I feel terrible for missing Yuki's piano recital last Tuesday. My wife Sakura has been handling everything at home alone - cooking, helping with homework, taking care of obaachan. I don't know how to make it up to them, but maybe I should start by being home for dinner at least twice a week. What do you think I should prioritize first?", "tasks": {"emotions": ["parental_guilt", "remorse", "overwhelmed", "sadness", "worry", "love"], "sentiment": "very_negative", "ner_family": [{"start": 95, "end": 99, "label": "PERSON", "token": "Yuki"}, {"start": 102, "end": 115, "label": "FAMILY_EVENT", "token": "piano recital"}, {"start": 135, "end": 139, "label": "KINSHIP", "token": "wife"}, {"start": 140, "end": 146, "label": "PERSON", "token": "Sakura"}, {"start": 239, "end": 247, "label": "KINSHIP", "token": "obaachan"}], "safety_familyos": "AMBER", "intent": "seek_advice", "ingress": "WORK", "relations": [{"subject": "user", "predicate": "spouse_of", "object": "Sakura"}, {"subject": "user", "predicate": "parent_of", "object": "Yuki"}], "temporal": [{"start": 31, "end": 40, "label": "FREQUENCY", "token": "every day"}, {"start": 53, "end": 65, "label": "DURATION", "token": "three weeks"}, {"start": 116, "end": 128, "label": "DATE_REL", "token": "last Tuesday"}, {"start": 343, "end": 356, "label": "FREQUENCY", "token": "twice a week"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": true, "TASK": true}}
```

### Example 5: Mexican family - Financial planning with tradition (3 sentences)
```json
{"id": "gold_00005", "text": "Abuela's 80th birthday quinceaera-style party is going to cost us around $5000, and I'm not sure how to split it fairly between all the siblings. Mi hermano Carlos thinks we should each pay equally, but Tia Rosa has been struggling since losing her job in March. Can you help me draft a budget proposal that accounts for everyone's situation before our family meeting next Sunday?", "tasks": {"emotions": ["worry", "caring", "frustration", "love", "patience"], "sentiment": "negative", "ner_family": [{"start": 0, "end": 6, "label": "KINSHIP", "token": "Abuela"}, {"start": 21, "end": 47, "label": "FAMILY_EVENT", "token": "quinceaera-style party"}, {"start": 140, "end": 148, "label": "KINSHIP", "token": "siblings"}, {"start": 150, "end": 160, "label": "KINSHIP", "token": "Mi hermano"}, {"start": 161, "end": 167, "label": "PERSON", "token": "Carlos"}, {"start": 205, "end": 213, "label": "KINSHIP", "token": "Tia Rosa"}], "safety_familyos": "AMBER", "intent": "seek_advice", "ingress": "FINANCE", "relations": [{"subject": "user", "predicate": "grandchild_of", "object": "Abuela"}, {"subject": "user", "predicate": "sibling_of", "object": "Carlos"}, {"subject": "user", "predicate": "niece_nephew_of", "object": "Tia Rosa"}], "temporal": [{"start": 9, "end": 13, "label": "AGE", "token": "80th"}, {"start": 253, "end": 258, "label": "DATE_ABS", "token": "March"}, {"start": 368, "end": 379, "label": "DATE_REL", "token": "next Sunday"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": false, "TASK": true}}
```

### Example 6: Korean family - Intergenerational memory (3 sentences)
```json
{"id": "gold_00006", "text": "Harabeoji used to take me fishing at the Han River every summer when I was 7 or 8 years old, and I still remember the smell of the doenjang-jjigae halmeoni would make when we came home. Now that I have my own son Min-jun, I want to continue that tradition with him. Please remind me to call appa this weekend to ask where harabeoji's old fishing gear is stored.", "tasks": {"emotions": ["nostalgia", "warmth", "love", "longing", "joy", "bittersweet"], "sentiment": "positive", "ner_family": [{"start": 0, "end": 9, "label": "KINSHIP", "token": "Harabeoji"}, {"start": 131, "end": 145, "label": "TRADITION", "token": "doenjang-jjigae"}, {"start": 146, "end": 154, "label": "KINSHIP", "token": "halmeoni"}, {"start": 203, "end": 206, "label": "KINSHIP", "token": "son"}, {"start": 207, "end": 214, "label": "PERSON", "token": "Min-jun"}, {"start": 296, "end": 300, "label": "KINSHIP", "token": "appa"}, {"start": 327, "end": 336, "label": "KINSHIP", "token": "harabeoji"}, {"start": 343, "end": 354, "label": "HEIRLOOM", "token": "fishing gear"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "MEMORY", "relations": [{"subject": "user", "predicate": "grandchild_of", "object": "Harabeoji"}, {"subject": "user", "predicate": "grandchild_of", "object": "halmeoni"}, {"subject": "user", "predicate": "parent_of", "object": "Min-jun"}, {"subject": "user", "predicate": "child_of", "object": "appa"}], "temporal": [{"start": 55, "end": 67, "label": "FREQUENCY", "token": "every summer"}, {"start": 79, "end": 93, "label": "AGE", "token": "7 or 8 years old"}, {"start": 306, "end": 318, "label": "DATE_REL", "token": "this weekend"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": true, "TASK": true}}
```

---

## IMPORTANT RULES FOR GOLDEN TEST SET

1. **LONG MULTI-SENTENCE TEXTS** - Each sample MUST be 2-5 sentences, 50-150 words
2. **WORLD CULTURES** - Include: Indian (dadi/nani/bhai), Chinese (mama/baba), Japanese (obaachan/ojiichan), Korean (halmeoni/harabeoji), Nigerian, Mexican (abuela/tio), Arabic, etc.
3. **COMPLEX SCENARIOS** - Multiple emotions, mixed feelings, nuanced situations
4. **MULTIPLE ENTITIES** - 3-8 NER entities per sample, multiple relations
5. **ACCURATE OFFSETS** - Count character positions VERY carefully
6. **REALISTIC SPEECH** - Natural dialogue, cultural expressions, code-switching
7. **BALANCED SENTIMENT** - Equal distribution of all 5 sentiment classes
8. **MIXED HUB ROUTING** - Most samples should trigger multiple hubs
9. **CHALLENGING CASES** - Edge cases, ambiguous emotions, complex family dynamics

---

## OUTPUT

Generate the requested number of LONG, MULTI-SENTENCE, MULTI-CULTURAL samples in JSONL format.
One complete JSON object per line. No markdown, no explanations.
Start output immediately:"""


# =============================================================================
# Dynamic Distribution Analyzer & Prompt Generator
# =============================================================================

# Target distributions (ideal balanced %)
TARGET_DISTRIBUTIONS = {
    "emotions": {
        # Rare emotions that need boosting (< 3% currently)
        "tier1_critical": [
            "disgust",
            "embarrassment",
            "remorse",
            "homesickness",
            "belonging",
            "anger",
            "surprise",
            "fear",
            "disapproval",
        ],
        "tier2_low": [
            "tenderness",
            "patience",
            "parental_guilt",
            "nervousness",
            "admiration",
            "protectiveness",
            "grief",
            "emptiness",
            "amusement",
            "parental_pride",
            "playfulness",
            "approval",
        ],
    },
    "sentiment": {
        "very_positive": 20,
        "positive": 20,
        "neutral": 20,
        "negative": 20,
        "very_negative": 20,
    },
    "safety": {"GREEN": 60, "AMBER": 30, "RED": 8, "CRISIS": 2},
    "intent": {
        "log_memory": 12,
        "query_memory": 12,
        "set_reminder": 15,
        "express_feeling": 12,
        "seek_advice": 15,
        "share_news": 12,
        "reflect": 12,
        "other": 10,
    },
    "ingress": {
        "DIARY": 8,
        "TASK": 12,
        "HEALTH": 10,
        "FINANCE": 8,
        "RELATIONSHIP": 8,
        "WORK": 8,
        "META": 8,
        "MEMORY": 8,
        "PLANNING": 10,
        "CELEBRATION": 8,
        "CONCERN": 6,
        "GRATITUDE": 8,
    },
    "relations": {
        "parent_of": 10,
        "child_of": 10,
        "spouse_of": 8,
        "sibling_of": 8,
        "grandparent_of": 8,
        "grandchild_of": 10,
        "aunt_uncle_of": 8,
        "niece_nephew_of": 8,
        "cousin_of": 10,
        "pet_of": 6,
        "friend_of": 8,
        "colleague_of": 8,
        "lives_at": 8,
        "owns": 8,
    },
    "ner": {
        "KINSHIP": 20,
        "PERSON": 15,
        "FAMILY_EVENT": 10,
        "ROUTINE": 10,
        "HOME_LOC": 10,
        "TRADITION": 10,
        "PET": 8,
        "NICKNAME": 8,
        "MILESTONE": 8,
        "HEIRLOOM": 8,
    },
    "temporal": {
        "DATE_REL": 25,
        "TIME": 20,
        "FREQUENCY": 20,
        "DATE_ABS": 15,
        "DURATION": 10,
        "AGE": 10,
    },
}


def load_current_distribution() -> dict:
    """Load current distribution from existing data."""
    from collections import Counter

    output_dirs = [
        Path("D:/Modeling_studio/data/familyos/unified/output"),
        Path("D:/Modeling_studio/data/familyos/unified/output_synthetic"),
    ]

    stats = {
        "total": 0,
        "emotions": Counter(),
        "sentiment": Counter(),
        "safety": Counter(),
        "intent": Counter(),
        "ingress": Counter(),
        "relations": Counter(),
        "ner": Counter(),
        "temporal": Counter(),
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
                        tasks = sample.get("tasks", {})

                        for e in tasks.get("emotions", []):
                            stats["emotions"][e] += 1
                        stats["sentiment"][tasks.get("sentiment", "neutral")] += 1
                        stats["safety"][tasks.get("safety_familyos", "GREEN")] += 1
                        stats["intent"][tasks.get("intent", "other")] += 1
                        stats["ingress"][tasks.get("ingress", "DIARY")] += 1
                        for r in tasks.get("relations", []):
                            stats["relations"][r.get("predicate", "")] += 1
                        for e in tasks.get("ner_family", []):
                            stats["ner"][e.get("label", "")] += 1
                        for t in tasks.get("temporal", []):
                            stats["temporal"][t.get("label", "")] += 1

                    except (json.JSONDecodeError, KeyError):
                        pass

    return stats


def calculate_gaps(current_stats: dict) -> dict:
    """Calculate what's underrepresented vs target distribution."""
    gaps = {}
    total = max(current_stats["total"], 1)

    # Emotions - find those below threshold
    gaps["emotions_critical"] = []
    gaps["emotions_low"] = []
    for emotion in VALID_EMOTIONS:
        pct = (current_stats["emotions"].get(emotion, 0) / total) * 100
        if pct < 2.2:
            gaps["emotions_critical"].append((emotion, pct))
        elif pct < 3.5:
            gaps["emotions_low"].append((emotion, pct))
    gaps["emotions_critical"].sort(key=lambda x: x[1])
    gaps["emotions_low"].sort(key=lambda x: x[1])

    # Sentiment gaps
    gaps["sentiment"] = []
    for sent, target_pct in TARGET_DISTRIBUTIONS["sentiment"].items():
        current_pct = (current_stats["sentiment"].get(sent, 0) / total) * 100
        if current_pct < target_pct - 5:
            gaps["sentiment"].append((sent, current_pct, target_pct))

    # Safety gaps - use relative threshold (50% of target) to catch low-percentage items like CRISIS
    gaps["safety"] = []
    for safety, target_pct in TARGET_DISTRIBUTIONS["safety"].items():
        current_pct = (current_stats["safety"].get(safety, 0) / total) * 100
        # Flag if current is less than 50% of target OR more than 3% below (whichever catches more)
        if current_pct < target_pct * 0.5 or current_pct < target_pct - 3:
            gaps["safety"].append((safety, current_pct, target_pct))

    # Intent gaps
    gaps["intent"] = []
    for intent, target_pct in TARGET_DISTRIBUTIONS["intent"].items():
        current_pct = (current_stats["intent"].get(intent, 0) / total) * 100
        if current_pct < target_pct - 3:
            gaps["intent"].append((intent, current_pct, target_pct))
    gaps["intent"].sort(key=lambda x: x[1])

    # Ingress gaps
    gaps["ingress"] = []
    for ing, target_pct in TARGET_DISTRIBUTIONS["ingress"].items():
        current_pct = (current_stats["ingress"].get(ing, 0) / total) * 100
        if current_pct < target_pct - 2:
            gaps["ingress"].append((ing, current_pct, target_pct))
    gaps["ingress"].sort(key=lambda x: x[1])

    # Relations gaps
    rel_total = sum(current_stats["relations"].values()) or 1
    gaps["relations"] = []
    for rel, target_pct in TARGET_DISTRIBUTIONS["relations"].items():
        current_pct = (current_stats["relations"].get(rel, 0) / rel_total) * 100
        if current_pct < target_pct - 2:
            gaps["relations"].append((rel, current_pct, target_pct))
    gaps["relations"].sort(key=lambda x: x[1])

    # NER gaps
    ner_total = sum(current_stats["ner"].values()) or 1
    gaps["ner"] = []
    for ner, target_pct in TARGET_DISTRIBUTIONS["ner"].items():
        current_pct = (current_stats["ner"].get(ner, 0) / ner_total) * 100
        if current_pct < target_pct - 2:
            gaps["ner"].append((ner, current_pct, target_pct))
    gaps["ner"].sort(key=lambda x: x[1])

    # Temporal gaps
    temp_total = sum(current_stats["temporal"].values()) or 1
    gaps["temporal"] = []
    for temp, target_pct in TARGET_DISTRIBUTIONS["temporal"].items():
        current_pct = (current_stats["temporal"].get(temp, 0) / temp_total) * 100
        if current_pct < target_pct - 3:
            gaps["temporal"].append((temp, current_pct, target_pct))
    gaps["temporal"].sort(key=lambda x: x[1])

    return gaps


def generate_dynamic_worker_prompts(num_workers: int = 20) -> list[str]:
    """Generate worker prompts dynamically based on current distribution gaps.

    ALL workers get the SAME comprehensive gap-filling prompt.
    This ensures balanced generation regardless of worker count.
    """

    logger.debug("=" * 60)
    logger.debug("DYNAMIC PROMPT GENERATION")
    logger.debug("=" * 60)

    # Use cached stats to avoid re-reading all shards
    current_stats = get_cached_stats()
    total = max(current_stats["total"], 1)

    logger.debug("Calculating distribution gaps...")
    gaps = calculate_gaps(current_stats)

    # Build a SINGLE comprehensive prompt with all gaps
    # Every worker gets the same prompt = balanced generation

    prompt_sections = []

    # === SENTIMENT SECTION ===
    sentiment_gaps = gaps.get("sentiment", [])
    if sentiment_gaps:
        sent_lines = []
        for sent, current_pct, target_pct in sentiment_gaps:
            need = target_pct - current_pct
            sent_lines.append(f"  - {sent}: {current_pct:.1f}% (need +{need:.1f}%)")
        prompt_sections.append(
            f"""🎯 SENTIMENT (priority: HIGH)
{chr(10).join(sent_lines)}
Generate MORE: {', '.join([s[0] for s in sentiment_gaps])}

Emotion mapping:
- very_positive → joy, excitement, love, celebration, gratitude, pride
- positive → contentment, hope, relief, approval, admiration
- neutral → neutral, patience
- negative → annoyance, disappointment, frustration, worry
- very_negative → anger, fear, grief, overwhelmed, emptiness"""
        )

    # === EMOTIONS SECTION ===
    critical_emotions = gaps.get("emotions_critical", [])
    low_emotions = gaps.get("emotions_low", [])
    if critical_emotions or low_emotions:
        emo_lines = []
        for emo, pct in (critical_emotions + low_emotions)[:10]:
            emo_lines.append(f"  - {emo}: {pct:.1f}%")
        prompt_sections.append(
            f"""🎯 EMOTIONS (priority: MEDIUM)
{chr(10).join(emo_lines)}
Include these underrepresented emotions in your samples."""
        )

    # === RELATIONS SECTION ===
    rel_gaps = gaps.get("relations", [])
    if rel_gaps:
        rel_lines = [f"  - {r[0]}: {r[1]:.1f}%" for r in rel_gaps[:6]]
        prompt_sections.append(
            f"""🎯 RELATIONS (priority: MEDIUM)
{chr(10).join(rel_lines)}
Use extended family: dadi, nani, cousin, colleague, friend"""
        )

    # === INTENT SECTION ===
    intent_gaps = gaps.get("intent", [])
    if intent_gaps:
        int_lines = [f"  - {i[0]}: {i[1]:.1f}%" for i in intent_gaps[:4]]
        prompt_sections.append(
            f"""🎯 INTENT (priority: LOW)
{chr(10).join(int_lines)}"""
        )

    # === INGRESS SECTION ===
    ingress_gaps = gaps.get("ingress", [])
    if ingress_gaps:
        ing_lines = [f"  - {i[0]}: {i[1]:.1f}%" for i in ingress_gaps[:4]]
        prompt_sections.append(
            f"""🎯 INGRESS (priority: LOW)
{chr(10).join(ing_lines)}"""
        )

    # === NER SECTION ===
    ner_gaps = gaps.get("ner", [])
    if ner_gaps:
        ner_lines = [f"  - {n[0]}: {n[1]:.1f}%" for n in ner_gaps[:5]]
        prompt_sections.append(
            f"""🎯 NER ENTITIES (priority: LOW)
{chr(10).join(ner_lines)}
Examples: HEIRLOOM (ring), MILESTONE (first steps), NICKNAME, ROUTINE, TRADITION"""
        )

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
            _DYNAMIC_PROMPTS_CACHE = generate_dynamic_worker_prompts(20)

    prompt = _DYNAMIC_PROMPTS_CACHE[worker_id % len(_DYNAMIC_PROMPTS_CACHE)]
    return f"Generate exactly {num_samples} samples with this focus:\n\n{prompt}"


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


class SyntheticDataManager:
    """Thread-safe manager for synthetic output data with cross-run deduplication."""

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

        self.stats = {
            "total_samples": 0,
            "task_hub_samples": 0,
            "intent_distribution": Counter(),
            "ingress_distribution": Counter(),
            "emotions_distribution": Counter(),
            "safety_distribution": Counter(),
            "relations_distribution": Counter(),
            "ner_distribution": Counter(),
            "temporal_distribution": Counter(),
            "sentiment_distribution": Counter(),
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
                                text = sample.get("text", "").lower().strip()
                                sample_hash = hashlib.md5(text.encode()).hexdigest()
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
                text = sample.get("text", "").lower().strip()
                sample_hash = hashlib.md5(text.encode()).hexdigest()

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

                # Track comprehensive stats
                self.stats["total_samples"] += 1
                tasks = sample.get("tasks", {})

                if sample.get("hub_routing", {}).get("TASK", False):
                    self.stats["task_hub_samples"] += 1

                # Track all distributions for incremental stats updates
                intent = tasks.get("intent", "")
                ingress = tasks.get("ingress", "")
                sentiment = tasks.get("sentiment", "")
                safety = tasks.get("safety_familyos", "")

                self.stats["intent_distribution"][intent] += 1
                self.stats["ingress_distribution"][ingress] += 1
                self.stats["sentiment_distribution"][sentiment] += 1
                self.stats["safety_distribution"][safety] += 1

                for emotion in tasks.get("emotions", []):
                    self.stats["emotions_distribution"][emotion] += 1
                for rel in tasks.get("relations", []):
                    self.stats["relations_distribution"][rel.get("predicate", "")] += 1
                for ner in tasks.get("ner_family", []):
                    self.stats["ner_distribution"][ner.get("label", "")] += 1
                for temp in tasks.get("temporal", []):
                    self.stats["temporal_distribution"][temp.get("label", "")] += 1

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


# =============================================================================
# Validation & Parsing (Copied from unified_data_generator.py)
# =============================================================================

# Valid schemas
VALID_NER_LABELS = {
    "PERSON",
    "KINSHIP",
    "NICKNAME",
    "PET",
    "HOME_LOC",
    "FAMILY_EVENT",
    "ROUTINE",
    "TRADITION",
    "MILESTONE",
    "HEIRLOOM",
}
VALID_TEMPORAL_LABELS = {"DATE_ABS", "DATE_REL", "TIME", "DURATION", "FREQUENCY", "AGE"}
VALID_EMOTIONS = {
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "love",
    "disgust",
    "admiration",
    "amusement",
    "approval",
    "caring",
    "excitement",
    "gratitude",
    "optimism",
    "pride",
    "relief",
    "contentment",
    "hope",
    "tenderness",
    "annoyance",
    "disappointment",
    "disapproval",
    "embarrassment",
    "grief",
    "nervousness",
    "remorse",
    "frustration",
    "overwhelmed",
    "emptiness",
    "nostalgia",
    "protectiveness",
    "togetherness",
    "longing",
    "warmth",
    "playfulness",
    "celebration",
    "belonging",
    "parental_pride",
    "parental_guilt",
    "patience",
    "worry",
    "bittersweet",
    "homesickness",
}
# 5-class sentiment matching SENTIMENT_LABELS in labels.py
VALID_SENTIMENTS = {"very_negative", "negative", "neutral", "positive", "very_positive"}
VALID_SAFETY = {"GREEN", "AMBER", "RED", "CRISIS"}
VALID_INTENTS = {
    "log_memory",
    "query_memory",
    "set_reminder",
    "express_feeling",
    "seek_advice",
    "share_news",
    "reflect",
    "other",
}
VALID_INGRESS = {
    "DIARY",
    "TASK",
    "HEALTH",
    "FINANCE",
    "RELATIONSHIP",
    "WORK",
    "META",
    "MEMORY",
    "PLANNING",
    "CELEBRATION",
    "CONCERN",
    "GRATITUDE",
}
VALID_RELATIONS = {
    "no_relation",
    "parent_of",
    "child_of",
    "spouse_of",
    "sibling_of",
    "grandparent_of",
    "grandchild_of",
    "aunt_uncle_of",
    "niece_nephew_of",
    "cousin_of",
    "pet_of",
    "friend_of",
    "colleague_of",
    "lives_at",
    "owns",
}


def clean_and_validate_sample(sample: dict) -> tuple[bool, str]:
    """Clean and validate a synthetic sample - removes invalid labels."""
    if "text" not in sample or not sample["text"]:
        return False, "Missing text"

    if "tasks" not in sample:
        return False, "Missing tasks"

    tasks = sample["tasks"]

    # Validate intent
    intent = tasks.get("intent", "")
    if intent not in VALID_INTENTS:
        return False, f"Invalid intent: {intent}"

    # Validate ingress
    ingress = tasks.get("ingress", "")
    if ingress not in VALID_INGRESS:
        return False, f"Invalid ingress: {ingress}"

    # Validate hub_routing
    if "hub_routing" not in sample:
        return False, "Missing hub_routing"

    # CLEAN EMOTIONS - remove invalid ones
    if "emotions" in tasks:
        valid_emotions = [e for e in tasks["emotions"] if e in VALID_EMOTIONS]
        if not valid_emotions:
            valid_emotions = ["neutral"]
        tasks["emotions"] = valid_emotions

    # CLEAN NER - remove invalid labels
    if "ner_family" in tasks and tasks["ner_family"]:
        cleaned_ner = []
        for entity in tasks["ner_family"]:
            if entity.get("label") in VALID_NER_LABELS:
                cleaned_ner.append(entity)
        tasks["ner_family"] = cleaned_ner

    # CLEAN TEMPORAL - remove invalid labels
    if "temporal" in tasks and tasks["temporal"]:
        cleaned_temporal = []
        for temp in tasks["temporal"]:
            if temp.get("label") in VALID_TEMPORAL_LABELS:
                cleaned_temporal.append(temp)
        tasks["temporal"] = cleaned_temporal

    # CLEAN RELATIONS - remove invalid predicates
    if "relations" in tasks and tasks["relations"]:
        cleaned_relations = []
        for rel in tasks["relations"]:
            pred = rel.get("predicate", "")
            # Normalize common mistakes
            if pred == "nephew_niece_of":
                pred = "niece_nephew_of"
            elif pred == "owner_of":
                pred = "owns"
            elif pred == "family_of":
                continue  # Skip, too generic

            if pred in VALID_RELATIONS:
                rel["predicate"] = pred
                cleaned_relations.append(rel)
        tasks["relations"] = cleaned_relations

    # Validate sentiment - with smart normalization
    sentiment = tasks.get("sentiment", "")
    if sentiment not in VALID_SENTIMENTS:
        # Smart mapping for common LLM mistakes
        emotions = set(tasks.get("emotions", []))

        # Strong positive emotions → very_positive
        strong_positive = {"excitement", "joy", "love", "celebration", "gratitude", "pride"}
        # Strong negative emotions → very_negative
        strong_negative = {"anger", "fear", "disgust", "grief", "overwhelmed", "emptiness"}
        # Mild positive → positive
        mild_positive = {"contentment", "hope", "relief", "approval", "admiration", "amusement"}
        # Mild negative → negative
        mild_negative = {"annoyance", "disappointment", "frustration", "worry", "nervousness"}

        if emotions & strong_positive:
            tasks["sentiment"] = "very_positive"
        elif emotions & strong_negative:
            tasks["sentiment"] = "very_negative"
        elif emotions & mild_positive:
            tasks["sentiment"] = "positive"
        elif emotions & mild_negative:
            tasks["sentiment"] = "negative"
        else:
            tasks["sentiment"] = "neutral"

    # Validate safety
    safety = tasks.get("safety_familyos", "")
    if safety not in VALID_SAFETY:
        tasks["safety_familyos"] = "GREEN"

    return True, ""


def validate_sample(sample: dict) -> tuple[bool, str]:
    """Validate a synthetic sample (legacy - use clean_and_validate_sample)."""
    return clean_and_validate_sample(sample)


def parse_synthetic_response(response_text: str) -> list[dict]:
    """Parse JSONL from LLM response."""
    valid_samples = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue

        try:
            if line.startswith("{"):
                sample = json.loads(line)
            else:
                match = re.search(r"\{.*\}", line, re.DOTALL)
                if match:
                    sample = json.loads(match.group())
                else:
                    continue

            is_valid, error = validate_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")

        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue

    return valid_samples


# =============================================================================
# Synthetic Generator Agent
# =============================================================================


class SyntheticTaskGenerator:
    """Generate synthetic TASK-oriented samples."""

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
    ):
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests

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
                system_prompt=SYSTEM_PROMPT,
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

        self.output_manager = SyntheticDataManager()
        self.batch_counter = 0
        self.batch_lock = threading.Lock()

    def _get_next_batch_id(self) -> int:
        with self.batch_lock:
            batch_id = self.batch_counter
            self.batch_counter += 1
            return batch_id

    def _generate_batch(self, client, user_prompt: str) -> int:
        """Generate one batch of synthetic samples using worker-specific prompt."""
        batch_id = self._get_next_batch_id()

        try:
            response = client.generate(
                model=MODEL if hasattr(client, "api_key") else client.model_name,
                system_prompt=SYSTEM_PROMPT,
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
        """Run parallel synthetic generation."""
        start_time = datetime.now()
        samples_per_worker = target_samples // len(self.clients)

        logger.info("=" * 60)
        logger.info("SYNTHETIC TASK GENERATION")
        logger.info("=" * 60)
        logger.info(f"Target: {target_samples:,} samples")
        logger.info(f"Workers: {len(self.clients)}")
        logger.info(f"Per worker: {samples_per_worker:,} samples")
        logger.info("=" * 60)

        stats = {
            "start_time": start_time.isoformat(),
            "target_samples": target_samples,
            "generated_samples": 0,
            "task_hub_samples": 0,
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
        logger.info("GENERATION COMPLETE")
        logger.info(f"Total samples: {final_stats['total_samples']:,}")
        logger.info(
            f"TASK hub: {final_stats['task_hub_samples']:,} ({100*final_stats['task_hub_samples']/max(final_stats['total_samples'],1):.1f}%)"
        )
        logger.info(f"Duration: {stats['duration_minutes']:.1f} minutes")
        logger.info(f"{'='*60}")

        for client in self.clients:
            client.close()

        return stats


# =============================================================================
# CLI
# =============================================================================


def show_stats():
    """Show generation statistics."""
    if not OUTPUT_DIR.exists():
        print("No synthetic data generated yet.")
        return

    shards = list(OUTPUT_DIR.glob("shard_*.jsonl"))

    if not shards:
        print("No synthetic data found.")
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
    task_count = sum(1 for s in all_samples if s.get("hub_routing", {}).get("TASK", False))

    intent_dist = Counter(s.get("tasks", {}).get("intent", "") for s in all_samples)
    ingress_dist = Counter(s.get("tasks", {}).get("ingress", "") for s in all_samples)

    print("\n" + "=" * 60)
    print("SYNTHETIC TASK DATA STATISTICS")
    print("=" * 60)
    print(f"\nTotal samples: {total:,}")
    print(f"TASK hub samples: {task_count:,} ({100*task_count/total:.1f}%)")
    print(f"Number of shards: {len(shards)}")

    print("\n📋 Intent Distribution:")
    for intent, count in intent_dist.most_common():
        print(f"  {intent:20s} {count:6,} ({100*count/total:5.1f}%)")

    print("\n🎯 Ingress Distribution:")
    for ingress, count in ingress_dist.most_common():
        print(f"  {ingress:20s} {count:6,} ({100*count/total:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Synthetic TASK-Oriented Data Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic samples")
    gen_parser.add_argument(
        "--count", type=int, default=10000, help="Number of samples to generate"
    )
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=20, help="Samples per API call"
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

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

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

        logger.info(f"Speed settings: {args.speed} preset, delay={delay}s, rpm={rpm}")

        generator = SyntheticTaskGenerator(
            samples_per_request=args.samples_per_request,
            delay_between_requests=delay,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
        )

        stats = generator.run(target_samples=args.count)
        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "stats":
        show_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
