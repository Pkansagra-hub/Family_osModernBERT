"""
Unified Multi-Task Data Generator (LLM Enrichment)

Takes existing FamilyOS per-task data and enriches it into unified multi-task format
using LLM. Each sample gets all 8 task annotations + hub_routing.

Workflow:
1. Merge all train.jsonl from familyos folders → add _source_task field
2. Split into 6 parts (for 6 API keys)
3. Pass samples to LLM with full schema → LLM enriches with all tasks
4. Output: unified format ready for ModernBERT v3.3

Usage:
    python unified_data_generator.py prepare           # Merge and split data
    python unified_data_generator.py generate          # Start LLM enrichment
    python unified_data_generator.py generate --part 1 # Process specific part
    python unified_data_generator.py stats             # Show progress
    python unified_data_generator.py export            # Export final unified data
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

import httpx
from dotenv import load_dotenv

# Optional: Google Cloud Vertex AI
try:
    from google import genai
    from google.genai import types as genai_types

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
# Free models on OpenRouter (try in order)
# Options: "google/gemini-2.0-flash-exp:free", "meta-llama/llama-3.1-8b-instruct:free", "tngtech/deepseek-r1t-chimera:free"
MODEL = "qwen/qwen3-coder:free"

# GCP Vertex AI Configuration
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
# Vertex AI models - gemini-2.5-flash for best quality
# Pricing: ~$0.15/1M input, ~$0.60/1M output = ~$35 for 230K samples
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "false").lower() == "true"

# Rate limiting
REQUESTS_PER_MINUTE_PER_KEY = 10
REQUESTS_PER_DAY_PER_KEY = 900
DELAY_BETWEEN_REQUESTS = 6.0

# Paths
FAMILYOS_DIR = Path("D:/Modeling_studio/data/familyos")
UNIFIED_DIR = Path("D:/Modeling_studio/data/familyos/unified")
PARTS_DIR = UNIFIED_DIR / "parts"
OUTPUT_DIR = UNIFIED_DIR / "output"
PROGRESS_FILE = UNIFIED_DIR / "progress.json"
INCOMPLETE_FILE = UNIFIED_DIR / "incomplete_samples.jsonl"  # Track samples that failed

# Processing settings
SAMPLES_PER_REQUEST = 15  # Fewer samples for complex multi-task enrichment
NUM_PARTS = 6  # Split into 6 parts for 6 API keys


# =============================================================================
# Progress Tracker (Persistent State)
# =============================================================================


class ProgressTracker:
    """
    Tracks generation progress per API key with persistent storage.

    Each key processes its own part file (key 0 → part_1, key 1 → part_2, etc.)
    Progress is saved after each batch so we can resume exactly where we left off.

    State stored in progress.json:
    {
        "key_0": {"part": 1, "processed": 1500, "total": 38418, "last_update": "..."},
        "key_1": {"part": 2, "processed": 2300, "total": 38418, "last_update": "..."},
        ...
    }
    """

    def __init__(self, progress_file: Path = PROGRESS_FILE):
        self.progress_file = progress_file
        self.lock = threading.Lock()
        self.state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load progress from disk."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, encoding="utf-8") as f:
                    self.state = json.load(f)
                logger.info(f"Loaded progress from {self.progress_file}")
                for key_id, info in self.state.items():
                    logger.info(
                        f"  {key_id}: Part {info['part']}, "
                        f"{info['processed']}/{info['total']} processed"
                    )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load progress: {e}, starting fresh")
                self.state = {}
        else:
            logger.info("No existing progress file, starting fresh")

    def _save(self) -> None:
        """Save progress to disk."""
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)

    def initialize_key(self, key_id: int, part_id: int, total_samples: int) -> None:
        """Initialize progress for a key if not already tracked."""
        key_name = f"key_{key_id}"
        with self.lock:
            if key_name not in self.state:
                self.state[key_name] = {
                    "part": part_id,
                    "processed": 0,
                    "total": total_samples,
                    "successful": 0,
                    "failed": 0,
                    "last_update": datetime.now().isoformat(),
                }
                self._save()
                logger.info(f"[Key {key_id}] Initialized: Part {part_id}, {total_samples} samples")

    def get_processed_count(self, key_id: int) -> int:
        """Get how many samples this key has already processed."""
        key_name = f"key_{key_id}"
        with self.lock:
            return self.state.get(key_name, {}).get("processed", 0)

    def update_progress(
        self,
        key_id: int,
        processed_increment: int,
        successful_increment: int = 0,
        failed_increment: int = 0,
    ) -> None:
        """Update progress for a key after processing a batch."""
        key_name = f"key_{key_id}"
        with self.lock:
            if key_name in self.state:
                self.state[key_name]["processed"] += processed_increment
                self.state[key_name]["successful"] += successful_increment
                self.state[key_name]["failed"] += failed_increment
                self.state[key_name]["last_update"] = datetime.now().isoformat()
                self._save()

    def get_all_progress(self) -> dict:
        """Get progress for all keys."""
        with self.lock:
            return dict(self.state)

    def reset_key(self, key_id: int) -> None:
        """Reset progress for a specific key."""
        key_name = f"key_{key_id}"
        with self.lock:
            if key_name in self.state:
                del self.state[key_name]
                self._save()
                logger.info(f"[Key {key_id}] Progress reset")

    def reset_all(self) -> None:
        """Reset all progress."""
        with self.lock:
            self.state = {}
            if self.progress_file.exists():
                self.progress_file.unlink()
            logger.info("All progress reset")


# =============================================================================
# Task Schemas (from READMEs)
# =============================================================================

# Emotions (44 classes)
EMOTIONS = [
    # Core (8)
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "love",
    "disgust",
    # Positive (12)
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
    # Negative (10)
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
    # Family (14)
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
]

# Safety bands
SAFETY_BANDS = ["GREEN", "AMBER", "RED", "CRISIS"]

# Intents (8)
INTENTS = [
    "log_memory",
    "query_memory",
    "set_reminder",
    "express_feeling",
    "seek_advice",
    "share_news",
    "reflect",
    "other",
]

# Ingress domains (12)
INGRESS_DOMAINS = [
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
]

# Relations (15)
RELATIONS = [
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
]

# NER Family labels
NER_LABELS = [
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
]

# Temporal labels
TEMPORAL_LABELS = [
    "DATE_ABS",
    "DATE_REL",
    "TIME",
    "DURATION",
    "FREQUENCY",
    "AGE",
]

# Sentiments
SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

# =============================================================================
# System Prompt (Approved)
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for FamilyOS, a family-focused AI system. Your task is to take existing per-task annotated samples and enrich them into a unified multi-task format.

## INPUT FORMAT
You will receive samples that already have ONE task annotation. Each sample has a `_source_task` field indicating which task it came from.

## OUTPUT FORMAT
For each input sample, generate a complete multi-task annotation in this exact JSON format:

```json
{
  "id": "fam_XXXXX",
  "text": "<the original text>",
  "tasks": {
    "emotions": ["emotion1", "emotion2", ...],
    "sentiment": "positive" | "negative" | "neutral" | "mixed",
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

---

## TASK SCHEMAS (USE EXACTLY THESE VALUES)

### 1. EMOTIONS (44 classes, multi-label)
**Core (8):** neutral, joy, sadness, anger, fear, surprise, love, disgust
**Positive (12):** admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness
**Negative (10):** annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness
**Family (14):** nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

→ Most samples have 2-4 emotions. Use multi-label.

### 2. SENTIMENT (4 classes)
- `positive` - Happy, joyful, grateful content
- `negative` - Sad, angry, worried content
- `neutral` - Factual, no strong emotion
- `mixed` - Both positive and negative feelings (e.g., bittersweet)

### 3. NER_FAMILY (10 entity types ONLY - DO NOT USE ANY OTHER LABELS)
| Label | Description | Examples |
|-------|-------------|----------|
| PERSON | Named individuals | "John", "Sarah", "Emma" |
| KINSHIP | Family terms | "mom", "dad", "didi", "nana", "bhai", "chacha" |
| NICKNAME | Family nicknames | "Panda", "Bunny", "Sweetie" |
| PET | Pets | "Max", "Whiskers" |
| HOME_LOC | Home locations | "kitchen", "backyard", "Emma's room" |
| FAMILY_EVENT | Family occasions | "birthday party", "graduation" |
| ROUTINE | Regular activities | "school run", "dinner time" |
| TRADITION | Family rituals | "Sunday brunch", "Diwali celebration" |
| MILESTONE | Life events | "first steps", "graduation day" |
| HEIRLOOM | Sentimental objects | "grandma's necklace", "dad's watch" |

**CRITICAL**: Use ONLY these 10 labels. DO NOT use: TIME, DATE_ABS, DATE_REL, DURATION, FREQUENCY, AGE (those are TEMPORAL), RELATIONSHIP, FRIEND, COLLEAGUE (those are RELATIONS), or any other labels like OTHER, MEMORY, EMOTION, TASK, HEALTH, PLANNING.

→ Return character offsets (start, end) and the token text.

### 4. SAFETY_FAMILYOS (4 policy bands)
| Band | When to Use |
|------|-------------|
| GREEN | Safe, routine content |
| AMBER | Mild stress, frustration, health mentions |
| RED | Persistent sadness, isolation, hopelessness |
| CRISIS | Self-harm, suicide ideation, abuse disclosure |

### 5. INTENT (8 classes)
| Intent | Examples |
|--------|----------|
| log_memory | "Had dinner with family tonight" |
| query_memory | "What did we do last Sunday?" |
| set_reminder | "Remind me to call mom tomorrow" |
| express_feeling | "Feeling grateful today" |
| seek_advice | "What should I do about..." |
| share_news | "Guess what happened today!" |
| reflect | "Thinking about the past..." |
| other | General conversation |

### 6. INGRESS (12 domains)
| Domain | Examples |
|--------|----------|
| DIARY | Personal reflections |
| TASK | To-dos, reminders |
| HEALTH | Medical, wellness |
| FINANCE | Money, bills |
| RELATIONSHIP | Family dynamics |
| WORK | Job, career |
| META | System queries |
| MEMORY | Recalling past events |
| PLANNING | Future events |
| CELEBRATION | Achievements, milestones |
| CONCERN | Worries, anxieties |
| GRATITUDE | Appreciation |

### 7. RELATIONS (15 types)
| Relation | Description |
|----------|-------------|
| no_relation | No relationship |
| parent_of | X is parent of Y |
| child_of | X is child of Y |
| spouse_of | X is married to Y |
| sibling_of | X is sibling of Y |
| grandparent_of | X is grandparent of Y |
| grandchild_of | X is grandchild of Y |
| aunt_uncle_of | X is aunt/uncle of Y |
| niece_nephew_of | X is niece/nephew of Y |
| cousin_of | X is cousin of Y |
| pet_of | X is pet of Y |
| friend_of | X is friend of Y |
| colleague_of | X works with Y |
| lives_at | X lives at location Y |
| owns | X owns Y (heirloom) |

→ Only include if there are 2+ entities with a relationship.

### 8. TEMPORAL (6 types ONLY - DO NOT USE ANY OTHER LABELS)
| Type | Examples |
|------|----------|
| DATE_ABS | "January 15", "2024" |
| DATE_REL | "yesterday", "last week", "tomorrow" |
| TIME | "3pm", "morning", "at noon" |
| DURATION | "for 2 hours", "all day" |
| FREQUENCY | "every Sunday", "weekly" |
| AGE | "when she was 5", "at age 10" |

**CRITICAL**: Use ONLY these 6 labels. DO NOT use: ROUTINE, FAMILY_EVENT, MILESTONE, HOME_LOC, TRADITION, TASK, MEMORY, TEMPORAL_TYPE (those belong in NER_FAMILY or other tasks).

→ Return character offsets (start, end) and the token text.

---

## HUB ROUTING RULES

Determine which hub tokens should be trained based on content:

| Hub | Set TRUE when... |
|-----|------------------|
| EMO | Text expresses emotions, feelings, sentiment |
| REL | Text contains family relationships, entity pairs |
| MEM | Text is about memories, past events, nostalgia, milestones |
| TASK | Text is about tasks, reminders, queries, commands |

**Examples:**
- "Remember when Mom made curry for Dad's birthday" → EMO: true, REL: true, MEM: true, TASK: false
- "Remind me to call the dentist tomorrow" → EMO: false, REL: false, MEM: false, TASK: true
- "Feeling so grateful for my family today" → EMO: true, REL: true, MEM: false, TASK: false

---

## IMPORTANT RULES

1. **Keep the original `_source_task` annotation** - If the input had emotions, those are ground truth
2. **Infer missing tasks** - Add annotations for tasks not in the original
3. **Character offsets must be accurate** - Count characters carefully for NER and temporal
4. **STRICT LABEL ADHERENCE**:
   - NER: Use ONLY the 10 labels listed (PERSON, KINSHIP, NICKNAME, PET, HOME_LOC, FAMILY_EVENT, ROUTINE, TRADITION, MILESTONE, HEIRLOOM)
   - TEMPORAL: Use ONLY the 6 labels listed (DATE_ABS, DATE_REL, TIME, DURATION, FREQUENCY, AGE)
   - RELATIONS: Use ONLY the 15 predicate types listed (parent_of, child_of, spouse_of, sibling_of, etc.)
   - DO NOT create new labels, mix labels across tasks, or use generic labels like "OTHER", "other", "MEMORY", "RELATIONSHIP"
4. **Empty arrays are valid** - If no relations exist, use `"relations": []`
5. **Multi-label emotions** - Most texts have 2-4 emotions
6. **Indian English** - Recognize kinship terms: didi, bhai, nana, nani, dada, dadi, chacha, masi, etc.

---

## COMPLETE EXAMPLES

### Example 1: Memory with emotions
**Input:**
{"text": "Remember when Mom made her special curry for Dad's birthday last year? The whole house smelled amazing.", "_source_task": "emotions", "_emotions": ["nostalgia", "joy"]}

**Output:**
{"id": "fam_00001", "text": "Remember when Mom made her special curry for Dad's birthday last year? The whole house smelled amazing.", "tasks": {"emotions": ["nostalgia", "joy", "warmth", "love"], "sentiment": "positive", "ner_family": [{"start": 14, "end": 17, "label": "KINSHIP", "token": "Mom"}, {"start": 40, "end": 43, "label": "KINSHIP", "token": "Dad"}, {"start": 46, "end": 54, "label": "FAMILY_EVENT", "token": "birthday"}, {"start": 73, "end": 78, "label": "HOME_LOC", "token": "house"}], "safety_familyos": "GREEN", "intent": "reflect", "ingress": "MEMORY", "relations": [{"subject": "Mom", "predicate": "spouse_of", "object": "Dad"}], "temporal": [{"start": 55, "end": 64, "label": "DATE_REL", "token": "last year"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": true, "TASK": false}}

### Example 2: Task with temporal
**Input:**
{"text": "Remind me to call dadi tomorrow at 3pm", "_source_task": "intents", "_intent_id": 2}

**Output:**
{"id": "fam_00002", "text": "Remind me to call dadi tomorrow at 3pm", "tasks": {"emotions": ["neutral"], "sentiment": "neutral", "ner_family": [{"start": 18, "end": 22, "label": "KINSHIP", "token": "dadi"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "TASK", "relations": [], "temporal": [{"start": 23, "end": 31, "label": "DATE_REL", "token": "tomorrow"}, {"start": 35, "end": 38, "label": "TIME", "token": "3pm"}]}, "hub_routing": {"EMO": false, "REL": false, "MEM": false, "TASK": true}}

### Example 3: Safety concern
**Input:**
{"text": "I've been feeling really isolated since the kids moved away. Sometimes I wonder if anyone cares about me anymore.", "_source_task": "safety", "_safety": "RED"}

**Output:**
{"id": "fam_00003", "text": "I've been feeling really isolated since the kids moved away. Sometimes I wonder if anyone cares about me anymore.", "tasks": {"emotions": ["sadness", "loneliness", "emptiness", "longing"], "sentiment": "negative", "ner_family": [{"start": 43, "end": 47, "label": "KINSHIP", "token": "kids"}], "safety_familyos": "RED", "intent": "express_feeling", "ingress": "CONCERN", "relations": [], "temporal": []}, "hub_routing": {"EMO": true, "REL": false, "MEM": false, "TASK": false}}

### Example 4: Relationship with multiple entities
**Input:**
{"text": "Nana and nani are visiting from Delhi next week to see little Arjun take his first steps!", "_source_task": "relations", "_entity1": "Nana", "_entity2": "Arjun"}

**Output:**
{"id": "fam_00004", "text": "Nana and nani are visiting from Delhi next week to see little Arjun take his first steps!", "tasks": {"emotions": ["excitement", "joy", "love", "parental_pride"], "sentiment": "positive", "ner_family": [{"start": 0, "end": 4, "label": "KINSHIP", "token": "Nana"}, {"start": 9, "end": 13, "label": "KINSHIP", "token": "nani"}, {"start": 32, "end": 37, "label": "HOME_LOC", "token": "Delhi"}, {"start": 62, "end": 67, "label": "PERSON", "token": "Arjun"}, {"start": 78, "end": 89, "label": "MILESTONE", "token": "first steps"}], "safety_familyos": "GREEN", "intent": "share_news", "ingress": "CELEBRATION", "relations": [{"subject": "Nana", "predicate": "spouse_of", "object": "nani"}, {"subject": "Nana", "predicate": "grandparent_of", "object": "Arjun"}, {"subject": "nani", "predicate": "grandparent_of", "object": "Arjun"}], "temporal": [{"start": 38, "end": 47, "label": "DATE_REL", "token": "next week"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": false, "TASK": false}}

---

## OUTPUT

Generate unified JSONL for each sample. One JSON object per line. No markdown, no explanations."""


def get_user_prompt(samples: list[dict], batch_id: int) -> str:
    """Generate user prompt with samples to enrich."""
    samples_json = "\n".join(json.dumps(s, ensure_ascii=False) for s in samples)

    return f"""Enrich these {len(samples)} samples into unified multi-task format.

Each sample has `_source_task` indicating the original annotation type. Keep those annotations as ground truth, and infer the missing task annotations.

## INPUT SAMPLES (Batch {batch_id})

{samples_json}

---

## OUTPUT

Generate unified JSONL. One complete JSON object per line with ALL task fields. No markdown, no explanations.
Start output immediately:"""


# =============================================================================
# Data Preparation
# =============================================================================


def normalize_sample(sample: dict, source_task: str) -> dict:
    """Normalize sample format and add _source_task."""
    # Extract text from different formats
    if "text" in sample:
        text = sample["text"]
    elif "tokens" in sample:
        text = " ".join(sample["tokens"])
    else:
        return None

    if not text or len(text.strip()) < 5:
        return None

    # Build normalized sample
    normalized = {
        "text": text.strip(),
        "_source_task": source_task,
    }

    # Preserve original annotations based on source task
    if source_task == "emotions":
        normalized["_emotions"] = sample.get("emotions", [])
        normalized["_primary_emotion"] = sample.get("primary_emotion")
        normalized["_intensity"] = sample.get("intensity")

    elif source_task == "safety":
        label_map = {0: "GREEN", 1: "AMBER", 2: "RED", 3: "CRISIS"}
        normalized["_safety"] = label_map.get(sample.get("label"), "GREEN")

    elif source_task == "intents":
        normalized["_intent_id"] = sample.get("label")

    elif source_task == "ingress":
        normalized["_ingress_id"] = sample.get("label")

    elif source_task == "relations":
        normalized["_entity1"] = sample.get("entity1")
        normalized["_entity2"] = sample.get("entity2")
        normalized["_relation_id"] = sample.get("relation")

    elif source_task == "ner_family":
        normalized["_tokens"] = sample.get("tokens", [])
        normalized["_ner_tags"] = sample.get("ner_tags", [])

    elif source_task == "temporal":
        normalized["_tokens"] = sample.get("tokens", [])
        normalized["_temporal_tags"] = sample.get("temporal_tags", [])

    elif source_task == "embeddings":
        normalized["_cluster"] = sample.get("cluster")

    return normalized


def merge_all_data() -> list[dict]:
    """Merge all train.jsonl files with _source_task field."""
    all_samples = []

    tasks = [
        "emotions",
        "safety",
        "intents",
        "ingress",
        "relations",
        "ner_family",
        "temporal",
        "embeddings",
    ]

    for task in tasks:
        for tier in ["gold", "silver"]:
            train_file = FAMILYOS_DIR / task / tier / "train.jsonl"
            if not train_file.exists():
                continue

            logger.info(f"Loading {task}/{tier}/train.jsonl...")
            count = 0

            with open(train_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        normalized = normalize_sample(sample, task)
                        if normalized:
                            normalized["_tier"] = tier
                            all_samples.append(normalized)
                            count += 1
                    except json.JSONDecodeError:
                        continue

            logger.info(f"  Loaded {count} samples from {task}/{tier}")

    logger.info(f"Total merged: {len(all_samples)} samples")
    return all_samples


def split_into_parts(samples: list[dict], num_parts: int = NUM_PARTS) -> None:
    """Split samples into N parts for parallel processing."""
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Shuffle for diversity in each part
    random.shuffle(samples)

    part_size = len(samples) // num_parts

    for i in range(num_parts):
        start = i * part_size
        end = start + part_size if i < num_parts - 1 else len(samples)
        part_samples = samples[start:end]

        part_file = PARTS_DIR / f"part_{i+1}.jsonl"
        with open(part_file, "w", encoding="utf-8") as f:
            for sample in part_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"Created {part_file.name} with {len(part_samples)} samples")


def prepare_data() -> None:
    """Merge all data and split into parts."""
    logger.info("=" * 60)
    logger.info("STEP 1: Merging all FamilyOS data...")
    logger.info("=" * 60)

    all_samples = merge_all_data()

    logger.info("=" * 60)
    logger.info("STEP 2: Splitting into 6 parts...")
    logger.info("=" * 60)

    split_into_parts(all_samples)

    # Also save merged file
    merged_file = UNIFIED_DIR / "merged_all.jsonl"
    with open(merged_file, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Saved merged file: {merged_file}")
    logger.info("=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Parts ready in: {PARTS_DIR}")
    logger.info("=" * 60)


# =============================================================================
# Validation
# =============================================================================


def validate_unified_sample(sample: dict) -> tuple[bool, str]:
    """Validate a unified multi-task sample."""
    required_fields = ["id", "text", "tasks", "hub_routing"]

    for field in required_fields:
        if field not in sample:
            return False, f"Missing '{field}' field"

    tasks = sample.get("tasks", {})

    # Validate emotions
    emotions = tasks.get("emotions", [])
    if not isinstance(emotions, list):
        return False, "emotions must be a list"
    for e in emotions:
        if e not in EMOTIONS:
            return False, f"Invalid emotion: {e}"

    # Validate sentiment
    sentiment = tasks.get("sentiment")
    if sentiment and sentiment not in SENTIMENTS:
        return False, f"Invalid sentiment: {sentiment}"

    # Validate safety
    safety = tasks.get("safety_familyos")
    if safety and safety not in SAFETY_BANDS:
        return False, f"Invalid safety band: {safety}"

    # Validate intent
    intent = tasks.get("intent")
    if intent and intent not in INTENTS:
        return False, f"Invalid intent: {intent}"

    # Validate ingress
    ingress = tasks.get("ingress")
    if ingress and ingress not in INGRESS_DOMAINS:
        return False, f"Invalid ingress: {ingress}"

    # Validate hub_routing
    hub_routing = sample.get("hub_routing", {})
    for hub in ["EMO", "REL", "MEM", "TASK"]:
        if hub not in hub_routing:
            return False, f"Missing hub_routing.{hub}"
        if not isinstance(hub_routing[hub], bool):
            return False, f"hub_routing.{hub} must be boolean"

    return True, ""


def fix_ner_offsets(text: str, ner_list: list[dict]) -> list[dict]:
    """
    Fix NER character offsets by finding the actual token position in text.

    LLMs often get offsets wrong by 1-2 characters. This function:
    1. Takes the token text from the LLM output
    2. Finds the actual position in the original text
    3. Returns corrected offsets
    """
    if not ner_list or not text:
        return ner_list

    fixed = []
    text_lower = text.lower()
    used_positions = set()  # Track used positions to handle duplicates

    for ner in ner_list:
        token = ner.get("token", "")
        label = ner.get("label", "")

        if not token:
            continue

        token_lower = token.lower()

        # Try to find the token in the text
        search_start = 0
        found = False

        while search_start < len(text):
            pos = text_lower.find(token_lower, search_start)

            if pos == -1:
                # Try partial match (token might have extra chars)
                # E.g., "Dad's" vs "Dad"
                for i in range(len(text) - len(token_lower) + 1):
                    if text_lower[i : i + len(token_lower)] == token_lower:
                        if i not in used_positions:
                            pos = i
                            break
                if pos == -1:
                    break

            # Check if this position is already used
            if pos not in used_positions:
                used_positions.add(pos)
                # Get the actual token from text (preserve original case)
                actual_token = text[pos : pos + len(token)]
                fixed.append(
                    {
                        "start": pos,
                        "end": pos + len(token),
                        "label": label,
                        "token": actual_token,
                    }
                )
                found = True
                break

            search_start = pos + 1

        if not found:
            # Keep original if we can't find it (but log warning)
            logger.debug(f"Could not find token '{token}' in text: {text[:50]}...")
            # Still include it with original offsets
            fixed.append(ner)

    return fixed


def fix_temporal_offsets(text: str, temporal_list: list[dict]) -> list[dict]:
    """
    Fix temporal expression offsets similar to NER.
    """
    if not temporal_list or not text:
        return temporal_list

    fixed = []
    text_lower = text.lower()
    used_positions = set()

    for temp in temporal_list:
        token = temp.get("token", "")
        label = temp.get("label", "")

        if not token:
            continue

        token_lower = token.lower()
        pos = text_lower.find(token_lower)

        if pos != -1 and pos not in used_positions:
            used_positions.add(pos)
            actual_token = text[pos : pos + len(token)]
            fixed.append(
                {
                    "start": pos,
                    "end": pos + len(token),
                    "label": label,
                    "token": actual_token,
                }
            )
        else:
            # Keep original
            fixed.append(temp)

    return fixed


def postprocess_sample(sample: dict) -> dict:
    """
    Post-process a unified sample to fix common LLM errors:
    1. Fix NER character offsets
    2. Fix temporal character offsets
    3. Ensure consistent formatting
    """
    text = sample.get("text", "")
    tasks = sample.get("tasks", {})

    # Fix NER offsets
    if "ner_family" in tasks and tasks["ner_family"]:
        tasks["ner_family"] = fix_ner_offsets(text, tasks["ner_family"])

    # Fix temporal offsets
    if "temporal" in tasks and tasks["temporal"]:
        tasks["temporal"] = fix_temporal_offsets(text, tasks["temporal"])

    sample["tasks"] = tasks
    return sample


def parse_unified_response(response_text: str) -> list[dict]:
    """Parse JSONL from LLM response and fix offsets."""
    valid_samples = []

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue

        try:
            # Try to find JSON object
            if line.startswith("{"):
                sample = json.loads(line)
            else:
                # Try to extract JSON from line
                match = re.search(r"\{.*\}", line, re.DOTALL)
                if match:
                    sample = json.loads(match.group())
                else:
                    continue

            is_valid, error = validate_unified_sample(sample)
            if is_valid:
                # Post-process to fix NER/temporal offsets
                sample = postprocess_sample(sample)
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")

        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue

    return valid_samples


def compute_sample_hash(sample: dict) -> str:
    """Compute hash for deduplication."""
    text = sample.get("text", "").lower().strip()
    return hashlib.md5(text.encode()).hexdigest()


# =============================================================================
# OpenRouter Client (Reused from emotion_data_generator)
# =============================================================================


class ProxyManager:
    """Manages proxy rotation."""

    def __init__(self, proxies: list[str] | None = None):
        self.proxies = proxies or []
        self.current_index = 0
        self.lock = threading.Lock()
        self.failed_proxies: set[str] = set()

    def add_proxy(self, proxy: str) -> None:
        with self.lock:
            if proxy not in self.proxies:
                self.proxies.append(proxy)

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
            pass
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
            self.failed_proxies.clear()
            return self.proxies[0] if self.proxies else None

    def mark_failed(self, proxy: str) -> None:
        with self.lock:
            self.failed_proxies.add(proxy)

    def has_proxies(self) -> bool:
        return len(self.proxies) > 0


_proxy_manager: ProxyManager | None = None


def get_proxy_manager() -> ProxyManager:
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = ProxyManager()
    return _proxy_manager


class OpenRouterClient:
    """Client for OpenRouter API with rate limiting."""

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
            return True
        return False

    def _wait_for_rate_limit(self) -> None:
        with self.lock:
            now = datetime.now()
            if now >= self.daily_reset:
                self.daily_count = 0
                self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)

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
        temperature: float = 0.7,
        max_tokens: int = 16000,
    ) -> str:
        self._wait_for_rate_limit()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/familyos",
            "X-Title": "FamilyOS Unified Data Generator",
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
                    f"[Key {self.key_id}] API call successful "
                    f"(daily: {self.daily_count}/{self.requests_per_day})"
                )
                return content

            except httpx.HTTPStatusError as e:
                logger.error(f"[Key {self.key_id}] HTTP error: {e.response.status_code}")
                if e.response.status_code == 429:
                    if self.rotate_proxy():
                        continue
                    time.sleep(60)
                raise
            except httpx.ProxyError:
                if self.rotate_proxy():
                    continue
                raise

        raise RuntimeError(f"[Key {self.key_id}] Max retries exceeded")

    def close(self):
        self.client.close()


# =============================================================================
# Vertex AI Client (GCP - uses $300 credit)
# =============================================================================


class VertexAIClient:
    """
    Client for Google Cloud Vertex AI (Gemini models) using google-genai library.

    Costs with $300 GCP credit:
    - Gemini 2.5 Flash Lite: ~$0.10/1M input, ~$0.40/1M output
    - Gemini 2.5 Flash: ~$0.30/1M input, ~$2.50/1M output

    Uses EXPLICIT CACHING for system prompt to guarantee 90% discount.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash-lite",
        key_id: int = 0,
        api_key: str | None = None,
        system_prompt: str | None = None,  # For explicit caching
        cache_ttl: str = "86400s",  # 24 hours default
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
        self.total_cached_tokens = 0

        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("GOOGLE_CLOUD_API_KEY")

        # Initialize client with Vertex AI backend
        if self.api_key:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                api_key=self.api_key,
            )
            logger.info(f"[Vertex AI] Initialized with API key")
        else:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )
            logger.info(f"[Vertex AI] Initialized with ADC")

        logger.info(
            f"[Vertex AI] Config: project={project_id}, location={location}, model={model_name}"
        )

        # Create explicit cache for system prompt (90% discount guaranteed)
        self.cached_content_name = None
        if system_prompt:
            self._create_cache(system_prompt, cache_ttl)

    def _create_cache(self, system_prompt: str, ttl: str = "86400s") -> None:
        """Create explicit cache for system prompt."""
        try:
            cached_content = self.client.caches.create(
                model=self.model_name,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl=ttl,
                ),
            )
            self.cached_content_name = cached_content.name
            logger.info(
                f"[Vertex AI] Created cache: {cached_content.name}, "
                f"expires: {cached_content.expire_time}"
            )
        except Exception as e:
            logger.warning(f"[Vertex AI] Failed to create cache: {e}. Will use uncached requests.")
            self.cached_content_name = None

    def refresh_cache(self, system_prompt: str, ttl: str = "3600s") -> None:
        """Refresh/extend cache TTL."""
        if self.cached_content_name:
            try:
                self.client.caches.update(
                    name=self.cached_content_name,
                    config=genai_types.CreateCachedContentConfig(
                        system_instruction=system_prompt,
                        ttl=ttl,
                    ),
                )
                logger.info(f"[Vertex AI] Cache TTL extended by {ttl}")
            except Exception as e:
                logger.warning(f"[Vertex AI] Failed to refresh cache: {e}")
                # Try to create a new cache
                self._create_cache(system_prompt, ttl)

    def delete_cache(self) -> None:
        """Delete the cache when done."""
        if self.cached_content_name:
            try:
                self.client.caches.delete(name=self.cached_content_name)
                logger.info(f"[Vertex AI] Deleted cache: {self.cached_content_name}")
            except Exception as e:
                logger.warning(f"[Vertex AI] Failed to delete cache: {e}")

    def generate(
        self,
        model: str,  # Ignored, uses self.model_name
        system_prompt: str,  # Ignored if cache exists
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> str:
        """Generate response using Vertex AI Gemini model with explicit caching."""
        # Safety settings to disable content filtering
        safety_settings = [
            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        # Build config - use cached content if available
        if self.cached_content_name:
            config = genai_types.GenerateContentConfig(
                cached_content=self.cached_content_name,
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                safety_settings=safety_settings,
            )
            contents = user_prompt  # Only user prompt, system is in cache
        else:
            # Fallback: combine system + user prompt
            config = genai_types.GenerateContentConfig(
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
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        input_tokens = (
                            getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        )
                        output_tokens = (
                            getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                        )
                        cached_tokens = (
                            getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
                        )
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_cached_tokens += cached_tokens

                content = response.text

                # Log with cache info
                cache_info = (
                    f", cached: {self.total_cached_tokens}" if self.total_cached_tokens > 0 else ""
                )
                logger.info(
                    f"[Vertex AI] Request {self.request_count} successful "
                    f"(tokens: {self.total_input_tokens}/{self.total_output_tokens}{cache_info})"
                )

                return content

            except Exception as e:
                logger.error(f"[Vertex AI] Error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        raise RuntimeError("[Vertex AI] Max retries exceeded")

    def get_cost_estimate(self) -> dict:
        """Estimate cost based on token usage."""
        # Pricing per 1M tokens (Gemini 2.0 Flash)
        input_price = 0.10
        output_price = 0.40

        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }

    def close(self):
        """No cleanup needed for Vertex AI."""
        cost = self.get_cost_estimate()
        logger.info(
            f"[Vertex AI] Session complete. "
            f"Requests: {self.request_count}, "
            f"Estimated cost: ${cost['total_cost_usd']:.4f}"
        )


# =============================================================================
# Unified Data Manager
# =============================================================================


class UnifiedDataManager:
    """Thread-safe manager for unified output data."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        shard_size: int = 5000,
        enable_deduplication: bool = False,
    ):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()
        self.enable_deduplication = enable_deduplication

        # ALWAYS load existing hashes to prevent duplicates on resume
        # Even in enrichment mode, we don't want to re-generate samples we already created
        if enable_deduplication or self.output_dir.exists():
            existing_count = self._load_existing_hashes()
            if existing_count > 0:
                logger.info(
                    f"Checkpoint: Loaded {existing_count} existing samples - will skip duplicates"
                )
        else:
            logger.info("Starting fresh - no existing data found")

        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

        # Track task coverage
        self.task_counts: dict[str, int] = defaultdict(int)

    def _load_existing_hashes(self) -> int:
        """Load existing sample hashes to prevent duplicates on resume."""
        count = 0
        for shard_file in self.output_dir.glob("shard_*.jsonl"):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.seen_hashes.add(compute_sample_hash(sample))
                        count += 1
                    except json.JSONDecodeError:
                        continue
        return count

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
        skipped = 0

        with self.lock:
            for sample in samples:
                sample_hash = compute_sample_hash(sample)

                # ALWAYS check for duplicates (checkpoint resume protection)
                # This prevents re-generating samples when resuming from checkpoint
                if sample_hash in self.seen_hashes:
                    skipped += 1
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

                # Track hub routing
                hub_routing = sample.get("hub_routing", {})
                for hub, active in hub_routing.items():
                    if active:
                        self.task_counts[hub] += 1

                added += 1

        if skipped > 0:
            logger.debug(f"Skipped {skipped} duplicate samples (already in dataset)")

        return added

    def get_total_samples(self) -> int:
        with self.lock:
            return len(self.seen_hashes)

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "total_samples": len(self.seen_hashes),
                "num_shards": len(list(self.output_dir.glob("shard_*.jsonl"))),
                "hub_coverage": dict(self.task_counts),
            }


# =============================================================================
# Unified Generator Agent (With Progress Tracking)
# =============================================================================


class UnifiedDataGeneratorAgent:
    """
    Multi-key parallel unified data generator with persistent progress tracking.

    Supports two backends:
    1. OpenRouter (free tier with 6 API keys)
    2. GCP Vertex AI (uses $300 credit, faster and more reliable)

    Each API key/client is assigned to its own part file:
    - Key 0 → Part 1
    - Key 1 → Part 2
    - etc.

    Progress is saved after each batch, so generation can be stopped and resumed.
    """

    def __init__(
        self,
        api_keys: list[str] | None = None,
        samples_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
        use_vertex_ai: bool = False,
        gcp_project_id: str | None = None,
        gcp_location: str = "us-central1",
        vertex_model: str = "gemini-1.5-flash-002",
        num_parallel: int = 1,  # For Vertex AI: number of parallel workers
    ):
        self.use_vertex_ai = use_vertex_ai or USE_VERTEX_AI
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests

        if self.use_vertex_ai:
            # Use GCP Vertex AI with explicit caching
            project_id = gcp_project_id or GCP_PROJECT_ID
            if not project_id:
                raise ValueError("GCP_PROJECT_ID not set. Set via env var or --gcp-project")

            # Create clients with cached system prompt (90% discount)
            self.clients = [
                VertexAIClient(
                    project_id=project_id,
                    location=gcp_location or GCP_LOCATION,
                    model_name=vertex_model or VERTEX_MODEL,
                    key_id=i,
                    system_prompt=SYSTEM_PROMPT,  # Creates explicit cache
                    cache_ttl="86400s",  # 24 hours
                )
                for i in range(num_parallel)
            ]
            logger.info(f"Using Vertex AI with {num_parallel} parallel worker(s)")
            logger.info(f"  Model: {vertex_model}")
            logger.info(f"  Project: {project_id}")
            logger.info(f"  Explicit caching: ENABLED (90% discount on system prompt)")
        else:
            # Use OpenRouter
            api_keys = api_keys or OPENROUTER_API_KEYS
            self.api_keys = [k for k in api_keys if k and "REPLACE" not in k]
            if not self.api_keys:
                raise ValueError("No valid API keys provided!")
            self.clients = [
                OpenRouterClient(api_key=key, key_id=i) for i, key in enumerate(self.api_keys)
            ]
            logger.info(f"Using OpenRouter with {len(self.clients)} API keys")

        # Disable deduplication during enrichment - we want to allow re-enriching texts
        self.output_manager = UnifiedDataManager(enable_deduplication=False)
        self.progress_tracker = ProgressTracker()
        self.batch_counter = 0
        self.batch_lock = threading.Lock()

    def _get_next_batch_id(self) -> int:
        with self.batch_lock:
            batch_id = self.batch_counter
            self.batch_counter += 1
            return batch_id

    def _load_part_samples_with_offset(self, part_id: int, skip_count: int = 0) -> list[dict]:
        """
        Load samples from a part file, skipping already processed ones.

        Args:
            part_id: The part file to load (1-6)
            skip_count: Number of samples to skip (already processed)

        Returns:
            List of samples to process
        """
        part_file = PARTS_DIR / f"part_{part_id}.jsonl"
        samples = []
        total_in_file = 0

        if not part_file.exists():
            logger.error(f"Part file not found: {part_file}")
            return samples

        with open(part_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                total_in_file += 1
                # Skip already processed samples
                if idx < skip_count:
                    continue
                try:
                    sample = json.loads(line.strip())
                    # Add line number for tracking
                    sample["_line_idx"] = idx
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

        logger.info(
            f"Part {part_id}: Total {total_in_file}, "
            f"Skipped {skip_count}, Remaining {len(samples)}"
        )
        return samples, total_in_file

    def _save_incomplete_samples(self, samples: list[dict]) -> None:
        """Save samples that were not processed to incomplete file for retry."""
        if not samples:
            return
        INCOMPLETE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INCOMPLETE_FILE, "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.warning(f"Saved {len(samples)} incomplete samples to {INCOMPLETE_FILE}")

    def _find_missing_samples(
        self, input_samples: list[dict], output_samples: list[dict]
    ) -> list[dict]:
        """Find which input samples were not returned in output."""
        # Create a set of output texts (normalized)
        output_texts = set()
        for s in output_samples:
            text = s.get("text", "").strip().lower()
            if text:
                output_texts.add(text)

        # Find inputs not in outputs
        missing = []
        for s in input_samples:
            text = s.get("text", "").strip().lower()
            if text and text not in output_texts:
                missing.append(s)

        return missing

    def _process_batch(
        self,
        client: OpenRouterClient,
        samples: list[dict],
    ) -> tuple[int, int, list[dict]]:
        """
        Process a batch of samples using LLM.

        Returns:
            (added_count, processed_count, missing_samples)
        """
        batch_id = self._get_next_batch_id()

        # Remove internal tracking field before sending to LLM
        clean_samples = [{k: v for k, v in s.items() if k != "_line_idx"} for s in samples]
        user_prompt = get_user_prompt(clean_samples, batch_id)

        try:
            response = client.generate(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )

            unified_samples = parse_unified_response(response)
            added = self.output_manager.add_samples(unified_samples)

            # Check for incomplete batch - return missing samples for next batch
            missing = []
            if len(unified_samples) < len(samples):
                missing = self._find_missing_samples(clean_samples, unified_samples)
                if missing:
                    logger.warning(
                        f"[Key {client.key_id}] Batch {batch_id}: "
                        f"{len(missing)} samples missing, will retry in next batch"
                    )

            logger.info(
                f"[Key {client.key_id}] Batch {batch_id}: "
                f"Input {len(samples)}, Output {len(unified_samples)}, Added {added}. "
                f"Total: {self.output_manager.get_total_samples()}"
            )

            return added, len(samples), missing

        except Exception as e:
            logger.error(f"[Key {client.key_id}] Batch {batch_id} failed: {e}")
            # Return all samples as missing for retry
            return 0, len(samples), clean_samples

    def _worker_with_progress(
        self,
        client: OpenRouterClient,
        part_id: int,
        samples: list[dict],
        stop_event: threading.Event,
        stats_queue: Queue,
    ) -> None:
        """
        Worker thread for one API key with progress tracking.

        Args:
            client: The API client for this worker
            part_id: The part file this worker is processing
            samples: Pre-loaded samples for this worker
            stop_event: Signal to stop processing
            stats_queue: Queue for reporting stats to main thread
        """
        sample_idx = 0
        carryover_samples: list[dict] = []  # Missing samples from previous batch

        while not stop_event.is_set() and (sample_idx < len(samples) or carryover_samples):
            try:
                # Build batch: carryover samples first, then new samples
                batch = carryover_samples.copy()
                carryover_samples = []  # Clear after using

                # Fill remaining slots with new samples
                remaining_slots = self.samples_per_request - len(batch)
                if remaining_slots > 0 and sample_idx < len(samples):
                    batch_end = min(sample_idx + remaining_slots, len(samples))
                    batch.extend(samples[sample_idx:batch_end])
                    sample_idx = batch_end

                if not batch:
                    break

                added, processed, missing = self._process_batch(client, batch)

                # Carry missing samples to next batch
                if missing:
                    carryover_samples = missing

                # Update progress tracker (persisted to disk)
                self.progress_tracker.update_progress(
                    key_id=client.key_id,
                    processed_increment=processed - len(missing),  # Only count successful
                    successful_increment=added,
                    failed_increment=0 if added > 0 else processed,
                )

                # Report to main thread
                stats_queue.put(
                    {
                        "key_id": client.key_id,
                        "part_id": part_id,
                        "added": added,
                        "processed": processed - len(missing),
                        "requests": 1,
                        "errors": 0 if added > 0 else 1,
                    }
                )

                time.sleep(self.delay_between_requests)

            except RuntimeError as e:
                if "rate limit" in str(e).lower():
                    logger.warning(f"[Key {client.key_id}] Daily limit reached")
                    # Save any remaining carryover samples
                    if carryover_samples:
                        self._save_incomplete_samples(carryover_samples)
                    break
                raise
            except Exception as e:
                logger.error(f"[Key {client.key_id}] Worker error: {e}")
                stats_queue.put(
                    {
                        "key_id": client.key_id,
                        "part_id": part_id,
                        "added": 0,
                        "processed": 0,
                        "requests": 1,
                        "errors": 1,
                    }
                )
                time.sleep(self.delay_between_requests)

        # Save any remaining carryover samples at the end
        if carryover_samples:
            self._save_incomplete_samples(carryover_samples)
            logger.warning(
                f"[Key {client.key_id}] Saved {len(carryover_samples)} incomplete samples at end"
            )

        logger.info(f"[Key {client.key_id}] Finished Part {part_id}")

    def run(
        self,
        part_ids: list[int] | None = None,
        target_samples: int | None = None,
        max_requests: int | None = None,
    ) -> dict:
        """
        Run parallel generation with progress tracking.

        Each API key processes its assigned part file. Progress is saved
        after each batch, so generation can be safely stopped and resumed.
        """
        start_time = datetime.now()

        # Determine which parts to process
        # Default: Key 0 → Part 1, Key 1 → Part 2, etc.
        if part_ids:
            key_to_part = {i: part_ids[i % len(part_ids)] for i in range(len(self.clients))}
        else:
            key_to_part = {i: i + 1 for i in range(len(self.clients))}

        # Load samples for each key with progress offset
        key_samples: dict[int, list[dict]] = {}
        total_to_process = 0

        for key_id, client in enumerate(self.clients):
            part_id = key_to_part[key_id]

            # Get how many samples this key has already processed
            already_processed = self.progress_tracker.get_processed_count(key_id)

            # Load remaining samples
            samples, total_in_part = self._load_part_samples_with_offset(part_id, already_processed)
            key_samples[key_id] = samples
            total_to_process += len(samples)

            # Initialize progress tracker for this key
            self.progress_tracker.initialize_key(key_id, part_id, total_in_part)

        if total_to_process == 0:
            logger.info("No samples to process! All parts may be complete.")
            return {"status": "complete", "total_unified": self.output_manager.get_total_samples()}

        logger.info(f"Total samples to process: {total_to_process}")
        logger.info("=" * 60)
        logger.info("KEY ASSIGNMENTS:")
        for key_id in range(len(self.clients)):
            part_id = key_to_part[key_id]
            remaining = len(key_samples[key_id])
            progress = self.progress_tracker.get_all_progress().get(f"key_{key_id}", {})
            processed = progress.get("processed", 0)
            total = progress.get("total", 0)
            logger.info(
                f"  Key {key_id} → Part {part_id}: {processed}/{total} done, {remaining} remaining"
            )
        logger.info("=" * 60)

        stats = {
            "start_time": start_time.isoformat(),
            "num_keys": len(self.clients),
            "input_samples": total_to_process,
            "new_samples": 0,
            "processed": 0,
            "requests": 0,
            "errors": 0,
            "key_stats": {},
        }

        stop_event = threading.Event()
        stats_queue: Queue = Queue()

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [
                executor.submit(
                    self._worker_with_progress,
                    client,
                    key_to_part[key_id],
                    key_samples[key_id],
                    stop_event,
                    stats_queue,
                )
                for key_id, client in enumerate(self.clients)
            ]

            try:
                while True:
                    if target_samples and stats["new_samples"] >= target_samples:
                        logger.info(f"Reached target: {target_samples}")
                        break

                    if max_requests and stats["requests"] >= max_requests:
                        logger.info(f"Reached max requests: {max_requests}")
                        break

                    # Collect stats from queue
                    while not stats_queue.empty():
                        batch_stats = stats_queue.get_nowait()
                        stats["new_samples"] += batch_stats["added"]
                        stats["processed"] += batch_stats["processed"]
                        stats["requests"] += batch_stats["requests"]
                        stats["errors"] += batch_stats["errors"]

                        # Track per-key stats
                        key_id = batch_stats.get("key_id", 0)
                        if key_id not in stats["key_stats"]:
                            stats["key_stats"][key_id] = {"added": 0, "processed": 0, "errors": 0}
                        stats["key_stats"][key_id]["added"] += batch_stats["added"]
                        stats["key_stats"][key_id]["processed"] += batch_stats["processed"]
                        stats["key_stats"][key_id]["errors"] += batch_stats["errors"]

                    if all(f.done() for f in futures):
                        break

                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("INTERRUPTED BY USER - Progress saved!")
                logger.info("Run 'python unified_data_generator.py stats' to see progress")
                logger.info("Run 'python unified_data_generator.py generate' to resume")
                logger.info("=" * 60)
            finally:
                stop_event.set()

        for client in self.clients:
            client.close()

        stats["end_time"] = datetime.now().isoformat()
        stats["total_unified"] = self.output_manager.get_total_samples()
        stats["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60
        stats["hub_coverage"] = self.output_manager.get_stats()["hub_coverage"]
        stats["progress"] = self.progress_tracker.get_all_progress()

        logger.info(f"\n{'='*60}")
        logger.info("Session complete!")
        logger.info(f"Total unified samples: {stats['total_unified']}")
        logger.info(f"New samples this session: {stats['new_samples']}")
        logger.info(f"Duration: {stats['duration_minutes']:.1f} minutes")
        logger.info(f"{'='*60}")

        return stats


# =============================================================================
# Export Functions
# =============================================================================


def export_unified_data(output_file: Path, shuffle: bool = True, seed: int = 42) -> int:
    """Export unified data to single file."""
    if shuffle:
        random.seed(seed)

    samples = []

    for shard_path in sorted(OUTPUT_DIR.glob("shard_*.jsonl")):
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue

    if shuffle:
        random.shuffle(samples)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"Exported {len(samples)} unified samples to {output_file}")
    return len(samples)


def show_stats() -> None:
    """Show generation statistics including per-key progress."""
    print("\n" + "=" * 60)
    print("UNIFIED DATA GENERATION STATISTICS")
    print("=" * 60)

    # Parts stats
    print("\n📁 Source Parts:")
    for i in range(1, NUM_PARTS + 1):
        part_file = PARTS_DIR / f"part_{i}.jsonl"
        if part_file.exists():
            count = sum(1 for _ in open(part_file, encoding="utf-8"))
            print(f"   Part {i}: {count:,} samples")
        else:
            print(f"   Part {i}: Not created yet")

    # Progress tracker stats
    print("\n🔑 API Key Progress:")
    progress_tracker = ProgressTracker()
    progress = progress_tracker.get_all_progress()

    if progress:
        for key_name, info in sorted(progress.items()):
            part = info.get("part", "?")
            processed = info.get("processed", 0)
            total = info.get("total", 0)
            successful = info.get("successful", 0)
            failed = info.get("failed", 0)
            last_update = info.get("last_update", "N/A")

            pct = (processed / total * 100) if total > 0 else 0
            remaining = total - processed

            print(f"   {key_name} → Part {part}:")
            print(f"      Progress: {processed:,}/{total:,} ({pct:.1f}%)")
            print(f"      Remaining: {remaining:,}")
            print(f"      Success/Fail: {successful:,}/{failed:,}")
            print(f"      Last update: {last_update}")
    else:
        print("   No progress recorded yet. Run 'generate' to start.")

    # Output stats (enable deduplication to get accurate unique counts)
    print("\n📊 Unified Output:")
    manager = UnifiedDataManager(enable_deduplication=True)
    stats = manager.get_stats()

    print(f"   Total unified samples: {stats['total_samples']:,}")
    print(f"   Number of shards: {stats['num_shards']}")

    # Incomplete samples
    if INCOMPLETE_FILE.exists():
        incomplete_count = sum(1 for _ in open(INCOMPLETE_FILE, encoding="utf-8"))
        print(f"   Incomplete samples (for retry): {incomplete_count:,}")
        print(
            "   Run 'python unified_data_generator.py retry --vertex-ai --gcp-project <project>' to process them"
        )

    if stats["hub_coverage"]:
        print("\n🎯 Hub Routing Coverage:")
        for hub, count in sorted(stats["hub_coverage"].items()):
            print(f"   {hub}: {count:,}")

    # Resumption info
    print("\n💡 To resume generation:")
    print("   python unified_data_generator.py generate")
    print("\n💡 To reset progress and start over:")
    print("   python unified_data_generator.py reset")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified Multi-Task Data Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Prepare command
    subparsers.add_parser("prepare", help="Merge data and split into parts")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Start LLM enrichment")
    gen_parser.add_argument("--part", type=int, nargs="+", help="Specific part(s) to process")
    gen_parser.add_argument("--target", type=int, help="Target samples")
    gen_parser.add_argument("--max-requests", type=int, help="Max API requests")
    gen_parser.add_argument(
        "--samples-per-request", type=int, default=10, help="Samples per API call"
    )
    gen_parser.add_argument("--delay", type=float, default=6.0, help="Delay between requests")
    gen_parser.add_argument("--proxy", type=str, help="Proxy URL")
    gen_parser.add_argument("--proxy-file", type=str, help="File with proxy URLs")

    # GCP Vertex AI options
    gen_parser.add_argument(
        "--vertex-ai",
        action="store_true",
        help="Use GCP Vertex AI instead of OpenRouter (requires GCP auth)",
    )
    gen_parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help="GCP Project ID (or set GCP_PROJECT_ID env var)",
    )
    gen_parser.add_argument(
        "--gcp-location", type=str, default="us-central1", help="GCP region (default: us-central1)"
    )
    gen_parser.add_argument(
        "--vertex-model",
        type=str,
        default="gemini-2.5-flash",
        help="Vertex AI model (default: gemini-2.5-flash)",
    )
    gen_parser.add_argument(
        "--num-parallel",
        type=int,
        default=4,
        help="Number of parallel workers for Vertex AI (default: 4)",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset progress tracking")
    reset_parser.add_argument(
        "--key", type=int, help="Reset specific key (0-5), or all if not specified"
    )
    reset_parser.add_argument("--confirm", action="store_true", help="Confirm reset without prompt")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export unified data")
    export_parser.add_argument(
        "--output",
        type=str,
        default="D:/Modeling_studio/data/familyos/unified/family_training.jsonl",
        help="Output file",
    )
    export_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Retry command - process incomplete samples
    retry_parser = subparsers.add_parser("retry", help="Retry incomplete samples")
    retry_parser.add_argument("--vertex-ai", action="store_true", help="Use Vertex AI")
    retry_parser.add_argument("--gcp-project", type=str, help="GCP Project ID")
    retry_parser.add_argument("--vertex-model", type=str, default="gemini-2.5-flash")
    retry_parser.add_argument(
        "--clear", action="store_true", help="Clear incomplete file after retry"
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data()

    elif args.command == "generate":
        # Setup proxies (OpenRouter only)
        if not args.vertex_ai:
            if args.proxy:
                get_proxy_manager().add_proxy(args.proxy)
            if args.proxy_file:
                get_proxy_manager().add_proxies_from_file(args.proxy_file)

        agent = UnifiedDataGeneratorAgent(
            samples_per_request=args.samples_per_request,
            delay_between_requests=args.delay,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
        )

        stats = agent.run(
            part_ids=args.part,
            target_samples=args.target,
            max_requests=args.max_requests,
        )

        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "stats":
        show_stats()

    elif args.command == "reset":
        tracker = ProgressTracker()

        if args.key is not None:
            # Reset specific key
            if not args.confirm:
                confirm = input(f"Reset progress for key_{args.key}? (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return
            tracker.reset_key(args.key)
            print(f"Reset progress for key_{args.key}")
        else:
            # Reset all
            if not args.confirm:
                confirm = input("Reset ALL progress? This cannot be undone. (y/N): ")
                if confirm.lower() != "y":
                    print("Cancelled.")
                    return
            tracker.reset_all()
            print("Reset all progress.")

    elif args.command == "export":
        output = Path(args.output)
        count = export_unified_data(output, seed=args.seed)
        print(f"Exported {count} samples to {output}")

    elif args.command == "retry":
        # Load incomplete samples
        if not INCOMPLETE_FILE.exists():
            print("No incomplete samples to retry.")
            return

        incomplete_samples = []
        with open(INCOMPLETE_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        incomplete_samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not incomplete_samples:
            print("No incomplete samples to retry.")
            return

        print(f"Found {len(incomplete_samples)} incomplete samples to retry")

        # Initialize client
        if args.vertex_ai:
            client = VertexAIClient(
                project_id=args.gcp_project or GCP_PROJECT_ID,
                location=GCP_LOCATION,
                model_name=args.vertex_model,
                key_id=0,
            )
        else:
            if not OPENROUTER_API_KEYS:
                print("No API keys available")
                return
            client = OpenRouterClient(api_key=OPENROUTER_API_KEYS[0], key_id=0)

        # Process in batches (disable deduplication for retry)
        output_mgr = UnifiedDataManager(enable_deduplication=False)
        total_added = 0

        for i in range(0, len(incomplete_samples), 10):
            batch = incomplete_samples[i : i + 10]
            user_prompt = get_user_prompt(batch, i // 10)

            try:
                response = client.generate(
                    model=args.vertex_model if args.vertex_ai else MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                unified = parse_unified_response(response)
                added = output_mgr.add_samples(unified)
                total_added += added
                print(
                    f"Batch {i // 10 + 1}: Input {len(batch)}, Output {len(unified)}, Added {added}"
                )
            except Exception as e:
                print(f"Batch {i // 10 + 1} failed: {e}")

        print(f"\nRetry complete. Added {total_added} samples.")

        if args.clear and total_added > 0:
            INCOMPLETE_FILE.unlink()
            print("Cleared incomplete samples file.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
