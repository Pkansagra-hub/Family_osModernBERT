"""
Synthetic TASK-Oriented Data Generator

Generates brand new synthetic samples focused on TASK hub activities.
Unlike unified_data_generator.py which enriches existing samples, this creates
completely new conversations from scratch.

Focus Areas:
- TASK hub activities (reminders, queries, planning)
- Underrepresented intents: set_reminder, query_memory, seek_advice
- Underrepresented ingress: TASK, PLANNING, HEALTH, WORK, FINANCE
- Underrepresented relations: grandchild_of, friend_of, cousin_of, colleague_of
- All NER entities with focus on: ROUTINE, HOME_LOC, TRADITION, PET, NICKNAME

Target: Generate 15,000-20,000 high-quality TASK-oriented samples

Usage:
    python synthetic_task_generator.py generate --count 15000 --vertex-ai --gcp-project <project>
    python synthetic_task_generator.py generate --count 5000  # OpenRouter (slower)
    python synthetic_task_generator.py stats
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
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
MODEL = "amazon/nova-2-lite-v1:free"

# GCP Vertex AI Configuration
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
USE_VERTEX_AI = os.environ.get("USE_VERTEX_AI", "false").lower() == "true"

# Rate limiting
REQUESTS_PER_MINUTE_PER_KEY = 10
REQUESTS_PER_DAY_PER_KEY = 900
DELAY_BETWEEN_REQUESTS = 6.0

# Paths
BASE_DIR = Path("D:/Modeling_studio")
OUTPUT_DIR = BASE_DIR / "data" / "familyos" / "unified" / "output_synthetic"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Processing settings
SAMPLES_PER_REQUEST = 20  # Generate 20 new samples per API call


# =============================================================================
# System Prompt - Synthetic Generation
# =============================================================================

SYSTEM_PROMPT = """You are an expert synthetic data generator for FamilyOS, a family-focused AI assistant. Your task is to generate REALISTIC, diverse family-related conversations from scratch.

## TASK: Generate Realistic Family Assistant Conversations

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

## GENERATION GUIDELINES

### Priority Intent Distribution (FOCUS ON TASK HUB):
- **set_reminder** (30%) - "Remind me to...", "Set alarm for...", "Don't forget to..."
- **query_memory** (20%) - "When did we...", "What time is...", "Where did I..."
- **seek_advice** (15%) - "Should I...", "How do I...", "What's the best way..."
- **reflect** (15%) - "I've been thinking about...", "Looking back on..."
- **other** (10%) - General queries and commands
- **log_memory** (5%) - Recording events
- **share_news** (3%) - Sharing updates
- **express_feeling** (2%) - Emotional expressions

### Priority Ingress Distribution:
- **TASK** (25%) - To-dos, reminders, commands
- **PLANNING** (20%) - Future events, schedules
- **HEALTH** (15%) - Medical, wellness, fitness
- **WORK** (10%) - Job, productivity, career
- **GRATITUDE** (10%) - Appreciation, thankfulness
- **FINANCE** (8%) - Money, bills, budgets
- **RELATIONSHIP** (7%) - Family dynamics
- **META** (5%) - System queries

### Include These Underrepresented Elements:
**Relations:** grandchild_of, friend_of, cousin_of, lives_at, colleague_of, niece_nephew_of
**NER:** ROUTINE, HOME_LOC, TRADITION, PET, NICKNAME, MILESTONE, HEIRLOOM
**Temporal:** Always include dates/times for task-oriented samples

### Diversity Requirements:
- Mix of Indian and Western family structures
- Use Indian kinship terms: dadi, nani, nana, dada, bhai, didi, chacha, masi, tau, bua
- Include pets with names
- Use nicknames (Bunny, Champ, Princess, etc.)
- Include family traditions (Sunday brunch, Diwali celebration, weekly calls)
- Include routines (morning walk, dinner time, school pickup)

---

## TASK SCHEMAS (USE EXACTLY THESE VALUES)

### 1. EMOTIONS (44 classes ONLY, multi-label)
🚨 **CRITICAL: Use ONLY these 44 emotions. DO NOT create new emotions like "responsibility", "planning", "curiosity", "concern", etc.**

**Core (8):** neutral, joy, sadness, anger, fear, surprise, love, disgust
**Positive (12):** admiration, amusement, approval, caring, excitement, gratitude, optimism, pride, relief, contentment, hope, tenderness
**Negative (10):** annoyance, disappointment, disapproval, embarrassment, grief, nervousness, remorse, frustration, overwhelmed, emptiness
**Family (14):** nostalgia, protectiveness, togetherness, longing, warmth, playfulness, celebration, belonging, parental_pride, parental_guilt, patience, worry, bittersweet, homesickness

**DO NOT USE:** responsibility, planning, curiosity, anticipation, fondness, concern, stress, discipline, health, routine, organization, memory, reflection, determination, guilt, uncertainty, care, anxiety, supportive, friendship, accomplishment, wellness, support, contemplation, indecision, professionalism, appreciation, prudence, etc.

### 2. SENTIMENT (4 ONLY)
positive, negative, neutral, mixed

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

**CRITICAL: Most samples should have TASK=true**

---

## EXAMPLES OF TASK-ORIENTED SAMPLES

### Example 1: Reminder with temporal
```json
{"id": "syn_00001", "text": "Remind me to call Nana tomorrow at 3pm to check on his doctor's appointment", "tasks": {"emotions": ["caring", "responsibility"], "sentiment": "neutral", "ner_family": [{"start": 18, "end": 22, "label": "KINSHIP", "token": "Nana"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "TASK", "relations": [], "temporal": [{"start": 23, "end": 31, "label": "DATE_REL", "token": "tomorrow"}, {"start": 35, "end": 38, "label": "TIME", "token": "3pm"}]}, "hub_routing": {"EMO": false, "REL": false, "MEM": false, "TASK": true}}
```

### Example 2: Query with grandchild relation
```json
{"id": "syn_00002", "text": "When is little Arjun's next checkup? His dadi wants to come along", "tasks": {"emotions": ["caring", "warmth"], "sentiment": "positive", "ner_family": [{"start": 14, "end": 19, "label": "PERSON", "token": "Arjun"}, {"start": 41, "end": 45, "label": "KINSHIP", "token": "dadi"}], "safety_familyos": "GREEN", "intent": "query_memory", "ingress": "HEALTH", "relations": [{"subject": "dadi", "predicate": "grandparent_of", "object": "Arjun"}], "temporal": [{"start": 24, "end": 28, "label": "DATE_REL", "token": "next"}]}, "hub_routing": {"EMO": true, "REL": true, "MEM": false, "TASK": true}}
```

### Example 3: Planning with routine
```json
{"id": "syn_00003", "text": "Set up weekly grocery shopping every Saturday morning with my cousin Priya", "tasks": {"emotions": ["neutral"], "sentiment": "neutral", "ner_family": [{"start": 8, "end": 29, "label": "ROUTINE", "token": "weekly grocery shopping"}, {"start": 63, "end": 69, "label": "KINSHIP", "token": "cousin"}, {"start": 70, "end": 75, "label": "PERSON", "token": "Priya"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "PLANNING", "relations": [{"subject": "user", "predicate": "cousin_of", "object": "Priya"}], "temporal": [{"start": 30, "end": 44, "label": "FREQUENCY", "token": "every Saturday"}, {"start": 45, "end": 52, "label": "TIME", "token": "morning"}]}, "hub_routing": {"EMO": false, "REL": true, "MEM": false, "TASK": true}}
```

### Example 4: Work-life balance advice
```json
{"id": "syn_00004", "text": "How do I balance work deadlines with spending quality time with my kids on weekends?", "tasks": {"emotions": ["worry", "guilt", "overwhelmed"], "sentiment": "mixed", "ner_family": [{"start": 65, "end": 69, "label": "KINSHIP", "token": "kids"}], "safety_familyos": "AMBER", "intent": "seek_advice", "ingress": "WORK", "relations": [], "temporal": [{"start": 77, "end": 85, "label": "FREQUENCY", "token": "weekends"}]}, "hub_routing": {"EMO": true, "REL": false, "MEM": false, "TASK": true}}
```

### Example 5: Pet care routine
```json
{"id": "syn_00005", "text": "Don't let me forget to take Max to the vet next Tuesday for his vaccination", "tasks": {"emotions": ["responsibility", "caring"], "sentiment": "neutral", "ner_family": [{"start": 28, "end": 31, "label": "PET", "token": "Max"}], "safety_familyos": "GREEN", "intent": "set_reminder", "ingress": "TASK", "relations": [], "temporal": [{"start": 47, "end": 58, "label": "DATE_REL", "token": "next Tuesday"}]}, "hub_routing": {"EMO": false, "REL": false, "MEM": false, "TASK": true}}
```

---

## IMPORTANT RULES

1. **Generate REALISTIC conversations** - Sound natural, like real users
2. **Focus on TASK hub** - 80% of samples should have TASK=true
3. **Include temporal expressions** - Dates, times, frequencies for task samples
4. **Use diverse names** - Mix Indian and Western names
5. **Include underrepresented elements** - grandchild_of, cousin_of, colleague_of, ROUTINE, TRADITION, PET
6. **Accurate character offsets** - Count carefully for NER and temporal
7. **Safety conscious** - Most should be GREEN, some AMBER (stress, health)
8. **Multi-label emotions** - 2-4 emotions per sample
9. **Indian English support** - Use kinship terms: dadi, nani, bhai, didi, etc.

---

## OUTPUT

Generate {num_samples} diverse, realistic TASK-oriented samples in JSONL format.
One complete JSON object per line. No markdown, no explanations.
Start output immediately:"""


def get_generation_prompt(num_samples: int, batch_id: int, focus_areas: dict = None) -> str:
    """Generate prompt for synthetic data generation."""
    focus_str = ""
    if focus_areas:
        focus_str = "\n\n## SPECIAL FOCUS FOR THIS BATCH:\n"
        if "intents" in focus_areas:
            focus_str += f"- Prioritize these intents: {', '.join(focus_areas['intents'])}\n"
        if "ingress" in focus_areas:
            focus_str += f"- Prioritize these domains: {', '.join(focus_areas['ingress'])}\n"
        if "relations" in focus_areas:
            focus_str += f"- Include these relations: {', '.join(focus_areas['relations'])}\n"
        if "ner" in focus_areas:
            focus_str += f"- Include these entities: {', '.join(focus_areas['ner'])}\n"

    prompt = SYSTEM_PROMPT.replace("{num_samples}", str(num_samples))
    return prompt + focus_str


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
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
                api_key=self.api_key,
            )
        else:
            self.client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location,
            )

        logger.info(f"[Vertex AI] Initialized for synthetic generation")

        # Create cache for system prompt
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
            logger.info(f"[Vertex AI] Created cache: {cached_content.name}")
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
            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        if self.cached_content_name:
            config = genai_types.GenerateContentConfig(
                cached_content=self.cached_content_name,
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                safety_settings=safety_settings,
            )
            contents = user_prompt
        else:
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
    """Thread-safe manager for synthetic output data."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, shard_size: int = 5000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()
        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

        self.stats = {
            "total_samples": 0,
            "task_hub_samples": 0,
            "intent_distribution": Counter(),
            "ingress_distribution": Counter(),
        }

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
                self.current_shard_count += 1

                # Track stats
                self.stats["total_samples"] += 1
                if sample.get("hub_routing", {}).get("TASK", False):
                    self.stats["task_hub_samples"] += 1

                intent = sample.get("tasks", {}).get("intent", "")
                ingress = sample.get("tasks", {}).get("ingress", "")
                self.stats["intent_distribution"][intent] += 1
                self.stats["ingress_distribution"][ingress] += 1

                added += 1

        return added

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
VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
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

    # Validate sentiment
    sentiment = tasks.get("sentiment", "")
    if sentiment not in VALID_SENTIMENTS:
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

            self.clients = [
                VertexAIClient(
                    project_id=project_id,
                    location=gcp_location or GCP_LOCATION,
                    model_name=vertex_model or VERTEX_MODEL,
                    key_id=i,
                    system_prompt=SYSTEM_PROMPT,
                    cache_ttl="86400s",
                )
                for i in range(num_parallel)
            ]
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

    def _generate_batch(self, client, focus_areas: dict = None) -> int:
        """Generate one batch of synthetic samples."""
        batch_id = self._get_next_batch_id()
        user_prompt = get_generation_prompt(self.samples_per_request, batch_id, focus_areas)

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
                f"[Key {client.key_id}] Batch {batch_id}: "
                f"Generated {len(samples)}, Added {added}. "
                f"Total: {self.output_manager.stats['total_samples']}"
            )

            return added

        except Exception as e:
            logger.error(f"[Key {client.key_id}] Batch {batch_id} failed: {e}")
            return 0

    def _worker(
        self,
        client,
        target_samples: int,
        stop_event: threading.Event,
        stats_queue: Queue,
    ) -> None:
        """Worker thread for one API key."""
        samples_generated = 0

        while not stop_event.is_set() and samples_generated < target_samples:
            try:
                # Rotate focus areas to ensure diversity
                focus_areas = None
                if samples_generated % 100 == 0:
                    focus_areas = {
                        "intents": ["set_reminder", "query_memory", "seek_advice"],
                        "ingress": ["TASK", "PLANNING", "HEALTH"],
                        "relations": ["grandchild_of", "cousin_of", "friend_of", "colleague_of"],
                        "ner": ["ROUTINE", "TRADITION", "PET", "NICKNAME"],
                    }

                added = self._generate_batch(client, focus_areas)
                samples_generated += added

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

        # Final stats
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

    manager = SyntheticDataManager()
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
                except:
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
    gen_parser.add_argument("--delay", type=float, default=6.0, help="Delay between requests")

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

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if args.command == "generate":
        generator = SyntheticTaskGenerator(
            samples_per_request=args.samples_per_request,
            delay_between_requests=args.delay,
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
