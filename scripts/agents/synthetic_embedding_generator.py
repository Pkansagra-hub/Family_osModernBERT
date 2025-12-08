"""
Synthetic Embedding Triplet Data Generator

Generates synthetic embedding triplets (anchor, positive, negative) for contrastive learning.
Each triplet contains semantically similar anchor-positive pairs and hard negatives from different clusters.

Cluster Categories (30 family-centric clusters):
- Immediate Family: spouse_partner, my_children, my_parents, my_siblings
- Extended Family: grandparents, in_laws, extended_relatives
- Home & Daily Life: morning_routines, evening_family, household_chores, home_management, family_meals
- Milestones & Traditions: birthdays, festivals_traditions, weddings_ceremonies, life_milestones
- Family Responsibilities: family_finances, kids_education, family_health, legal_documents
- Work-Family Balance: work_family_balance, childcare
- Emotional & Relational: family_conflicts, family_bonding, grief_loss, gratitude_love
- Family Extensions: family_pets, long_distance, family_memories, family_planning

Target: Generate 200,000 high-quality embedding triplets with hard negatives

Usage:
    python synthetic_embedding_generator.py generate --count 200000 --vertex-ai --gcp-project <project>
    python synthetic_embedding_generator.py generate --count 10000  # OpenRouter (slower)
    python synthetic_embedding_generator.py stats
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
OUTPUT_DIR = BASE_DIR / "data" / "familyos" / "embeddings" / "silver_synthetic"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
HASH_INDEX_FILE = OUTPUT_DIR / "hash_index.jsonl"

# Processing settings
SAMPLES_PER_REQUEST = 15  # Generate 15 triplets per API call


# =============================================================================
# System Prompt - Embedding Triplet Generation
# =============================================================================

SYSTEM_PROMPT = """You are an expert synthetic data generator for FamilyOS embedding training. Your task is to generate high-quality TRIPLETS for contrastive learning.

## TASK: Generate Embedding Triplets

Each triplet contains:
- **anchor**: A realistic family-related message
- **positive**: A semantically SIMILAR message (paraphrase or closely related topic from SAME cluster)
- **negative**: A semantically DIFFERENT message (hard negative from a DIFFERENT cluster)

## OUTPUT FORMAT

Generate triplets in this exact JSON format (one per line, JSONL):

```json
{"anchor": "<message about family topic>", "positive": "<similar message, same cluster>", "negative": "<different topic, different cluster>", "anchor_cluster": "<cluster_name>", "negative_cluster": "<different_cluster_name>"}
```

## 30 FAMILY-CENTRIC CLUSTERS

### IMMEDIATE FAMILY CORE (4)
1. **spouse_partner** - Marriage, relationship with husband/wife, date nights, arguments, anniversaries, decisions together, intimacy
2. **my_children** - Parenting, kids' school, tantrums, proud moments, bedtime struggles, homework help, milestones
3. **my_parents** - Mom/Dad's health, visits to parents, their advice, caring for aging parents, their opinions
4. **my_siblings** - Brother/sister dynamics, childhood memories with siblings, rivalry, adult sibling friendship

### EXTENDED FAMILY (3)
5. **grandparents** - Dadi/Nani/Dada/Nana, their stories, health concerns, wisdom, visits, grandparent-grandchild bond
6. **in_laws** - Saas/Sasur relationships, boundaries, expectations, festivals together, mother-in-law dynamics
7. **extended_relatives** - Chacha/Chachi, Mama/Mami, cousins, family gatherings, WhatsApp groups, weddings

### HOME & DAILY LIFE (5)
8. **morning_routines** - Wake up chaos, school drop-off, breakfast rush, getting ready, morning prayers
9. **evening_family** - Dinner together, homework time, bedtime stories, winding down, family TV time
10. **household_chores** - Cleaning, laundry, dishes, division of labor, "whose turn is it", maid/help
11. **home_management** - Bills, repairs, groceries, organizing, home projects, appliances, utilities
12. **family_meals** - Cooking together, mom's recipes, picky eaters, special dishes, meal prep, diet

### MILESTONES & TRADITIONS (4)
13. **birthdays** - Birthday parties, gifts, cake, growing older, surprise celebrations, aging
14. **festivals_traditions** - Diwali, Holi, Christmas, Eid, Thanksgiving, Raksha Bandhan, family customs
15. **weddings_ceremonies** - Family weddings, planning, shaadi, relatives gathering, ceremonies, sangeet
16. **life_milestones** - First words, graduations, new jobs, retirements, births, deaths, anniversaries

### FAMILY RESPONSIBILITIES (4)
17. **family_finances** - Budget, saving for kids, parents' expenses, joint decisions, EMIs, investments
18. **kids_education** - School choice, tuition fees, grades, college planning, extracurriculars, homework
19. **family_health** - Doctor visits, medications, vaccinations, health scares, hospital, chronic conditions
20. **legal_documents** - Wills, insurance, property papers, inheritance, nominations, power of attorney

### WORK-FAMILY BALANCE (2)
21. **work_family_balance** - Missing kids' events, guilt, WFH, office deadlines vs family, career vs home
22. **childcare** - Daycare, nanny, creche, grandparents helping, school pickup, babysitting

### EMOTIONAL & RELATIONAL (4)
23. **family_conflicts** - Arguments, misunderstandings, silent treatment, making up, setting boundaries
24. **family_bonding** - Quality time, outings, games together, conversations, laughter, weekend fun
25. **grief_loss** - Losing family members, mourning, death anniversary, memories of departed, healing
26. **gratitude_love** - Expressing appreciation, thankfulness, love, recognition, heartfelt messages

### FAMILY EXTENSIONS (4)
27. **family_pets** - Dog/cat as family member, vet visits, walks, pet food, pet responsibilities
28. **long_distance** - Family abroad, video calls, missing them, time zone struggles, NRI life
29. **family_memories** - Old photos, "remember when", nostalgia, family albums, childhood stories
30. **family_planning** - Vacation planning, event organizing, future plans, family decisions, moving

---

## TRIPLET QUALITY RULES

### Positive (MUST be semantically similar):
- Paraphrase of anchor with different words
- Same topic/cluster, different phrasing
- Related sub-topic within same cluster
- Similar emotional context

### Negative (MUST be semantically different - HARD NEGATIVE):
- DIFFERENT cluster than anchor
- Still family-related (not random)
- Plausible but clearly different topic
- Should be challenging for the model to distinguish

---

## EXAMPLES

### Example 1: spouse_partner vs family_finances
```json
{"anchor": "Had a long talk with my husband about our future plans last night", "positive": "My spouse and I discussed where we want to be in five years", "negative": "Need to review the monthly budget before the credit card bill is due", "anchor_cluster": "spouse_partner", "negative_cluster": "family_finances"}
```

### Example 2: my_children vs grandparents
```json
{"anchor": "Arjun got his first gold star at school today, so proud!", "positive": "My son received an award in class and I couldn't be happier", "negative": "Dadi's arthritis is getting worse, need to schedule her doctor visit", "anchor_cluster": "my_children", "negative_cluster": "grandparents"}
```

### Example 3: morning_routines vs evening_family
```json
{"anchor": "The morning rush to get everyone ready for school is exhausting", "positive": "Waking up the kids and making breakfast before the school bus arrives is chaos", "negative": "Love our after-dinner walks in the park as a family", "anchor_cluster": "morning_routines", "negative_cluster": "evening_family"}
```

### Example 4: festivals_traditions vs grief_loss
```json
{"anchor": "Can't wait for Diwali celebrations with the whole family this year", "positive": "Planning the Lakshmi puja and sweets distribution for Deepavali", "negative": "It's been a year since Papa passed, still feels empty at home", "anchor_cluster": "festivals_traditions", "negative_cluster": "grief_loss"}
```

### Example 5: in_laws vs family_bonding
```json
{"anchor": "My mother-in-law keeps commenting on how I raise my kids", "positive": "Saas ji has opinions about everything from food to parenting", "negative": "Sunday board game nights with the kids are the best part of my week", "anchor_cluster": "in_laws", "negative_cluster": "family_bonding"}
```

### Example 6: family_health vs kids_education
```json
{"anchor": "Worried about Dad's blood pressure readings, they're too high", "positive": "Papa's health check showed elevated BP, need to monitor diet", "negative": "Researching the best coaching classes for Priya's board exams", "anchor_cluster": "family_health", "negative_cluster": "kids_education"}
```

---

## IMPORTANT RULES

1. **DIVERSE CLUSTERS** - Rotate through all 30 clusters evenly
2. **HARD NEGATIVES** - Negative must be from a DIFFERENT cluster (not random text)
3. **REALISTIC LANGUAGE** - Sound like real Indian/Western family conversations
4. **MIX NAMES** - Use Indian names (Arjun, Priya, Dadi) and Western names (Emma, John, Grandma)
5. **KINSHIP TERMS** - Use: dadi, nani, dada, nana, bhai, didi, chacha, mama, saas, sasur
6. **VARIED LENGTH** - Mix short (10 words) and longer (30 words) messages
7. **EMOTIONAL RANGE** - Include happy, worried, frustrated, nostalgic, grateful tones

---

## OUTPUT

Generate the requested number of triplets in JSONL format.
One complete JSON object per line. No markdown, no explanations.
Each line must have: anchor, positive, negative, anchor_cluster, negative_cluster
Start output immediately:"""


# =============================================================================
# Cluster Distribution Analyzer for Embedding Triplets
# =============================================================================

# 30 Family-Centric Clusters
VALID_CLUSTERS = {
    # Immediate Family Core (4)
    "spouse_partner",
    "my_children",
    "my_parents",
    "my_siblings",
    # Extended Family (3)
    "grandparents",
    "in_laws",
    "extended_relatives",
    # Home & Daily Life (5)
    "morning_routines",
    "evening_family",
    "household_chores",
    "home_management",
    "family_meals",
    # Milestones & Traditions (4)
    "birthdays",
    "festivals_traditions",
    "weddings_ceremonies",
    "life_milestones",
    # Family Responsibilities (4)
    "family_finances",
    "kids_education",
    "family_health",
    "legal_documents",
    # Work-Family Balance (2)
    "work_family_balance",
    "childcare",
    # Emotional & Relational (4)
    "family_conflicts",
    "family_bonding",
    "grief_loss",
    "gratitude_love",
    # Family Extensions (4)
    "family_pets",
    "long_distance",
    "family_memories",
    "family_planning",
}

# Target: uniform distribution across all 30 clusters (3.33% each)
TARGET_CLUSTER_PCT = 100.0 / len(VALID_CLUSTERS)  # ~3.33%


def load_current_cluster_distribution() -> dict:
    """Load current cluster distribution from existing embedding data."""
    from collections import Counter

    output_dirs = [
        Path("D:/Modeling_studio/data/familyos/embeddings/silver"),
        Path("D:/Modeling_studio/data/familyos/embeddings/gold"),
    ]

    stats = {
        "total": 0,
        "anchor_clusters": Counter(),
        "negative_clusters": Counter(),
    }

    for output_dir in output_dirs:
        if not output_dir.exists():
            continue
        for shard in output_dir.glob("*.jsonl"):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        stats["total"] += 1
                        anchor_cluster = sample.get("anchor_cluster", "")
                        negative_cluster = sample.get("negative_cluster", "")
                        if anchor_cluster:
                            stats["anchor_clusters"][anchor_cluster] += 1
                        if negative_cluster:
                            stats["negative_clusters"][negative_cluster] += 1
                    except (json.JSONDecodeError, KeyError):
                        pass

    return stats


def calculate_cluster_gaps(current_stats: dict) -> dict:
    """Calculate which clusters are underrepresented."""
    gaps = {"underrepresented": [], "overrepresented": []}
    total = max(current_stats["total"], 1)

    for cluster in VALID_CLUSTERS:
        anchor_count = current_stats["anchor_clusters"].get(cluster, 0)
        neg_count = current_stats["negative_clusters"].get(cluster, 0)
        combined_pct = ((anchor_count + neg_count) / (2 * total)) * 100

        if combined_pct < TARGET_CLUSTER_PCT - 1.0:
            gaps["underrepresented"].append((cluster, combined_pct))
        elif combined_pct > TARGET_CLUSTER_PCT + 2.0:
            gaps["overrepresented"].append((cluster, combined_pct))

    gaps["underrepresented"].sort(key=lambda x: x[1])
    gaps["overrepresented"].sort(key=lambda x: -x[1])

    return gaps


def generate_dynamic_worker_prompts(num_workers: int = 20) -> list[str]:
    """Generate worker prompts dynamically based on cluster distribution gaps.

    All workers get the same comprehensive prompt for balanced generation.
    """
    logger.debug("=" * 60)
    logger.debug("DYNAMIC PROMPT GENERATION FOR EMBEDDING TRIPLETS")
    logger.debug("=" * 60)

    # Load current stats
    current_stats = load_current_cluster_distribution()
    total = max(current_stats["total"], 1)

    logger.debug(f"Current total triplets: {total}")

    gaps = calculate_cluster_gaps(current_stats)

    # Build prompt sections
    prompt_sections = []

    underrep = gaps.get("underrepresented", [])
    if underrep:
        cluster_lines = [f"  - {c}: {pct:.1f}%" for c, pct in underrep[:10]]
        prompt_sections.append(
            f"""PRIORITY CLUSTERS (underrepresented - generate MORE):
{chr(10).join(cluster_lines)}

Focus on generating triplets with these clusters as anchors."""
        )

    overrep = gaps.get("overrepresented", [])
    if overrep:
        cluster_lines = [f"  - {c}: {pct:.1f}%" for c, pct in overrep[:5]]
        prompt_sections.append(
            f"""REDUCE CLUSTERS (overrepresented - generate LESS):
{chr(10).join(cluster_lines)}

Avoid over-generating these clusters."""
        )

    if prompt_sections:
        combined_prompt = """CLUSTER DISTRIBUTION GUIDANCE

""" + "\n\n".join(
            prompt_sections
        )
    else:
        combined_prompt = """BALANCED GENERATION

No major cluster gaps. Generate diverse triplets across all 30 clusters evenly.
Rotate through: immediate family, extended family, daily life, milestones,
responsibilities, work-life, emotions, and extensions."""

    # Log summary
    logger.info(f"Cluster gaps: underrep={len(underrep)}, overrep={len(overrep)}, total={total}")

    # All workers get same prompt
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
            _CURRENT_STATS_CACHE = load_current_cluster_distribution()
            logger.info(f"Loaded {_CURRENT_STATS_CACHE['total']:,} existing triplets (cached)")
        return _CURRENT_STATS_CACHE


def update_cached_stats(new_triplets: list[dict]) -> None:
    """Incrementally update cached stats from newly generated triplets."""
    global _CURRENT_STATS_CACHE
    with _STATS_LOCK:
        if _CURRENT_STATS_CACHE is not None:
            for triplet in new_triplets:
                _CURRENT_STATS_CACHE["total"] += 1
                anchor_cluster = triplet.get("anchor_cluster", "")
                negative_cluster = triplet.get("negative_cluster", "")
                if anchor_cluster:
                    _CURRENT_STATS_CACHE["anchor_clusters"][anchor_cluster] += 1
                if negative_cluster:
                    _CURRENT_STATS_CACHE["negative_clusters"][negative_cluster] += 1


def refresh_prompts_cache(num_workers: int = 20, force_reload_stats: bool = False) -> None:
    """Refresh the prompts cache (called by worker 0 during periodic refresh)."""
    global _DYNAMIC_PROMPTS_CACHE, _CURRENT_STATS_CACHE
    with _PROMPTS_LOCK:
        if force_reload_stats:
            with _STATS_LOCK:
                _CURRENT_STATS_CACHE = load_current_cluster_distribution()
                logger.info(f"Reloaded stats: {_CURRENT_STATS_CACHE['total']:,} triplets")
        _DYNAMIC_PROMPTS_CACHE = generate_dynamic_worker_prompts(num_workers)
        logger.info(f"Refreshed {num_workers} worker prompts based on current cluster gaps")


def get_worker_user_prompt(worker_id: int, num_triplets: int = 20) -> str:
    """Get the dynamically generated user prompt for a specific worker.

    The num_triplets parameter is the ONLY place that specifies how many triplets to generate.
    Dynamic prompts focus on WHAT clusters to generate, this function specifies HOW MANY.
    """
    global _DYNAMIC_PROMPTS_CACHE

    with _PROMPTS_LOCK:
        if _DYNAMIC_PROMPTS_CACHE is None:
            _DYNAMIC_PROMPTS_CACHE = generate_dynamic_worker_prompts(20)

    prompt = _DYNAMIC_PROMPTS_CACHE[worker_id % len(_DYNAMIC_PROMPTS_CACHE)]
    return f"Generate exactly {num_triplets} embedding triplets with this focus:\n\n{prompt}"


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
        max_tokens: int = 30000,
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
        max_tokens: int = 30000,
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
    """Thread-safe manager for embedding triplet output with cross-run deduplication."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, shard_size: int = 10000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()
        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_triplets(self.current_shard_id)

        # Load existing hashes for cross-run deduplication
        self._load_hash_index()

        self.stats = {
            "total_triplets": 0,
            "anchor_clusters": Counter(),
            "negative_clusters": Counter(),
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
            logger.info("Building hash index from existing triplets...")
            for shard in self.output_dir.glob("*.jsonl"):
                if shard.name == "hash_index.jsonl":
                    continue
                try:
                    with open(shard, encoding="utf-8") as f:
                        for line in f:
                            try:
                                triplet = json.loads(line.strip())
                                # Hash based on anchor text (primary identifier)
                                anchor = triplet.get("anchor", "").lower().strip()
                                triplet_hash = hashlib.md5(anchor.encode()).hexdigest()
                                self.seen_hashes.add(triplet_hash)
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
        return self.output_dir / f"triplets_{shard_id:04d}.jsonl"

    def _count_shard_triplets(self, shard_id: int) -> int:
        shard_path = self._get_shard_path(shard_id)
        if not shard_path.exists():
            return 0
        with open(shard_path, encoding="utf-8") as f:
            return sum(1 for _ in f)

    def _get_next_shard_id(self) -> int:
        existing = list(self.output_dir.glob("triplets_*.jsonl"))
        if not existing:
            return 0
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
        if self._count_shard_triplets(max_id) >= self.shard_size:
            return max_id + 1
        return max_id

    def add_triplets(self, triplets: list[dict]) -> int:
        """Add triplets to output, deduplicating by anchor text."""
        added = 0

        with self.lock:
            for triplet in triplets:
                anchor = triplet.get("anchor", "").lower().strip()
                triplet_hash = hashlib.md5(anchor.encode()).hexdigest()

                if triplet_hash in self.seen_hashes:
                    continue

                if self.current_shard_count >= self.shard_size:
                    self.current_shard_id += 1
                    self.current_shard_count = 0
                    logger.info(f"Started new shard: triplets_{self.current_shard_id:04d}")

                shard_path = self._get_shard_path(self.current_shard_id)
                with open(shard_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(triplet, ensure_ascii=False) + "\n")

                self.seen_hashes.add(triplet_hash)
                self._new_hashes.append(triplet_hash)
                self.current_shard_count += 1

                # Track stats
                self.stats["total_triplets"] += 1
                anchor_cluster = triplet.get("anchor_cluster", "")
                negative_cluster = triplet.get("negative_cluster", "")
                self.stats["anchor_clusters"][anchor_cluster] += 1
                self.stats["negative_clusters"][negative_cluster] += 1

                added += 1

                # Periodically save hashes
                if len(self._new_hashes) >= self._hash_save_threshold:
                    self._save_hash_index(self._new_hashes)
                    self._new_hashes = []

        return added

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self.lock:
            return {
                "total_triplets": self.stats["total_triplets"],
                "anchor_clusters": dict(self.stats["anchor_clusters"]),
                "negative_clusters": dict(self.stats["negative_clusters"]),
                "shard_count": self.current_shard_id + 1,
            }

    def flush_hashes(self) -> None:
        """Save any remaining hashes."""
        with self.lock:
            if self._new_hashes:
                self._save_hash_index(self._new_hashes)
                self._new_hashes = []


# =============================================================================
# Triplet Validation
# =============================================================================


def validate_triplet(triplet: dict) -> tuple[bool, str]:
    """Validate an embedding triplet has required fields and valid clusters."""
    # Check required fields
    required_fields = ["anchor", "positive", "negative", "anchor_cluster", "negative_cluster"]
    for field in required_fields:
        if field not in triplet or not triplet[field]:
            return False, f"Missing or empty field: {field}"

    # Check text fields are strings with reasonable length
    for text_field in ["anchor", "positive", "negative"]:
        text = triplet[text_field]
        if not isinstance(text, str):
            return False, f"{text_field} must be a string"
        if len(text) < 5:
            return False, f"{text_field} too short: {len(text)} chars"
        if len(text) > 500:
            return False, f"{text_field} too long: {len(text)} chars"

    # Validate clusters
    anchor_cluster = triplet["anchor_cluster"]
    negative_cluster = triplet["negative_cluster"]

    if anchor_cluster not in VALID_CLUSTERS:
        return False, f"Invalid anchor_cluster: {anchor_cluster}"
    if negative_cluster not in VALID_CLUSTERS:
        return False, f"Invalid negative_cluster: {negative_cluster}"

    # Ensure anchor and negative are from DIFFERENT clusters
    if anchor_cluster == negative_cluster:
        return False, f"anchor_cluster and negative_cluster must be different: {anchor_cluster}"

    return True, ""


def parse_triplet_response(response_text: str) -> list[dict]:
    """Parse JSONL triplets from LLM response."""
    valid_triplets = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue

        try:
            if line.startswith("{"):
                triplet = json.loads(line)
            else:
                match = re.search(r"\{.*\}", line, re.DOTALL)
                if match:
                    triplet = json.loads(match.group())
                else:
                    continue

            is_valid, error = validate_triplet(triplet)
            if is_valid:
                valid_triplets.append(triplet)
            else:
                logger.debug(f"Invalid triplet: {error}")

        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error: {e}")
            continue

    return valid_triplets


# =============================================================================
# Embedding Triplet Generator Agent
# =============================================================================


class EmbeddingTripletGenerator:
    """Generate synthetic embedding triplets for contrastive learning."""

    def __init__(
        self,
        api_keys: list[str] | None = None,
        triplets_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
        use_vertex_ai: bool = False,
        gcp_project_id: str | None = None,
        gcp_location: str = "us-central1",
        vertex_model: str = "gemini-2.5-flash",
        num_parallel: int = 1,
    ):
        self.triplets_per_request = triplets_per_request
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
        """Generate one batch of embedding triplets using worker-specific prompt."""
        batch_id = self._get_next_batch_id()

        try:
            response = client.generate(
                model=MODEL if hasattr(client, "api_key") else client.model_name,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.85,
            )

            triplets = parse_triplet_response(response)
            added = self.output_manager.add_triplets(triplets)

            # Update cached stats incrementally
            update_cached_stats(triplets)

            logger.info(
                f"[Worker {client.key_id}] Batch {batch_id}: "
                f"Generated {len(triplets)}, Added {added}. "
                f"Total: {self.output_manager.stats['total_triplets']}"
            )

            return added

        except Exception as e:
            logger.error(f"[Worker {client.key_id}] Batch {batch_id} failed: {e}")
            return 0

    def _worker(
        self,
        client,
        target_triplets: int,
        stop_event: threading.Event,
        stats_queue: Queue,
        refresh_interval: int = 333,
    ) -> None:
        """Worker thread with dedicated focus area based on worker_id.

        Args:
            refresh_interval: Re-calculate cluster gaps every N batches (default: 333 = ~5000 triplets)
        """
        worker_id = client.key_id
        triplets_generated = 0
        batch_count = 0

        # Get worker-specific prompt (dynamically generated based on cluster gaps)
        user_prompt = get_worker_user_prompt(worker_id, self.triplets_per_request)

        logger.debug(f"[Worker {worker_id}] Starting with focus: {user_prompt[:80]}...")

        while not stop_event.is_set() and triplets_generated < target_triplets:
            try:
                # Batch progress tracking every 100 batches
                if batch_count > 0 and batch_count % 100 == 0:
                    progress_pct = (triplets_generated / target_triplets) * 100
                    logger.info(
                        f"[Worker {worker_id}] PROGRESS: batch {batch_count}, "
                        f"{triplets_generated:,}/{target_triplets:,} triplets ({progress_pct:.1f}%)"
                    )

                # Periodic refresh: Worker 0 triggers re-calculation for all workers
                if worker_id == 0 and batch_count > 0 and batch_count % refresh_interval == 0:
                    logger.info(
                        f"PERIODIC REFRESH at batch {batch_count} - Re-analyzing cluster gaps..."
                    )
                    refresh_prompts_cache(20, force_reload_stats=True)

                # Get potentially updated prompt
                user_prompt = get_worker_user_prompt(worker_id, self.triplets_per_request)

                added = self._generate_batch(client, user_prompt)
                triplets_generated += added
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

        logger.info(f"[Key {client.key_id}] Finished. Generated {triplets_generated} triplets")

    def run(self, target_triplets: int = 10000) -> dict:
        """Run parallel embedding triplet generation."""
        start_time = datetime.now()
        triplets_per_worker = target_triplets // len(self.clients)

        logger.info("=" * 60)
        logger.info("EMBEDDING TRIPLET GENERATION")
        logger.info("=" * 60)
        logger.info(f"Target: {target_triplets:,} triplets")
        logger.info(f"Workers: {len(self.clients)}")
        logger.info(f"Per worker: {triplets_per_worker:,} triplets")
        logger.info("=" * 60)

        stats = {
            "start_time": start_time.isoformat(),
            "target_triplets": target_triplets,
            "generated_triplets": 0,
        }

        stop_event = threading.Event()
        stats_queue: Queue = Queue()

        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            futures = [
                executor.submit(
                    self._worker,
                    client,
                    triplets_per_worker,
                    stop_event,
                    stats_queue,
                )
                for client in self.clients
            ]

            try:
                while not all(f.done() for f in futures):
                    while not stats_queue.empty():
                        batch_stats = stats_queue.get_nowait()
                        stats["generated_triplets"] += batch_stats["added"]
                    time.sleep(1)

            except KeyboardInterrupt:
                logger.info("\n" + "=" * 60)
                logger.info("INTERRUPTED BY USER")
                logger.info("=" * 60)
                stop_event.set()

        # Final stats and cleanup
        final_stats = self.output_manager.get_stats()
        stats.update(final_stats)
        stats["end_time"] = datetime.now().isoformat()
        stats["duration_minutes"] = (datetime.now() - start_time).total_seconds() / 60

        logger.info(f"\n{'='*60}")
        logger.info("GENERATION COMPLETE")
        logger.info(f"Total triplets: {final_stats['total_triplets']:,}")
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
        print("No embedding triplets generated yet.")
        return

    shards = list(OUTPUT_DIR.glob("*.jsonl"))

    if not shards:
        print("No embedding triplets found.")
        return

    # Load all triplets for stats
    all_triplets = []
    for shard in sorted(shards):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    all_triplets.append(json.loads(line.strip()))
                except (json.JSONDecodeError, KeyError):
                    pass

    total = len(all_triplets)
    anchor_clusters = Counter(t.get("anchor_cluster", "") for t in all_triplets)
    negative_clusters = Counter(t.get("negative_cluster", "") for t in all_triplets)

    print("\n" + "=" * 60)
    print("EMBEDDING TRIPLET STATISTICS")
    print("=" * 60)
    print(f"\nTotal triplets: {total:,}")
    print(f"Number of shards: {len(shards)}")

    print("\nAnchor Cluster Distribution:")
    for cluster, count in anchor_clusters.most_common():
        print(f"  {cluster:25s} {count:6,} ({100*count/total:5.1f}%)")

    print("\nNegative Cluster Distribution:")
    for cluster, count in negative_clusters.most_common()[:10]:
        print(f"  {cluster:25s} {count:6,} ({100*count/total:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Embedding Triplet Generator for FamilyOS")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate embedding triplets")
    gen_parser.add_argument(
        "--count", type=int, default=10000, help="Number of triplets to generate"
    )
    gen_parser.add_argument(
        "--triplets-per-request", type=int, default=20, help="Triplets per API call"
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
        help="Re-analyze cluster gaps every N batches (default: 100)",
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

        generator = EmbeddingTripletGenerator(
            triplets_per_request=args.triplets_per_request,
            delay_between_requests=delay,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
        )

        stats = generator.run(target_triplets=args.count)
        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2, default=str))

    elif args.command == "stats":
        show_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
