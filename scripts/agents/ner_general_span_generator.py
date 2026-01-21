"""
NER General Span Data Generator (GlobalPointer Format)

Generates synthetic NER training data for ner_general task in span format.
Uses Google Vertex AI (Gemini) for generation.

Output format: {"text": "...", "entities": [{"start": 0, "end": 4, "label": "PER"}]}

Labels (4 types - CoNLL-2003 compatible):
- PER: Person names (Emma, John Smith, Dr. Sarah)
- ORG: Organizations (Lincoln School, Google, St. Mary's Hospital)
- LOC: Locations (New York, backyard, kitchen, Delhi)
- MISC: Other proper nouns (iPhone, Christmas, COVID-19, Diwali)

Usage:
    python ner_general_span_generator.py generate --samples 1000
    python ner_general_span_generator.py generate --samples 50000 --batch-size 20
    python ner_general_span_generator.py stats
    python ner_general_span_generator.py validate
"""

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

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
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Google AI Configuration (using API key, not Vertex AI)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Paths
DATA_DIR = Path("D:/Modeling_studio/data/ner_general_span")
OUTPUT_DIR = DATA_DIR / "familyos_synthetic"
PROGRESS_FILE = DATA_DIR / "generation_progress.json"
BATCH_INPUT_DIR = DATA_DIR / "batch_input"
BATCH_OUTPUT_DIR = DATA_DIR / "batch_output"

# Processing settings
SAMPLES_PER_REQUEST = 25  # Samples per API call
SHARD_SIZE = 10000  # Samples per output shard
NUM_WORKERS = 12  # Number of parallel workers

# Batch mode settings
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")  # e.g., "gs://my-bucket/ner-gen"
BATCH_SIZE = 500  # Number of requests per batch file

# NER General Labels (CoNLL-2003 compatible)
NER_GENERAL_LABELS = ["PER", "ORG", "LOC", "MISC"]

# =============================================================================
# Diverse Prompt Templates for Quality & Variety
# =============================================================================

FAMILY_CONTEXTS = [
    "a Western nuclear family (mom, dad, two kids)",
    "an Indian joint family (parents, grandparents, uncle, aunt, cousins)",
    "a mixed Western-Indian family",
    "a single-parent family with grandparents helping",
    "a multigenerational household",
    "a family celebrating Indian festivals (Diwali, Holi, Raksha Bandhan)",
    "a family celebrating Western holidays (Christmas, Thanksgiving, Easter)",
    "a family with young children (toddlers, babies)",
    "a family with teenagers going to school",
    "a professional family with working parents",
]

SCENARIOS = [
    "morning routines and breakfast",
    "school drop-off and activities",
    "dinner time conversations",
    "weekend family activities",
    "birthday party planning and celebration",
    "holiday celebrations and traditions",
    "doctor visits and health appointments",
    "shopping trips and errands",
    "vacation planning and travel",
    "work and career updates",
    "school events and parent-teacher meetings",
    "religious and cultural celebrations",
    "family reunions and gatherings",
    "home improvement and moving",
    "technology and gadget discussions",
]

LANGUAGE_STYLES = [
    "casual everyday speech",
    "diary or journal entries",
    "text message style (short, informal)",
    "calendar reminders and notes",
    "formal announcements",
    "photo captions and social media posts",
    "email-style family updates",
    "storytelling and narratives",
]

# =============================================================================
# System Prompt for ner_general Span Generation
# =============================================================================

SYSTEM_PROMPT = """You are an expert data annotator for Named Entity Recognition (NER) tasks.
Your job is to generate realistic, diverse English sentences that a family might say or write in daily life, and annotate them with SPAN-BASED entity labels.

## LABEL SCHEMA (4 types - CoNLL-2003 compatible)

| Label | Description | Examples |
|-------|-------------|----------|
| PER | Person names | "Emma", "John Smith", "Dr. Sarah Wilson", "Uncle Raj" |
| ORG | Organizations | "Lincoln School", "Google", "St. Mary's Hospital", "Mumbai University" |
| LOC | Locations | "New York", "kitchen", "Delhi", "Central Park", "India" |
| MISC | Other proper nouns | "iPhone", "Christmas", "Diwali", "COVID-19", "Toyota Camry" |

## OUTPUT FORMAT (SPAN-BASED - NOT BIO)

Output JSON lines with:
- "text": The complete sentence as a string
- "entities": Array of entity spans with:
  - "start": Character start position (0-indexed, inclusive)
  - "end": Character end position (exclusive)
  - "label": One of PER, ORG, LOC, MISC
  - "token": The actual text of the entity

Example output:
{"text": "Emma got accepted to Lincoln School today!", "entities": [{"start": 0, "end": 4, "label": "PER", "token": "Emma"}, {"start": 21, "end": 35, "label": "ORG", "token": "Lincoln School"}]}

## CRITICAL RULES

1. **Character offsets must be EXACT**:
   - Count characters carefully
   - "start" is the index of the first character
   - "end" is the index AFTER the last character (exclusive)
   - Verify: text[start:end] == token

2. **Multi-word entities are ONE span**:
   - "New York" is ONE LOC entity, not two
   - "Dr. John Smith" is ONE PER entity
   - "St. Mary's Hospital" is ONE ORG entity

3. **Include Indian English naturally**:
   - Mix Hindi/regional names: "Priya", "Arjun", "Rajesh"
   - Indian cities: "Mumbai", "Delhi", "Bangalore"
   - Indian festivals as MISC: "Diwali", "Holi", "Raksha Bandhan"
   - Indian organizations: "Tata Motors", "IIT Delhi", "Apollo Hospital"

4. **Entity distribution (per batch)**:
   - PER: 35-40% of entities (people are mentioned most)
   - LOC: 25-30% of entities (places, home locations)
   - ORG: 20-25% of entities (schools, companies, hospitals)
   - MISC: 10-15% of entities (products, events, diseases)

5. **Sentence variety**:
   - Simple sentences with 1-2 entities
   - Complex sentences with 3-5 entities
   - Multi-sentence examples (10-20%)
   - Include sentences with NO entities (5-10%)

## EXAMPLES

{"text": "Sarah picked up the kids from Lincoln Elementary this afternoon.", "entities": [{"start": 0, "end": 5, "label": "PER", "token": "Sarah"}, {"start": 30, "end": 48, "label": "ORG", "token": "Lincoln Elementary"}]}

{"text": "We're flying to Mumbai next week to visit Grandma at Apollo Hospital.", "entities": [{"start": 15, "end": 21, "label": "LOC", "token": "Mumbai"}, {"start": 42, "end": 49, "label": "PER", "token": "Grandma"}, {"start": 53, "end": 68, "label": "ORG", "token": "Apollo Hospital"}]}

{"text": "Arjun bought a new iPhone 15 from the Apple Store in Delhi.", "entities": [{"start": 0, "end": 5, "label": "PER", "token": "Arjun"}, {"start": 19, "end": 28, "label": "MISC", "token": "iPhone 15"}, {"start": 38, "end": 49, "label": "ORG", "token": "Apple Store"}, {"start": 53, "end": 58, "label": "LOC", "token": "Delhi"}]}

{"text": "The Diwali celebration at our house was wonderful this year.", "entities": [{"start": 4, "end": 10, "label": "MISC", "token": "Diwali"}]}

{"text": "Had a quiet evening at home today.", "entities": []}

## OUTPUT

Generate the requested number of samples in JSONL format.
Output ONLY valid JSON lines, no explanations or markdown."""


def get_user_prompt(num_samples: int, batch_id: int) -> str:
    """Generate diverse user prompts to avoid repetitive data."""
    context = FAMILY_CONTEXTS[batch_id % len(FAMILY_CONTEXTS)]
    scenario = SCENARIOS[batch_id % len(SCENARIOS)]
    style = LANGUAGE_STYLES[batch_id % len(LANGUAGE_STYLES)]

    return f"""Generate {num_samples} NER examples in SPAN format.

Context: {context}
Scenario: {scenario}
Style: {style}

Requirements:
- Use EXACT character offsets (verify text[start:end] == token)
- Include mix of PER, ORG, LOC, MISC entities
- Include some sentences with no entities
- Include multi-word entities (e.g., "New York", "Dr. Smith")

Output JSONL only (one JSON object per line):"""


# =============================================================================
# Span Validation
# =============================================================================


def validate_span_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single span-format NER sample."""
    if "text" not in sample:
        return False, "Missing 'text' key"

    if "entities" not in sample:
        return False, "Missing 'entities' key"

    text = sample["text"]
    entities = sample["entities"]

    if not isinstance(text, str) or len(text) == 0:
        return False, "text must be a non-empty string"

    if not isinstance(entities, list):
        return False, "entities must be a list"

    for i, entity in enumerate(entities):
        if not isinstance(entity, dict):
            return False, f"Entity {i} must be a dict"

        required_keys = {"start", "end", "label"}
        if not required_keys.issubset(entity.keys()):
            return False, f"Entity {i} missing keys: {required_keys - entity.keys()}"

        start = entity["start"]
        end = entity["end"]
        label = entity["label"]

        if not isinstance(start, int) or not isinstance(end, int):
            return False, f"Entity {i}: start/end must be integers"

        if start < 0 or end < 0:
            return False, f"Entity {i}: negative offset"

        if start >= end:
            return False, f"Entity {i}: start >= end"

        if end > len(text):
            return False, f"Entity {i}: end ({end}) > text length ({len(text)})"

        if label not in NER_GENERAL_LABELS:
            return False, f"Entity {i}: invalid label '{label}'"

        # Verify token matches text slice
        expected_token = text[start:end]
        if "token" in entity:
            if entity["token"] != expected_token:
                return False, f"Entity {i}: token mismatch - expected '{expected_token}', got '{entity['token']}'"

    # Check for overlapping entities
    sorted_entities = sorted(entities, key=lambda e: e["start"])
    for i in range(len(sorted_entities) - 1):
        if sorted_entities[i]["end"] > sorted_entities[i + 1]["start"]:
            return False, f"Overlapping entities at positions {i} and {i+1}"

    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    """Compute hash for deduplication."""
    text = sample.get("text", "").lower().strip()
    return hashlib.md5(text.encode()).hexdigest()


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response, handling various formatting issues."""
    valid_samples = []

    # Clean up response
    response_text = response_text.strip()

    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(
            line for line in lines
            if not line.startswith("```")
        )

    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue

        try:
            # Try to extract JSON object
            json_match = re.search(r'\{[^{}]*"text"[^{}]*"entities"[^{}]*\[.*?\][^{}]*\}', line, re.DOTALL)
            if json_match:
                sample = json.loads(json_match.group())
            else:
                sample = json.loads(line)

            # Add token field if missing
            if "entities" in sample and "text" in sample:
                for entity in sample["entities"]:
                    if "token" not in entity and "start" in entity and "end" in entity:
                        entity["token"] = sample["text"][entity["start"]:entity["end"]]

            is_valid, error = validate_span_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")

        except json.JSONDecodeError:
            continue

    return valid_samples


# =============================================================================
# Google AI Client (using API key with prompt caching)
# =============================================================================


class GoogleAIClient:
    """Client for Google AI (Gemini models) using API key."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        system_prompt: str | None = None,
        cache_ttl: int = 3600,  # seconds
    ):
        if not HAS_VERTEX_AI:
            raise ImportError("Google GenAI SDK not installed. Run: pip install google-genai")

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.lock = threading.Lock()
        self.request_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0

        # Get API key from environment
        api_key = GOOGLE_API_KEY
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")

        # Initialize client with API key (NOT Vertex AI)
        self.client = genai.Client(api_key=api_key)
        logger.info(f"[Google AI] Initialized with API key")
        logger.info(f"[Google AI] Model: {model_name}")

        # Create cache for system prompt (prompt caching for cost savings)
        self.cached_content_name = None
        if system_prompt:
            self._create_cache(system_prompt, cache_ttl)

    def _create_cache(self, system_prompt: str, ttl: int = 3600) -> None:
        """Create explicit cache for system prompt (75%% discount on cached tokens)."""
        try:
            # Note: Caching requires minimum 32K tokens for system prompt
            # For smaller prompts, we'll use inline system_instruction instead
            cached_content = self.client.caches.create(
                model=self.model_name,
                config=genai_types.CreateCachedContentConfig(
                    system_instruction=system_prompt,
                    ttl=f"{ttl}s",
                ),
            )
            self.cached_content_name = cached_content.name
            logger.info(f"[Google AI] Created cache: {cached_content.name}")
        except Exception as e:
            logger.warning(f"[Google AI] Cache creation failed (will use inline): {e}")
            self.cached_content_name = None

    def generate(
        self,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 8192,
    ) -> str:
        """Generate response using Google AI Gemini model."""
        # Safety settings - allow all content for data generation
        safety_settings = [
            genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]

        if self.cached_content_name:
            # Use cached content
            config = genai_types.GenerateContentConfig(
                cached_content=self.cached_content_name,
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                safety_settings=safety_settings,
            )
        else:
            # Use inline system instruction
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                max_output_tokens=max_tokens,
                system_instruction=self.system_prompt or SYSTEM_PROMPT,
                safety_settings=safety_settings,
            )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )

                with self.lock:
                    self.request_count += 1
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                        cached_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_cached_tokens += cached_tokens

                logger.debug(f"[Google AI] Request {self.request_count} successful")
                return response.text

            except Exception as e:
                logger.error(f"[Google AI] Error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise

        raise RuntimeError("[Google AI] Max retries exceeded")

    def delete_cache(self) -> None:
        """Delete the cache when done."""
        if self.cached_content_name:
            try:
                self.client.caches.delete(name=self.cached_content_name)
                logger.info(f"[Google AI] Deleted cache")
            except Exception as e:
                logger.warning(f"[Google AI] Failed to delete cache: {e}")

    def get_cost_estimate(self) -> dict:
        """Estimate cost based on token usage (Gemini 2.5 Flash pricing)."""
        # Gemini 2.5 Flash pricing (per 1M tokens)
        input_price = 0.15  # $0.15/1M input tokens
        output_price = 0.60  # $0.60/1M output tokens
        cached_price = 0.0375  # 75%% discount on cached tokens

        input_cost = (self.total_input_tokens / 1_000_000) * input_price
        output_cost = (self.total_output_tokens / 1_000_000) * output_price
        cached_savings = (self.total_cached_tokens / 1_000_000) * (input_price - cached_price)

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cached_tokens": self.total_cached_tokens,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "cached_savings_usd": round(cached_savings, 4),
            "total_cost_usd": round(input_cost + output_cost - cached_savings, 4),
        }


# =============================================================================
# Data Manager
# =============================================================================


class SpanDataManager:
    """Manages span-format NER data with sharding and deduplication."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        shard_size: int = SHARD_SIZE,
    ):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.seen_hashes: set[str] = set()

        # Load existing hashes
        existing_count = self._load_existing_hashes()
        if existing_count > 0:
            logger.info(f"Loaded {existing_count} existing samples for deduplication")

        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)

        # Track label distribution
        self.label_counts: dict[str, int] = defaultdict(int)

    def _load_existing_hashes(self) -> int:
        """Load existing sample hashes."""
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

    def add_samples(self, samples: list[dict]) -> tuple[int, int]:
        """Add samples, return (added, skipped) counts."""
        added = 0
        skipped = 0

        with self.lock:
            for sample in samples:
                sample_hash = compute_sample_hash(sample)

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
                added += 1

                # Track labels
                for entity in sample.get("entities", []):
                    self.label_counts[entity.get("label", "UNKNOWN")] += 1

        return added, skipped

    def get_total_samples(self) -> int:
        return len(self.seen_hashes)

    def get_stats(self) -> dict:
        """Get generation statistics."""
        total = 0
        label_counts: dict[str, int] = defaultdict(int)
        samples_with_entities = 0
        total_entities = 0

        for shard_file in sorted(self.output_dir.glob("shard_*.jsonl")):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        total += 1
                        entities = sample.get("entities", [])
                        if entities:
                            samples_with_entities += 1
                            total_entities += len(entities)
                            for entity in entities:
                                label_counts[entity.get("label", "UNKNOWN")] += 1
                    except json.JSONDecodeError:
                        continue

        return {
            "total_samples": total,
            "samples_with_entities": samples_with_entities,
            "samples_without_entities": total - samples_with_entities,
            "total_entities": total_entities,
            "avg_entities_per_sample": round(total_entities / max(total, 1), 2),
            "label_distribution": dict(label_counts),
            "shard_count": len(list(self.output_dir.glob("shard_*.jsonl"))),
        }


# =============================================================================
# Progress Tracker
# =============================================================================


class ProgressTracker:
    """Tracks generation progress with persistence."""

    def __init__(self, progress_file: Path = PROGRESS_FILE):
        self.progress_file = progress_file
        self.lock = threading.Lock()
        self.state: dict = {}
        self._load()

    def _load(self) -> None:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, encoding="utf-8") as f:
                    self.state = json.load(f)
                logger.info(f"Loaded progress: {self.state.get('generated', 0)} samples")
            except (json.JSONDecodeError, KeyError):
                self.state = {}

    def _save(self) -> None:
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, default=str)

    def update(self, generated: int, target: int) -> None:
        with self.lock:
            self.state["generated"] = generated
            self.state["target"] = target
            self.state["last_update"] = datetime.now().isoformat()
            self._save()

    def get_generated(self) -> int:
        return self.state.get("generated", 0)


# =============================================================================
# Generator
# =============================================================================


def worker_generate(
    worker_id: int,
    batch_queue: "queue.Queue",
    result_queue: "queue.Queue",
    stop_event: threading.Event,
) -> None:
    """Worker thread for parallel generation."""
    import queue as queue_module

    # Each worker gets its own client (no caching to avoid conflicts)
    client = genai.Client(api_key=GOOGLE_API_KEY)

    safety_settings = [
        genai_types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
        genai_types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ]

    config = genai_types.GenerateContentConfig(
        temperature=0.8,
        top_p=0.95,
        max_output_tokens=8192,
        system_instruction=SYSTEM_PROMPT,
        safety_settings=safety_settings,
    )

    while not stop_event.is_set():
        try:
            batch_id, batch_size = batch_queue.get(timeout=1)
        except queue_module.Empty:
            continue

        user_prompt = get_user_prompt(batch_size, batch_id)

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=config,
            )
            samples = parse_jsonl_response(response.text)
            result_queue.put((worker_id, batch_id, samples, None))
        except Exception as e:
            result_queue.put((worker_id, batch_id, [], str(e)))

        batch_queue.task_done()


def generate_samples_parallel(
    target_samples: int = 100000,
    samples_per_request: int = SAMPLES_PER_REQUEST,
    num_workers: int = NUM_WORKERS,
) -> None:
    """Generate span-format NER samples using parallel workers."""
    import queue

    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return

    logger.info(f"Starting parallel generation: target={target_samples}, workers={num_workers}")

    # Initialize components
    data_manager = SpanDataManager()
    progress = ProgressTracker()

    # Get current count
    current_count = data_manager.get_total_samples()
    logger.info(f"Current samples: {current_count}")

    if current_count >= target_samples:
        logger.info(f"Already have {current_count} samples (target: {target_samples})")
        return

    # Create queues
    batch_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()

    # Start workers
    workers = []
    for i in range(num_workers):
        t = threading.Thread(target=worker_generate, args=(i, batch_queue, result_queue, stop_event))
        t.daemon = True
        t.start()
        workers.append(t)
        logger.info(f"Started worker {i}")

    # Fill initial batches
    batch_id = 0
    batches_in_flight = 0
    max_in_flight = num_workers * 2

    try:
        while current_count < target_samples:
            # Add batches to queue
            while batches_in_flight < max_in_flight and current_count + (batches_in_flight * samples_per_request) < target_samples:
                batch_queue.put((batch_id, samples_per_request))
                batch_id += 1
                batches_in_flight += 1

            # Process results
            try:
                worker_id, completed_batch_id, samples, error = result_queue.get(timeout=30)
                batches_in_flight -= 1

                if error:
                    logger.warning(f"Worker {worker_id} batch {completed_batch_id} error: {error}")
                elif samples:
                    added, skipped = data_manager.add_samples(samples)
                    current_count = data_manager.get_total_samples()
                    progress.update(current_count, target_samples)
                    logger.info(
                        f"W{worker_id} B{completed_batch_id}: +{added} samples, "
                        f"Total: {current_count}/{target_samples} ({100*current_count/target_samples:.1f}%%)"
                    )
                else:
                    logger.warning(f"Worker {worker_id} batch {completed_batch_id}: No valid samples")

            except queue.Empty:
                logger.warning("Timeout waiting for results")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        logger.info("Stopping workers...")

    # Final stats
    stats = data_manager.get_stats()
    logger.info(f"Final stats: {json.dumps(stats, indent=2)}")


def generate_samples(
    target_samples: int = 50000,
    samples_per_request: int = SAMPLES_PER_REQUEST,
) -> None:
    """Generate span-format NER samples using Google AI."""
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY environment variable not set")
        return

    logger.info(f"Starting generation: target={target_samples} samples")

    # Initialize components
    data_manager = SpanDataManager()
    progress = ProgressTracker()

    # Get current count
    current_count = data_manager.get_total_samples()
    logger.info(f"Current samples: {current_count}")

    if current_count >= target_samples:
        logger.info(f"Already have {current_count} samples (target: {target_samples})")
        return

    # Initialize Google AI client with system prompt caching
    client = GoogleAIClient(
        model_name=GEMINI_MODEL,
        system_prompt=SYSTEM_PROMPT,
    )

    batch_id = 0
    try:
        while current_count < target_samples:
            remaining = target_samples - current_count
            batch_size = min(samples_per_request, remaining)

            # Generate prompt
            user_prompt = get_user_prompt(batch_size, batch_id)

            # Call API
            try:
                response = client.generate(user_prompt)
                samples = parse_jsonl_response(response)

                if samples:
                    added, skipped = data_manager.add_samples(samples)
                    current_count = data_manager.get_total_samples()
                    progress.update(current_count, target_samples)

                    logger.info(
                        f"Batch {batch_id}: Generated {len(samples)}, "
                        f"Added {added}, Skipped {skipped}, "
                        f"Total: {current_count}/{target_samples}"
                    )
                else:
                    logger.warning(f"Batch {batch_id}: No valid samples parsed")

            except Exception as e:
                logger.error(f"Batch {batch_id} failed: {e}")
                time.sleep(10)

            batch_id += 1

            # Small delay between requests
            time.sleep(1)

    finally:
        # Cleanup
        client.delete_cache()
        cost = client.get_cost_estimate()
        logger.info(f"Generation complete. Estimated cost: ${cost['total_cost_usd']:.4f}")

    # Final stats
    stats = data_manager.get_stats()
    logger.info(f"Final stats: {json.dumps(stats, indent=2)}")


# =============================================================================
# Batch Mode Generation (Vertex AI Batch Prediction)
# =============================================================================


def create_batch_input_files(
    target_samples: int = 50000,
    samples_per_request: int = SAMPLES_PER_REQUEST,
    requests_per_file: int = BATCH_SIZE,
) -> list[Path]:
    """Create JSONL input files for batch processing.

    Each file contains multiple requests in the format:
    {"request": {"contents": [{"role": "user", "parts": [{"text": "..."}]}], "system_instruction": {"parts": [{"text": "..."}]}}}
    """
    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_requests = (target_samples + samples_per_request - 1) // samples_per_request
    total_files = (total_requests + requests_per_file - 1) // requests_per_file

    logger.info(f"Creating {total_files} batch input files for {total_requests} requests")

    created_files = []
    request_id = 0

    for file_idx in range(total_files):
        batch_file = BATCH_INPUT_DIR / f"batch_input_{file_idx:04d}.jsonl"

        with open(batch_file, "w", encoding="utf-8") as f:
            for _ in range(requests_per_file):
                if request_id >= total_requests:
                    break

                user_prompt = get_user_prompt(samples_per_request, request_id)

                # Vertex AI batch format
                request_obj = {
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": user_prompt}]
                            }
                        ],
                        "system_instruction": {
                            "parts": [{"text": SYSTEM_PROMPT}]
                        },
                        "generation_config": {
                            "temperature": 0.8,
                            "top_p": 0.95,
                            "max_output_tokens": 8192,
                        }
                    }
                }

                f.write(json.dumps(request_obj) + "\n")
                request_id += 1

        created_files.append(batch_file)
        logger.info(f"Created {batch_file.name}")

    logger.info(f"Total: {len(created_files)} batch files, {request_id} requests")
    return created_files


def submit_batch_job(input_uri: str, output_uri: str) -> str:
    """Submit a batch prediction job to Vertex AI.

    Args:
        input_uri: GCS URI for input JSONL (e.g., gs://bucket/input.jsonl)
        output_uri: GCS URI for output folder (e.g., gs://bucket/output/)

    Returns:
        Job name/ID
    """
    if not HAS_VERTEX_AI:
        raise ImportError("Google GenAI SDK not installed")

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )

    # Create batch job
    batch_job = client.batches.create(
        model=VERTEX_MODEL,
        src=input_uri,
        config=genai_types.CreateBatchJobConfig(
            dest=output_uri,
        ),
    )

    logger.info(f"Submitted batch job: {batch_job.name}")
    logger.info(f"State: {batch_job.state}")

    return batch_job.name


def check_batch_job(job_name: str) -> dict:
    """Check the status of a batch job."""
    if not HAS_VERTEX_AI:
        raise ImportError("Google GenAI SDK not installed")

    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
    )

    job = client.batches.get(name=job_name)

    return {
        "name": job.name,
        "state": str(job.state),
        "create_time": str(getattr(job, "create_time", "")),
        "update_time": str(getattr(job, "update_time", "")),
    }


def process_batch_output(output_dir: Path) -> None:
    """Process batch output files and add to data manager."""
    data_manager = SpanDataManager()

    total_added = 0
    total_skipped = 0
    total_errors = 0

    for output_file in sorted(output_dir.glob("*.jsonl")):
        logger.info(f"Processing {output_file.name}")

        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line.strip())

                    # Extract response text from batch output format
                    response_text = ""
                    if "response" in result:
                        candidates = result["response"].get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                response_text = parts[0].get("text", "")

                    if response_text:
                        samples = parse_jsonl_response(response_text)
                        if samples:
                            added, skipped = data_manager.add_samples(samples)
                            total_added += added
                            total_skipped += skipped
                        else:
                            total_errors += 1
                    else:
                        total_errors += 1

                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Error parsing line: {e}")
                    total_errors += 1

    logger.info(f"Batch processing complete:")
    logger.info(f"  Added: {total_added}")
    logger.info(f"  Skipped (duplicates): {total_skipped}")
    logger.info(f"  Errors: {total_errors}")
    logger.info(f"  Total samples: {data_manager.get_total_samples()}")


def batch_generate(
    target_samples: int = 50000,
    samples_per_request: int = SAMPLES_PER_REQUEST,
    gcs_bucket: str = "",
) -> None:
    """Generate samples using Vertex AI batch mode.

    Steps:
    1. Create batch input files locally
    2. Upload to GCS (manual step if no gsutil)
    3. Submit batch job
    4. Download results (manual step)
    5. Process results
    """
    if not gcs_bucket:
        gcs_bucket = GCS_BUCKET

    if not gcs_bucket:
        logger.error("GCS_BUCKET not set. Set it in .env or pass --gcs-bucket")
        logger.info("Batch mode requires Cloud Storage for input/output.")
        logger.info("Alternative: Use 'generate' command for streaming mode.")
        return

    # Step 1: Create batch input files
    logger.info("Step 1: Creating batch input files...")
    batch_files = create_batch_input_files(target_samples, samples_per_request)

    logger.info(f"\nBatch input files created in: {BATCH_INPUT_DIR}")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Upload to GCS:")
    logger.info(f"     gsutil -m cp {BATCH_INPUT_DIR}/*.jsonl {gcs_bucket}/input/")
    logger.info(f"  2. Submit batch job (or use 'submit-batch' command)")
    logger.info(f"  3. Wait for completion")
    logger.info(f"  4. Download results:")
    logger.info(f"     gsutil -m cp {gcs_bucket}/output/*.jsonl {BATCH_OUTPUT_DIR}/")
    logger.info(f"  5. Process results:")
    logger.info(f"     python ner_general_span_generator.py process-batch")


def show_stats() -> None:
    """Show generation statistics."""
    data_manager = SpanDataManager()
    stats = data_manager.get_stats()

    print("\n=== NER General Span Generation Stats ===\n")
    print(f"Total samples:           {stats['total_samples']:,}")
    print(f"Samples with entities:   {stats['samples_with_entities']:,}")
    print(f"Samples without entities:{stats['samples_without_entities']:,}")
    print(f"Total entities:          {stats['total_entities']:,}")
    print(f"Avg entities/sample:     {stats['avg_entities_per_sample']:.2f}")
    print(f"Number of shards:        {stats['shard_count']}")
    print("\nLabel distribution:")
    for label, count in sorted(stats["label_distribution"].items()):
        pct = 100 * count / max(stats["total_entities"], 1)
        print(f"  {label}: {count:,} ({pct:.1f}%)")


def validate_all() -> None:
    """Validate all generated samples."""
    data_manager = SpanDataManager()

    valid_count = 0
    invalid_count = 0
    errors: dict[str, int] = defaultdict(int)

    for shard_file in sorted(data_manager.output_dir.glob("shard_*.jsonl")):
        with open(shard_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    is_valid, error = validate_span_sample(sample)
                    if is_valid:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        errors[error] += 1
                except json.JSONDecodeError:
                    invalid_count += 1
                    errors["JSON parse error"] += 1

    print("\n=== Validation Results ===\n")
    print(f"Valid samples:   {valid_count:,}")
    print(f"Invalid samples: {invalid_count:,}")

    if errors:
        print("\nErrors:")
        for error, count in sorted(errors.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="NER General Span Data Generator")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command (streaming mode)
    gen_parser = subparsers.add_parser("generate", help="Generate samples (streaming mode)")
    gen_parser.add_argument("--samples", type=int, default=50000, help="Target number of samples")
    gen_parser.add_argument("--batch-size", type=int, default=25, help="Samples per API request")
    gen_parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (use 1 for sequential)")

    # Batch mode commands
    batch_parser = subparsers.add_parser("batch", help="Generate using batch mode (half price)")
    batch_parser.add_argument("--samples", type=int, default=50000, help="Target number of samples")
    batch_parser.add_argument("--gcs-bucket", type=str, default="", help="GCS bucket URI (e.g., gs://my-bucket/ner)")

    submit_parser = subparsers.add_parser("submit-batch", help="Submit batch job to Vertex AI")
    submit_parser.add_argument("--input-uri", type=str, required=True, help="GCS input URI")
    submit_parser.add_argument("--output-uri", type=str, required=True, help="GCS output URI")

    check_parser = subparsers.add_parser("check-batch", help="Check batch job status")
    check_parser.add_argument("--job-name", type=str, required=True, help="Batch job name")

    process_parser = subparsers.add_parser("process-batch", help="Process batch output files")
    process_parser.add_argument("--output-dir", type=str, default=str(BATCH_OUTPUT_DIR), help="Directory with batch output files")

    # Stats command
    subparsers.add_parser("stats", help="Show generation statistics")

    # Validate command
    subparsers.add_parser("validate", help="Validate all generated samples")

    args = parser.parse_args()

    if args.command == "generate":
        if args.workers > 1:
            generate_samples_parallel(
                target_samples=args.samples,
                samples_per_request=args.batch_size,
                num_workers=args.workers,
            )
        else:
            generate_samples(
                target_samples=args.samples,
                samples_per_request=args.batch_size,
            )
    elif args.command == "batch":
        batch_generate(
            target_samples=args.samples,
            gcs_bucket=args.gcs_bucket,
        )
    elif args.command == "submit-batch":
        job_name = submit_batch_job(args.input_uri, args.output_uri)
        print(f"Submitted batch job: {job_name}")
    elif args.command == "check-batch":
        status = check_batch_job(args.job_name)
        print(json.dumps(status, indent=2))
    elif args.command == "process-batch":
        process_batch_output(Path(args.output_dir))
    elif args.command == "stats":
        show_stats()
    elif args.command == "validate":
        validate_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
