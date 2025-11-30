"""
NER Family Data Generator Agent (v2)

Uses OpenRouter API with Grok to generate synthetic NER training data
for the FamilyOS Family NER dataset.

Data Strategy:
- GOLD: Human-verified, high-quality samples (dev/test + gold train subset)
- SILVER: LLM-generated, validated samples (large pool for sampling)

Rate Limits:
- OpenRouter free tier: ~1000 requests/day
- Conservative rate limiting to avoid errors

Sampling Strategy:
- Store silver data in shards (e.g., 10 x 10k = 100k)
- Each epoch samples 50k-100k from silver pool
- Gold samples always included
- Rotate shards across epochs for diversity
"""

import json
import time
import logging
import re
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from collections import defaultdict

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

OPENROUTER_API_KEY = "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"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "x-ai/grok-4.1-fast:free"

# Rate limiting settings (conservative for free tier)
REQUESTS_PER_MINUTE = 10  # Stay well under limits
REQUESTS_PER_DAY = 900  # Leave buffer from 1000 limit
DELAY_BETWEEN_REQUESTS = 6.0  # seconds (10 req/min = 6s between)

# Output settings
DATA_DIR = Path("D:/Modeling_studio/data/familyos/ner_family")
GOLD_DIR = DATA_DIR / "gold"
SILVER_DIR = DATA_DIR / "silver"
SHARD_SIZE = 10000  # Samples per silver shard

# Legacy files (for backward compatibility)
LEGACY_TRAIN = DATA_DIR / "train.jsonl"
LEGACY_VALIDATION = DATA_DIR / "validation.jsonl"

SAMPLES_PER_REQUEST = 50  # Ask for 50 samples per API call (free tier limit)

# =============================================================================
# Diverse Prompt Templates for Quality & Variety
# =============================================================================

# Different family contexts for diversity
FAMILY_CONTEXTS = [
    "a Western nuclear family (mom, dad, two kids)",
    "an Indian joint family (parents, grandparents, uncle, aunt, cousins)",
    "a mixed Western-Indian family",
    "a single-parent family with grandparents helping",
    "a family with multiple pets (dog, cat, fish)",
    "a multigenerational household with great-grandparents",
    "a family celebrating Indian festivals (Diwali, Holi, Raksha Bandhan)",
    "a family celebrating Western holidays (Christmas, Thanksgiving, Easter)",
    "a family with young children (toddlers, babies)",
    "a family with teenagers",
]

SCENARIOS = [
    "morning routines and breakfast",
    "school drop-off and pick-up",
    "dinner time conversations",
    "weekend family activities",
    "birthday party planning",
    "holiday celebrations",
    "bedtime stories and routines",
    "family game night",
    "visiting grandparents",
    "pet care and playing with pets",
    "cooking together",
    "family photos and memories",
    "milestone moments (first steps, first words, graduations)",
    "heirloom stories and family traditions",
    "video calls with distant relatives",
]

LANGUAGE_STYLES = [
    "casual everyday speech",
    "affectionate family nicknames",
    "Indian-English with Hindi/Tamil/Bengali words mixed in",
    "formal announcements (wedding, graduation)",
    "text message style (short, informal)",
    "diary or journal entries",
    "photo captions",
    "calendar reminders and notes",
]

SYSTEM_PROMPT = """You are an expert data annotator for a family-specific Named Entity Recognition (NER) task.
Your job is to generate realistic, diverse English sentences that a family might say or write in daily life, and then annotate them with the exact 21-class BIO label schema below.

### LABEL SCHEMA (you MUST follow this exactly)
Use only these integer IDs:

0  → O
1  → B-PERSON      2 → I-PERSON         # full names or named individuals (e.g. "Emma", "John Smith", "Sarah Smith")
3  → B-KINSHIP     4 → I-KINSHIP        # relationship terms (mom, dad, didi, nana, chacha, mummy, grandpa, etc.)
5  → B-NICKNAME    6 → I-NICKNAME       # cute family nicknames (Panda, Bunny, Sweetie, Baby Bear, Munchkin, etc.)
7  → B-PET         8 → I-PET            # pet names (Max, Whiskers, Milo, etc.)
9  → B-HOME_LOC   10 → I-HOME_LOC       # places inside or around home (kitchen, backyard, Emma's room, living room, garden, garage, etc.)
11 → B-FAMILY_EVENT 12 → I-FAMILY_EVENT # one-time family occasions (birthday party, anniversary dinner, graduation, wedding day, etc.)
13 → B-ROUTINE     14 → I-ROUTINE       # daily/regular activities (school run, bedtime story, dinner time, morning walk, etc.)
15 → B-TRADITION   16 → I-TRADITION     # recurring rituals (Sunday brunch, movie night, Diwali celebration, Christmas Eve dinner, etc.)
17 → B-MILESTONE   18 → I-MILESTONE     # important life events (first steps, lost his first tooth, graduation day, first word, etc.)
19 → B-HEIRLOOM    20 → I-HEIRLOOM      # sentimental objects (grandma's necklace, dad's old watch, family photo album, etc.)

### RULES YOU MUST OBEY STRICTLY
- Output ONLY valid JSONL lines (one JSON object per line).
- Never output explanations, markdown, or extra text outside the JSON.
- Each line must have exactly two keys: "tokens" (list of strings) and "ner_tags" (list of integers, same length).
- Use proper BIO rules: every I- tag must be preceded by the corresponding B- tag of the same entity type. No orphan I-tags.
- Entities can be 1–4 tokens long.
- Include possessive forms correctly (e.g. "Panda's birthday party", "Emma's room").
- Vary sentence length and complexity.
- Make sure all 10 entity types appear regularly across the generated data.
- Nicknames, pets, and heirlooms should feel personal and cute.

### EXAMPLE OUTPUT (follow this format exactly)
{"tokens": ["Panda", "took", "her", "first", "steps", "in", "the", "kitchen", "today"], "ner_tags": [5, 0, 0, 17, 18, 0, 0, 9, 0]}
{"tokens": ["Mom", "made", "her", "famous", "Sunday", "brunch", "again"], "ner_tags": [3, 0, 0, 0, 15, 16, 0]}
{"tokens": ["Chacha", "and", "Chachi", "arrived", "for", "Diwali", "celebration", "yesterday"], "ner_tags": [3, 0, 3, 0, 0, 15, 16, 0]}"""


def get_diverse_user_prompt(
    num_samples: int,
    focus_entities: list[str] | None = None,
    batch_id: int = 0,
) -> str:
    """Generate diverse user prompts to avoid repetitive data."""
    # Rotate through contexts and scenarios
    context = FAMILY_CONTEXTS[batch_id % len(FAMILY_CONTEXTS)]
    scenario = SCENARIOS[batch_id % len(SCENARIOS)]
    style = LANGUAGE_STYLES[batch_id % len(LANGUAGE_STYLES)]
    
    prompt = f"""Generate {num_samples} NER examples. Context: {context}. Scenario: {scenario}. Style: {style}."""
    
    if focus_entities:
        prompt += f" Priority entities: {', '.join(focus_entities)}."
    
    prompt += " Output JSONL only:"
    return prompt


# =============================================================================
# NER Data Validation
# =============================================================================

ENTITY_TAGS = {
    "PERSON": (1, 2),
    "KINSHIP": (3, 4),
    "NICKNAME": (5, 6),
    "PET": (7, 8),
    "HOME_LOC": (9, 10),
    "FAMILY_EVENT": (11, 12),
    "ROUTINE": (13, 14),
    "TRADITION": (15, 16),
    "MILESTONE": (17, 18),
    "HEIRLOOM": (19, 20),
}

TAG_INFO: dict[int, tuple[str, bool]] = {0: ("O", True)}
for entity, (b_tag, i_tag) in ENTITY_TAGS.items():
    TAG_INFO[b_tag] = (entity, True)
    TAG_INFO[i_tag] = (entity, False)


def validate_ner_sample(sample: dict[str, Any]) -> tuple[bool, str]:
    """Validate a single NER sample with strict BIO rules."""
    if "tokens" not in sample or "ner_tags" not in sample:
        return False, "Missing 'tokens' or 'ner_tags' key"
    
    tokens = sample["tokens"]
    tags = sample["ner_tags"]
    
    if not isinstance(tokens, list) or not isinstance(tags, list):
        return False, "tokens and ner_tags must be lists"
    
    if len(tokens) != len(tags):
        return False, f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags"
    
    if len(tokens) == 0:
        return False, "Empty sample"
    
    if not all(isinstance(t, str) for t in tokens):
        return False, "All tokens must be strings"
    
    if not all(isinstance(t, int) and 0 <= t <= 20 for t in tags):
        return False, f"Invalid tag values (must be 0-20): {tags}"
    
    # Check BIO consistency
    prev_entity = None
    for i, tag in enumerate(tags):
        if tag not in TAG_INFO:
            return False, f"Unknown tag ID: {tag}"
        
        entity, is_beginning = TAG_INFO[tag]
        
        if entity == "O":
            prev_entity = None
        elif is_beginning:
            prev_entity = entity
        else:
            if prev_entity != entity:
                return False, f"Orphan I-{entity} tag at position {i}"
    
    return True, ""


def compute_sample_hash(sample: dict[str, Any]) -> str:
    """Compute hash for deduplication."""
    text = " ".join(sample["tokens"]).lower()
    return hashlib.md5(text.encode()).hexdigest()


def get_entity_coverage(sample: dict[str, Any]) -> dict[str, int]:
    """Count entity types in a sample."""
    counts: dict[str, int] = defaultdict(int)
    for tag in sample.get("ner_tags", []):
        if tag in TAG_INFO and tag != 0:
            entity, is_beginning = TAG_INFO[tag]
            if is_beginning:
                counts[entity] += 1
    return dict(counts)


def parse_jsonl_response(response_text: str) -> list[dict[str, Any]]:
    """Parse JSONL from model response, handling various formatting issues."""
    valid_samples = []
    
    lines = response_text.strip().split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        
        try:
            json_match = re.search(r'\{[^{}]*"tokens"[^{}]*"ner_tags"[^{}]*\}', line)
            if json_match:
                sample = json.loads(json_match.group())
            else:
                sample = json.loads(line)
            
            is_valid, error = validate_ner_sample(sample)
            if is_valid:
                valid_samples.append(sample)
            else:
                logger.debug(f"Invalid sample: {error}")
                
        except json.JSONDecodeError:
            continue
    
    return valid_samples


# =============================================================================
# OpenRouter API Client
# =============================================================================

class OpenRouterClient:
    """Client for OpenRouter API with rate limiting."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        requests_per_minute: int = REQUESTS_PER_MINUTE,
        requests_per_day: int = REQUESTS_PER_DAY,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        
        self.request_times: list[datetime] = []
        self.daily_count = 0
        self.daily_reset = datetime.now().replace(hour=0, minute=0, second=0) + timedelta(days=1)
        
        self.client = httpx.Client(timeout=120.0)
    
    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = datetime.now()
        
        if now >= self.daily_reset:
            self.daily_count = 0
            self.daily_reset = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            logger.info("Daily rate limit reset")
        
        if self.daily_count >= self.requests_per_day:
            raise RuntimeError("Daily rate limit reached")
        
        minute_ago = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > minute_ago]
        
        if len(self.request_times) >= self.requests_per_minute:
            oldest = min(self.request_times)
            wait_seconds = (oldest + timedelta(minutes=1) - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"Rate limiting: waiting {wait_seconds:.1f}s")
                time.sleep(wait_seconds)
    
    def generate(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.9,  # Higher for diversity
        max_tokens: int = 16000,
    ) -> str:
        """Generate text using OpenRouter API."""
        self._wait_for_rate_limit()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/modeling-studio",
            "X-Title": "FamilyOS NER Data Generator",
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
        
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            
            self.request_times.append(datetime.now())
            self.daily_count += 1
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            logger.info(f"API call successful (daily: {self.daily_count}/{self.requests_per_day})")
            return content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            if e.response.status_code == 429:
                logger.warning("Rate limited by API. Waiting 60 seconds...")
                time.sleep(60)
            raise
    
    def close(self):
        self.client.close()


# =============================================================================
# Silver Data Manager (Sharded Storage)
# =============================================================================

class SilverDataManager:
    """Manages sharded silver data storage."""
    
    def __init__(self, silver_dir: Path = SILVER_DIR, shard_size: int = SHARD_SIZE):
        self.silver_dir = silver_dir
        self.shard_size = shard_size
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        
        # Track hashes for deduplication across shards
        self.seen_hashes: set[str] = set()
        self._load_existing_hashes()
        
        # Current shard being written to
        self.current_shard_id = self._get_next_shard_id()
        self.current_shard_count = self._count_shard_samples(self.current_shard_id)
    
    def _load_existing_hashes(self) -> None:
        """Load hashes from all existing shards for deduplication."""
        for shard_file in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        self.seen_hashes.add(compute_sample_hash(sample))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(self.seen_hashes)} existing hashes from silver shards")
    
    def _get_shard_path(self, shard_id: int) -> Path:
        return self.silver_dir / f"shard_{shard_id:04d}.jsonl"
    
    def _count_shard_samples(self, shard_id: int) -> int:
        shard_path = self._get_shard_path(shard_id)
        if not shard_path.exists():
            return 0
        with open(shard_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    
    def _get_next_shard_id(self) -> int:
        """Find the current shard to write to (last incomplete or new)."""
        existing = list(self.silver_dir.glob("shard_*.jsonl"))
        if not existing:
            return 0
        
        # Get highest shard ID
        max_id = max(int(p.stem.split("_")[1]) for p in existing)
        
        # Check if it's full
        if self._count_shard_samples(max_id) >= self.shard_size:
            return max_id + 1
        return max_id
    
    def add_samples(self, samples: list[dict[str, Any]]) -> int:
        """Add samples to silver storage, returns count added."""
        added = 0
        
        for sample in samples:
            sample_hash = compute_sample_hash(sample)
            if sample_hash in self.seen_hashes:
                continue
            
            # Check if current shard is full
            if self.current_shard_count >= self.shard_size:
                self.current_shard_id += 1
                self.current_shard_count = 0
                logger.info(f"Started new shard: shard_{self.current_shard_id:04d}")
            
            # Write to current shard
            shard_path = self._get_shard_path(self.current_shard_id)
            with open(shard_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
            self.seen_hashes.add(sample_hash)
            self.current_shard_count += 1
            added += 1
        
        return added
    
    def get_total_samples(self) -> int:
        return len(self.seen_hashes)
    
    def get_shard_count(self) -> int:
        return len(list(self.silver_dir.glob("shard_*.jsonl")))
    
    def sample_for_epoch(
        self,
        num_samples: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample N samples from silver pool for an epoch."""
        if seed is not None:
            random.seed(seed)
        
        all_shards = sorted(self.silver_dir.glob("shard_*.jsonl"))
        if not all_shards:
            return []
        
        # Load all samples (for now - could optimize with reservoir sampling)
        all_samples = []
        for shard_path in all_shards:
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        all_samples.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        
        # Sample
        if len(all_samples) <= num_samples:
            return all_samples
        
        return random.sample(all_samples, num_samples)
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about silver data."""
        entity_counts: dict[str, int] = defaultdict(int)
        total = 0
        
        for shard_path in self.silver_dir.glob("shard_*.jsonl"):
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        for entity, count in get_entity_coverage(sample).items():
                            entity_counts[entity] += count
                        total += 1
                    except json.JSONDecodeError:
                        continue
        
        return {
            "total_samples": total,
            "num_shards": self.get_shard_count(),
            "entity_counts": dict(entity_counts),
        }


# =============================================================================
# Data Generator Agent (v2)
# =============================================================================

class NERDataGeneratorAgent:
    """Agent that generates NER training data using OpenRouter."""
    
    def __init__(
        self,
        samples_per_request: int = SAMPLES_PER_REQUEST,
        delay_between_requests: float = DELAY_BETWEEN_REQUESTS,
    ):
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests
        
        self.client = OpenRouterClient(api_key=OPENROUTER_API_KEY)
        self.silver_manager = SilverDataManager()
        
        # Track entity coverage for balanced generation
        self.entity_counts: dict[str, int] = defaultdict(int)
        self._load_existing_counts()
        
        # Batch counter for prompt diversity
        self.batch_id = 0
    
    def _load_existing_counts(self) -> None:
        """Load existing entity counts from silver data."""
        stats = self.silver_manager.get_stats()
        self.entity_counts = defaultdict(int, stats.get("entity_counts", {}))
        logger.info(f"Existing entity counts: {dict(self.entity_counts)}")
    
    def _get_underrepresented_entities(self, n: int = 3) -> list[str]:
        """Get the n most underrepresented entity types."""
        all_entities = list(ENTITY_TAGS.keys())
        sorted_entities = sorted(all_entities, key=lambda e: self.entity_counts.get(e, 0))
        return sorted_entities[:n]
    
    def generate_batch(self) -> int:
        """Generate a batch of samples. Returns count of new samples added."""
        focus = self._get_underrepresented_entities(3)
        
        user_prompt = get_diverse_user_prompt(
            self.samples_per_request,
            focus_entities=focus,
            batch_id=self.batch_id,
        )
        
        try:
            response = self.client.generate(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            
            samples = parse_jsonl_response(response)
            added = self.silver_manager.add_samples(samples)
            
            # Update entity counts
            for sample in samples:
                for entity, count in get_entity_coverage(sample).items():
                    self.entity_counts[entity] += count
            
            self.batch_id += 1
            
            logger.info(
                f"Generated {len(samples)} valid, added {added} new. "
                f"Focus: {focus}. Total silver: {self.silver_manager.get_total_samples()}"
            )
            
            return added
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return 0
    
    def run(
        self,
        target_samples: int | None = None,
        max_requests: int | None = None,
        run_time_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Run the data generation agent."""
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=run_time_minutes) if run_time_minutes else None
        
        existing_count = self.silver_manager.get_total_samples()
        
        stats = {
            "start_time": start_time.isoformat(),
            "existing_samples": existing_count,
            "new_samples": 0,
            "requests": 0,
            "errors": 0,
        }
        
        logger.info(f"Starting NER data generation. Existing silver samples: {existing_count}")
        
        try:
            while True:
                if target_samples and stats["new_samples"] >= target_samples:
                    logger.info(f"Reached target of {target_samples} new samples")
                    break
                
                if max_requests and stats["requests"] >= max_requests:
                    logger.info(f"Reached max requests: {max_requests}")
                    break
                
                if end_time and datetime.now() >= end_time:
                    logger.info(f"Reached time limit: {run_time_minutes} minutes")
                    break
                
                stats["requests"] += 1
                added = self.generate_batch()
                
                if added > 0:
                    stats["new_samples"] += added
                else:
                    stats["errors"] += 1
                
                total = existing_count + stats["new_samples"]
                logger.info(
                    f"Progress: {stats['new_samples']} new ({total} total) "
                    f"from {stats['requests']} requests"
                )
                
                time.sleep(self.delay_between_requests)
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except RuntimeError as e:
            if "rate limit" in str(e).lower():
                logger.warning("Stopping due to rate limit")
            else:
                raise
        finally:
            self.client.close()
        
        stats["end_time"] = datetime.now().isoformat()
        stats["final_entity_counts"] = dict(self.entity_counts)
        stats["total_samples"] = self.silver_manager.get_total_samples()
        stats["num_shards"] = self.silver_manager.get_shard_count()
        
        logger.info(f"Generation complete. Stats: {json.dumps(stats, indent=2)}")
        return stats


# =============================================================================
# Training Data Sampler
# =============================================================================

def prepare_epoch_data(
    gold_file: Path | None = None,
    silver_samples: int = 50000,
    epoch_seed: int = 42,
    output_file: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Prepare training data for one epoch.
    
    Strategy:
    - Include all gold samples
    - Sample N samples from silver pool
    - Shuffle and optionally save to file
    
    Args:
        gold_file: Path to gold JSONL file (all included)
        silver_samples: Number of samples to draw from silver pool
        epoch_seed: Random seed for reproducibility
        output_file: Optional path to save the epoch data
    
    Returns:
        List of training samples for this epoch
    """
    random.seed(epoch_seed)
    samples = []
    
    # Load gold samples
    if gold_file and gold_file.exists():
        with open(gold_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    sample["_source"] = "gold"
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Loaded {len(samples)} gold samples")
    
    # Sample from silver
    silver_manager = SilverDataManager()
    silver = silver_manager.sample_for_epoch(silver_samples, seed=epoch_seed)
    for sample in silver:
        sample["_source"] = "silver"
    samples.extend(silver)
    logger.info(f"Sampled {len(silver)} silver samples")
    
    # Shuffle
    random.shuffle(samples)
    
    # Optionally save
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(samples)} samples to {output_file}")
    
    return samples


def migrate_legacy_to_gold():
    """Migrate existing train.jsonl to gold directory."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold_train = GOLD_DIR / "train.jsonl"
    
    if LEGACY_TRAIN.exists() and not gold_train.exists():
        logger.info(f"Migrating {LEGACY_TRAIN} to {gold_train}")
        import shutil
        shutil.copy(LEGACY_TRAIN, gold_train)
        logger.info("Migration complete")
    
    if LEGACY_VALIDATION.exists():
        gold_val = GOLD_DIR / "validation.jsonl"
        if not gold_val.exists():
            import shutil
            shutil.copy(LEGACY_VALIDATION, gold_val)
            logger.info(f"Migrated validation to {gold_val}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Run the NER data generator agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NER training data using OpenRouter")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new silver data")
    gen_parser.add_argument("--target", type=int, default=None, help="Target number of new samples")
    gen_parser.add_argument("--max-requests", type=int, default=None, help="Maximum API requests")
    gen_parser.add_argument("--run-time", type=int, default=None, help="Run time in minutes")
    gen_parser.add_argument("--samples-per-request", type=int, default=25, help="Samples per API call")
    gen_parser.add_argument("--delay", type=float, default=6.0, help="Delay between requests (seconds)")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show silver data statistics")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Sample data for an epoch")
    sample_parser.add_argument("--silver", type=int, default=50000, help="Number of silver samples")
    sample_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sample_parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate legacy data to gold")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        agent = NERDataGeneratorAgent(
            samples_per_request=args.samples_per_request,
            delay_between_requests=args.delay,
        )
        stats = agent.run(
            target_samples=args.target,
            max_requests=args.max_requests,
            run_time_minutes=args.run_time,
        )
        print("\n=== Final Statistics ===")
        print(json.dumps(stats, indent=2))
        
    elif args.command == "stats":
        manager = SilverDataManager()
        stats = manager.get_stats()
        print("\n=== Silver Data Statistics ===")
        print(json.dumps(stats, indent=2))
        
    elif args.command == "sample":
        output = Path(args.output) if args.output else None
        gold_file = GOLD_DIR / "train.jsonl"
        samples = prepare_epoch_data(
            gold_file=gold_file if gold_file.exists() else None,
            silver_samples=args.silver,
            epoch_seed=args.seed,
            output_file=output,
        )
        print(f"\nPrepared {len(samples)} samples for epoch")
        
    elif args.command == "migrate":
        migrate_legacy_to_gold()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
