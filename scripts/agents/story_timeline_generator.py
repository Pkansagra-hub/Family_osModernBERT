"""Story timeline data generator.

Generates 3-day narrative timelines using LLM. Each timeline has:
- A cast of characters with roles and relationships
- Recurring arcs (wedding, work, health, family)
- Day slots (Morning Routine, Work, Lunch, Evening, Night)
- Thread continuity (reminders created and resolved)

Usage:
    python story_timeline_generator.py generate --days 3 --output data/generated/timeline.jsonl
    python story_timeline_generator.py generate --days 1 --events 25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

try:
    from google import genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# World Model Configuration
# =============================================================================

CAST: dict[str, dict[str, str]] = {
    "Prince": {"role": "self", "home": "Home", "work": "Home Office"},
    "Panda": {"role": "partner", "home": "Home"},
    "Sharvi": {"role": "friend", "home": "Neighborhood"},
    "Mom": {"role": "parent", "home": "Parents House"},
    "Dad": {"role": "parent", "home": "Parents House"},
    "Aarav": {"role": "friend", "home": "City"},
    "Dadi": {"role": "grandparent", "home": "India"},
    "Nana": {"role": "grandparent", "home": "India"},
}

CAST_NAMES: set[str] = set(CAST.keys()) - {"Prince"}

PLACES: list[str] = [
    "Home",
    "Home Office",
    "Gym",
    "Starbucks",
    "Micro Center",
    "Medical Center",
    "Neighborhood",
    "Grocery Store",
    "Parents House",
]

ARCS: list[dict[str, Any]] = [
    {"id": "wedding", "people": ["Panda"], "weight": 1.2, "topics": ["venue", "budget", "guest list", "catering", "decorations"]},
    {"id": "family_calls", "people": ["Mom", "Dad", "Dadi", "Nana"], "weight": 1.0, "topics": ["health", "news", "visits", "advice"]},
    {"id": "k0k1_dev", "people": [], "weight": 1.4, "topics": ["debugging", "refactoring", "PRs", "tests", "deployment"]},
    {"id": "health", "people": [], "weight": 1.1, "topics": ["GERD", "sleep", "headache", "nasal", "gym", "diet"]},
    {"id": "social", "people": ["Sharvi", "Aarav"], "weight": 0.8, "topics": ["hangout", "dinner", "movies", "games"]},
    {"id": "errands", "people": [], "weight": 0.7, "topics": ["shopping", "returns", "repairs", "bills"]},
]

ARC_IDS: set[str] = {a["id"] for a in ARCS}

SLOTS: list[str] = [
    "Morning Routine",
    "Morning Work",
    "Lunch",
    "Afternoon Work",
    "Gym",
    "Evening",
    "Night",
]

SLOT_DISTRIBUTION: dict[str, tuple[int, int]] = {
    "Morning Routine": (2, 4),
    "Morning Work": (4, 7),
    "Lunch": (1, 3),
    "Afternoon Work": (4, 7),
    "Gym": (1, 2),
    "Evening": (3, 6),
    "Night": (2, 4),
}

# Thread detection triggers (strict)
THREAD_TRIGGERS: tuple[str, ...] = (
    "remind me",
    "don't let me forget",
    "don't forget",
    "set a reminder",
    "need to remember",
    "should remember",
    "make sure to",
    "have to remember",
)

# Slot to allowed locations mapping (soft constraint for realism)
SLOT_ALLOWED_LOCATIONS: dict[str, list[str]] = {
    "Morning Routine": ["Home", "Parents House"],
    "Morning Work": ["Home Office"],
    "Lunch": ["Home", "Starbucks", "Grocery Store", "Home Office"],
    "Afternoon Work": ["Home Office", "Micro Center", "Starbucks"],
    "Gym": ["Gym"],
    "Evening": ["Home", "Neighborhood", "Parents House", "Grocery Store", "Starbucks"],
    "Night": ["Home"],
}


# =============================================================================
# Slot Allocation (with min-reserve logic)
# =============================================================================

def allocate_slot_counts(total: int) -> dict[str, int]:
    """Allocate event counts across slots respecting min/max constraints.

    Args:
        total: Total number of events to allocate.

    Returns:
        Dictionary mapping slot name to event count.
    """
    mins = {s: SLOT_DISTRIBUTION[s][0] for s in SLOTS}
    maxs = {s: SLOT_DISTRIBUTION[s][1] for s in SLOTS}

    # Start with minimums
    counts = dict(mins)
    min_sum = sum(counts.values())

    # If total is less than sum of mins, scale down proportionally
    if total < min_sum:
        scale = total / min_sum
        counts = {s: max(1, int(mins[s] * scale)) for s in SLOTS}
        # Adjust to hit exact total
        diff = total - sum(counts.values())
        for s in SLOTS:
            if diff == 0:
                break
            if diff > 0:
                counts[s] += 1
                diff -= 1
            elif counts[s] > 1:
                counts[s] -= 1
                diff += 1
        return counts

    # Distribute remaining events without exceeding max
    remaining = total - min_sum
    available_slots = [s for s in SLOTS if counts[s] < maxs[s]]

    while remaining > 0 and available_slots:
        s = random.choice(available_slots)
        if counts[s] < maxs[s]:
            counts[s] += 1
            remaining -= 1
            if counts[s] >= maxs[s]:
                available_slots.remove(s)
        else:
            available_slots.remove(s)

    # If still remaining (all maxed), distribute anyway
    if remaining > 0:
        for s in SLOTS:
            if remaining <= 0:
                break
            counts[s] += 1
            remaining -= 1

    return counts


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for story timeline generation."""

    days: int = 3
    min_events_per_day: int = 18
    max_events_per_day: int = 35
    start_date: str = "2026-01-23"
    is_weekday: bool = True


@dataclass(frozen=True)
class GoogleLLMConfig:
    """Configuration for the Google LLM client."""

    model_name: str = "gemini-2.0-flash"
    api_key_env_var: str = "GOOGLE_API_KEY"


@dataclass(frozen=True)
class EventRecord:
    """Single event record for JSONL output."""

    text: str
    participants: Sequence[str]
    location_name: str
    routine_time_slot: str
    day: str = ""
    arc: str = ""
    thread_id: str = ""


@dataclass
class Thread:
    """Represents a pending thread that needs resolution."""

    id: str
    due_slot: str
    person: str
    arc: str
    action: str
    created_day: int
    status: str = "pending"


# =============================================================================
# System Prompt for LLM
# =============================================================================

SYSTEM_PROMPT = """You are a life event generator for a personal AI assistant. Generate realistic daily events for a software engineer named Prince.

RULES:
1. Events should feel natural and personal, like diary entries
2. Use first person ("I", "my", "me")
3. Include specific details (names, times, places, feelings)
4. Create continuity - if something is mentioned, follow up on it later
5. Mix mundane routine with meaningful moments
6. Include health tracking (GERD, sleep, headaches)
7. Include work details (K0/K1 systems, debugging, PRs)
8. Include relationship moments (Panda, family, friends)

OUTPUT FORMAT:
Return ONLY valid JSON array. Each object must have:
{
  "text": "Event description in first person",
  "participants": ["Name1", "Name2"],
  "location_name": "Location",
  "routine time slot": "Slot Name",
  "arc": "arc_id",
  "thread_id": "optional_thread_id or null"
}

SLOTS (use EXACTLY one of these): Morning Routine, Morning Work, Lunch, Afternoon Work, Gym, Evening, Night

ARCS (use EXACTLY one of these): wedding, family_calls, k0k1_dev, health, social, errands

PARTICIPANTS (use only these names, exclude Prince): Panda, Mom, Dad, Sharvi, Aarav, Dadi, Nana

LOCATIONS (use only these): Home, Home Office, Gym, Starbucks, Micro Center, Medical Center, Neighborhood, Grocery Store, Parents House

THREAD RULES:
- When creating a new thread/reminder, use thread_id: "thread_new_1", "thread_new_2", etc.
- When resolving a pending thread, reuse its exact thread_id from the pending list
- Use phrases like "Remind me to...", "Don't forget to...", "Need to remember..." for new threads"""


# =============================================================================
# Google LLM Client
# =============================================================================

class GoogleLLMClient:
    """Google LLM client wrapper."""

    def __init__(self, config: GoogleLLMConfig) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is not installed. Install with: pip install google-genai"
            )

        api_key = os.environ.get(config.api_key_env_var, "")
        if not api_key:
            raise ValueError(f"Missing API key in {config.api_key_env_var}")

        self._model_name = config.model_name
        self._client = genai.Client(api_key=api_key)
        logger.info(f"Initialized Google LLM client with model: {self._model_name}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from the model."""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": 0.9,
                "max_output_tokens": 8192,
            },
        )
        return response.text


# =============================================================================
# Story Timeline Generator
# =============================================================================

class StoryTimelineGenerator:
    """Generator for multi-day story timelines using LLM."""

    def __init__(self, config: GeneratorConfig, llm_config: GoogleLLMConfig) -> None:
        self._config = config
        self._llm_config = llm_config
        self._logger = logging.getLogger(__name__)
        load_dotenv()
        self._llm_client = GoogleLLMClient(self._llm_config)
        self._threads: list[Thread] = []
        self._thread_counter = 0

    def _normalize_item(self, item: dict) -> dict:
        """Normalize and validate a single event item.

        Args:
            item: Raw event dictionary from LLM.

        Returns:
            Normalized event dictionary with valid enum values.
        """
        # Clean text: strip thread markers accidentally included in text
        text = item.get("text", "")
        text = re.sub(r"\bthread_new_\d*\b", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\bthread_id\s*:?\s*(thread_\d*)?\b", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s{2,}", " ", text)  # Collapse multiple spaces
        text = text.rstrip(".")  # Remove trailing period left by cleanup
        text = text.strip() + "." if text and not text.endswith((".", "!", "?")) else text.strip()
        item["text"] = text

        # Handle both key formats
        slot = item.get("routine time slot") or item.get("routine_time_slot") or ""
        arc = item.get("arc", "")
        loc = item.get("location_name", "Home")
        parts = item.get("participants", [])

        # Clamp to valid values
        if slot not in SLOTS:
            # Try fuzzy match
            slot_lower = slot.lower()
            matched = False
            for valid_slot in SLOTS:
                if valid_slot.lower() in slot_lower or slot_lower in valid_slot.lower():
                    slot = valid_slot
                    matched = True
                    break
            if not matched:
                slot = "Evening"  # Default fallback

        if arc not in ARC_IDS:
            arc = "errands"  # Default fallback

        if loc not in PLACES:
            # Try fuzzy match
            loc_lower = loc.lower()
            matched = False
            for valid_loc in PLACES:
                if valid_loc.lower() in loc_lower or loc_lower in valid_loc.lower():
                    loc = valid_loc
                    matched = True
                    break
            if not matched:
                loc = "Home"

        # Filter participants to valid cast members
        if isinstance(parts, list):
            parts = [p for p in parts if p in CAST_NAMES]
        else:
            parts = []

        # Clamp location based on slot (realism constraint)
        allowed_locs = SLOT_ALLOWED_LOCATIONS.get(slot, PLACES)
        if loc not in allowed_locs:
            # Pick random allowed location for variety
            loc = random.choice(allowed_locs) if allowed_locs else "Home"

        # Override slot based on text content (realism constraint)
        text_lower = item.get("text", "").lower()
        # Morning cues - things only done in morning
        morning_cues = ("woke up", "breakfast", "morning routine", "7:00 am", "7 am", "nasal rinse", "showered and dressed")
        # Night cues - things only done at night (avoid "sleep" as it appears in "sleep quality")
        night_cues = ("before bed", "lights off", "set my alarm", "going to sleep", "got ready for bed", "turned off the lights", "heading to bed")

        # Check morning cues first (higher priority)
        has_morning_cue = any(cue in text_lower for cue in morning_cues)
        has_night_cue = any(cue in text_lower for cue in night_cues)

        # Only override if unambiguous
        if has_morning_cue and not has_night_cue and slot not in ("Morning Routine", "Morning Work"):
            self._logger.debug(f"Slot override: '{slot}' -> 'Morning Routine'")
            slot = "Morning Routine"
            # Re-clamp location for new slot
            allowed_locs = SLOT_ALLOWED_LOCATIONS.get(slot, PLACES)
            if loc not in allowed_locs:
                loc = random.choice(allowed_locs) if allowed_locs else "Home"
        elif has_night_cue and not has_morning_cue and slot != "Night":
            self._logger.debug(f"Slot override: '{slot}' -> 'Night'")
            slot = "Night"
            allowed_locs = SLOT_ALLOWED_LOCATIONS.get(slot, PLACES)
            if loc not in allowed_locs:
                loc = random.choice(allowed_locs) if allowed_locs else "Home"

        item["routine time slot"] = slot
        item["arc"] = arc
        item["location_name"] = loc
        item["participants"] = parts
        return item

    def _detect_thread_due_slot(self, text: str) -> str:
        """Determine when a thread should be resolved based on text.

        Args:
            text: Event text.

        Returns:
            Slot name for resolution.
        """
        text_lower = text.lower()

        if "tonight" in text_lower or "this evening" in text_lower:
            return "Night"
        if "tomorrow morning" in text_lower:
            return "Morning Routine"
        if "tomorrow" in text_lower:
            return "Evening"
        if "this weekend" in text_lower or "weekend" in text_lower:
            return "Evening"
        if "after work" in text_lower:
            return "Evening"
        if "at lunch" in text_lower:
            return "Lunch"
        if "before bed" in text_lower:
            return "Night"

        return "Evening"  # Default

    def _build_day_prompt(
        self, day_num: int, day_date: str, events_count: int
    ) -> tuple[str, dict[str, int]]:
        """Build the user prompt for generating a day's events.

        Returns:
            Tuple of (prompt_text, slot_counts).
        """
        # Select arcs for this day weighted by priority
        arc_weights = [(arc["id"], arc["weight"]) for arc in ARCS]
        selected_arcs = random.choices(
            [a[0] for a in arc_weights],
            weights=[a[1] for a in arc_weights],
            k=min(4, len(ARCS)),
        )

        # Use proper slot allocation
        slot_counts = allocate_slot_counts(events_count)

        # Get pending threads for this day, prioritized by due_slot
        pending_threads = [t for t in self._threads if t.status == "pending"]
        # Sort by due_slot order (earlier slots first) and creation day
        pending_threads.sort(
            key=lambda t: (
                SLOTS.index(t.due_slot) if t.due_slot in SLOTS else 99,
                t.created_day,
            )
        )
        thread_context = ""
        threads_to_resolve = []
        if pending_threads:
            threads_to_resolve = pending_threads[:3]
            thread_items = [
                f'- thread_id: "{t.id}" | action: {t.action[:80]}... | person: {t.person or "none"}'
                for t in threads_to_resolve
            ]
            thread_context = f"""

PENDING THREADS TO RESOLVE (reuse these exact thread_ids when resolving):
{chr(10).join(thread_items)}"""

        # Determine how many new threads to create
        new_thread_count = 2 if day_num < self._config.days else 1  # Fewer on last day
        resolve_count = min(len(threads_to_resolve), 2) if day_num > 1 else 0

        prompt = f"""Generate EXACTLY {events_count} events for Day {day_num} ({day_date}).

SLOT DISTRIBUTION (generate this many events per slot):
{json.dumps(slot_counts, indent=2)}

FOCUS ARCS FOR TODAY: {', '.join(set(selected_arcs))}
{thread_context}

REQUIREMENTS:
1. Generate EXACTLY {events_count} events total (matching slot distribution above)
2. Create {new_thread_count} NEW threads using thread_id: "thread_new_1", "thread_new_2"
3. {"Resolve " + str(resolve_count) + " pending threads by reusing their exact thread_id" if resolve_count > 0 else "No pending threads to resolve"}
4. Include at least 1 reflection event in Night slot
5. Include health tracking in Morning Routine (GERD, sleep quality, nasal)
6. Make events flow naturally through the day
7. Use ONLY the allowed slot names, arc ids, locations, and participant names

Generate the events as a JSON array. Start immediately with ["""

        return prompt, slot_counts

    def _has_slot_cue(self, text: str, slot: str) -> bool:
        """Check if text has strong cues for a specific slot.

        Args:
            text: Event text.
            slot: Slot name to check.

        Returns:
            True if text strongly indicates this slot.
        """
        t = text.lower()
        morning_cues = ("woke up", "breakfast", "morning routine", "7:00 am", "7 am", "nasal rinse")
        night_cues = ("before bed", "lights off", "set my alarm", "going to sleep", "got ready for bed", "heading to bed")
        gym_cues = ("gym", "workout", "cardio", "weights", "treadmill")
        lunch_cues = ("lunch", "midday meal", "grabbed a bite")

        if slot == "Morning Routine" and any(c in t for c in morning_cues):
            return True
        if slot == "Night" and any(c in t for c in night_cues):
            return True
        if slot == "Gym" and any(c in t for c in gym_cues):
            return True
        if slot == "Lunch" and any(c in t for c in lunch_cues):
            return True
        return False

    def _enforce_slot_distribution(
        self, events: list[EventRecord], target: dict[str, int]
    ) -> list[EventRecord]:
        """Redistribute events to match target slot counts.

        Args:
            events: List of events to redistribute.
            target: Target slot counts.

        Returns:
            List of events with adjusted slots.
        """
        # Count current distribution
        cur = {s: 0 for s in SLOTS}
        for e in events:
            cur[e.routine_time_slot] += 1

        under = [s for s in SLOTS if cur[s] < target[s]]
        over = [s for s in SLOTS if cur[s] > target[s]]

        if not under or not over:
            return events

        # Index events by slot, excluding those with strong slot cues (can't be moved)
        by_slot: dict[str, list[int]] = {s: [] for s in SLOTS}
        locked_indices: set[int] = set()
        for i, e in enumerate(events):
            if self._has_slot_cue(e.text, e.routine_time_slot):
                locked_indices.add(i)
            else:
                by_slot[e.routine_time_slot].append(i)

        events_mut = list(events)
        u = 0

        for s_over in over:
            while cur[s_over] > target[s_over] and under:
                s_under = under[u % len(under)]
                if cur[s_under] >= target[s_under]:
                    under = [s for s in under if cur[s] < target[s]]
                    if not under:
                        break
                    u = 0
                    continue

                if not by_slot[s_over]:
                    break

                idx = by_slot[s_over].pop()
                e = events_mut[idx]

                # Skip if event text has cues for the target slot (wrong direction)
                if self._has_slot_cue(e.text, s_over):
                    continue

                # Get allowed location for new slot (randomize for variety)
                allowed_locs = SLOT_ALLOWED_LOCATIONS.get(s_under, PLACES)
                new_loc = (
                    e.location_name
                    if e.location_name in allowed_locs
                    else random.choice(allowed_locs) if allowed_locs else "Home"
                )

                events_mut[idx] = EventRecord(
                    text=e.text,
                    participants=e.participants,
                    location_name=new_loc,
                    routine_time_slot=s_under,
                    day=e.day,
                    arc=e.arc,
                    thread_id=e.thread_id,
                )
                cur[s_over] -= 1
                cur[s_under] += 1
                by_slot[s_under].append(idx)
                u += 1

        return events_mut

    def _trim_to_count(self, events: list[EventRecord], n: int) -> list[EventRecord]:
        """Trim events to exact count, preserving important ones.

        Args:
            events: List of events.
            n: Target count.

        Returns:
            Trimmed list of events.
        """
        if len(events) <= n:
            return events

        # Keep important events: thread resolutions, Night reflections
        important = []
        others = []
        for e in events:
            is_important = (
                e.thread_id
                or (e.routine_time_slot == "Night" and "reflect" in e.text.lower())
                or e.routine_time_slot == "Morning Routine"
            )
            if is_important:
                important.append(e)
            else:
                others.append(e)

        keep = important[:]
        for e in others:
            if len(keep) >= n:
                break
            keep.append(e)

        # Sort by slot order for natural flow
        slot_order = {s: i for i, s in enumerate(SLOTS)}
        keep.sort(key=lambda e: slot_order.get(e.routine_time_slot, 99))

        return keep[:n]

    def _parse_events(self, response: str, day_date: str, day_num: int) -> list[EventRecord]:
        """Parse LLM response into EventRecord objects.

        Args:
            response: Raw LLM response text.
            day_date: Date string for this day.
            day_num: Day number (1-indexed).

        Returns:
            List of validated EventRecord objects.
        """
        events = []

        # Clean up response - remove markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)

        # Ensure it starts with [ and ends with ]
        if not cleaned.startswith("["):
            cleaned = "[" + cleaned
        if not cleaned.endswith("]"):
            # Try to find the last complete object
            last_brace = cleaned.rfind("}")
            if last_brace != -1:
                cleaned = cleaned[: last_brace + 1] + "]"

        try:
            data = json.loads(cleaned)
            if not isinstance(data, list):
                data = [data]

            # Track which thread_new_X IDs we see for remapping
            new_thread_map: dict[str, str] = {}

            for item in data:
                if not isinstance(item, dict):
                    continue
                if "text" not in item:
                    continue

                # Normalize the item
                item = self._normalize_item(item)

                # Handle thread_id remapping
                raw_thread_id = item.get("thread_id", "") or ""
                thread_id = ""

                if raw_thread_id:
                    if raw_thread_id.startswith("thread_new_"):
                        # New thread - assign real ID
                        if raw_thread_id not in new_thread_map:
                            self._thread_counter += 1
                            new_thread_map[raw_thread_id] = f"thread_{self._thread_counter}"
                        thread_id = new_thread_map[raw_thread_id]
                    else:
                        # Existing thread ID (resolution) - validate it exists
                        known_ids = {t.id for t in self._threads}
                        if raw_thread_id not in known_ids:
                            # Unknown thread id: drop it (phantom resolution)
                            self._logger.debug(f"Dropping unknown thread_id: {raw_thread_id}")
                            raw_thread_id = ""
                        else:
                            thread_id = raw_thread_id
                            # Mark as resolved
                            for t in self._threads:
                                if t.id == thread_id and t.status == "pending":
                                    t.status = "resolved"
                                    self._logger.debug(f"Resolved thread: {thread_id}")
                                    break

                event = EventRecord(
                    text=item.get("text", ""),
                    participants=item["participants"],
                    location_name=item["location_name"],
                    routine_time_slot=item["routine time slot"],
                    day=day_date,
                    arc=item["arc"],
                    thread_id=thread_id,
                )
                events.append(event)

                # Only create threads from explicit thread_new_* IDs (not from text detection)
                # This prevents false positives from resolution events containing trigger phrases
                if raw_thread_id.startswith("thread_new_") and thread_id:
                    existing_ids = {t.id for t in self._threads}
                    if thread_id not in existing_ids:
                        due_slot = self._detect_thread_due_slot(event.text)
                        thread = Thread(
                            id=thread_id,
                            due_slot=due_slot,
                            person=event.participants[0] if event.participants else "",
                            arc=event.arc,
                            action=event.text[:100],
                            created_day=day_num,
                        )
                        self._threads.append(thread)
                        self._logger.debug(f"Created thread: {thread_id} (due: {due_slot})")

        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse LLM response: {e}")
            self._logger.debug(f"Response was: {cleaned[:500]}...")

        return events

    def generate(self) -> Iterable[EventRecord]:
        """Generate event records for all days."""
        all_events: list[EventRecord] = []
        start = date.fromisoformat(self._config.start_date)

        for day_num in range(1, self._config.days + 1):
            day_date = (start + timedelta(days=day_num - 1)).isoformat()
            events_count = random.randint(
                self._config.min_events_per_day,
                self._config.max_events_per_day,
            )

            self._logger.info(f"Generating Day {day_num} ({day_date}): {events_count} events...")

            # Log pending threads
            pending = [t for t in self._threads if t.status == "pending"]
            if pending:
                self._logger.info(f"  Pending threads: {len(pending)}")

            prompt, slot_counts = self._build_day_prompt(day_num, day_date, events_count)

            try:
                response = self._llm_client.generate(SYSTEM_PROMPT, prompt)
                events = self._parse_events(response, day_date, day_num)

                # Log raw parse results
                raw_count = len(events)
                raw_slots = {s: 0 for s in SLOTS}
                for e in events:
                    raw_slots[e.routine_time_slot] += 1
                self._logger.info(f"  Raw parse: {raw_count} events")
                self._logger.debug(f"  Raw slot distribution: {raw_slots}")

                # Trim first (preserves important events)
                events = self._trim_to_count(events, events_count)

                # Then enforce slot distribution (final pass)
                events = self._enforce_slot_distribution(events, slot_counts)

                # Sort events by slot order for natural timeline flow
                slot_order = {s: i for i, s in enumerate(SLOTS)}
                events = sorted(events, key=lambda e: slot_order.get(e.routine_time_slot, 99))

                # Log final slot distribution and check for mismatch
                final_slots = {s: 0 for s in SLOTS}
                for e in events:
                    final_slots[e.routine_time_slot] += 1
                self._logger.info(f"  Final: {len(events)} events, slots: {final_slots}")

                if final_slots != slot_counts:
                    self._logger.warning(
                        f"  Slot mismatch! target={slot_counts} got={final_slots}"
                    )

                all_events.extend(events)

                # Stats
                resolved_count = sum(1 for t in self._threads if t.status == "resolved")
                pending_count = sum(1 for t in self._threads if t.status == "pending")
                self._logger.info(
                    f"  Threads: {pending_count} pending, {resolved_count} resolved"
                )

            except Exception as e:
                self._logger.error(f"Failed to generate Day {day_num}: {e}")

        # Final stats
        self._logger.info(f"Total threads created: {len(self._threads)}")
        self._logger.info(f"  Resolved: {sum(1 for t in self._threads if t.status == 'resolved')}")
        self._logger.info(f"  Pending: {sum(1 for t in self._threads if t.status == 'pending')}")

        return all_events


# =============================================================================
# Serialization
# =============================================================================

def serialize_event(event: EventRecord) -> dict:
    """Serialize an EventRecord to a JSON-compatible dict."""
    result = {
        "text": event.text,
        "participants": list(event.participants),
        "location_name": event.location_name,
        "routine time slot": event.routine_time_slot,
    }
    if event.day:
        result["day"] = event.day
    if event.arc:
        result["arc"] = event.arc
    if event.thread_id:
        result["thread_id"] = event.thread_id
    return result


def write_jsonl(events: Iterable[EventRecord], output_path: Path) -> int:
    """Write events to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(serialize_event(event), ensure_ascii=False) + "\n")
            count += 1
    return count


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate story timeline data using LLM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate timeline data")
    gen_parser.add_argument("--days", type=int, default=3, help="Number of days to generate")
    gen_parser.add_argument("--min-events", type=int, default=18, help="Min events per day")
    gen_parser.add_argument("--max-events", type=int, default=35, help="Max events per day")
    gen_parser.add_argument("--start-date", type=str, default="2026-01-23", help="Start date (YYYY-MM-DD)")
    gen_parser.add_argument("--output", type=str, default="data/generated/timeline.jsonl", help="Output file")
    gen_parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="LLM model name")

    args = parser.parse_args()

    if args.command == "generate":
        config = GeneratorConfig(
            days=args.days,
            min_events_per_day=args.min_events,
            max_events_per_day=args.max_events,
            start_date=args.start_date,
        )
        llm_config = GoogleLLMConfig(model_name=args.model)

        generator = StoryTimelineGenerator(config, llm_config)
        events = generator.generate()

        output_path = Path(args.output)
        count = write_jsonl(events, output_path)

        logger.info(f"Wrote {count} events to {output_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "GeneratorConfig",
    "GoogleLLMConfig",
    "EventRecord",
    "StoryTimelineGenerator",
    "serialize_event",
    "GoogleLLMClient",
    "write_jsonl",
    "allocate_slot_counts",
]
