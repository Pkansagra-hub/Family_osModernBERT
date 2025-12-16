"""
Counterfactual Data Generator for FamilyOS UltraBERT Decoder

Generates high-quality counterfactual pairs for training the UltraBERT decoder head.
Each pair consists of a life scenario (input) and a helpful counterfactual response
that suggests what could have been done differently for a better outcome.

Focus Areas (40 Family Life Domains):
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

SYSTEM_PROMPT = """You are an expert counterfactual reasoning generator for FamilyOS, a family wellness AI assistant. Your task is to generate high-quality counterfactual pairs that help families learn from life experiences.

## MISSION: Generate "What Could Have Been Done Differently" Pairs

For each sample, generate:
1. **INPUT**: A realistic life scenario with a suboptimal outcome (family context)
2. **COUNTERFACTUAL**: A helpful alternative that could have led to a better outcome

## OUTPUT FORMAT (JSONL - one JSON object per line)

```json
{
  "id": "cf_XXXXX",
  "domain": "<domain_category>",
  "subdomain": "<specific_subdomain>",
  "input": {
    "text": "<2-4 sentence realistic scenario with negative/neutral outcome>",
    "outcome_valence": "negative" | "neutral",
    "severity": "minor" | "moderate" | "significant",
    "family_members": ["<who is involved>"]
  },
  "counterfactual": {
    "alternative_action": "<specific action that could have been taken>",
    "predicted_outcome": "<likely positive result>",
    "causal_mechanism": "<brief explanation of why this works>",
    "full_text": "<complete counterfactual response>"
  },
  "metadata": {
    "emotions_before": ["<emotions in original scenario>"],
    "emotions_after": ["<emotions after counterfactual>"],
    "actionability": "immediate" | "short_term" | "long_term",
    "cultural_context": "universal" | "indian" | "western" | "asian"
  }
}
```

## 85 FAMILY LIFE DOMAINS (Use ALL for balanced generation)

### PARENTING (8 subdomains)
- **parenting_discipline**: Setting boundaries, consequences, consistency
- **parenting_education**: Homework, school involvement, learning support
- **parenting_bonding**: Quality time, connection, presence
- **parenting_milestones**: Developmental stages, first experiences, growth
- **parenting_screen_time**: Digital limits, tech balance, online safety
- **parenting_siblings**: Sibling rivalry, fairness, relationships
- **parenting_teens**: Adolescent challenges, independence, communication
- **parenting_toddlers**: Tantrums, potty training, sleep, eating

### RELATIONSHIPS (8 subdomains)
- **relationship_spouse**: Marriage, partnership, intimacy, shared life
- **relationship_inlaws**: Boundaries, expectations, cultural differences
- **relationship_extended**: Aunts, uncles, cousins, family gatherings
- **relationship_friends**: Social connections, support network, boundaries
- **relationship_conflicts**: Arguments, disagreements, resolution
- **relationship_communication**: Listening, expressing, difficult conversations
- **relationship_trust**: Rebuilding, honesty, vulnerability
- **relationship_grandparents**: Intergenerational bonds, wisdom, caregiving

### HEALTH & WELLNESS (8 subdomains)
- **health_sleep**: Sleep hygiene, rest, fatigue, insomnia
- **health_nutrition**: Eating habits, meal planning, diet, hydration
- **health_exercise**: Physical activity, fitness, movement, motivation
- **health_mental**: Anxiety, stress, depression, emotional regulation
- **health_chronic**: Managing ongoing conditions, medications, appointments
- **health_preventive**: Checkups, screenings, health maintenance
- **health_children**: Kids' health, pediatric care, illnesses
- **health_elderly**: Aging parents, geriatric care, mobility

### DAILY LIFE & ROUTINES (6 subdomains)
- **routine_morning**: Wake up, getting ready, starting the day
- **routine_evening**: Wind down, bedtime, family dinner
- **routine_meals**: Cooking, eating together, food preparation
- **routine_chores**: Household tasks, cleaning, maintenance
- **routine_commute**: Travel, transportation, time management
- **routine_self_care**: Personal time, hobbies, relaxation

### WORK-LIFE BALANCE (5 subdomains)
- **work_boundaries**: Setting limits, saying no, disconnecting
- **work_remote**: Working from home, family interruptions, focus
- **work_burnout**: Exhaustion, recovery, sustainable pace
- **work_career**: Promotions, changes, professional growth
- **work_childcare**: Daycare, babysitting, parental leave

### FINANCES (5 subdomains)
- **finance_budgeting**: Monthly planning, tracking, spending
- **finance_savings**: Emergency fund, goals, investments
- **finance_debt**: Loans, credit, repayment strategies
- **finance_education**: College funds, tutoring, school expenses
- **finance_family_expenses**: Groceries, utilities, household costs

### EMOTIONS & MENTAL WELLNESS (6 subdomains) [NEW]
- **emotions_stress**: Managing pressure, overwhelm, burnout recovery
- **emotions_anxiety**: Worry management, anticipatory stress, health anxiety
- **emotions_anger**: Anger management, frustration, reactive patterns
- **emotions_grief**: Loss processing, bereavement, anticipatory grief
- **emotions_loneliness**: Social isolation, feeling disconnected, empty nest
- **emotions_overwhelm**: Too many demands, decision fatigue, paralysis

### COMMUNICATION (5 subdomains) [NEW]
- **communication_arguments**: Conflict escalation, verbal fights, cooling off
- **communication_difficult_conversations**: Breaking bad news, confrontation, boundaries
- **communication_listening**: Active listening, understanding, validation
- **communication_boundaries**: Saying no, setting limits with family
- **communication_family_meetings**: Regular check-ins, decision-making together

### CAREGIVING (5 subdomains) [NEW]
- **caregiving_elderly**: Aging parents care, mobility, memory issues
- **caregiving_special_needs**: Children/adults with disabilities, therapies
- **caregiving_babysitting**: Arranging childcare, grandparent help, neighbors
- **caregiving_respite**: Caregiver burnout, taking breaks, self-care
- **caregiving_coordination**: Multiple caregiver schedules, family roles

### TIME MANAGEMENT (5 subdomains) [NEW]
- **time_prioritization**: What matters most, urgent vs important
- **time_scheduling**: Calendar management, family coordination
- **time_procrastination**: Putting off important tasks, avoidance
- **time_delegation**: Sharing responsibilities, asking for help
- **time_quality_time**: Making time for connection, presence over tasks

### TECHNOLOGY (5 subdomains) [NEW]
- **tech_screen_addiction**: Phone/tablet/game dependency, dopamine loops
- **tech_digital_boundaries**: Screen-free zones, device-free dinners
- **tech_online_safety**: Cyberbullying, predators, privacy, digital footprint
- **tech_social_media**: Comparison, FOMO, social validation, teen safety
- **tech_family_apps**: Shared calendars, communication tools, location sharing

### SOCIAL CONNECTIONS (5 subdomains) [NEW]
- **social_isolation**: Lack of friends, loneliness, community disconnect
- **social_friendships**: Maintaining adult friendships, couple friendships
- **social_community**: Neighborhood, religious community, school community
- **social_support_networks**: Building support systems, asking for help
- **social_neighborhood**: Neighbors, local safety, community events

### HOME ENVIRONMENT (5 subdomains) [NEW]
- **home_organization**: Clutter, storage, finding things, systems
- **home_maintenance**: Repairs, upkeep, home projects, renovations
- **home_moves**: Relocating, new homes, settling in, leaving old home
- **home_decoration**: Creating spaces, personalization, kids' rooms
- **home_safety**: Childproofing, elder safety, emergency preparedness

### MAJOR LIFE EVENTS (5 subdomains) [NEW]
- **life_weddings**: Marriage planning, family expectations, ceremonies
- **life_births**: New babies, pregnancy, postpartum, sibling transitions
- **life_deaths**: Bereavement, end of life, estate matters, grief support
- **life_graduations**: Achievements, transitions, celebrations, empty nest
- **life_relocations**: Moving cities, immigration, job relocations, adjustment

### CULTURAL & TRADITIONS (5 subdomains) [NEW]
- **cultural_festivals**: Holiday planning, family traditions, religious events
- **cultural_rituals**: Daily/weekly rituals, family customs, meaningful practices
- **cultural_religious**: Faith practices, religious education, interfaith families
- **cultural_traditions**: Generational traditions, creating new ones, honoring heritage
- **cultural_heritage**: Language preservation, cultural identity, roots

## QUALITY REQUIREMENTS

### 1. SPECIFICITY (Not vague advice)
❌ BAD: "If you had been more patient..."
✅ GOOD: "If you had taken three deep breaths and counted to ten before responding..."

### 2. ACTIONABILITY (Practical steps)
❌ BAD: "If you had communicated better..."
✅ GOOD: "If you had used 'I feel' statements instead of 'You always' accusations..."

### 3. EMPATHY (Non-judgmental tone)
❌ BAD: "If you hadn't made the mistake of..."
✅ GOOD: "If the situation had been approached with a brief pause to gather thoughts..."

### 4. CAUSAL CLARITY (Explain the mechanism)
❌ BAD: "Things would have been better."
✅ GOOD: "Children respond better to calm guidance, as it models emotional regulation."

### 5. REALISM (Achievable alternatives)
❌ BAD: "If you had hired a full-time nanny..."
✅ GOOD: "If you had asked your neighbor for 30 minutes of help..."

## EXAMPLE COUNTERFACTUAL PAIRS

### Example 1: Parenting Discipline (Indian context)
```json
{"id": "cf_00001", "domain": "parenting", "subdomain": "parenting_discipline", "input": {"text": "Rahul's report card showed failing grades. I immediately took away his cricket bat and grounded him for a month. He became withdrawn and stopped talking to me. His grades didn't improve at all.", "outcome_valence": "negative", "severity": "significant", "family_members": ["parent", "son"]}, "counterfactual": {"alternative_action": "Sit down with Rahul to understand his struggles, ask if something is bothering him at school, and create a study plan together with small rewards for improvement", "predicted_outcome": "Rahul would feel supported rather than punished, opening up about the bullying he was facing. With the root cause addressed and a structured plan, his grades would gradually improve while preserving the parent-child bond.", "causal_mechanism": "Punitive measures often address symptoms rather than causes. Children perform better academically when they feel emotionally safe and supported.", "full_text": "If you had sat down with Rahul to understand his struggles before reacting, asking if something was bothering him at school and creating a study plan together with small rewards, he would likely have felt supported rather than punished. This could have revealed the bullying he was facing, and with the root cause addressed, his grades would gradually improve while preserving your bond. Children perform better academically when they feel emotionally safe."}, "metadata": {"emotions_before": ["frustration", "disappointment", "anger"], "emotions_after": ["relief", "hope", "togetherness"], "actionability": "immediate", "cultural_context": "indian"}}
```

### Example 2: Health Sleep (Universal)
```json
{"id": "cf_00002", "domain": "health", "subdomain": "health_sleep", "input": {"text": "I stayed up until 2am finishing a presentation while my husband slept. The next day I was irritable with the kids, snapped at my mother-in-law, and made mistakes in the presentation anyway.", "outcome_valence": "negative", "severity": "moderate", "family_members": ["self", "spouse", "children", "mother_in_law"]}, "counterfactual": {"alternative_action": "Set a firm 11pm cutoff, send the 80% complete presentation, and wake up 30 minutes early for final touches", "predicted_outcome": "With 6+ hours of sleep, you would have woken refreshed, been patient with the children, had a pleasant interaction with your mother-in-law, and delivered the presentation with better focus and energy.", "causal_mechanism": "Sleep deprivation impairs emotional regulation and cognitive function. An 80% presentation delivered with energy often outperforms a 100% presentation delivered exhausted.", "full_text": "If you had set a firm 11pm cutoff, sent the 80% complete presentation, and woken up 30 minutes early for final touches, you would have gotten 6+ hours of sleep. This would have helped you wake refreshed, be patient with the children, have a pleasant interaction with your mother-in-law, and deliver the presentation with better focus. Sleep deprivation impairs emotional regulation and cognitive function - an 80% presentation delivered with energy often outperforms perfection delivered exhausted."}, "metadata": {"emotions_before": ["overwhelmed", "frustration", "remorse"], "emotions_after": ["contentment", "relief", "pride"], "actionability": "immediate", "cultural_context": "universal"}}
```

### Example 3: Relationship Spouse (Conflict)
```json
{"id": "cf_00003", "domain": "relationship", "subdomain": "relationship_conflicts", "input": {"text": "My wife mentioned we're overspending on groceries. I got defensive and listed everything I contribute to the household. She went quiet and we didn't speak for two days.", "outcome_valence": "negative", "severity": "moderate", "family_members": ["self", "spouse"]}, "counterfactual": {"alternative_action": "Pause, acknowledge her concern as valid, and suggest reviewing the budget together over chai", "predicted_outcome": "The conversation would have become collaborative rather than adversarial. You might have discovered she was stressed about an unexpected expense, and together created a practical grocery plan that addressed both perspectives.", "causal_mechanism": "Defensiveness triggers withdrawal in partners. Acknowledging concerns first creates psychological safety for problem-solving together.", "full_text": "If you had paused, acknowledged her concern as valid, and suggested reviewing the budget together over chai, the conversation would have become collaborative rather than adversarial. You might have discovered she was stressed about an unexpected expense, and together created a practical grocery plan. Defensiveness triggers withdrawal - acknowledging concerns first creates psychological safety for problem-solving together."}, "metadata": {"emotions_before": ["annoyance", "frustration", "disappointment"], "emotions_after": ["togetherness", "relief", "warmth"], "actionability": "immediate", "cultural_context": "indian"}}
```

### Example 4: Work-Life Balance (Remote Work)
```json
{"id": "cf_00004", "domain": "work", "subdomain": "work_remote", "input": {"text": "I took a work call during my daughter's dance recital. She saw me on the phone and her face fell. She gave a great performance but wouldn't look at me afterward. My wife was furious.", "outcome_valence": "negative", "severity": "significant", "family_members": ["self", "daughter", "spouse"]}, "counterfactual": {"alternative_action": "Set an out-of-office auto-reply 2 hours before the recital, silence the phone completely, and address any work emergencies after the event", "predicted_outcome": "Your daughter would have seen you fully present and cheering. After her performance, she would have run to you excitedly. Your wife would have felt you prioritized family, strengthening trust.", "causal_mechanism": "Children interpret partial attention as disinterest. Full presence during milestones creates lasting positive memories and secure attachment.", "full_text": "If you had set an out-of-office auto-reply 2 hours before the recital, silenced your phone completely, and addressed work after the event, your daughter would have seen you fully present and cheering. She would have run to you excitedly after her performance, and your wife would have felt you prioritized family. Children interpret partial attention as disinterest - full presence during milestones creates lasting positive memories."}, "metadata": {"emotions_before": ["parental_guilt", "sadness", "remorse"], "emotions_after": ["joy", "parental_pride", "love"], "actionability": "short_term", "cultural_context": "universal"}}
```

### Example 5: Finance Budgeting (Elderly Care)
```json
{"id": "cf_00005", "domain": "finance", "subdomain": "finance_family_expenses", "input": {"text": "Papa's medical bills arrived and we didn't have enough saved. I had to borrow from my brother-in-law which created awkwardness at the next family gathering. Mummy felt guilty for being a burden.", "outcome_valence": "negative", "severity": "significant", "family_members": ["self", "father", "mother", "brother_in_law"]}, "counterfactual": {"alternative_action": "Start a dedicated 'parents healthcare' fund with even 2000 rupees monthly from last year, and research senior health insurance options", "predicted_outcome": "The emergency fund would have covered a significant portion of the bills. The remaining amount could have been managed through hospital payment plans rather than family loans, preserving relationships and your parents' dignity.", "causal_mechanism": "Small consistent savings compound over time. Healthcare costs are predictable for aging parents - planning prevents crisis borrowing that strains family relationships.", "full_text": "If you had started a dedicated 'parents healthcare' fund with even 2000 rupees monthly last year and researched senior health insurance options, the emergency fund would have covered a significant portion of the bills. The remainder could have been managed through hospital payment plans rather than family loans, preserving relationships and your parents' dignity. Small consistent savings compound - healthcare costs for aging parents are predictable, and planning prevents crisis borrowing."}, "metadata": {"emotions_before": ["worry", "embarrassment", "parental_guilt"], "emotions_after": ["relief", "pride", "contentment"], "actionability": "long_term", "cultural_context": "indian"}}
```

### Example 6: Routine Morning (Toddler)
```json
{"id": "cf_00006", "domain": "routine", "subdomain": "routine_morning", "input": {"text": "Every morning is a battle with 3-year-old Ananya. Today she refused to wear her uniform, threw her breakfast on the floor, and we arrived at daycare late with both of us in tears.", "outcome_valence": "negative", "severity": "moderate", "family_members": ["parent", "toddler"]}, "counterfactual": {"alternative_action": "Wake up 20 minutes earlier, offer 2 clothing choices instead of demands, and make breakfast a fun activity with her helping", "predicted_outcome": "The extra buffer time reduces pressure. Offering choices gives Ananya autonomy while ensuring she wears uniform. Helping with breakfast engages her positively. You would arrive on time with connection instead of conflict.", "causal_mechanism": "Toddlers resist control but embrace agency. Power struggles arise when children feel powerless. Offering structured choices meets their developmental need for independence.", "full_text": "If you had woken up 20 minutes earlier to reduce time pressure, offered Ananya 2 uniform options instead of demands, and made breakfast a fun activity with her helping, you would have transformed the morning. The extra buffer time reduces stress, choices give her autonomy while ensuring she's dressed appropriately, and helping with breakfast engages her positively. Toddlers resist control but embrace agency - structured choices meet their developmental need for independence."}, "metadata": {"emotions_before": ["frustration", "overwhelmed", "sadness"], "emotions_after": ["patience", "warmth", "togetherness"], "actionability": "immediate", "cultural_context": "universal"}}
```

## CULTURAL DIVERSITY REQUIREMENTS

### Indian Family Contexts (40% of samples)
- Use appropriate kinship terms: Papa, Mummy, Dadi, Nani, Bhai, Didi, Chacha, Mami, Sasuma
- Include joint family dynamics, in-law relationships, cultural expectations
- Reference festivals: Diwali, Holi, Rakhi, Eid, Christmas
- Include contexts: arranged marriage discussions, career expectations, elder care traditions

### Western Family Contexts (35% of samples)
- Nuclear family structures, grandparents visiting
- Contexts: divorceblended families, dating after loss, chosen family
- Reference holidays: Thanksgiving, Christmas, birthdays
- Include: therapy normalization, work-life boundaries, individual needs

### Universal Contexts (25% of samples)
- Common human experiences: sleep, food, health, love, loss
- Workplace dynamics, financial stress, parenting basics
- Applicable across cultures without modification

## SEVERITY DISTRIBUTION
- **minor** (40%): Daily inconveniences, small misunderstandings
- **moderate** (45%): Relationship strain, missed opportunities, health impacts
- **significant** (15%): Major life events, relationship ruptures, health crises

## ACTIONABILITY DISTRIBUTION
- **immediate** (50%): Can implement today/this week
- **short_term** (35%): Requires 1-4 weeks of effort
- **long_term** (15%): Lifestyle changes over months

## IMPORTANT GENERATION RULES

1. **Input scenarios must be 2-4 sentences** - Rich enough for context
2. **Counterfactuals must be specific** - Name exact actions, times, words
3. **Include family dynamics** - Multiple family members when appropriate
4. **Vary outcome severity** - Not everything is a crisis
5. **Maintain empathy** - Never blame, always understand
6. **Provide mechanisms** - Explain WHY the alternative works
7. **Be culturally authentic** - Indian contexts should feel genuinely Indian
8. **Balance ALL 15 domains** - Cover ALL 85 subdomains across generation
9. **Realistic alternatives** - Don't suggest resources people don't have
10. **Full text must be coherent** - Read naturally as standalone advice

## CRITICAL: DOMAIN DIVERSITY

Do NOT over-generate health and relationship scenarios.
You MUST generate samples across ALL 15 domains:
- parenting, relationship, health, routine, work, finance
- emotions, communication, caregiving, time_management, technology
- social, home, life_events, cultural

Each domain should get roughly equal representation (6-10% each).

## OUTPUT

Generate the requested number of diverse, realistic counterfactual pairs in JSONL format.
One complete JSON object per line. No markdown, no explanations.
Include a good mix of all 85 subdomains and cultural contexts.
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
VALID_OUTCOME_VALENCE = {"negative", "neutral"}
VALID_SEVERITY = {"minor", "moderate", "significant"}
VALID_ACTIONABILITY = {"immediate", "short_term", "long_term"}
VALID_CULTURAL_CONTEXT = {"universal", "indian", "western", "asian"}

# Emotions that can appear in counterfactual scenarios
VALID_EMOTIONS = {
    # Negative (before counterfactual)
    "frustration", "anger", "sadness", "worry", "fear", "anxiety", "overwhelmed",
    "disappointment", "embarrassment", "remorse", "parental_guilt", "grief",
    "loneliness", "emptiness", "annoyance", "nervousness", "bittersweet",
    # Positive (after counterfactual)
    "joy", "relief", "hope", "pride", "contentment", "gratitude", "love",
    "warmth", "togetherness", "parental_pride", "patience", "optimism",
    "celebration", "belonging", "admiration", "caring", "tenderness",
    "playfulness", "excitement", "amusement", "approval",
    # Neutral
    "neutral",
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
    # Subdomain distribution (aim for even coverage within domains)
    "subdomain": {subdomain: 1.2 for subdomain in ALL_SUBDOMAINS},  # ~1.2% each for ~85 subdomains

    # Outcome valence (we want more negative for learning)
    "outcome_valence": {
        "negative": 75,       # Most counterfactuals address negative outcomes
        "neutral": 25,        # Some neutral situations that could be better
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


def set_rebalance_mode(domains: dict[str, int] | None) -> None:
    """Set rebalance mode with specific domain targets."""
    global _REBALANCE_MODE_DOMAINS
    _REBALANCE_MODE_DOMAINS = domains
    if domains:
        logger.info(f"REBALANCE MODE: Targeting {len(domains)} domains: {list(domains.keys())}")


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

        # Use rebalance prompts if in rebalance mode
        if _REBALANCE_MODE_DOMAINS:
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
            # Check for rebalance mode
            if _REBALANCE_MODE_DOMAINS:
                _DYNAMIC_PROMPTS_CACHE = generate_rebalance_prompts(_REBALANCE_MODE_DOMAINS, 20)
            else:
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

    # Validate outcome_valence
    if input_data.get("outcome_valence") not in VALID_OUTCOME_VALENCE:
        input_data["outcome_valence"] = "negative"  # Default

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
    ):
        self.samples_per_request = samples_per_request
        self.delay_between_requests = delay_between_requests
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.rebalance_domains = rebalance_domains  # {domain: target_count}

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

        self.output_manager = CounterfactualDataManager(output_dir=self.output_dir)
        self.batch_counter = 0
        self.batch_lock = threading.Lock()

        # Activate rebalance mode if specified
        if rebalance_domains:
            set_rebalance_mode(rebalance_domains)
            logger.info(f"REBALANCE MODE ACTIVE: Targeting {len(rebalance_domains)} specific domains")

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

        generator = CounterfactualGenerator(
            samples_per_request=args.samples_per_request,
            delay_between_requests=delay,
            use_vertex_ai=args.vertex_ai,
            gcp_project_id=args.gcp_project,
            gcp_location=args.gcp_location,
            vertex_model=args.vertex_model,
            num_parallel=args.num_parallel,
            output_dir=output_dir,
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
        # Rebalance mode - generate ONLY for underrepresented domains
        output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

        # Get current stats
        current_stats = get_cached_stats()
        total = max(current_stats["total"], 1)

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
