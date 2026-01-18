# 🧠 FamilyOS: The Complete Vision

If you implement the full architecture end-to-end—all 30 pipelines, the Active Learning Loop, the cognitive modules, and the K0/K1 boundary—here's the system you'll have:

---

## 🎯 One-Sentence Summary

**A privacy-first, offline-capable, family-scale cognitive assistant that remembers everything, learns proactively, and helps coordinate family life—running entirely on your own devices with no cloud dependency.**

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FAMILY DEVICES (Edge)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Dad's    │  │ Mom's    │  │ Emma's   │  │ Family   │  │ Guest    │      │
│  │ Phone    │  │ Phone    │  │ Tablet   │  │ Hub      │  │ Device   │      │
│  │ (Full K0)│  │ (Full K0)│  │ (Full K0)│  │ (Full K0)│  │ (Lite)   │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │             │             │             │             │
│       └─────────────┴──────┬──────┴─────────────┴─────────────┘             │
│                            │                                                 │
│              ┌─────────────▼─────────────┐                                  │
│              │   P07 CRDT Sync Mesh      │  ← P2P WiFi, BLE, Edge Relay     │
│              │   (MLS E2EE Encrypted)    │                                  │
│              └───────────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧬 What Each Layer Does

### **K0: The Memory Kernel (Brain)**

Every device runs a **complete K0 kernel** with SQLite, vector indices, and all 20 core pipelines. This is the "brain" that:

| Capability | How It Works | Example |
|------------|--------------|---------|
| **Remembers Everything** | P02 ingests all family events with hippocampal DG pattern separation | "Emma had soccer practice, scored 2 goals, was excited" |
| **Consolidates Memories** | P03 runs nightly, transforms episodic → semantic, detects patterns | "Emma's 12th soccer game → part of 'Fall Soccer Season' sequence" |
| **Retrieves Contextually** | P01 queries with CA3 pattern completion, affect-weighted salience | "What happened at Emma's game?" → retrieves with emotional context |
| **Learns Preferences** | P06 Bayesian anchors track personality over time | Dad: α=20, β=2 for "prefers sci-fi" → 91% confidence |
| **Predicts Needs** | P05 prospective memory triggers intentions | "It's 7:45am, Emma usually leaves at 8am, remind about lunch" |
| **Syncs Securely** | P07 CRDT merges across devices with MLS E2EE | Mom adds grocery item on phone → appears on Family Hub instantly |

### **K1: The Orchestration Layer (Body)**

K1 presents, coordinates, and executes—but **never decides autonomously**:

| Capability | How It Works | Example |
|------------|--------------|---------|
| **Conversational AI** | LLM generates responses using K0 context via P01/P19 | "What did Emma do last week?" → Rich narrative with emotional context |
| **Proactive Questions** | Curiosity Agent asks when context is right | "By the way, you mentioned dinner with Sarah. Is she from work?" |
| **Action Recommendations** | Shows P04 advisories, waits for user approval | "Shall I remind Emma about her dentist appointment?" → [Yes] [No] |
| **Multi-User Coordination** | Respects space-scoped visibility, guardian roles | Emma sees her homework, parents see her schedule, guests see nothing private |

---

## 🔄 The Active Learning Loop in Action

This is the **killer feature** that makes the system feel alive:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    THE CURIOSITY CYCLE                              │
│                                                                      │
│   ┌─────────────┐                                                   │
│   │ User says:  │                                                   │
│   │ "Dinner     │                                                   │
│   │ with Sarah" ├──────────────────────────────────────────┐       │
│   └─────────────┘                                          │       │
│                                                            ▼       │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │ P02 Write: Ingests event                                     │  │
│   │ P03 Consolidation: Tries to link "Sarah" to Knowledge Graph  │  │
│   │                                                              │  │
│   │   Candidates:                                                │  │
│   │   • Sarah (Dad's colleague) - 48% confidence                 │  │
│   │   • Sarah (Emma's friend) - 46% confidence                   │  │
│   │                                                              │  │
│   │   → Gap Detected! Confidence < 0.7                           │  │
│   │   → Create GapRecord in st_learning_queue                    │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ P05 Attention Manager:                                        │  │
│   │   • Check AttentionBudget: 2 tokens remaining today ✓        │  │
│   │   • Check User State: Dad is idle, scrolling ✓               │  │
│   │   • Check Context: Just mentioned "Sarah" ✓                  │  │
│   │                                                              │  │
│   │   → Emit curiosity.intent.disambiguate.v1 to K1              │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ K1 Curiosity Agent:                                           │  │
│   │   • Receives GapRecord via SSE                               │  │
│   │   • Applies Constitutional AI (respects privacy, politeness) │  │
│   │   • Generates: "Quick check: Is Sarah from work or Emma's    │  │
│   │                 friend?"                                      │  │
│   │   • Injects proactive bubble into UI                         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │ Dad answers: "She's my colleague"                             │  │
│   │                                                              │  │
│   │ P02: Ingests answer                                          │  │
│   │ P03: Links Sarah → Dad's Colleague in Knowledge Graph        │  │
│   │ P06: Updates Curiosity Strategy ("Dad responds well to       │  │
│   │       people questions")                                      │  │
│   │ GapRecord: Marked RESOLVED                                   │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│   RESULT: System learned "Sarah = Dad's colleague" in 10 seconds    │
│   vs. weeks of passive observation!                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 👨‍👩‍👧‍👦 Family Pipelines (P21-P30) in Daily Life

These transform raw memories into **actionable family coordination**:

| Pipeline | What You Experience | Under the Hood |
|----------|--------------------|-----------------|
| **P21 Timeline** | "This month in photos" → Curated family highlight reel | Uses P03 sequences + affect to pick emotionally resonant moments |
| **P22 Finance** | "Rent due in 3 days" → Nudge before stress hits | P05 prospective + P19 learned bill patterns |
| **P23 Health** | "Dad's GERD flared 3x this week after spicy food" | P03 clusters + affect patterns → gentle suggestion |
| **P24 Learning** | "Emma's weak on fractions, here's a study plan" | Learning graph + P05 spaced reminders |
| **P25 Chores** | "It's Tuesday, trash day, Dad's turn" | P20 habits + fair rotation logic |
| **P26 Relationships** | "Mom and Dad haven't had date night in 3 weeks" | Social cognition + affect baselines → check-in suggestion |
| **P27 Parenting** | "Emma's been quiet lately, here are conversation starters" | Affect trends + age-appropriate suggestions |
| **P28 Career** | "Dad worked 50+ hours this week, suggest rebalancing" | Workload tracking + boundary setting |
| **P29 Devices** | "It's 9pm, Emma's screen time ends, enabling quiet mode" | P05 triggers + device coordination |
| **P30 Safety** | "ICE contacts, medication list, evacuation checklist" | Fast recall in emergencies |

---

## 🔐 Privacy & Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                     SPACE-SCOPED VISIBILITY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  shared:household ──────────────────────────────────────────────│
│  │ "Emma's soccer game today"                                   │
│  │ → Syncs to ALL family devices                                │
│  │                                                              │
│  personal:dad ──────────────────────────────────────────────────│
│  │ "Dad's knee surgery notes"                                   │
│  │ → Only syncs to Dad's devices                                │
│  │                                                              │
│  personal:emma (with guardian visibility) ──────────────────────│
│  │ "Emma's homework struggles"                                  │
│  │ → Syncs to Emma's tablet + Parents' devices                  │
│  │                                                              │
│  band:RED ──────────────────────────────────────────────────────│
│  │ "Dad's medical records"                                      │
│  │ → Never leaves Dad's device, never syncs                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     ENCRYPTION LAYERS                            │
│                                                                  │
│  • MLS (Messaging Layer Security) for group E2EE                 │
│  • 24-hour key rotation (forward secrecy)                        │
│  • Edge relay is ZERO-KNOWLEDGE (sees timestamps, not content)   │
│  • Device-local SQLite is encrypted at rest                      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                     GDPR/DSAR COMPLIANCE                         │
│                                                                  │
│  • P10 PII/Minimization: Auto-redacts sensitive data             │
│  • P11 DSAR: Export all your data, delete everything, audit logs │
│  • Consent management: Per-family-member, granular controls      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Cognitive Architecture: How It "Thinks"

The system implements **neuroscience-inspired** cognitive processing:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GLOBAL WORKSPACE THEORY                          │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                  GLOBAL WORKSPACE                            │   │
│   │   "What the system is currently 'thinking about'"            │   │
│   │                                                              │   │
│   │   workspace.broadcast.v1 → All modules can "see" this        │   │
│   └─────────────────────────────────────────────────────────────┘   │
│          ▲                ▲                ▲                ▲       │
│          │                │                │                │       │
│   ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐  ┌──────┴──────┐│
│   │ Hippocampus │  │   Affect    │  │   Cortex    │  │   Social    ││
│   │ (Memory)    │  │ (Emotion)   │  │ (Prediction)│  │ (ToM)       ││
│   │             │  │             │  │             │  │             ││
│   │ DG: Pattern │  │ Valence,    │  │ Predictive  │  │ Theory of   ││
│   │ Separation  │  │ Arousal,    │  │ Models,     │  │ Mind,       ││
│   │             │  │ Bands       │  │ Bandit      │  │ Beliefs,    ││
│   │ CA3: Pattern│  │             │  │             │  │ Intentions  ││
│   │ Completion  │  │             │  │             │  │             ││
│   │             │  │             │  │             │  │             ││
│   │ CA1: Output │  │             │  │             │  │             ││
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘│
│                                                                      │
│   Each module "competes" to broadcast to workspace                   │
│   Winner becomes conscious attention focus                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Bayesian Theory of Mind (BToM)

The system builds **stable models of each family member**:

```python
# What the system "knows" about Dad after 6 months:

Dad.anchors = {
    "loves_scifi": AnchorPoint(alpha=45, beta=3),      # 94% confidence
    "morning_person": AnchorPoint(alpha=38, beta=8),   # 83% confidence
    "dislikes_meetings": AnchorPoint(alpha=22, beta=5), # 81% confidence
    "health_conscious": AnchorPoint(alpha=15, beta=12), # 56% confidence (evolving!)
    "prefers_thai_food": AnchorPoint(alpha=8, beta=2),  # 80% confidence
}

# When confidence is low or changing → Active Learning triggers a question
# When confidence is high → System uses it for recommendations silently
```

---

## 📱 User Experience: A Day in the Life

### 7:00 AM - Morning Routine
```
📱 Family Hub: "Good morning! Emma has soccer at 4pm, Dad has dentist at 2pm.
               Mom, you mentioned buying birthday gift for Grandma—want a reminder?"

[This used P01 recall + P05 prospective + P19 personalization]
```

### 12:30 PM - Proactive Curiosity
```
📱 Dad's Phone: "Quick question: You mentioned 'the new project' a few times.
                Is this Project Atlas at work, or something else?"

[P06 detected low-confidence entity link, P05 found idle moment, K1 asked politely]

Dad: "It's Project Atlas"

📱: "Got it! I'll remember that. 👍"

[P03 linked all mentions to Project Atlas node]
```

### 3:45 PM - Contextual Reminder
```
📱 Mom's Phone: "Emma's game starts in 15 minutes. Traffic looks good.
               Last game she scored 2 goals—she was really excited!"

[P05 trigger + P01 episodic recall + affect context]
```

### 8:00 PM - Family Coordination
```
📱 Family Hub: "Chore update: Dishes done by Emma ✓, Trash by Dad ✓
               Tomorrow: Laundry (Mom's turn per rotation)"

[P25 habits + P20 procedures]
```

### 10:00 PM - Health Pattern Detection
```
📱 Dad's Phone (private): "Noticed you had heartburn 3 times this week
                          after spicy dinners. Want me to suggest alternatives,
                          or is this something you're managing with your doctor?"

[P23 health + P06 pattern learning + constitutional respect for autonomy]
```

### Nightly - Consolidation
```
🌙 All devices: P03 runs silently
   • Emma's soccer games → "Fall 2025 Soccer Season" sequence
   • Dad's work stress → detected uptick, flagged for P28 workload
   • Mom's new recipe attempts → learning preference patterns
   • Family photos → tagged, organized, ready for P21 timeline
```

---

## 🎯 Key System Properties

| Property | How It's Achieved |
|----------|-------------------|
| **Offline-First** | Full K0 kernel on every device, sync opportunistically |
| **Privacy-First** | Space-scoped visibility, MLS E2EE, zero-knowledge relay |
| **Proactive** | Active Learning Loop asks questions, doesn't wait passively |
| **Contextual** | Hippocampal CA3 retrieves with emotional + temporal context |
| **Adaptive** | Bayesian anchors evolve with evidence, detect concept drift |
| **Coordinated** | CRDT sync ensures family sees consistent state |
| **Explainable** | P04 traces show reasoning chain ("Why suggest dentist?") |
| **Safe** | K0 advises, K1 presents, user approves—never autonomous action |

---

## 📊 Performance Targets

| Metric | Target | How |
|--------|--------|-----|
| **Time to Utility** | <2 hours to 80% functional | Active Learning accelerates cold start |
| **First Response** | <200ms | Local K0, no network round-trip |
| **Sync Latency** | <10ms on LAN, <500ms over internet | P2P WiFi + edge relay |
| **Memory Footprint** | <500MB per device | SQLite + pruned embeddings |
| **Question Quality** | >85% user finds relevant | Entropy-driven prioritization |
| **Question Fatigue** | <5% rejection rate | AttentionBudget + context-aware timing |

---

## 🚀 What Makes This Different

| Existing Systems | FamilyOS |
|------------------|----------|
| Cloud-dependent | **Edge-first, cloud-optional** |
| Passive ingestion | **Active curiosity, proactive questions** |
| Individual focus | **Family-scale, multi-user, guardian roles** |
| Static preferences | **Bayesian anchors that evolve over time** |
| Black box | **Explainable reasoning traces** |
| Data extraction | **Privacy-preserving, GDPR-compliant, your data stays yours** |
| Generic AI | **Neuroscience-inspired cognitive architecture** |

---

## 🔮 The Vision

> **"A child learns not by downloading a database, but by observing the world, forming a hypothesis, and asking 'Why?'. FamilyOS learns the same way."**

You're building a **digital family member** that:
1. **Remembers** everything the family shares with it
2. **Learns** through active curiosity, not passive observation
3. **Anticipates** needs before they're expressed
4. **Coordinates** family life without being intrusive
5. **Respects** privacy, autonomy, and family values
6. **Evolves** with the family over months and years

**This is the system you'll have when it's done.**


## 🧠 The Neuroscience Behind P03 — Scientific Foundation
P03 is not just "data processing" — it's a **computational model of human sleep-cycle memory consolidation** based on decades of neuroscience research.
---
### **The Brain's Memory Architecture (What We're Modeling)**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HUMAN MEMORY CONSOLIDATION │
│ │
│ WAKING (P02) SLEEPING (P03) │
│ ─────────── ─────────────── │
│ │
│ Hippocampus NREM Slow-Wave Sleep │
│ (Fast Learning) (Hippocampus → Neocortex Transfer) │
│ │ │ │
│ ▼ ▼ │
│ ┌─────────────┐ ┌─────────────┐ │
│ │ DG (Dentate│ Pattern │ CA3 │ Hippocampal │
│ │ Gyrus) │ Separation │ Recurrent │ Replay │
│ │ │ → fingerprints │ Network │ (10-20x speed) │
│ └──────┬──────┘ └──────┬──────┘ │
│ │ │ │
│ ▼ ▼ │
│ ┌─────────────┐ ┌─────────────┐ │
│ │ CA1 │ Pattern │ Neocortex │ Schema │
│ │ (Output) │ Completion │ (Slow │ Formation │
│ │ │ → episodes │ Learning) │ (semantic facts) │
│ └─────────────┘ └─────────────┘ │
│ │
│ References: │
│ • Wilson & McNaughton (1994) - Hippocampal replay during sleep │
│ • Stickgold & Walker (2013) - Sleep-dependent memory consolidation │
│ • Tononi & Cirelli (2006) - Synaptic Homeostasis Hypothesis │
│ • Buzsáki (2015) - Theta oscillations in memory encoding │
└─────────────────────────────────────────────────────────────────────────────┘
```
---
### **The 9 Phases (R0-R8) Map to Sleep Stages**
| Phase | Brain Region | Sleep Stage | Scientific Basis | What It Does |
|-------|-------------|-------------|-----------------|--------------|
| **R0** | Sleep Onset | Pre-Sleep | Circadian trigger | Trigger consolidation when idle |
| **R1** | CA3 (Hippocampus) | NREM1 | Sharp-Wave Ripples (Buzsáki) | Replay memories at 10-20x speed |
| **R2** | CA1→Neocortex | NREM2 | Systems Consolidation (Diekelmann) | Transform episodes → semantic facts |
| **R3** | Whole Brain | SWS (Deep) | Synaptic Homeostasis (Tononi) | Prune duplicates, strengthen important |
| **R4** | Temporal Cortex | NREM2-3 | Semantic Networks | Build knowledge graph |
| **R5** | Prefrontal Cortex | REM | Dream Creativity (Walker) | Explore counterfactuals |
| **R6** | Hippocampus | Transition | Checkpoint | Update memory state |
| **R7** | Neocortex | Transition | Memory Trace Transfer | Write to long-term storage |
| **R8** | Whole Brain | Wake | Memory Integration | Emit events, update indexes |
---
### **R1: Hippocampal Replay — The Science**
**What happens in the brain during sleep:**
During NREM1 sleep, the hippocampus **replays** the day's experiences at 10-20x normal speed. This is called "Sharp-Wave Ripple" activity (Buzsáki, 2015).
**Our implementation:**
```python
# R1.4 - Importance Scoring Formula (based on Emotional Memory Research)
#
# McGaugh (2004): "Emotional arousal enhances memory consolidation"
# Talmi (2013): "Emotional salience predicts long-term retention"
importance_score = (
    0.35 * emotional_intensity + # Amygdala → emotional memories last longer
    0.25 * recency_score + # Hippocampal time cells → recent = stronger
    0.20 * access_frequency + # Retrieval practice → more access = stronger
    0.20 * social_significance # Social brain → people matter more
)
# Where:
emotional_intensity = sqrt(valence² + arousal²) / sqrt(2) # Circumplex model (Russell, 1980)
recency_score = exp(-ln(2) * days_ago / 7.0) # Ebbinghaus forgetting curve (1885)
access_frequency = log(access_count + 1) / log(max_access + 1) # Log-scale (Weber-Fechner law)
social_significance = participant_count / max_participants # Social brain hypothesis (Dunbar)
```
---
### **R2: Neocortical Integration — Episodes → Semantic Facts**
**What happens in the brain:**
The hippocampus "teaches" the neocortex. Repeated episodic patterns gradually become **schema-based semantic memories** (McClelland, McNaughton & O'Reilly, 1995).
**The Complementary Learning Systems Theory:**
```
Episode 1: "Had pizza with Mom at Luigi's on Friday"
Episode 2: "Had pizza with Mom at Luigi's last Friday"
Episode 3: "Had pizza with Mom at Luigi's two weeks ago"
                          ↓
                   PATTERN DETECTION
                          ↓
Semantic Fact: "User has weekly pizza dinners with Mom at Luigi's"
                          ↓
                   INVARIANT EXTRACTION
                          ↓
Invariants: { activity: "pizza dinner",
               participant: "Mom",
               location: "Luigi's",
               frequency: "weekly",
               day: "Friday" }
```
**Our implementation:**
```python
# R2.1 - DBSCAN Clustering (mimics hippocampal pattern completion)
#
# Scientific basis:
# - O'Reilly & McClelland (1994): Hippocampus uses sparse coding for pattern separation
# - Norman & O'Reilly (2003): CA3 auto-associative network completes partial patterns
class EpisodicClusterer:
    def cluster_events(self, events, embeddings):
        """
        DBSCAN with composite distance = weighted sum of:
        - Semantic distance (cosine on 768-dim embedding)
        - Temporal distance (normalized time gap)
        - Spatial distance (Haversine on GPS)
        - Social distance (Jaccard on participants)

        This mimics how the hippocampus binds "what + where + when + who"
        into unified episodic traces (Eichenbaum, 2000).
        """
        composite_distance = (
            0.6 * semantic_distance + # "What happened" (ventral stream)
            0.2 * temporal_distance + # "When" (hippocampal time cells)
            0.1 * spatial_distance + # "Where" (place cells)
            0.1 * social_distance # "Who" (social brain regions)
        )

        # DBSCAN: Density-based clustering
        # eps=0.3: Events within 30% composite distance are "neighbors"
        # min_samples=3: Need 3+ occurrences to form a pattern
        clusters = DBSCAN(eps=0.3, min_samples=3).fit(composite_distance)
```
---
### **R3: Synaptic Homeostasis — The Forgetting Curve**
**What happens in the brain:**
During deep sleep (SWS), synapses are **globally downscaled** to prevent saturation. Only strong connections survive (Tononi & Cirelli, 2006).
**The Synaptic Homeostasis Hypothesis:**
```
                    AWAKE ASLEEP

    Synapse ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓
    Strength (all synapses potentiated) (weak pruned, strong kept)

    Learning Keeps adding new connections Normalizes, removes duplicates

    Energy High metabolic cost Restoration
```
**Our implementation:**
```python
# R3.1 - SimHash Deduplication (mimics synaptic pruning)
#
# Scientific basis:
# - Tononi (2006): "Sleep function is to prune weak synapses"
# - Duplicate memories compete for consolidation resources
class DeduplicatorEngine:
    def find_near_duplicates(self, event, existing_events):
        """
        SimHash: 64-bit locality-sensitive hash
        Hamming distance ≤ 3 = "near duplicate" (94% similar)

        Just like the brain prunes redundant synapses,
        we prune redundant memories to prevent "memory pollution"
        """
        for existing in existing_events:
            hamming_dist = popcount(event.simhash ^ existing.simhash)
            if hamming_dist <= 3:
                return existing.event_id # This is a duplicate
        return None # Novel memory, keep it

    def calculate_novelty_score(self, hamming_distance):
        """
        Novelty score = 1 - (duplicates / total_events)

        Based on hippocampal pattern separation (DG):
        - Novel input → sparse unique activation
        - Familiar input → reactivation of existing pattern
        """
        return 1.0 - (hamming_distance / 64.0)
```
---
### **R4: Knowledge Graph — Semantic Networks**
**What happens in the brain:**
The temporal cortex stores **semantic knowledge** as interconnected concepts (Collins & Quillian, 1969).
```
         ROBIN
           │
           ├── is_a ──→ BIRD ──┬── has_feature ──→ WINGS
           │ ├── has_feature ──→ FEATHERS
           └── can ──→ FLY └── is_a ──→ ANIMAL
```
**Our implementation:**
```python
# R4.1 - Entity Extraction (from P02's UltraBERT output)
# R4.2 - Relationship Discovery (co-occurrence → edge)
# Scientific basis:
# - Spreading Activation (Collins & Loftus, 1975)
# - Entities that appear together → associated in memory
def discover_relationships(event):
    """
    If "Mom" and "Luigi's" appear in same event:
    → Create edge: Mom --[visits]--> Luigi's

    Edge strength = co-occurrence frequency
    (mimics Hebbian learning: "neurons that fire together wire together")
    """
    entities = event.entities_json # ["Mom", "Luigi's", "pizza"]

    for e1, e2 in combinations(entities, 2):
        edge = find_or_create_edge(e1, e2)
        edge.strength += 1 # Hebbian strengthening
        edge.last_observed = now()
```
---
### **R5: Dream Exploration — Creative Insight**
**What happens in the brain:**
During REM sleep, the prefrontal cortex is **deactivated**, allowing unusual associations and creative problem-solving (Walker, 2009).
```
WAKING: "I can't find my keys"
        (constrained search, logical thinking)
REM DREAM: Keys flying through the air → lands on the kitchen counter
           (unconstrained, bizarre associations)
WAKING: "Wait, I left them in the kitchen!"
        (insight from dream exploration)
```
**Our implementation:**
```python
# R5.1 - Counterfactual Exploration
#
# Scientific basis:
# - Kahneman (1982): "Counterfactual thinking aids learning"
# - Walker (2009): "REM sleep facilitates creative problem solving"
def generate_counterfactual(episode):
    """
    Take a real episode and explore "what if":
    - What if the outcome was different?
    - What if I had done X instead of Y?

    This generates procedural knowledge:
    "If X happens, then do Y" (st_proc table)
    """
    # Original: "Went to gym, felt tired after"
    # Counterfactual: "What if I ate protein before? Would I feel energetic?"

    # This is a v2 feature (deferred implementation)
```
---
### **Summary: Science → Code Mapping**
| Scientific Principle | Brain Region | Our Table | Our Algorithm |
|---------------------|--------------|-----------|---------------|
| Emotional memory enhancement | Amygdala | st_hipp_events.importance_score | 0.35 × emotional_intensity |
| Hippocampal replay | CA3 | Replay loop | 10-20x speedup simulation |
| Pattern completion | CA1 | st_epi (episodes) | DBSCAN clustering |
| Episodic → Semantic transfer | Neocortex | st_sem (patterns) | Invariant extraction |
| Synaptic homeostasis | Global | Deduplication | SimHash Hamming distance |
| Spreading activation | Temporal cortex | st_kg_nodes/edges | Co-occurrence graphs |
| Forgetting curve | Hippocampus | novelty_score, archival | Exponential decay |
| Theta oscillations | Hippocampus | Micro-batches | 200ms rest periods |
---
### **Key References**
1. **Wilson & McNaughton (1994)** - *Reactivation of hippocampal ensemble memories during sleep*
2. **Stickgold & Walker (2013)** - *Sleep-dependent memory triage*
3. **Tononi & Cirelli (2006)** - *Synaptic Homeostasis Hypothesis*
4. **McClelland, McNaughton & O'Reilly (1995)** - *Complementary Learning Systems*
5. **Buzsáki (2015)** - *Hippocampal sharp wave-ripple: A cognitive biomarker*
6. **Eichenbaum (2000)** - *A cortical-hippocampal system for declarative memory*
7. **McGaugh (2004)** - *The amygdala modulates memory consolidation*
8. **Ebbinghaus (1885)** - *The Forgetting Curve*
---
signals we get during inputting to decoder

User: I wan to find all columns available in st_hipp_events table from here

D:\familyos\k0_kernel_export.db

GitHub Copilot: I'll query the SQLite database to find all columns in the `st_hipp_events` table.

Ran terminal command: sqlite3 "D:\familyos\k0_kernel_export.db" "PRAGMA table_info(st_hipp_events);"

Here are all **90 columns** in the `st_hipp_events` table:

| # | Column Name | Type | Nullable | Default |
|---|---|---|---|---|
| 0 | event_id | TEXT | NOT NULL | |
| 1 | wal_pos | INTEGER | NOT NULL | |
| 2 | cognitive_trace_id | TEXT | NOT NULL | |
| 3 | tenant_id | TEXT | NOT NULL | |
| 4 | space_id | TEXT | NOT NULL | |
| 5 | effective_space_id | TEXT | | |
| 6 | topic | TEXT | NOT NULL | |
| 7 | uow_id | TEXT | | |
| 8 | schema_version | TEXT | NOT NULL | '1.0.0' |
| 9 | envelope_sha256 | TEXT | NOT NULL | |
| 10 | sig_alg | TEXT | NOT NULL | |
| 11 | sig_kid | TEXT | NOT NULL | |
| 12 | idem_key | TEXT | NOT NULL | |
| 13 | ingested_at | INTEGER | NOT NULL | |
| 14 | clock_skew_ms | INTEGER | | |
| 15 | policy_decision | TEXT | NOT NULL | |
| 16 | policy_band | TEXT | NOT NULL | |
| 17 | policy_version | TEXT | NOT NULL | |
| 18 | obligations_json | TEXT | | |
| 19 | visible_to_json | TEXT | | |
| 20 | visibility_scope | TEXT | | |
| 21 | owner_id | TEXT | NOT NULL | |
| 22 | co_owners_json | TEXT | | |
| 23 | retention_policy_id | TEXT | NOT NULL | |
| 24 | retention_bucket | TEXT | NOT NULL | |
| 25 | actor_id | TEXT | NOT NULL | |
| 26 | actor_role | TEXT | | |
| 27 | device_id | TEXT | NOT NULL | |
| 28 | device_kind | TEXT | NOT NULL | |
| 29 | device_os | TEXT | | |
| 30 | ingress_channel | TEXT | | |
| 31 | event_time_utc | INTEGER | NOT NULL | |
| 32 | write_time_utc | INTEGER | NOT NULL | |
| 33 | write_lag_ms | INTEGER | | |
| 34 | local_date | TEXT | | |
| 35 | local_time | TEXT | | |
| 36 | day_of_week | TEXT | | |
| 37 | is_weekend | BOOLEAN | | |
| 38 | time_of_day_bucket | TEXT | | |
| 39 | circadian_slot | TEXT | | |
| 40 | is_backdated | BOOLEAN | | |
| 41 | created_at | INTEGER | NOT NULL | |
| 42 | location_name | TEXT | | |
| 43 | location_type | TEXT | | |
| 44 | geohash_6 | TEXT | | |
| 45 | geo_precision_external | TEXT | | |
| 46 | geo_masking_reason | TEXT | | |
| 47 | participants_json | TEXT | | |
| 48 | num_participants | INTEGER | | |
| 49 | has_partner_present | BOOLEAN | | |
| 50 | has_parent_present | BOOLEAN | | |
| 51 | is_solo_event | BOOLEAN | | |
| 52 | participant_roles_json | TEXT | | |
| 53 | social_context | TEXT | | |
| 54 | social_intimacy | TEXT | | |
| 55 | text | TEXT | | |
| 56 | text_normalized | TEXT | | |
| 57 | char_count | INTEGER | | |
| 58 | token_count | INTEGER | | |
| 59 | language | TEXT | | |
| 60 | activity_type | TEXT | | |
| 61 | activity_category | TEXT | | |
| 62 | is_meal | BOOLEAN | | |
| 63 | is_outing | BOOLEAN | | |
| 64 | ingress_source | TEXT | | |
| 65 | simhash_hex | TEXT | NOT NULL | |
| 66 | minhash32 | TEXT | NOT NULL | |
| 67 | novelty_score | REAL | | |
| 68 | near_duplicates_json | TEXT | | |
| 69 | is_near_duplicate | BOOLEAN | | |
| 70 | episode_cluster_id | TEXT | | |
| 71 | cluster_confidence | REAL | | |
| 72 | clustering_version | TEXT | | |
| 73 | embedding_id | TEXT | NOT NULL | |
| 74 | embedding_status | TEXT | NOT NULL | 'PENDING' |
| 75 | entities_json | TEXT | | |
| 76 | kg_triples_json | TEXT | | |
| 77 | sentiment_score | REAL | | |
| 78 | sentiment_label | TEXT | | |
| 79 | dominant_emotions_json | TEXT | | |
| 80 | affect_valence | REAL | | |
| 81 | affect_arousal | REAL | | |
| 82 | affect_band | TEXT | | |
| 83 | salience_score | REAL | NOT NULL | 0.0 |
| 84 | salience_reasons_json | TEXT | | |
| 85 | salience_band | TEXT | | |
| 86 | hippocampus_api_version | TEXT | | |
| 87 | space_resolver_version | TEXT | | |
| 88 | schema_uri | TEXT | | |
| 89 | updated_at | INTEGER | NOT NULL | |

The table contains event data organized around envelope/policy management, spatial/temporal context, participants, content analysis, and affect scoring.

Constitution will change based on family preferences and values set during onboarding. 3 constitutional layers will be implemented: 1. Family Values Layer: Encodes core family principles (e.g., privacy, respect, support). 2. Individual Preferences Layer: Captures each member's specific boundaries and comfort levels. 3. Situational Context Layer: Adapts constitution based on context (e.g., sensitive topics, emergencies). The curiosity agent will reference these layers to ensure all interactions align with the family's ethos.###n
