#!/usr/bin/env python3
"""Analyze generated unified data distribution."""

import json
from collections import Counter
from pathlib import Path


def analyze_generated_data(
    output_dirs: list[str] = None,
):
    """Analyze all generated shards from multiple directories and print distribution stats."""
    if output_dirs is None:
        output_dirs = [
            "D:/Modeling_studio/data/familyos/unified/output",
        ]

    # Load all samples from all directories
    samples = []
    total_by_dir = {}

    print("\n" + "=" * 70)
    print("LOADING DATA FROM MULTIPLE SOURCES")
    print("=" * 70)

    for output_dir in output_dirs:
        output_path = Path(output_dir)
        shards = sorted(output_path.glob("shard_*.jsonl"))

        dir_name = output_path.name
        dir_samples = []

        if not shards:
            print(f"  {dir_name}: No shards found")
            continue

        for shard in shards:
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    try:
                        sample = json.loads(line.strip())
                        dir_samples.append(sample)
                    except json.JSONDecodeError:
                        pass

        samples.extend(dir_samples)
        total_by_dir[dir_name] = len(dir_samples)
        print(f"  {dir_name}: {len(dir_samples):,} samples ({len(shards)} shards)")

    total = len(samples)
    print(f"\n{'='*70}")
    print(f"TOTAL SAMPLES (COMBINED): {total:,}")
    print(f"{'='*70}")
    for dir_name, count in total_by_dir.items():
        print(f"  {dir_name}: {count:,} ({100*count/total:.1f}%)")
    print()

    # 1. EMOTIONS
    print("=" * 70)
    print("1. EMOTIONS (all unique labels)")
    print("=" * 70)
    emotions = Counter()
    for s in samples:
        for e in s.get("tasks", {}).get("emotions", []):
            emotions[e] += 1
    print(f"Unique emotions: {len(emotions)}")
    for e, c in emotions.most_common():
        print(f"  {e:20} {c:5} ({c/total*100:5.1f}%)")

    # 2. SENTIMENT
    print()
    print("=" * 70)
    print("2. SENTIMENT")
    print("=" * 70)
    sentiments = Counter()
    for s in samples:
        sentiments[s.get("tasks", {}).get("sentiment", "MISSING")] += 1
    for label, count in sentiments.most_common():
        print(f"  {label:12} {count:5} ({count/total*100:5.1f}%)")

    # 3. SAFETY
    print()
    print("=" * 70)
    print("3. SAFETY_FAMILYOS")
    print("=" * 70)
    safety = Counter()
    for s in samples:
        safety[s.get("tasks", {}).get("safety_familyos", "MISSING")] += 1
    for label, count in safety.most_common():
        print(f"  {label:12} {count:5} ({count/total*100:5.1f}%)")

    # 4. INTENT
    print()
    print("=" * 70)
    print("4. INTENT")
    print("=" * 70)
    intents = Counter()
    for s in samples:
        intents[s.get("tasks", {}).get("intent", "MISSING")] += 1
    for label, count in intents.most_common():
        print(f"  {label:16} {count:5} ({count/total*100:5.1f}%)")

    # 5. INGRESS
    print()
    print("=" * 70)
    print("5. INGRESS")
    print("=" * 70)
    ingress = Counter()
    for s in samples:
        ingress[s.get("tasks", {}).get("ingress", "MISSING")] += 1
    for label, count in ingress.most_common():
        print(f"  {label:14} {count:5} ({count/total*100:5.1f}%)")

    # 6. NER_FAMILY
    print()
    print("=" * 70)
    print("6. NER_FAMILY (entity types)")
    print("=" * 70)
    ner_labels = Counter()
    ner_count = 0
    for s in samples:
        ents = s.get("tasks", {}).get("ner_family", [])
        if ents:
            ner_count += 1
        for e in ents:
            ner_labels[e.get("label", "UNKNOWN")] += 1
    print(f"Samples with NER: {ner_count} ({ner_count/total*100:.1f}%)")
    print("Entity type distribution:")
    for label, count in ner_labels.most_common():
        print(f"  {label:14} {count:5}")

    # 7. RELATIONS
    print()
    print("=" * 70)
    print("7. RELATIONS (predicate types)")
    print("=" * 70)
    rel_labels = Counter()
    rel_count = 0
    for s in samples:
        rels = s.get("tasks", {}).get("relations", [])
        if rels:
            rel_count += 1
        for r in rels:
            rel_labels[r.get("predicate", "UNKNOWN")] += 1
    print(f"Samples with relations: {rel_count} ({rel_count/total*100:.1f}%)")
    print("Relation type distribution:")
    for label, count in rel_labels.most_common():
        print(f"  {label:18} {count:5}")

    # 8. TEMPORAL
    print()
    print("=" * 70)
    print("8. TEMPORAL (tag types)")
    print("=" * 70)
    temp_labels = Counter()
    temp_count = 0
    for s in samples:
        temps = s.get("tasks", {}).get("temporal", [])
        if temps:
            temp_count += 1
        for t in temps:
            temp_labels[t.get("label", "UNKNOWN")] += 1
    print(f"Samples with temporal: {temp_count} ({temp_count/total*100:.1f}%)")
    print("Temporal type distribution:")
    for label, count in temp_labels.most_common():
        print(f"  {label:14} {count:5}")

    # 9. HUB_ROUTING
    print()
    print("=" * 70)
    print("9. HUB_ROUTING")
    print("=" * 70)
    hub_emo = hub_rel = hub_mem = hub_task = 0
    hub_combos = Counter()
    for s in samples:
        hub = s.get("hub_routing", {})
        if hub.get("EMO"):
            hub_emo += 1
        if hub.get("REL"):
            hub_rel += 1
        if hub.get("MEM"):
            hub_mem += 1
        if hub.get("TASK"):
            hub_task += 1
        combo = tuple(sorted([k for k, v in hub.items() if v]))
        hub_combos[combo] += 1
    print(f"  EMO:  {hub_emo:5} ({hub_emo/total*100:.1f}%)")
    print(f"  REL:  {hub_rel:5} ({hub_rel/total*100:.1f}%)")
    print(f"  MEM:  {hub_mem:5} ({hub_mem/total*100:.1f}%)")
    print(f"  TASK: {hub_task:5} ({hub_task/total*100:.1f}%)")
    print()
    print("Top hub combinations:")
    for combo, count in hub_combos.most_common(10):
        print(f"  {str(combo):40} {count:5} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    analyze_generated_data()
