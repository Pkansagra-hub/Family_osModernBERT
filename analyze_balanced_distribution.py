#!/usr/bin/env python3
"""Analyze domain and subdomain distribution in balanced training dataset."""

import json
from pathlib import Path
from collections import defaultdict

# Load the balanced dataset
dataset_path = Path('data/counterfactual/training_v2/samples.jsonl')

print("=" * 80)
print("DOMAIN & SUBDOMAIN DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"\nDataset: {dataset_path}")
print(f"Exists: {dataset_path.exists()}\n")

if not dataset_path.exists():
    print("[ERROR] Dataset not found!")
    exit(1)

# Load samples
domain_counts = defaultdict(int)
subdomain_counts = defaultdict(int)
domain_subdomains = defaultdict(set)

total_samples = 0
with open(dataset_path, encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            domain = data.get('domain', 'unknown')
            subdomain = data.get('subdomain', 'unknown')

            domain_counts[domain] += 1
            subdomain_counts[subdomain] += 1
            domain_subdomains[domain].add(subdomain)
            total_samples += 1
        except json.JSONDecodeError:
            pass

print(f"Total samples: {total_samples:,}\n")

# Domain-level distribution
print("=" * 80)
print("DOMAIN-LEVEL DISTRIBUTION")
print("=" * 80)
print(f"{'Domain':30s} {'Samples':>10s} {'Subdomains':>15s} {'Avg/Sub':>10s}")
print("-" * 80)

domain_list = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
for domain, count in domain_list:
    num_subs = len(domain_subdomains[domain])
    avg_per_sub = count / num_subs if num_subs > 0 else 0
    print(f"{domain:30s} {count:10,} {num_subs:15} {avg_per_sub:10.0f}")

print("-" * 80)
print(f"{'Total':30s} {total_samples:10,} {len(subdomain_counts):15}")

# Subdomain-level distribution
print("\n" + "=" * 80)
print("SUBDOMAIN-LEVEL DISTRIBUTION (Top 20)")
print("=" * 80)
print(f"{'Subdomain':45s} {'Count':>10s} {'Domain':30s}")
print("-" * 80)

subdomain_list = sorted(subdomain_counts.items(), key=lambda x: x[1], reverse=True)
for i, (subdomain, count) in enumerate(subdomain_list):
    # Find domain for this subdomain
    domain = 'unknown'
    for d, subs in domain_subdomains.items():
        if subdomain in subs:
            domain = d
            break
    print(f"{subdomain:45s} {count:10,} {domain:30s}")
    if i >= 19:
        break

if len(subdomain_list) > 20:
    print(f"\n... and {len(subdomain_list) - 20} more subdomains")

# Statistics
print("\n" + "=" * 80)
print("DISTRIBUTION STATISTICS")
print("=" * 80)

counts = list(subdomain_counts.values())
avg_count = sum(counts) / len(counts) if counts else 0
min_count = min(counts) if counts else 0
max_count = max(counts) if counts else 0
std_dev = (sum((x - avg_count) ** 2 for x in counts) / len(counts)) ** 0.5 if counts else 0

print(f"Total subdomains: {len(subdomain_counts)}")
print(f"Average samples/subdomain: {avg_count:.0f}")
print(f"Min samples: {min_count:,}")
print(f"Max samples: {max_count:,}")
print(f"Std deviation: {std_dev:.1f}")

# Balance check
perfect_balance = all(c == avg_count for c in counts)
near_balance = all(abs(c - avg_count) < 10 for c in counts)

print(f"\nBalance status:")
print(f"  Perfect balance (all equal): {perfect_balance}")
print(f"  Near balance (±10): {near_balance}")
print(f"  Deviation range: {max_count - min_count:,}")

# Subdomains with gaps (if not perfectly balanced)
if not perfect_balance:
    print(f"\n{'Subdomain':45s} {'Count':>10s} {'Gap':>10s} {'Status':20s}")
    print("-" * 80)
    target = 1000
    gaps = []
    for subdomain, count in sorted(subdomain_counts.items()):
        if count != target:
            gap = target - count
            status = "GAP" if gap > 0 else "EXCESS"
            gaps.append((subdomain, count, abs(gap)))
            print(f"{subdomain:45s} {count:10,} {gap:10,} {status:20s}")

    if gaps:
        print(f"\nTotal subdomains with gaps: {len(gaps)}")
        total_gap = sum(g[2] for g in gaps)
        print(f"Total gap: {total_gap:,}")

print("\n" + "=" * 80)
