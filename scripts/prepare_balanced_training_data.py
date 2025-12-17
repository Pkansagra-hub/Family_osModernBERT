"""
Prepare balanced training dataset for fine-tuning.

Strategy:
1. Load newly generated samples from synthetic folder (underrepresented subdomains)
2. Sample from merged folder (well-represented subdomains to prevent forgetting)
3. Combine into a balanced training dataset
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import argparse

# Target: equal distribution across all 86 subdomains
TARGET_PER_SUBDOMAIN = 1000  # Target samples per subdomain in final dataset

EXPECTED_SUBDOMAINS = {
    'parenting': ['parenting_toddlers', 'parenting_teens', 'parenting_discipline', 'parenting_education', 'parenting_bonding', 'parenting_siblings', 'parenting_milestones', 'parenting_screen_time'],
    'relationship': ['relationship_spouse', 'relationship_inlaws', 'relationship_grandparents', 'relationship_extended', 'relationship_conflicts', 'relationship_trust', 'relationship_communication', 'relationship_friends'],
    'health': ['health_mental', 'health_children', 'health_elderly', 'health_chronic', 'health_nutrition', 'health_sleep', 'health_exercise', 'health_preventive'],
    'emotions': ['emotions_stress', 'emotions_anger', 'emotions_anxiety', 'emotions_grief', 'emotions_loneliness', 'emotions_overwhelm'],
    'communication': ['communication_arguments', 'communication_listening', 'communication_boundaries', 'communication_difficult_conversations', 'communication_family_meetings'],
    'work': ['work_career', 'work_burnout', 'work_childcare', 'work_remote', 'work_boundaries'],
    'time': ['time_scheduling', 'time_prioritization', 'time_delegation', 'time_quality_time', 'time_procrastination'],
    'routine': ['routine_morning', 'routine_evening', 'routine_meals', 'routine_chores', 'routine_self_care', 'routine_commute'],
    'finance': ['finance_budgeting', 'finance_savings', 'finance_debt', 'finance_education', 'finance_family_expenses'],
    'caregiving': ['caregiving_elderly', 'caregiving_special_needs', 'caregiving_respite', 'caregiving_babysitting', 'caregiving_coordination'],
    'cultural': ['cultural_traditions', 'cultural_festivals', 'cultural_religious', 'cultural_heritage', 'cultural_rituals'],
    'social': ['social_isolation', 'social_community', 'social_friendships', 'social_support_networks', 'social_neighborhood'],
    'home': ['home_organization', 'home_maintenance', 'home_safety', 'home_decoration', 'home_moves'],
    'tech': ['tech_screen_addiction', 'tech_social_media', 'tech_online_safety', 'tech_digital_boundaries', 'tech_family_apps'],
    'life': ['life_weddings', 'life_births', 'life_deaths', 'life_graduations', 'life_relocations'],
}

ALL_SUBDOMAINS = set()
for subs in EXPECTED_SUBDOMAINS.values():
    ALL_SUBDOMAINS.update(subs)


def load_samples_by_subdomain(folder_path: Path) -> dict[str, list[dict]]:
    """Load all samples grouped by subdomain."""
    samples_by_subdomain = defaultdict(list)
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Warning: Folder {folder} does not exist")
        return dict(samples_by_subdomain)

    for f in folder.glob('*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                try:
                    data = json.loads(line.strip())
                    subdomain = data.get('subdomain', 'unknown')
                    samples_by_subdomain[subdomain].append(data)
                except Exception:
                    pass

    return dict(samples_by_subdomain)


def analyze_and_plan(merged_path: Path, synthetic_path: Path, target_per_subdomain: int = 1000):
    """Analyze current distribution and create sampling plan."""

    print("=" * 90)
    print("LOADING DATA")
    print("=" * 90)

    # Load samples
    print(f"Loading from merged: {merged_path}")
    merged = load_samples_by_subdomain(merged_path)
    merged_total = sum(len(v) for v in merged.values())
    print(f"  Loaded {merged_total:,} samples across {len(merged)} subdomains")

    print(f"Loading from synthetic: {synthetic_path}")
    synthetic = load_samples_by_subdomain(synthetic_path)
    synthetic_total = sum(len(v) for v in synthetic.values())
    print(f"  Loaded {synthetic_total:,} samples across {len(synthetic)} subdomains")

    print()
    print("=" * 90)
    print(f"SAMPLING PLAN (Target: {target_per_subdomain} per subdomain)")
    print("=" * 90)

    sampling_plan = {}
    total_from_merged = 0
    total_from_synthetic = 0

    print(f"\n{'Subdomain':45s} {'Merged':>8s} {'Synth':>8s} {'Take Merged':>12s} {'Take Synth':>12s} {'Total':>8s}")
    print("-" * 100)

    for subdomain in sorted(ALL_SUBDOMAINS):
        merged_count = len(merged.get(subdomain, []))
        synthetic_count = len(synthetic.get(subdomain, []))

        # Priority: take from synthetic first (new data), then fill from merged
        take_synthetic = min(synthetic_count, target_per_subdomain)
        remaining = target_per_subdomain - take_synthetic
        take_merged = min(merged_count, remaining)

        total = take_synthetic + take_merged

        sampling_plan[subdomain] = {
            'merged': take_merged,
            'synthetic': take_synthetic,
            'total': total
        }

        total_from_merged += take_merged
        total_from_synthetic += take_synthetic

        # Only print if there's something interesting
        if total > 0:
            status = "OK" if total >= target_per_subdomain else f"GAP: {target_per_subdomain - total}"
            print(f"{subdomain:45s} {merged_count:8,} {synthetic_count:8,} {take_merged:12,} {take_synthetic:12,} {total:8,}")

    print("-" * 100)
    print(f"{'TOTAL':45s} {merged_total:8,} {synthetic_total:8,} {total_from_merged:12,} {total_from_synthetic:12,} {total_from_merged + total_from_synthetic:8,}")

    return merged, synthetic, sampling_plan


def create_balanced_dataset(
    merged: dict[str, list[dict]],
    synthetic: dict[str, list[dict]],
    sampling_plan: dict,
    output_path: Path,
    seed: int = 42
):
    """Create the balanced dataset based on sampling plan."""
    random.seed(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    subdomain_counts = defaultdict(int)

    with open(output_path, 'w', encoding='utf-8') as fp:
        for subdomain, plan in sampling_plan.items():
            # Sample from synthetic
            if plan['synthetic'] > 0 and subdomain in synthetic:
                samples = random.sample(synthetic[subdomain], min(plan['synthetic'], len(synthetic[subdomain])))
                for sample in samples:
                    fp.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    total_samples += 1
                    subdomain_counts[subdomain] += 1

            # Sample from merged
            if plan['merged'] > 0 and subdomain in merged:
                samples = random.sample(merged[subdomain], min(plan['merged'], len(merged[subdomain])))
                for sample in samples:
                    fp.write(json.dumps(sample, ensure_ascii=False) + '\n')
                    total_samples += 1
                    subdomain_counts[subdomain] += 1

    print()
    print("=" * 90)
    print(f"DATASET CREATED: {output_path}")
    print("=" * 90)
    print(f"Total samples: {total_samples:,}")
    print(f"Subdomains covered: {len(subdomain_counts)}")
    print()

    return total_samples, dict(subdomain_counts)


def main():
    parser = argparse.ArgumentParser(description='Prepare balanced training dataset')
    parser.add_argument('--merged', type=str, default='D:/Modeling_studio/data/counterfactual/merged',
                       help='Path to merged folder')
    parser.add_argument('--synthetic', type=str, default='D:/Modeling_studio/data/counterfactual/synthetic',
                       help='Path to synthetic folder')
    parser.add_argument('--output', type=str, default='D:/Modeling_studio/data/counterfactual/balanced_training.jsonl',
                       help='Output path for balanced dataset')
    parser.add_argument('--target', type=int, default=1000,
                       help='Target samples per subdomain (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only show plan, do not create dataset')

    args = parser.parse_args()

    merged_path = Path(args.merged)
    synthetic_path = Path(args.synthetic)
    output_path = Path(args.output)

    # Analyze and plan
    merged, synthetic, sampling_plan = analyze_and_plan(
        merged_path, synthetic_path, args.target
    )

    if args.dry_run:
        print("\n[DRY RUN] Not creating dataset. Use without --dry-run to create.")
        return

    # Create dataset
    total, counts = create_balanced_dataset(
        merged, synthetic, sampling_plan, output_path, args.seed
    )

    print(f"\nDataset ready for training at: {output_path}")
    print(f"Use this for fine-tuning with 2-3 epochs on the existing model.")


if __name__ == '__main__':
    main()
