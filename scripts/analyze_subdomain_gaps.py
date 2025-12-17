"""Analyze subdomain distribution and calculate gaps for rebalancing."""

import json
from pathlib import Path
from collections import defaultdict

# All expected subdomains from COUNTERFACTUAL_DOMAINS
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


def analyze_folder(folder_path: Path) -> dict[str, int]:
    """Count samples per subdomain in a folder."""
    stats = defaultdict(int)
    folder = Path(folder_path)
    if not folder.exists():
        return dict(stats)

    for f in folder.glob('*.jsonl'):
        with open(f, 'r', encoding='utf-8') as fp:
            for line in fp:
                try:
                    data = json.loads(line.strip())
                    subdomain = data.get('subdomain', 'unknown')
                    stats[subdomain] += 1
                except Exception:
                    pass
    return dict(stats)


def main():
    """Main analysis function."""
    # Load both folders
    merged_path = Path('D:/Modeling_studio/data/counterfactual/merged')
    synthetic_path = Path('D:/Modeling_studio/data/counterfactual/synthetic')

    print("Loading data from folders...")
    merged = analyze_folder(merged_path)
    synthetic = analyze_folder(synthetic_path)

    # Combine
    combined = defaultdict(int)
    for sub, count in merged.items():
        combined[sub] += count
    for sub, count in synthetic.items():
        combined[sub] += count

    combined_total = sum(combined.values())
    num_subdomains = sum(len(subs) for subs in EXPECTED_SUBDOMAINS.values())
    target_per_subdomain = combined_total / num_subdomains  # Equal distribution target

    print('=' * 90)
    print('GAP ANALYSIS - Subdomains needing more data')
    print('=' * 90)
    print(f'Total samples: {combined_total:,}')
    print(f'Number of subdomains: {num_subdomains}')
    print(f'Target per subdomain (equal): {target_per_subdomain:,.0f}')
    print()

    # Calculate gaps
    gaps = []
    for domain, subdomains in EXPECTED_SUBDOMAINS.items():
        for sub in subdomains:
            current = combined.get(sub, 0)
            gap = max(0, int(target_per_subdomain) - current)
            if gap > 0:
                gaps.append((sub, current, gap, domain))

    # Sort by gap (biggest first)
    gaps.sort(key=lambda x: x[2], reverse=True)

    print(f'Found {len(gaps)} subdomains below target')
    print()
    print(f"{'Subdomain':45s} {'Current':>8s} {'Gap':>8s} {'Domain':>15s}")
    print('-' * 90)

    total_gap = 0
    rebalance_targets = {}
    for sub, current, gap, domain in gaps:
        print(f'{sub:45s} {current:8,} {gap:8,} {domain:>15s}')
        total_gap += gap
        if gap >= 500:  # Only include significant gaps
            rebalance_targets[sub] = gap

    print('-' * 90)
    print(f'Total samples needed to balance: {total_gap:,}')
    print()

    # Generate rebalance command
    if rebalance_targets:
        print('=' * 90)
        print('RECOMMENDED REBALANCE COMMAND (gaps >= 500):')
        print('=' * 90)
        # Sort by gap size
        sorted_targets = sorted(rebalance_targets.items(), key=lambda x: x[1], reverse=True)
        target_str = ','.join([f'{sub}:{count}' for sub, count in sorted_targets])
        print(f'Total to generate: {sum(rebalance_targets.values()):,}')
        print()
        print('python scripts/agents/counterfactual_data_generator.py rebalance \\')
        print(f'  --subdomains "{target_str}" \\')
        print('  --vertex-ai --gcp-project disco-axis-479800-k2 --vertex-model gemini-2.0-flash --num-parallel 10')


if __name__ == '__main__':
    main()
