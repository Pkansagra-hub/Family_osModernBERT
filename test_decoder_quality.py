"""
Decoder Counterfactual Generation Quality Test
Tests diverse family scenarios - life events, decisions, relationships
"""

from familyos_ultrabert import Client

def main():
    print("=" * 80)
    print("DECODER COUNTERFACTUAL GENERATION - DIVERSE FAMILY SCENARIOS")
    print("=" * 80)

    client = Client(backend='pytorch', device='cpu', verbose=False)

    scenarios = [
        # Negative emotions
        ("Frustration", "I yelled at my kids for making a mess in the living room"),
        ("Overwhelm", "I skipped my daughter's recital because I had too much work"),
        ("Anger", "I got furious when my teenager came home past curfew"),

        # Life decisions
        ("Career choice", "I took the promotion that required 60-hour weeks and missed my son's first steps"),
        ("Moving decision", "We moved to a new city for my job and my kids lost all their friends"),
        ("School choice", "I insisted my child go to the prestigious school instead of staying with their friends"),

        # Parenting moments
        ("Discipline", "I grounded my daughter for a month after she lied about where she was"),
        ("Homework battle", "I forced my son to redo his entire project because it wasn't perfect"),
        ("Screen time", "I took away all devices when grades dropped without discussing why"),

        # Relationship dynamics
        ("Partner conflict", "I blamed my spouse in front of the kids for forgetting the school pickup"),
        ("Extended family", "I cut off contact with my parents after they criticized my parenting"),
        ("Sibling rivalry", "I compared my children's grades in front of them"),

        # Missed opportunities
        ("Quality time", "I was on my phone during our entire family dinner"),
        ("Teaching moment", "I did my child's science project for them to make sure they got an A"),
        ("Independence", "I made all decisions for my 16-year-old because I knew better"),

        # Positive reframes
        ("Celebration", "We had a small birthday party because we couldn't afford a big one"),
        ("Failure handling", "My son failed his driving test on the first try"),
        ("Health scare", "I panicked when my child got a minor injury at the playground"),
    ]

    for category, scenario in scenarios:
        print(f"\n[{category.upper()}]")
        print(f"Original: {scenario}")
        try:
            alternative = client.suggest_alternative(scenario, max_new_tokens=100, temperature=0.7)
            # Clean up and truncate
            alt_clean = alternative.strip().replace("\n", " ")[:250]
            print(f"Alternative: {alt_clean}")
        except Exception as e:
            print(f"ERROR: {e}")
        print("-" * 80)


if __name__ == "__main__":
    main()
