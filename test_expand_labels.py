"""Test script for expand_labels functionality - label expansion for FamilyOS.

This demonstrates adding custom family-specific labels while preserving
trained performance on the original 8 intents.
"""
from familyos_ultrabert import Client

# Initialize client
client = Client()

# Check initial labels
print("=== Initial Intent Labels (V2 - Label Description Embedding) ===")
labels = client.get_intent_labels()
print(f"Count: {len(labels)}")
print(f"Labels: {labels}")

# Add new custom family labels with more specific descriptions
print("\n=== Adding Custom Family Labels ===")
new_labels = {
    "school_pickup": "Reminder about picking children up from school, school bus, after-school activities",
    "pet_care": "Taking care of pets including vet visits, veterinarian appointments, dog walking, pet feeding",
    "date_night": "Planning date night, anniversary dinner, romantic evening with spouse or partner",
}

all_labels = client.add_intent_labels(new_labels)
print(f"Labels after expansion: {len(all_labels)}")
print(f"All labels: {all_labels}")

# Test inference with both original and new labels
print("\n=== Testing Predictions ===")
print("Format: Text -> Primary (confidence) | Relevant trained labels")
print("-" * 70)

test_cases = [
    # NEW labels should win
    ("Pick up the kids from school at 3pm", "school_pickup"),
    ("Take the dog to the vet tomorrow", "pet_care"),
    ("Reserve a table for our anniversary dinner", "date_night"),
    # TRAINED labels should still work
    ("Remember when we went to Disneyland last summer?", "query_memory"),
    ("I'm feeling stressed about work", "express_feeling"),
    ("Schedule a dentist appointment for Tuesday", "set_reminder"),
    ("I heard that Sarah got a promotion!", "share_news"),
    ("What should I do about my teenager's behavior?", "seek_advice"),
    # Ambiguous cases
    ("Buy groceries for dinner", "other/set_reminder"),
    ("Call mom to wish her happy birthday", "set_reminder"),
]

correct = 0
total = len(test_cases)

for text, expected in test_cases:
    result = client.analyze(text, capabilities=["intent_v2"])
    primary = result.intent_v2_primary
    conf = result.intent_v2_confidence

    # Check if prediction matches expected (allow multiple valid answers)
    valid = expected.split("/")
    is_correct = primary in valid
    marker = "OK" if is_correct else "WRONG"
    if is_correct:
        correct += 1

    print(f"\n[{marker}] {text}")
    print(f"  Expected: {expected} | Got: {primary} ({conf:.3f})")

    # Show top 3 scores
    scores = result.intent_v2_scores
    top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3: {[(k, f'{v:.3f}') for k, v in top3]}")

print("\n" + "=" * 70)
print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 70)

# Also test ingress expansion
print("\n\n=== Testing Ingress Expansion ===")
ingress_labels = client.get_ingress_labels()
print(f"Initial ingress labels: {len(ingress_labels)}")
print(f"Labels: {ingress_labels[:5]}... (showing first 5)")

new_ingress = {
    "PETS": "Pet care activities, vet visits, pet food, animal-related",
    "SCHOOL": "School activities, homework, parent-teacher meetings, education",
}
all_ingress = client.add_ingress_labels(new_ingress)
print(f"After expansion: {len(all_ingress)} labels")
print(f"New labels added: {list(new_ingress.keys())}")
