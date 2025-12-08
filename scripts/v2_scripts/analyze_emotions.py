"""
FamilyOS Emotions Dataset - Comprehensive Analysis Script
Corporate-grade analysis for dataset documentation
"""

import json

# Load data
data = []
with open("D:/Modeling_studio/data/familyos/emotions/silver/train.jsonl", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))

print("=" * 80)
print("DATA QUALITY ANALYSIS")
print("=" * 80)

# Check for duplicates
texts = [d["text"] for d in data]
unique_texts = set(texts)
print("\n## 13. DUPLICATE ANALYSIS")
print(f"   Total samples: {len(texts):,}")
print(f"   Unique texts: {len(unique_texts):,}")
print(f"   Duplicates: {len(texts) - len(unique_texts):,}")
print(f"   Duplicate rate: {(len(texts) - len(unique_texts)) / len(texts) * 100:.2f}%")

# Check label consistency
print("\n## 14. LABEL CONSISTENCY CHECK")
primary_not_in_emotions = 0
for d in data:
    if d["primary_emotion"] not in d["emotions"]:
        primary_not_in_emotions += 1
print(
    f"   Primary emotion NOT in emotions list: {primary_not_in_emotions:,} ({primary_not_in_emotions/len(data)*100:.2f}%)"
)

# Emotion polarity/valence grouping
positive_emotions = [
    "joy",
    "love",
    "warmth",
    "pride",
    "gratitude",
    "amusement",
    "excitement",
    "contentment",
    "hope",
    "belonging",
    "approval",
    "optimism",
    "admiration",
    "parental_pride",
    "tenderness",
    "celebration",
    "playfulness",
    "caring",
    "relief",
]
negative_emotions = [
    "frustration",
    "sadness",
    "worry",
    "longing",
    "annoyance",
    "overwhelmed",
    "disgust",
    "emptiness",
    "disapproval",
    "disappointment",
    "remorse",
    "anger",
    "homesickness",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "parental_guilt",
]
neutral_emotions = [
    "neutral",
    "nostalgia",
    "bittersweet",
    "surprise",
    "togetherness",
    "protectiveness",
    "patience",
]

print("\n## 15. EMOTION VALENCE DISTRIBUTION")
positive_count = sum(1 for d in data if d["primary_emotion"] in positive_emotions)
negative_count = sum(1 for d in data if d["primary_emotion"] in negative_emotions)
neutral_count = sum(1 for d in data if d["primary_emotion"] in neutral_emotions)
other = len(data) - positive_count - negative_count - neutral_count

print(f"   Positive primary emotion: {positive_count:,} ({positive_count/len(data)*100:.1f}%)")
print(f"   Negative primary emotion: {negative_count:,} ({negative_count/len(data)*100:.1f}%)")
print(f"   Neutral/Mixed primary:    {neutral_count:,} ({neutral_count/len(data)*100:.1f}%)")
if other > 0:
    print(f"   Uncategorized:            {other:,} ({other/len(data)*100:.1f}%)")

# Sentence structure analysis
print("\n## 16. SENTENCE STRUCTURE ANALYSIS")
starts_with_pronoun = sum(
    1 for d in data if d["text"].split()[0].lower() in ["i", "we", "my", "our"]
)
contains_question = sum(1 for d in data if "?" in d["text"])
contains_exclamation = sum(1 for d in data if "!" in d["text"])
contains_ellipsis = sum(1 for d in data if "..." in d["text"])

print(
    f"   Starts with I/We/My/Our: {starts_with_pronoun:,} ({starts_with_pronoun/len(data)*100:.1f}%)"
)
print(f"   Contains question mark:  {contains_question:,} ({contains_question/len(data)*100:.1f}%)")
print(
    f"   Contains exclamation:    {contains_exclamation:,} ({contains_exclamation/len(data)*100:.1f}%)"
)
print(f"   Contains ellipsis:       {contains_ellipsis:,} ({contains_ellipsis/len(data)*100:.1f}%)")

# Family member mentions
print("\n## 17. FAMILY MEMBER MENTIONS")
family_terms = {
    "Kids/Children": ["kids", "children", "child", "beta"],
    "Emma (daughter)": ["emma"],
    "Jack (son)": ["jack"],
    "Mike (husband)": ["mike"],
    "Parents": ["mom", "dad", "papa", "mummy", "mother", "father"],
    "Grandparents": ["nani", "dadi", "grandma", "grandpa", "grandmother", "grandfather"],
    "Siblings": ["bhai", "didi", "brother", "sister"],
    "Extended": ["chacha", "chachi", "aunt", "uncle", "cousin"],
}

for category, terms in family_terms.items():
    count = sum(1 for d in data if any(t in d["text"].lower() for t in terms))
    print(f"   {category:25} {count:6,} ({count/len(data)*100:.1f}%)")

# Sample examples per emotion category
print("\n## 18. SAMPLE EXAMPLES (One per top emotion)")
top_emotions = ["joy", "love", "frustration", "worry", "grief"]
for emotion in top_emotions:
    for d in data:
        if d["primary_emotion"] == emotion:
            print(f"\n   [{emotion.upper()}]:")
            print(f'      "{d["text"]}"')
            print(f'      All emotions: {d["emotions"]}')
            print(f'      Intensity: {d["intensity"]}, Context: {d["context"]}')
            break
