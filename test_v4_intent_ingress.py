"""Test intent_v2 and ingress_v2 on the V4 checkpoint."""
from familyos_ultrabert import Client

client = Client(backend='pytorch', verbose=True)

# Test intent_v2 and ingress_v2
test_cases = [
    ('Remember that Mom bought flowers for the dinner party', 'log_memory', 'MEMORY'),
    ('What was the name of that restaurant we went to?', 'query_memory', 'MEMORY'),
    ('Set a reminder to call Dad tomorrow at 3pm', 'set_reminder', 'TASK'),
    ('I feel so grateful for my family today', 'express_feeling', 'GRATITUDE'),
    ('Should I accept the job offer or stay at my current company?', 'seek_advice', 'WORK'),
    ('Just got promoted at work!', 'share_news', 'WORK'),
    ('Looking back, that summer trip was amazing', 'reflect', 'MEMORY'),
    ('I need to pick up groceries after work', 'other', 'TASK'),
    ('We had a wonderful family reunion last weekend', 'log_memory', 'CELEBRATION'),
    ('Feeling anxious about the upcoming interview', 'express_feeling', 'CONCERN'),
    ('Can you remind me about my dentist appointment?', 'query_memory', 'HEALTH'),
    ('Thank you for helping me move last weekend', 'express_feeling', 'GRATITUDE'),
]

print('='*80)
print('INTENT_V2 & INGRESS_V2 EVALUATION')
print('='*80)
print()

intent_correct = 0
ingress_correct = 0
total = len(test_cases)

for text, expected_intent, expected_ingress in test_cases:
    result = client.analyze(text)

    # Use the ClientResult properties
    pred_intent = result.intent  # Uses V1 head by default
    pred_ingress = result.ingress  # Uses V1 head by default

    intent_match = '?' if pred_intent == expected_intent else '?'
    ingress_match = '?' if pred_ingress == expected_ingress else '?'

    if pred_intent == expected_intent:
        intent_correct += 1
        intent_match = '+'
    if pred_ingress == expected_ingress:
        ingress_correct += 1
        ingress_match = '+'

    print(f'Text: "{text[:60]}..."')
    print(f'  Intent:  {pred_intent:15} (expected: {expected_intent:15}) [{intent_match}]')
    print(f'  Ingress: {pred_ingress:15} (expected: {expected_ingress:15}) [{ingress_match}]')
    print()

print('='*80)
print('SUMMARY')
print('='*80)
print(f'Intent Accuracy:  {intent_correct}/{total} ({100*intent_correct/total:.1f}%)')
print(f'Ingress Accuracy: {ingress_correct}/{total} ({100*ingress_correct/total:.1f}%)')
print()

# Also test NER heads
print('='*80)
print('NER HEAD EVALUATION (GlobalPointer)')
print('='*80)
print()

ner_cases = [
    'Mom picked up Emma and Jake from Lincoln Elementary School',
    'Grandma visited us on Christmas morning at our house in Boston',
    'Uncle Raj gave Priya his antique watch for her 18th birthday',
    'Our family reunion is scheduled for next Sunday at 3pm',
]

for text in ner_cases:
    result = client.analyze(text)
    ner_general = result.capabilities.get('ner_general', {})
    ner_family = result.capabilities.get('ner_family', {})
    temporal = result.capabilities.get('temporal', {})

    print(f'Text: "{text}"')
    print(f'  ner_general: {ner_general.get("entities", [])}')
    print(f'  ner_family:  {ner_family.get("entities", [])}')
    print(f'  temporal:    {temporal.get("entities", [])}')
    print()
