"""Generate synthetic NLI data for MGRH dry run."""
import json
import os
import random

random.seed(42)

premises = [
    "The family went to the park on Saturday afternoon.",
    "My daughter started school last September.",
    "We had a big argument about finances yesterday.",
    "The baby slept through the night for the first time.",
    "Dad picked up the kids from soccer practice.",
    "Mom cooked dinner while helping with homework.",
    "The cat knocked over the vase in the living room.",
    "We celebrated her birthday with a surprise party.",
    "The plumber came to fix the kitchen sink.",
    "Everyone gathered for Thanksgiving dinner.",
    "She received her college acceptance letter today.",
    "The dog escaped from the backyard again.",
    "We had a productive family meeting about chores.",
    "The kids built a snowman in the front yard.",
    "Grandma arrived for her annual summer visit.",
    "The car broke down on the way to school.",
    "We signed up for a family cooking class.",
    "The children performed in the school play.",
    "Dad finally fixed the leaking faucet.",
    "We adopted a rescue dog from the shelter.",
]

entailment = [
    "The family spent time outdoors on the weekend.",
    "A child began her education in the fall.",
    "There was a disagreement about money recently.",
    "The infant managed to sleep without waking.",
    "The father collected children from sports.",
    "A parent was multitasking in the evening.",
    "A pet caused damage inside the home.",
    "There was a celebration for someone special.",
    "A repair worker visited the home.",
    "The family had a holiday meal together.",
    "Good news arrived regarding higher education.",
    "An animal left the property without permission.",
    "The household discussed responsibilities.",
    "Young people played in winter weather.",
    "An elderly relative came to stay.",
    "A vehicle had mechanical problems during commute.",
    "The family enrolled in a group activity.",
    "Students participated in a school event.",
    "Home maintenance was completed.",
    "A new pet joined the household.",
]

contradiction = [
    "The family stayed indoors all weekend.",
    "The child has never attended any school.",
    "Everyone agreed completely about the budget.",
    "The baby cried throughout the entire night.",
    "The children walked home alone from practice.",
    "Nobody was home in the evening.",
    "Nothing was damaged in the house.",
    "The birthday was completely forgotten.",
    "No repairs were needed in the kitchen.",
    "Nobody showed up for the holiday.",
    "The letter contained a rejection notice.",
    "The dog has never left the yard.",
    "Nobody discussed household duties.",
    "There was no snow at all this winter.",
    "No relatives visited during the summer.",
    "The car worked perfectly all day.",
    "The family canceled all planned activities.",
    "No students participated in school events.",
    "The faucet is still broken.",
    "The family decided against getting a pet.",
]

neutral = [
    "The park has a new playground section.",
    "The school is located three miles away.",
    "The financial advisor suggested new investments.",
    "The nursery was recently painted blue.",
    "Soccer practice ends at five thirty.",
    "The homework was about multiplication tables.",
    "The vase was a wedding present.",
    "The cake was chocolate flavored.",
    "The plumber charges seventy dollars per hour.",
    "Turkey was served as the main course.",
    "The university has a strong engineering program.",
    "The fence needs replacement.",
    "The meeting lasted about thirty minutes.",
    "The snowman had a carrot nose.",
    "Grandma lives in a different state.",
    "The mechanic quoted two hundred dollars.",
    "The cooking class meets on Wednesdays.",
    "The play was about American history.",
    "The faucet had been leaking for weeks.",
    "The shelter had many cats as well.",
]

hyp_map = {0: entailment, 1: neutral, 2: contradiction}

os.makedirs("d:/Modeling_studio/data/familyos/nli/general", exist_ok=True)

samples = []
for i in range(500):
    idx = i % 20
    label = i % 3
    samples.append({
        "premise": premises[idx],
        "hypothesis": hyp_map[label][idx],
        "label": label,
    })

random.shuffle(samples)

with open("d:/Modeling_studio/data/familyos/nli/general/dryrun_nli.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

labels = [s["label"] for s in samples]
print(f"Generated {len(samples)} NLI records")
print(f"  entailment={labels.count(0)}, neutral={labels.count(1)}, contradiction={labels.count(2)}")
