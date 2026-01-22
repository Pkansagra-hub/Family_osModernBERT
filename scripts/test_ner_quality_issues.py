#!/usr/bin/env python
"""
Test NER Quality Issues from the catalog against UltraBERT v4.

This script tests whether the v4 model with GlobalPointer resolves
the NER quality issues documented in the NER Quality Issues Catalog.
"""

from familyos_ultrabert import Client


def main():
    client = Client("pytorch", warmup=True, warmup_rounds=2, verbose=False)

    # Test cases from the NER Quality Issues document - REAL LIFE SENTENCES
    test_cases = {
        "NF-001: MILESTONE tags verbs": [
            (
                "My daughter Emma finally learned to ride a bike without training wheels last weekend.",
                "learned should NOT be MILESTONE",
            ),
            (
                "Sofia passed her driving test on the first try and we're so proud of her!",
                "passed should NOT be MILESTONE",
            ),
            (
                "I finally got promoted to senior manager after five years of hard work.",
                "promoted should NOT be MILESTONE",
            ),
            (
                "She accepted the job offer from Google and starts next month.",
                "accepted should NOT be MILESTONE",
            ),
            (
                "The baby started walking at 11 months old, which was earlier than expected.",
                "started should NOT be MILESTONE",
            ),
            (
                "He graduated from college with honors and now works at Microsoft.",
                "graduated should NOT be MILESTONE",
            ),
            (
                "They bought their first house together after saving for three years.",
                "bought should NOT be MILESTONE",
            ),
        ],
        "NF-002: FAMILY_EVENT tags pronouns": [
            (
                "Our trip to the lake last summer was the best family vacation we've ever had.",
                "our should NOT be FAMILY_EVENT",
            ),
            (
                "We celebrated grandma's 80th birthday for 3 hours with the whole family.",
                "3 should NOT be FAMILY_EVENT",
            ),
            (
                "They attended the ceremony at the church downtown yesterday afternoon.",
                "attended should NOT be FAMILY_EVENT",
            ),
            (
                "Her wedding reception lasted 4 hours and everyone had a great time.",
                "4 should NOT be FAMILY_EVENT",
            ),
            (
                "We gathered for 2 hours to watch the family reunion video together.",
                "2 should NOT be FAMILY_EVENT",
            ),
            (
                "Their anniversary dinner went on for 5 hours with lots of stories.",
                "5 should NOT be FAMILY_EVENT",
            ),
        ],
        "NF-003: HEIRLOOM tags prepositions": [
            (
                "Grandma's old ring has been in our family for over a hundred years.",
                "old should NOT be HEIRLOOM",
            ),
            (
                "The antique watch was passed down to me from my grandfather before he died.",
                "to should NOT be HEIRLOOM",
            ),
            (
                "Mom keeps the family photos from 1942 in a special album in the attic.",
                "from/in should NOT be HEIRLOOM",
            ),
            (
                "The silver locket from 1920 is kept safe in the drawer upstairs.",
                "from/in should NOT be HEIRLOOM",
            ),
            (
                "Dad's pocket watch from the 1800s is displayed on the mantelpiece.",
                "from should NOT be HEIRLOOM",
            ),
            (
                "The family bible from 1850 contains all our ancestors' records.",
                "from should NOT be HEIRLOOM",
            ),
        ],
        "NF-004: PET tags determiners": [
            (
                "Buddy the dog loves playing fetch with the kids in the backyard every evening.",
                "the should NOT be PET",
            ),
            (
                "We adopted Whiskers the cat from the local shelter last month.",
                "the should NOT be PET",
            ),
            (
                "Max the parrot can say hello and goodbye when visitors arrive.",
                "the should NOT be PET",
            ),
            (
                "Luna the rabbit enjoys eating carrots and lettuce every day.",
                "the should NOT be PET",
            ),
        ],
        "NG-001: PERSON tags verbs": [
            (
                "Met with John today at the coffee shop to discuss the wedding plans.",
                "Met should NOT be PERSON",
            ),
            (
                "Asked about the project status during this morning's team standup meeting.",
                "Asked should NOT be PERSON",
            ),
            (
                "Called the team to let them know about the schedule change for tomorrow.",
                "Called should NOT be PERSON",
            ),
            (
                "Thinking about taking a vacation with the family this coming summer.",
                "Thinking should NOT be PERSON",
            ),
            (
                "Went shopping for groceries after work yesterday evening.",
                "Went should NOT be PERSON",
            ),
            (
                "Came home early from the office to spend time with the kids.",
                "Came should NOT be PERSON",
            ),
            (
                "Worked late again on the quarterly report due tomorrow morning.",
                "Worked should NOT be PERSON",
            ),
            (
                "Drove to the airport to pick up grandma from her flight.",
                "Drove should NOT be PERSON",
            ),
        ],
        "NG-002: PERSON tags emotions": [
            (
                "I've been feeling really anxious today about the upcoming job interview.",
                "anxious should NOT be PERSON",
            ),
            (
                "So grateful for all the help my parents gave us during the move.",
                "grateful should NOT be PERSON",
            ),
            (
                "Really stressed about work lately with all the deadlines piling up.",
                "stressed should NOT be PERSON",
            ),
            (
                "Excited about the trip to Disneyland we're planning for the kids' birthday!",
                "Excited should NOT be PERSON",
            ),
            (
                "Worried about mom's health after her recent doctor's appointment.",
                "Worried should NOT be PERSON",
            ),
            (
                "Happy that dad finally retired and can relax now.",
                "Happy should NOT be PERSON",
            ),
            (
                "Sad about uncle Bob's passing last month, he was a great man.",
                "Sad should NOT be PERSON",
            ),
            (
                "Proud of sister's accomplishments in her new career.",
                "Proud should NOT be PERSON",
            ),
        ],
        "NG-003: ORG tags common nouns": [
            (
                "Had a meeting about Q2 targets with the sales team this afternoon.",
                "meeting should NOT be ORG",
            ),
            (
                "Checked my email and found a message from mom about Thanksgiving dinner.",
                "email should NOT be ORG",
            ),
            (
                "The afternoon was busy with errands - groceries, dry cleaning, and picking up the kids.",
                "afternoon should NOT be ORG",
            ),
            (
                "Coffee with my manager to discuss my career development went really well.",
                "manager should NOT be ORG",
            ),
            (
                "The conference room is booked for the client presentation tomorrow.",
                "conference should NOT be ORG",
            ),
            (
                "Sent the report to the accounting department for review.",
                "accounting should NOT be ORG",
            ),
            (
                "The kitchen needs cleaning after dinner preparation.",
                "kitchen should NOT be ORG",
            ),
            (
                "The hospital called about dad's test results from yesterday.",
                "hospital should NOT be ORG",
            ),
        ],
        "NG-004: Partial entity extraction": [
            (
                "The kids are doing great at Lincoln School and really love their teachers.",
                "Should extract full span Lincoln School",
            ),
            (
                "We're planning to visit San Francisco next month for our anniversary trip.",
                "Should extract full span San Francisco",
            ),
            (
                "Had dinner at Bella Notte restaurant to celebrate mom's retirement.",
                "Should extract full span Bella Notte",
            ),
            (
                "Working at Johnson and Johnson pharmaceuticals on new drug development.",
                "Should extract full span Johnson and Johnson",
            ),
            (
                "The concert at Madison Square Garden was absolutely amazing.",
                "Should extract full span Madison Square Garden",
            ),
            (
                "Flying to Los Angeles International Airport for the business meeting.",
                "Should extract full span Los Angeles International Airport",
            ),
            (
                "Graduated from Massachusetts Institute of Technology with honors.",
                "Should extract full span Massachusetts Institute of Technology",
            ),
        ],
        "NG-005: Time fragments": [
            (
                "The team meeting is scheduled at 3pm in the main conference room.",
                "3pm should NOT be PERSON",
            ),
            (
                "Please call me back at 10am tomorrow when you have a chance.",
                "10am should NOT be PERSON",
            ),
            (
                "We usually have lunch at 12pm when everyone's available.",
                "12pm should NOT be PERSON",
            ),
            (
                "The flight departs at 6pm and arrives at 9pm local time.",
                "6pm/9pm should NOT be PERSON",
            ),
            (
                "School starts at 8am and ends at 3pm every weekday.",
                "8am/3pm should NOT be PERSON",
            ),
            (
                "The movie begins at 7pm sharp, don't be late.",
                "7pm should NOT be PERSON",
            ),
        ],
        "NG-006: Verb forms": [
            (
                "Learned a lot from dad about fixing cars when I was growing up.",
                "Learned should NOT be entity",
            ),
            (
                "Working on a project for school that's due next Friday.",
                "Working should NOT be ORG",
            ),
            (
                "Organized the family photos into albums for grandma's birthday gift.",
                "Organized should NOT be PERSON",
            ),
            (
                "Developed new skills in programming during the summer internship.",
                "Developed should NOT be PERSON",
            ),
            (
                "Created beautiful artwork for the family room decoration.",
                "Created should NOT be PERSON",
            ),
            (
                "Managed the household budget carefully to save for vacation.",
                "Managed should NOT be PERSON",
            ),
        ],
        "HARD: Complex family contexts": [
            (
                "Aunt Sarah's husband Uncle Mike took their kids Timmy and Susie to Disney World last summer while grandma watched the dogs Buddy and Max at home.",
                "Complex family relationships and pets",
            ),
            (
                "Mom's antique silver tea set from 1890 that grandpa inherited from his grandmother is now being used by cousin Emily for her wedding reception next month.",
                "Multiple family generations and heirlooms",
            ),
            (
                "Dad's old pocket watch that was passed down from his father who fought in World War II is kept in the same drawer as mom's family photos from the 1950s.",
                "Historical family artifacts and time periods",
            ),
            (
                "Sister Mary's twins John and Jane celebrated their 10th birthday at Lincoln Elementary School with grandma's homemade cake recipe from 1972.",
                "Multiple entities: people, locations, events, heirlooms",
            ),
            (
                "Uncle Bob and Aunt Karen's golden retriever Charlie loves playing with cousin Peter's cat Whiskers in their backyard at 123 Maple Street.",
                "Pets, family members, addresses",
            ),
        ],
        "HARD: Ambiguous entities": [
            (
                "John Smith from accounting called about the quarterly report that's due tomorrow.",
                "John Smith could be person OR just 'called' verb context",
            ),
            (
                "The meeting with Google representatives went really well yesterday afternoon.",
                "Google could be ORG but 'meeting' should not be tagged",
            ),
            (
                "Sarah Johnson celebrated her promotion to senior manager with a dinner at Olive Garden.",
                "Sarah Johnson person, Olive Garden restaurant/org",
            ),
            (
                "Working late again on Friday night to finish the presentation for Monday morning.",
                "Time expressions mixed with work context",
            ),
            (
                "Grandpa's old Ford truck from 1965 is still running perfectly after all these years.",
                "1965 could be DATE, Ford could be ORG, truck is object",
            ),
        ],
        "HARD: Cultural and contextual challenges": [
            (
                "Aunty Priya made amazing chicken tikka masala for Diwali celebrations with the whole Indian family gathering.",
                "Cultural names, food, festivals, family terms",
            ),
            (
                "Abuela Rosa's traditional Mexican recipes for tamales have been passed down through generations since 1920.",
                "Cultural terms, traditional foods, multi-generational",
            ),
            (
                "Oma's German chocolate cake recipe from the old country is a family tradition for Christmas celebrations.",
                "Cultural terms, traditional foods, holidays",
            ),
            (
                "Tío Carlos taught me how to play guitar using the old flamenco techniques from Andalusia.",
                "Cultural family terms, skills, geographical references",
            ),
        ],
    }

    print("=" * 80)
    print("NER QUALITY TEST - UltraBERT v4 with GlobalPointer")
    print("=" * 80)

    issues_found = 0
    issues_resolved = 0
    total_tests = 0

    for issue_name, cases in test_cases.items():
        print(f"\n{issue_name}")
        print("-" * 60)

        for text, expected in cases:
            total_tests += 1
            result = client.analyze(text, ["ner_family", "ner_general"])
            family_ents = result.entities
            general_ents = result.general_entities

            all_ents = []
            for e in family_ents:
                all_ents.append((e["text"], e["label"], "family", e.get("score", 0)))
            for e in general_ents:
                all_ents.append((e["text"], e["label"], "general", e.get("score", 0)))

            # Check if issue is resolved
            has_garbage = False

            # Define garbage checks based on issue type
            if "MILESTONE tags verbs" in issue_name:
                for e in all_ents:
                    if e[1] == "MILESTONE" and e[0].lower() in [
                        "learned",
                        "passed",
                        "promoted",
                        "accepted",
                        "started",
                        "graduated",
                        "bought",
                    ]:
                        has_garbage = True
            elif "FAMILY_EVENT tags pronouns" in issue_name:
                for e in all_ents:
                    if e[0].lower() in ["our", "we", "3", "4", "2", "5"] and e[1] == "FAMILY_EVENT":
                        has_garbage = True
            elif "HEIRLOOM tags prepositions" in issue_name:
                for e in all_ents:
                    if (
                        e[0].lower() in ["old", "to", "from", "in", "safe", "displayed", "contains"]
                        and e[1] == "HEIRLOOM"
                    ):
                        has_garbage = True
            elif "PET tags determiners" in issue_name:
                for e in all_ents:
                    if e[0].lower() == "the" and e[1] == "PET":
                        has_garbage = True
            elif "PERSON tags verbs" in issue_name:
                for e in all_ents:
                    if e[0].lower() in [
                        "met",
                        "asked",
                        "called",
                        "thinking",
                        "went",
                        "came",
                        "worked",
                        "drove",
                    ] and e[1] in ["PER", "PERSON"]:
                        has_garbage = True
            elif "PERSON tags emotions" in issue_name:
                for e in all_ents:
                    if e[0].lower() in [
                        "anxious",
                        "grateful",
                        "stressed",
                        "excited",
                        "worried",
                        "happy",
                        "sad",
                        "proud",
                    ] and e[1] in [
                        "PER",
                        "PERSON",
                    ]:
                        has_garbage = True
            elif "ORG tags common nouns" in issue_name:
                for e in all_ents:
                    if (
                        e[0].lower()
                        in [
                            "meeting",
                            "email",
                            "afternoon",
                            "manager",
                            "conference",
                            "accounting",
                            "kitchen",
                            "hospital",
                        ]
                        and e[1] == "ORG"
                    ):
                        has_garbage = True
            elif "Partial entity extraction" in issue_name:
                # Check if full span is extracted (this is a POSITIVE test - we want full spans)
                if "Lincoln School" in text:
                    has_full = any(e[0] == "Lincoln School" for e in all_ents)
                    # Issue = partial extraction (just "Lincoln" without "School")
                    has_garbage = any(e[0] == "Lincoln" for e in all_ents) and not has_full
                elif "San Francisco" in text:
                    has_full = any(e[0] == "San Francisco" for e in all_ents)
                    has_garbage = any(e[0] == "San" for e in all_ents) and not has_full
                elif "Bella Notte" in text:
                    has_full = any("Bella Notte" in e[0] for e in all_ents)
                    has_garbage = any(e[0] == "Bella" for e in all_ents) and not has_full
                elif "Johnson and Johnson" in text:
                    has_full = any("Johnson and Johnson" in e[0] for e in all_ents)
                    has_garbage = any(e[0] in ["Johnson", "and"] for e in all_ents) and not has_full
                elif "Madison Square Garden" in text:
                    has_full = any("Madison Square Garden" in e[0] for e in all_ents)
                    has_garbage = (
                        any(
                            e[0] in ["Madison", "Square", "Garden"]
                            for e in all_ents
                            if len(e[0].split()) < 3
                        )
                        and not has_full
                    )
                elif "Los Angeles International Airport" in text:
                    has_full = any("Los Angeles International Airport" in e[0] for e in all_ents)
                    has_garbage = (
                        any(
                            e[0] in ["Los", "Angeles", "International", "Airport"]
                            for e in all_ents
                            if len(e[0].split()) < 4
                        )
                        and not has_full
                    )
                elif "Massachusetts Institute of Technology" in text:
                    has_full = any(
                        "Massachusetts Institute of Technology" in e[0] for e in all_ents
                    )
                    has_garbage = (
                        any(
                            e[0] in ["Massachusetts", "Institute", "of", "Technology"]
                            for e in all_ents
                            if len(e[0].split()) < 4
                        )
                        and not has_full
                    )
            elif "Time fragments" in issue_name:
                for e in all_ents:
                    if e[0].lower() in ["3pm", "10am", "12pm", "6pm", "9pm", "8am", "7pm"] and e[
                        1
                    ] in ["PER", "PERSON"]:
                        has_garbage = True
            elif "Verb forms" in issue_name:
                for e in all_ents:
                    if e[0].lower() in [
                        "learned",
                        "working",
                        "organized",
                        "developed",
                        "created",
                        "managed",
                    ] and e[1] in [
                        "PER",
                        "PERSON",
                        "ORG",
                    ]:
                        has_garbage = True
            elif "Complex family contexts" in issue_name:
                # For hard cases, check for obvious garbage like verbs/emotions being tagged as entities
                garbage_words = [
                    "took",
                    "watched",
                    "inherited",
                    "being",
                    "celebrated",
                    "loves",
                    "playing",
                    "called",
                    "went",
                    "working",
                    "finish",
                    "old",
                    "still",
                    "running",
                    "made",
                    "amazing",
                    "have",
                    "been",
                    "passed",
                    "through",
                    "taught",
                    "using",
                    "old",
                ]
                for e in all_ents:
                    if e[0].lower() in garbage_words and e[1] in [
                        "PER",
                        "PERSON",
                        "ORG",
                        "MILESTONE",
                        "FAMILY_EVENT",
                        "HEIRLOOM",
                        "PET",
                    ]:
                        has_garbage = True
            elif "Ambiguous entities" in issue_name:
                # Check for common false positives in ambiguous contexts
                garbage_words = [
                    "called",
                    "meeting",
                    "working",
                    "late",
                    "again",
                    "old",
                    "from",
                    "1965",
                ]
                for e in all_ents:
                    if e[0].lower() in garbage_words and e[1] in [
                        "PER",
                        "PERSON",
                        "ORG",
                        "MILESTONE",
                        "FAMILY_EVENT",
                        "HEIRLOOM",
                        "PET",
                    ]:
                        has_garbage = True
            elif "Cultural and contextual challenges" in issue_name:
                # Check for cultural terms being incorrectly tagged
                garbage_words = [
                    "made",
                    "amazing",
                    "traditional",
                    "have",
                    "been",
                    "passed",
                    "taught",
                    "old",
                ]
                for e in all_ents:
                    if e[0].lower() in garbage_words and e[1] in [
                        "PER",
                        "PERSON",
                        "ORG",
                        "MILESTONE",
                        "FAMILY_EVENT",
                        "HEIRLOOM",
                        "PET",
                    ]:
                        has_garbage = True

            status = "ISSUE" if has_garbage else "OK"
            if has_garbage:
                issues_found += 1
            else:
                issues_resolved += 1

            ents_str = (
                ", ".join([f"{e[0]}:{e[1]}({e[3]:.2f})" for e in all_ents])
                if all_ents
                else "(none)"
            )
            print(f'  [{status}] "{text}"')
            print(f"       Entities: {ents_str}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Issues resolved: {issues_resolved}")
    print(f"Issues remaining: {issues_found}")
    print(f"Resolution rate: {issues_resolved / total_tests * 100:.1f}%")

    if issues_found == 0:
        print("\nAll known NER quality issues are RESOLVED by v4 GlobalPointer!")
    else:
        print(f"\n{issues_found} issues still need post-processing filters.")


if __name__ == "__main__":
    main()
