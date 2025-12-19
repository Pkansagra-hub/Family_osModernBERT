"""Analyze counterfactual training data"""
import json
from pathlib import Path
from collections import Counter

def analyze_data():
    merged_path = Path("D:/Modeling_studio/data/counterfactual/merged")

    print("=== SAMPLE LIFE EVENTS ===")
    count = 0
    for shard in merged_path.glob("shard_*.jsonl"):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("domain") == "life_events" and count < 3:
                        inp = d.get("input", {})
                        cf = d.get("counterfactual", {})
                        text = inp.get("text", "")[:120]
                        output = cf.get("full_text", "")[:180]
                        print(f"INPUT: {text}...")
                        print(f"OUTPUT: {output}...")
                        print("-" * 60)
                        count += 1
                except:
                    pass
        if count >= 3:
            break

    print("\n=== SAMPLE PARENTING ===")
    count = 0
    for shard in merged_path.glob("shard_*.jsonl"):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("domain") == "parenting" and count < 3:
                        inp = d.get("input", {})
                        cf = d.get("counterfactual", {})
                        text = inp.get("text", "")[:120]
                        output = cf.get("full_text", "")[:180]
                        print(f"INPUT: {text}...")
                        print(f"OUTPUT: {output}...")
                        print("-" * 60)
                        count += 1
                except:
                    pass
        if count >= 3:
            break

    print("\n=== SAMPLE RELATIONSHIP ===")
    count = 0
    for shard in merged_path.glob("shard_*.jsonl"):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("domain") == "relationship" and count < 3:
                        inp = d.get("input", {})
                        cf = d.get("counterfactual", {})
                        text = inp.get("text", "")[:120]
                        output = cf.get("full_text", "")[:180]
                        print(f"INPUT: {text}...")
                        print(f"OUTPUT: {output}...")
                        print("-" * 60)
                        count += 1
                except:
                    pass
        if count >= 3:
            break

if __name__ == "__main__":
    analyze_data()
