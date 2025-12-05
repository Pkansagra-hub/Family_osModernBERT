"""
Smart Resume Checker for Unified Data Generator

Analyzes what's been processed and what remains:
- Checks existing output samples
- Checks progress tracker state
- Calculates true remaining work per part
- Shows whether to continue or reset

Usage:
    python check_resume_state.py
    python check_resume_state.py --reset  # Clear progress to start fresh
"""

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

# Paths
BASE_DIR = Path("D:/Modeling_studio")
UNIFIED_DIR = BASE_DIR / "data" / "familyos" / "unified"
PARTS_DIR = UNIFIED_DIR / "parts"
OUTPUT_DIR = UNIFIED_DIR / "output"
PROGRESS_FILE = UNIFIED_DIR / "progress.json"


def compute_hash(text: str) -> str:
    """Compute hash for a text string."""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()


def load_existing_output_hashes() -> set:
    """Load all text hashes from existing output."""
    hashes = set()

    if not OUTPUT_DIR.exists():
        return hashes

    for shard in OUTPUT_DIR.glob("shard_*.jsonl"):
        with open(shard, encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    text = sample.get("text", "")
                    if text:
                        hashes.add(compute_hash(text))
                except:
                    continue

    return hashes


def load_part_texts(part_id: int) -> list[str]:
    """Load all texts from a part file."""
    part_file = PARTS_DIR / f"part_{part_id}.jsonl"
    texts = []

    if not part_file.exists():
        return texts

    with open(part_file, encoding="utf-8") as f:
        for line in f:
            try:
                sample = json.loads(line.strip())
                text = sample.get("text", "")
                if text:
                    texts.append(text)
            except Exception:
                continue

    return texts


def analyze_resume_state():
    """Analyze what's processed and what remains."""

    print("\n" + "=" * 80)
    print("UNIFIED DATA GENERATOR - RESUME STATE ANALYSIS")
    print("=" * 80)

    # Load existing output
    print("\n1. Loading existing output...")
    output_hashes = load_existing_output_hashes()
    print(f"   Found {len(output_hashes):,} existing output samples")

    # Load progress tracker
    print("\n2. Loading progress tracker...")
    progress_state = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress_state = json.load(f)
        print(f"   Found progress file with {len(progress_state)} keys tracked")
        for key, info in progress_state.items():
            print(f"   - {key}: Part {info['part']}, {info['processed']}/{info['total']} processed")
    else:
        print("   No progress file found (starting fresh)")

    # Analyze each part
    print("\n3. Analyzing parts vs output...")
    print("\n" + "=" * 80)

    total_input = 0
    total_remaining = 0

    for part_id in range(1, 7):  # 6 parts
        print(f"\nPART {part_id}:")

        # Load part texts
        part_texts = load_part_texts(part_id)
        total_input += len(part_texts)

        if not part_texts:
            print(f"  ⚠️  Part file not found or empty")
            continue

        # Check how many are already in output
        already_done = 0
        for text in part_texts:
            if compute_hash(text) in output_hashes:
                already_done += 1

        remaining = len(part_texts) - already_done
        total_remaining += remaining

        # Check progress tracker claim
        key_id = part_id - 1
        key_name = f"key_{key_id}"
        claimed_processed = progress_state.get(key_name, {}).get("processed", 0)

        print(f"  Total input samples: {len(part_texts):,}")
        print(f"  Already in output: {already_done:,} ({100*already_done/len(part_texts):.1f}%)")
        print(f"  Remaining to process: {remaining:,}")
        print(f"  Progress tracker claims: {claimed_processed:,} processed")

        # Check consistency
        if claimed_processed > already_done:
            discrepancy = claimed_processed - already_done
            print(
                f"  ⚠️  INCONSISTENCY: Tracker claims {discrepancy:,} more than actually in output"
            )
            print(f"     (Likely duplicates were skipped during generation)")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total input samples (all parts): {total_input:,}")
    print(f"Already generated (in output): {len(output_hashes):,}")
    print(f"Remaining to generate: {total_remaining:,}")
    print(f"Progress: {100*len(output_hashes)/total_input:.1f}% complete")
    print("=" * 80)

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if total_remaining == 0:
        print("✅ ALL DONE! All input samples have been generated.")
        print("   You can safely export the final dataset.")
    elif total_remaining > 0:
        print(f"🔄 RESUME NEEDED: {total_remaining:,} samples remaining")
        print("\nOption 1: Continue from checkpoint (RECOMMENDED)")
        print("   The script will automatically skip already-generated samples.")
        print("   Just run: python unified_data_generator.py generate")
        print("\nOption 2: Fix progress tracker if needed")
        print(f"   Run: python check_resume_state.py --fix-progress")

    print("=" * 80)

    return {
        "total_input": total_input,
        "total_output": len(output_hashes),
        "total_remaining": total_remaining,
        "progress_state": progress_state,
    }


def fix_progress_tracker():
    """Reset progress tracker to match actual output."""

    print("\n" + "=" * 80)
    print("FIXING PROGRESS TRACKER")
    print("=" * 80)

    # Load existing output
    print("\nLoading existing output...")
    output_hashes = load_existing_output_hashes()
    print(f"Found {len(output_hashes):,} existing samples")

    # Calculate true progress for each part
    new_progress = {}

    for part_id in range(1, 7):
        part_texts = load_part_texts(part_id)
        if not part_texts:
            continue

        # Count how many are in output
        processed_count = sum(1 for text in part_texts if compute_hash(text) in output_hashes)

        key_id = part_id - 1
        key_name = f"key_{key_id}"

        new_progress[key_name] = {
            "part": part_id,
            "processed": processed_count,
            "total": len(part_texts),
            "successful": processed_count,
            "failed": 0,
            "last_update": "FIXED",
        }

        print(f"Part {part_id} (Key {key_id}): {processed_count}/{len(part_texts)} processed")

    # Save fixed progress
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(new_progress, f, indent=2)

    print(f"\n✅ Progress tracker fixed and saved to {PROGRESS_FILE}")
    print("=" * 80)


def reset_progress():
    """Delete progress file to start fresh."""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"✅ Deleted {PROGRESS_FILE}")
        print("   Next run will start from beginning (but skip existing output samples)")
    else:
        print("⚠️  No progress file found")


def main():
    parser = argparse.ArgumentParser(description="Check resume state for unified data generator")
    parser.add_argument(
        "--fix-progress",
        action="store_true",
        help="Fix progress tracker to match actual output",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete progress file (will restart from beginning)",
    )

    args = parser.parse_args()

    if args.reset:
        response = input("\n⚠️  This will delete progress.json. Continue? (yes/no): ")
        if response.lower() == "yes":
            reset_progress()
        else:
            print("Cancelled.")
    elif args.fix_progress:
        fix_progress_tracker()
    else:
        analyze_resume_state()


if __name__ == "__main__":
    main()
