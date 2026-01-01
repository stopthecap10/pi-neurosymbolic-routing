#!/usr/bin/env python3
"""
Smoke test script to inspect random samples from benchmark CSV.

Prints 2 random samples from each category for manual quality inspection.
"""

import argparse
import csv
import random
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Print random samples from each category for manual inspection"
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument(
        "--samples-per-cat",
        type=int,
        default=2,
        help="Number of samples to show per category (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    args = parser.parse_args()

    # Load CSV
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Group by category
    by_category = defaultdict(list)
    for row in rows:
        category = row["category"].strip()
        by_category[category].append(row)

    # Set random seed for reproducibility
    rng = random.Random(args.seed)

    # Print samples for each category
    for category in sorted(by_category.keys()):
        samples = by_category[category]
        selected = rng.sample(samples, min(args.samples_per_cat, len(samples)))

        print(f"\n{'='*80}")
        print(f"CATEGORY: {category} ({len(samples)} total samples)")
        print(f"{'='*80}")

        for i, sample in enumerate(selected, 1):
            print(f"\n--- Sample {i} (ID: {sample['id']}) ---")
            print(f"\nPrompt:")
            print(sample['prompt'])
            print(f"\nExpected Answer: {sample['expected_answer']}")
            print()

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
