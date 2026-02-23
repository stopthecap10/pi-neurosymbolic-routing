#!/usr/bin/env python3
"""
Evaluate A6 logic solver on all Tier-1 LOG prompts.

Usage:
    python scripts/eval_a6_on_log_t1.py
"""

import csv
import os
import sys

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from a6_logic_engine import solve_logic


def load_log_prompts(csv_path: str) -> list:
    """Load LOG category prompts from benchmark CSV."""
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('category', '') == 'LOG':
                rows.append(row)
    return rows


def main():
    # Find the benchmark CSV
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, 'data', 'splits', 'industry_tier1_40_v2.csv')

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    prompts = load_log_prompts(csv_path)
    print(f"Loaded {len(prompts)} LOG prompts from {csv_path}\n")

    correct = 0
    parsed = 0
    total = len(prompts)

    print(f"{'ID':>12} {'GT':>4} {'A6':>4} {'Match':>6} {'Parse':>6} "
          f"{'Facts':>6} {'Rules':>6} {'Derived':>8} {'ms':>8}")
    print("-" * 80)

    for row in prompts:
        pid = row.get('prompt_id', row.get('id', '?'))
        gt = row.get('answer', row.get('ground_truth', '')).strip()
        prompt_text = row.get('prompt_text', row.get('prompt', ''))

        result = solve_logic(prompt_text)
        a6_answer = result['answer']

        # Normalize for comparison
        gt_norm = gt.strip().lower()
        a6_norm = a6_answer.strip().lower()
        match = (gt_norm == a6_norm)

        if match:
            correct += 1
        if result['parse_success']:
            parsed += 1

        status = "OK" if match else "WRONG"
        print(f"{pid:>12} {gt:>4} {a6_answer:>4} {status:>6} "
              f"{'Y' if result['parse_success'] else 'N':>6} "
              f"{result['n_facts']:>6} {result['n_rules']:>6} "
              f"{result['n_derived']:>8} {result['latency_ms']:>7.2f}")

        if not match:
            print(f"    TRACE: {result['trace'][:200]}")
            print()

    print("-" * 80)
    print(f"\nResults: {correct}/{total} correct ({correct/total*100:.0f}%)")
    print(f"Parsed:  {parsed}/{total} ({parsed/total*100:.0f}%)")
    print(f"A1 baseline: 6/10 (60%)")
    if correct > 6:
        print(f"A6 BEATS A1 by {(correct-6)*10}pp!")
    elif correct == 6:
        print(f"A6 ties A1.")
    else:
        print(f"A6 loses to A1 by {(6-correct)*10}pp.")


if __name__ == "__main__":
    main()
