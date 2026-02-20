#!/usr/bin/env python3
"""Quick sanity check for Tier-1 CSV content."""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def compress_prompt(prompt: str, limit: int = 200) -> str:
    """Collapse whitespace and truncate for display."""
    flat = " ".join(prompt.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def expected_ok(category: str, expected: str) -> bool:
    if category == "LOG":
        return expected in {"Yes", "No"}
    return bool(re.match(r"^-?\d+$", expected))


def instruction_ok(category: str, prompt: str) -> bool:
    p = prompt.lower()
    if category == "LOG":
        return "answer with only yes or no." in p
    return "answer with only the final number." in p


def main() -> int:
    ap = argparse.ArgumentParser(description="Print sample Tier-1 prompts per category.")
    ap.add_argument("--csv", default="data/benchmarks/industry_tier1_40.csv", help="Benchmark CSV to inspect")
    ap.add_argument("--per_cat", type=int, default=2, help="Number of samples to show per category")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"CSV not found: {path}", file=sys.stderr)
        return 1

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        groups[row["category"].strip()].append(row)

    required_cats = {"AR", "ALG", "LOG", "WP"}
    missing = required_cats - set(groups)
    if missing:
        print(f"WARNING: missing categories in {path.name}: {', '.join(sorted(missing))}")

    issues = []

    print(f"Loaded {len(rows)} rows from {path}")
    for cat in ["AR", "ALG", "LOG", "WP"]:
        cat_rows = groups.get(cat, [])
        print(f"\n{cat}: total={len(cat_rows)} (showing up to {args.per_cat})")
        for sample in cat_rows[: args.per_cat]:
            pid = sample["id"].strip()
            prompt = sample["prompt"]
            expected = sample["expected_answer"].strip()

            if not expected_ok(cat, expected):
                issues.append(f"{pid} expected_answer invalid for {cat}: {expected!r}")
            if not instruction_ok(cat, prompt):
                issues.append(f"{pid} prompt does not end with required instruction for {cat}")

            preview = compress_prompt(prompt)
            print(f"- {pid} expect={expected} prompt={preview}")

    if issues:
        print("\nIssues detected:")
        for msg in issues:
            print(f"  - {msg}")
        return 1

    print("\nOK: samples look consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
