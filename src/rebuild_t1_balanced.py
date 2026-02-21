#!/usr/bin/env python3
"""
Rebuild T1 Split - Add 2 ALG Replacement Prompts

Reads the existing T1 CSV (38 rows, missing ALG_000004/ALG_000005),
pulls 2 replacement ALG prompts from the DeepMind Mathematics v1.0
tarball using the same pipeline as build_official_splits.py, and
outputs a balanced 40-row CSV as industry_tier1_40_v2.csv.

Rules:
- Replacements come from the same source (DeepMind Math algebra modules)
- Only single-integer answers (no commas, no multi-value)
- No overlap with any existing tier (T1/T2/T3)
- Deterministic seed for reproducibility
- Full provenance columns populated
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import tarfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Frozen config (same as build_official_splits.py)
DEEPMIND_MATH_URL = (
    "https://storage.googleapis.com/mathematics-dataset/"
    "mathematics_dataset-v1.0.tar.gz"
)
DEEPMIND_MATH_VERSION = "v1.0"

ALGEBRA_MODULES = [
    "algebra__linear_1d",
    "algebra__linear_2d",
    "algebra__polynomial_roots",
]

FIELD_MAP_VERSION = "fm_v1.0"

# Seed for replacement sampling (deterministic)
REPLACEMENT_SEED = 42 + 100  # offset from original seed to get fresh draws


def normalize_text(text: str) -> str:
    """Whitespace normalization only (Section 3.6)"""
    text = text.strip()
    text = text.replace('\r\n', '\n')
    text = re.sub(r' +', ' ', text)
    return text


def parse_deepmind_module(tarball_path: Path, module: str,
                          difficulty: str = "train-easy"):
    """Parse a single DeepMind Math module from tarball."""
    target_path = f"mathematics_dataset-v1.0/{difficulty}/{module}.txt"
    records = []

    with tarfile.open(str(tarball_path), 'r:gz') as tar:
        member = tar.getmember(target_path)
        f = tar.extractfile(member)
        if f is None:
            return []

        content = f.read().decode('utf-8')
        lines = content.strip().split('\n')

        idx = 0
        pair_idx = 0
        while idx + 1 < len(lines):
            question = lines[idx].strip()
            answer = lines[idx + 1].strip()
            idx += 2

            if not question or not answer:
                continue

            records.append({
                'question': question,
                'answer': answer,
                'pair_idx': pair_idx,
                'module': module,
            })
            pair_idx += 1

    return records


def load_all_existing_alg_ids(splits_dir: Path):
    """Load all ALG source_record_ids from all tiers to avoid overlap."""
    used_ids = set()
    for csv_file in splits_dir.glob("industry_tier*.csv"):
        with open(csv_file, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('category') == 'ALG':
                    used_ids.add(row['source_record_id'])
    return used_ids


def main():
    ap = argparse.ArgumentParser(description="Rebuild T1 with 2 ALG replacements")
    ap.add_argument("--splits_dir", default="data/splits", help="Splits directory")
    ap.add_argument("--cache_dir", default="data/cache", help="Cache directory")
    ap.add_argument("--dry_run", action="store_true", help="Print but don't write")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    cache_dir = Path(args.cache_dir)

    # Step 1: Load existing T1
    existing_t1_path = splits_dir / "industry_tier1_40.csv"
    if not existing_t1_path.exists():
        print(f"ERROR: {existing_t1_path} not found")
        sys.exit(1)

    with open(existing_t1_path, 'r', encoding='utf-8') as f:
        existing_rows = list(csv.DictReader(f))

    cat_counts = defaultdict(int)
    for r in existing_rows:
        cat_counts[r['category']] += 1

    print(f"Existing T1: {len(existing_rows)} rows")
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        print(f"  {cat}: {cat_counts[cat]}")

    needed_alg = 10 - cat_counts['ALG']
    if needed_alg <= 0:
        print(f"ALG already has {cat_counts['ALG']} rows, no replacements needed.")
        sys.exit(0)

    print(f"\nNeed {needed_alg} ALG replacement(s)")

    # Step 2: Load all existing ALG IDs across tiers
    used_alg_ids = load_all_existing_alg_ids(splits_dir)
    print(f"ALG source_record_ids already used across all tiers: {len(used_alg_ids)}")

    # Step 3: Load ALG pool from DeepMind Math tarball
    tarball_path = cache_dir / "mathematics_dataset-v1.0.tar.gz"
    if not tarball_path.exists():
        print(f"ERROR: Tarball not found at {tarball_path}")
        print("Run build_official_splits.py first to download it.")
        sys.exit(1)

    print(f"\nLoading ALG records from {tarball_path}...")
    all_alg_records = []
    filtered_counts = defaultdict(int)

    for module in ALGEBRA_MODULES:
        print(f"  Parsing {module}...")
        raw_pairs = parse_deepmind_module(tarball_path, module, "train-easy")

        for pair in raw_pairs:
            answer = pair['answer']

            # Filter: reject multi-value answers (commas)
            if ',' in answer:
                filtered_counts['multi_value'] += 1
                continue

            # Filter: single integer only
            answer_clean = answer.replace(' ', '')
            if not answer_clean.lstrip('-').isdigit():
                filtered_counts['non_integer'] += 1
                continue

            source_record_id = f"{module}_easy_{pair['pair_idx']}"

            # Filter: not already used in any tier
            if source_record_id in used_alg_ids:
                filtered_counts['already_used'] += 1
                continue

            all_alg_records.append({
                'question': pair['question'],
                'answer_clean': answer_clean,
                'answer_raw': answer,
                'module': module,
                'pair_idx': pair['pair_idx'],
                'source_record_id': source_record_id,
            })

        print(f"    {module}: {sum(1 for r in all_alg_records if r['module'] == module)} candidates")

    print(f"\nTotal ALG candidates (after filtering): {len(all_alg_records)}")
    print(f"Filtered out:")
    for reason, count in sorted(filtered_counts.items()):
        print(f"  {reason}: {count}")

    if len(all_alg_records) < needed_alg:
        print(f"ERROR: Only {len(all_alg_records)} candidates, need {needed_alg}")
        sys.exit(1)

    # Step 4: Sample replacements with deterministic seed
    rng = random.Random(REPLACEMENT_SEED)
    replacements = rng.sample(all_alg_records, needed_alg)

    print(f"\nSelected {needed_alg} replacement(s):")

    # Determine which prompt_ids to assign (fill the gaps)
    existing_alg_ids = sorted([
        r['prompt_id'] for r in existing_rows if r['category'] == 'ALG'
    ])
    all_expected = {f"ALG_{i:06d}" for i in range(1, 11)}
    missing_ids = sorted(all_expected - set(existing_alg_ids))

    replacement_rows = []
    for i, repl in enumerate(replacements):
        prompt_id = missing_ids[i] if i < len(missing_ids) else f"ALG_{10 + i + 1:06d}"

        row = {
            'prompt_id': prompt_id,
            'category': 'ALG',
            'dataset_name': 'deepmind_math',
            'dataset_source': DEEPMIND_MATH_URL,
            'dataset_version': DEEPMIND_MATH_VERSION,
            'source_split': 'train-easy',
            'source_record_id': repl['source_record_id'],
            'source_type': 'dataset_raw',
            'field_map_version': FIELD_MAP_VERSION,
            'prompt_text': normalize_text(repl['question']),
            'ground_truth': repl['answer_clean'],
            'source_answer_raw': repl['answer_raw'],
            'source_module': repl['module'],
            'source_meta_json': json.dumps({
                'pair_idx': repl['pair_idx'],
                'module': repl['module'],
                'difficulty': 'train-easy',
                'replacement': True,
                'replacement_reason': 'multi_value_answer_in_original',
                'replacement_seed': REPLACEMENT_SEED,
            }),
        }
        replacement_rows.append(row)

        print(f"  {prompt_id}: {repl['source_record_id']}")
        print(f"    Q: {repl['question'][:80]}...")
        print(f"    A: {repl['answer_clean']}")

    # Step 5: Merge and sort
    merged = existing_rows + replacement_rows

    # Sort by category then prompt_id for clean ordering
    cat_order = {'AR': 0, 'ALG': 1, 'WP': 2, 'LOG': 3}
    merged.sort(key=lambda r: (cat_order.get(r['category'], 9), r['prompt_id']))

    # Step 6: Validate
    print(f"\n{'='*60}")
    print("SPLIT INTEGRITY SUMMARY")
    print(f"{'='*60}")

    final_counts = defaultdict(int)
    for r in merged:
        final_counts[r['category']] += 1

    print(f"\nTotal rows: {len(merged)}")
    print(f"Per-category counts:")
    all_balanced = True
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        count = final_counts[cat]
        status = "OK" if count == 10 else "MISMATCH"
        if count != 10:
            all_balanced = False
        print(f"  {cat}: {count} {'(OK)' if count == 10 else '(EXPECTED 10)'}")

    # Check uniqueness
    prompt_ids = [r['prompt_id'] for r in merged]
    source_ids = [r['source_record_id'] for r in merged]
    dupes_prompt = len(prompt_ids) - len(set(prompt_ids))
    dupes_source = len(source_ids) - len(set(source_ids))
    print(f"\nUniqueness:")
    print(f"  Duplicate prompt_ids: {dupes_prompt}")
    print(f"  Duplicate source_record_ids: {dupes_source}")

    print(f"\nExclusions applied:")
    print(f"  Multi-value answers (commas): {filtered_counts.get('multi_value', 0)}")
    print(f"  Non-integer answers: {filtered_counts.get('non_integer', 0)}")
    print(f"  Already used in other tiers: {filtered_counts.get('already_used', 0)}")

    print(f"\nReplacement rows added:")
    for row in replacement_rows:
        print(f"  {row['prompt_id']}: {row['source_record_id']} (gt={row['ground_truth']})")

    print(f"\nFilters applied for ALG:")
    print(f"  1. Source: DeepMind Mathematics v1.0, algebra modules (linear_1d, linear_2d, polynomial_roots)")
    print(f"  2. Difficulty: train-easy")
    print(f"  3. Reject answers containing commas (multi-value polynomial roots)")
    print(f"  4. Reject non-integer answers")
    print(f"  5. Exclude source_record_ids already in T1/T2/T3")
    print(f"  6. Deterministic sample with seed={REPLACEMENT_SEED}")

    if not all_balanced or dupes_prompt > 0 or dupes_source > 0:
        print(f"\nERROR: Validation failed!")
        sys.exit(1)

    print(f"\nVALIDATION PASSED")

    if args.dry_run:
        print("\nDRY RUN - not writing files")
        return

    # Step 7: Write output
    output_path = splits_dir / "industry_tier1_40_v2.csv"
    fieldnames = [
        'prompt_id', 'category', 'dataset_name', 'dataset_source',
        'dataset_version', 'source_split', 'source_record_id',
        'source_type', 'field_map_version',
        'prompt_text', 'ground_truth',
        'source_answer_raw', 'source_module', 'source_meta_json',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    print(f"\nWrote: {output_path} ({len(merged)} rows)")

    # Step 8: Write manifest
    manifest_path = splits_dir / "industry_tier1_40_v2_manifest.json"
    manifest = {
        'split_file': 'industry_tier1_40_v2.csv',
        'split_version': 'v2',
        'previous_version': 'industry_tier1_40.csv (38 rows, 2 ALG removed)',
        'build_timestamp': datetime.now().isoformat(),
        'builder_script': 'src/rebuild_t1_balanced.py',
        'total_rows': len(merged),
        'per_category': dict(final_counts),
        'dropped_rows': {
            'ALG_000004': 'algebra__polynomial_roots multi-value answer (removed in v1)',
            'ALG_000005': 'algebra__polynomial_roots multi-value answer (removed in v1)',
        },
        'replacement_rows': {
            row['prompt_id']: {
                'source_record_id': row['source_record_id'],
                'source_module': row['source_module'],
                'ground_truth': row['ground_truth'],
                'reason': 'replacement for corrupted multi-value ALG row',
            }
            for row in replacement_rows
        },
        'replacement_seed': REPLACEMENT_SEED,
        'filtering_rules': [
            'Source: DeepMind Mathematics v1.0, algebra modules (linear_1d, linear_2d, polynomial_roots)',
            'Difficulty: train-easy',
            'Reject answers containing commas (multi-value polynomial roots)',
            'Reject non-integer answers',
            'Exclude source_record_ids already in T1/T2/T3',
            f'Deterministic sample with seed={REPLACEMENT_SEED}',
        ],
        'dataset_source': {
            'name': 'DeepMind Mathematics v1.0',
            'url': DEEPMIND_MATH_URL,
            'modules': list(ALGEBRA_MODULES),
            'difficulty': 'train-easy',
            'citation': 'Saxton et al., 2019 (arXiv:1904.01557)',
        },
        'validation': {
            'total_rows': len(merged),
            'balanced': all_balanced,
            'duplicate_prompt_ids': dupes_prompt,
            'duplicate_source_ids': dupes_source,
            'all_source_type_dataset_raw': all(r['source_type'] == 'dataset_raw' for r in merged),
        },
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
