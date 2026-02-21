#!/usr/bin/env python3
"""
Build easier benchmarks using EASIER problems from industry datasets.

Uses same datasets (GSM8K, SVAMP, BoolQ) but filters for SIMPLE problems
that edge models can actually solve.
"""

import csv
import random
import re
from pathlib import Path
from datasets import load_dataset

random.seed(42)

def extract_simple_number(text):
    """Extract a simple integer from answer text."""
    # Remove commas
    text = text.replace(',', '')
    # Try to find a simple integer
    match = re.search(r'\b(\d{1,5})\b', text)
    if match:
        return match.group(1)
    return None

def build_ar_from_gsm8k():
    """Get SIMPLE arithmetic from GSM8K (1-2 step problems)."""
    print("Loading simple arithmetic from GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    problems = []
    for ex in ds:
        question = ex['question']
        answer_text = ex['answer']

        # Extract final answer
        if '####' in answer_text:
            answer = answer_text.split('####')[-1].strip().replace(',', '')

            # Only simple integers < 1000 (easier problems)
            if answer.isdigit() and int(answer) < 1000:
                # Prefer 1-sentence questions (simpler)
                if question.count('.') <= 2 and len(question) < 120:
                    problems.append({
                        'question': question,
                        'answer': answer
                    })

    # Sort by answer magnitude (smaller = easier)
    problems.sort(key=lambda x: int(x['answer']))
    return problems[:15]

def build_alg_from_gsm8k():
    """Get slightly harder GSM8K for algebra category."""
    print("Loading algebra-like problems from GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    problems = []
    for ex in ds:
        question = ex['question']
        answer_text = ex['answer']

        # Extract final answer
        if '####' in answer_text:
            answer = answer_text.split('####')[-1].strip().replace(',', '')

            # Accept answers < 10000
            if answer.isdigit() and int(answer) < 10000:
                # Look for multi-step (2-3 sentences)
                if 2 < question.count('.') <= 4 and len(question) < 180:
                    problems.append({
                        'question': question,
                        'answer': answer
                    })

    # Shuffle and take random selection
    random.shuffle(problems)
    return problems[:15]

def build_wp_from_gsm8k():
    """Get word problems from GSM8K."""
    print("Loading word problems from GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    problems = []
    for ex in ds:
        question = ex['question']
        answer_text = ex['answer']

        # Extract final answer
        if '####' in answer_text:
            answer = answer_text.split('####')[-1].strip().replace(',', '')

            # Medium difficulty
            if answer.isdigit() and 10 < int(answer) < 5000:
                if len(question) < 200:
                    problems.append({
                        'question': question,
                        'answer': answer
                    })

    # Sort by length (shorter = easier)
    problems.sort(key=lambda x: len(x['question']))
    return problems[:15]

def build_log_from_boolq():
    """Get Yes/No from BoolQ."""
    print("Loading Yes/No questions from BoolQ...")
    ds = load_dataset("google/boolq", split="train")

    problems = []
    for ex in ds:
        question = ex['question']
        answer = "Yes" if ex['answer'] else "No"
        passage = ex['passage']

        # Prefer shorter passages (easier)
        if len(passage) < 250 and len(question) < 100:
            full_question = f"Passage: {passage}\n\nQuestion: {question}"
            problems.append({
                'question': full_question,
                'answer': answer
            })

    # Shuffle
    random.shuffle(problems)
    return problems[:15]

def main():
    output_dir = Path("data/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building EASIER benchmark from industry-standard datasets")
    print("=" * 70)
    print()

    # Load all categories
    ar_problems = build_ar_from_gsm8k()
    alg_problems = build_alg_from_gsm8k()
    wp_problems = build_wp_from_gsm8k()
    log_problems = build_log_from_boolq()

    print(f"\nCollected:")
    print(f"  AR: {len(ar_problems)} (simple arithmetic from GSM8K)")
    print(f"  ALG: {len(alg_problems)} (multi-step from GSM8K)")
    print(f"  WP: {len(wp_problems)} (word problems from GSM8K)")
    print(f"  LOG: {len(log_problems)} (Yes/No from BoolQ)")

    # Build Tier-1 (40 prompts)
    tier1_data = []

    for i, prob in enumerate(ar_problems[:10], 1):
        tier1_data.append({
            'id': f'AR{i:03d}',
            'category': 'AR',
            'prompt': f"{prob['question']}\nAnswer with only the final number.",
            'expected_answer': prob['answer']
        })

    for i, prob in enumerate(alg_problems[:10], 1):
        tier1_data.append({
            'id': f'ALG{i:03d}',
            'category': 'ALG',
            'prompt': f"{prob['question']}\nAnswer with only the final number.",
            'expected_answer': prob['answer']
        })

    for i, prob in enumerate(wp_problems[:10], 1):
        tier1_data.append({
            'id': f'WP{i:03d}',
            'category': 'WP',
            'prompt': f"{prob['question']}\nAnswer with only the final number.",
            'expected_answer': prob['answer']
        })

    for i, prob in enumerate(log_problems[:10], 1):
        tier1_data.append({
            'id': f'LOG{i:03d}',
            'category': 'LOG',
            'prompt': f"{prob['question']}\nAnswer with only Yes or No.",
            'expected_answer': prob['answer']
        })

    # Write Tier-1
    tier1_path = output_dir / "industry_tier1_40_easy.csv"
    with open(tier1_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'prompt', 'expected_answer'])
        writer.writeheader()
        writer.writerows(tier1_data)

    print(f"\n✓ Created Tier-1 benchmark: {tier1_path}")
    print(f"  Total: {len(tier1_data)} prompts")
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        count = sum(1 for x in tier1_data if x['category'] == cat)
        print(f"  {cat}: {count}")

    # Show samples
    print("\n" + "=" * 70)
    print("Sample problems:")
    print("=" * 70)
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        sample = next(x for x in tier1_data if x['category'] == cat)
        print(f"\n{cat} ({sample['id']}):")
        # Show first 120 chars of prompt
        prompt_preview = sample['prompt'].replace('\n', ' ')[:120]
        print(f"  {prompt_preview}...")
        print(f"  Answer: {sample['expected_answer']}")

    print("\n" + "=" * 70)
    print("✓ Benchmark ready to test!")
    print("=" * 70)

if __name__ == "__main__":
    main()
