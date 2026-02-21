#!/usr/bin/env python3
"""
Build easier benchmarks using simpler math problems.

This replaces the too-hard AQuA-RAT/SVAMP benchmarks with problems that
edge models can actually solve, while still being challenging enough to
show the value of neurosymbolic routing.
"""

import csv
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)

def clean_text(text):
    """Clean bytestring or string to plain text."""
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    return text.strip()

def extract_number(answer_str):
    """Extract a simple numeric answer from various formats."""
    import re

    # Handle bytes
    if isinstance(answer_str, bytes):
        answer_str = answer_str.decode('utf-8')

    answer_str = answer_str.strip()

    # Try to extract just a number (integer)
    match = re.search(r'^-?\d+$', answer_str)
    if match:
        return match.group(0)

    # Skip fractions and expressions - too complex
    return None

def is_simple_arithmetic(question):
    """Check if question is simple arithmetic (good for AR category)."""
    question = clean_text(question).lower()
    # Look for simple operations
    simple_keywords = ['what is', 'calculate', 'evaluate', 'compute']
    complex_keywords = ['probability', 'simplify', 'assuming', 'derive']

    has_simple = any(kw in question for kw in simple_keywords)
    has_complex = any(kw in question for kw in complex_keywords)

    return has_simple and not has_complex

def is_simple_algebra(question):
    """Check if question is simple algebra (good for ALG category)."""
    question = clean_text(question).lower()
    # Look for solve equations
    algebra_keywords = ['solve', 'let', 'for t', 'for x', 'for n']
    complex_keywords = ['probability', 'simplify k**']

    has_algebra = any(kw in question for kw in algebra_keywords)
    has_complex = any(kw in question for kw in complex_keywords)

    return has_algebra and not has_complex

def load_gsm8k_simple(n=10):
    """Load simple word problems from GSM8K."""
    print(f"Loading {n} simple word problems from GSM8K...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")

        problems = []
        for ex in ds:
            question = ex['question']
            answer_text = ex['answer']

            # Extract final answer (usually after ####)
            if '####' in answer_text:
                answer = answer_text.split('####')[-1].strip()
                # Remove commas from numbers
                answer = answer.replace(',', '')

                # Only use if answer is a simple integer
                if answer.isdigit() or (answer.startswith('-') and answer[1:].isdigit()):
                    # Prefer shorter questions (easier)
                    if len(question) < 200:
                        problems.append({
                            'question': question,
                            'answer': answer
                        })

        # Sort by question length (shorter = easier) and take first n
        problems.sort(key=lambda x: len(x['question']))
        return problems[:n]
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []

def load_boolq_simple(n=10):
    """Load Yes/No questions from BoolQ."""
    print(f"Loading {n} Yes/No questions from BoolQ...")
    try:
        ds = load_dataset("google/boolq", split="train")

        problems = []
        for ex in ds:
            question = ex['question']
            answer = "Yes" if ex['answer'] else "No"
            passage = ex['passage']

            # Create a combined prompt
            full_question = f"Passage: {passage}\n\nQuestion: {question}"

            # Prefer shorter passages (easier)
            if len(passage) < 300:
                problems.append({
                    'question': full_question,
                    'answer': answer
                })

        # Shuffle and take n
        random.shuffle(problems)
        return problems[:n]
    except Exception as e:
        print(f"Error loading BoolQ: {e}")
        return []

def main():
    output_dir = Path("data/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Building EASIER benchmark for edge models")
    print("=" * 70)

    # Load the deepmind math dataset
    print("\nLoading DeepMind mathematics dataset...")
    try:
        ds = load_dataset("mlfoundations-dev/a1_math_deepmind", split="train")
        print(f"Loaded {len(ds)} problems total")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Filter for simple problems - just need numeric answers
    ar_problems = []  # Arithmetic
    alg_problems = []  # Algebra
    other_numeric = []

    print("\nFiltering for problems with numeric answers...")
    for i, ex in enumerate(ds):
        if len(ar_problems) >= 20 and len(alg_problems) >= 20:
            break

        question = clean_text(ex['question'])
        answer = extract_number(ex['answer'])

        if answer is None:
            continue

        # Must be reasonably short (< 150 chars = easier)
        if len(question) > 150:
            continue

        # Check if it's simple arithmetic
        if len(ar_problems) < 20 and is_simple_arithmetic(question):
            ar_problems.append({
                'question': question,
                'answer': answer
            })

        # Check if it's simple algebra
        elif len(alg_problems) < 20 and is_simple_algebra(question):
            alg_problems.append({
                'question': question,
                'answer': answer
            })

        # Save others as backup
        elif len(other_numeric) < 30:
            other_numeric.append({
                'question': question,
                'answer': answer,
                'has_solve': 'solve' in question.lower()
            })

    # If we don't have enough, use the "other" bucket
    if len(ar_problems) < 10:
        # Take shortest problems as AR
        other_numeric.sort(key=lambda x: len(x['question']))
        for prob in other_numeric:
            if len(ar_problems) >= 10:
                break
            ar_problems.append(prob)

    if len(alg_problems) < 10:
        # Take problems with "solve" as ALG
        solve_probs = [p for p in other_numeric if p.get('has_solve')]
        for prob in solve_probs:
            if len(alg_problems) >= 10:
                break
            if prob not in ar_problems:
                alg_problems.append(prob)

    print(f"Found {len(ar_problems)} AR problems")
    print(f"Found {len(alg_problems)} ALG problems")

    # Load WP and LOG from other datasets
    wp_problems = load_gsm8k_simple(n=15)
    log_problems = load_boolq_simple(n=15)

    print(f"Found {len(wp_problems)} WP problems")
    print(f"Found {len(log_problems)} LOG problems")

    # Build Tier-1 (40 prompts: 10 each)
    tier1_data = []

    # Add problems with proper formatting
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
    print(f"  Total prompts: {len(tier1_data)}")
    print(f"  AR: {sum(1 for x in tier1_data if x['category'] == 'AR')}")
    print(f"  ALG: {sum(1 for x in tier1_data if x['category'] == 'ALG')}")
    print(f"  WP: {sum(1 for x in tier1_data if x['category'] == 'WP')}")
    print(f"  LOG: {sum(1 for x in tier1_data if x['category'] == 'LOG')}")

    # Show some samples
    print("\n" + "=" * 70)
    print("Sample problems from new benchmark:")
    print("=" * 70)
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        sample = next(x for x in tier1_data if x['category'] == cat)
        print(f"\n{cat} Sample ({sample['id']}):")
        print(f"  Question: {sample['prompt'][:150]}...")
        print(f"  Answer: {sample['expected_answer']}")

if __name__ == "__main__":
    main()
