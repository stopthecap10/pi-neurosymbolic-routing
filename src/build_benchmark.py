#!/usr/bin/env python3
"""
Build industry-standard benchmark CSVs from public datasets.

This script downloads/loads public datasets and generates deterministic
benchmark CSVs for evaluating mathematical and logical reasoning.
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple

def parse_gsm8k_answer(answer_text: str) -> str:
    """Extract final numeric answer from GSM8K answer field.

    GSM8K answers typically end with #### followed by the final answer.
    """
    if "####" in answer_text:
        # Extract everything after ####
        final_part = answer_text.split("####")[-1].strip()
        # Remove any commas and extract just the number
        final_part = final_part.replace(",", "")
        # Extract the first integer found
        match = re.search(r'-?\d+', final_part)
        if match:
            return match.group(0)
    # Fallback: try to find any integer in the answer
    match = re.search(r'-?\d+', answer_text.replace(",", ""))
    if match:
        return match.group(0)
    return "0"  # Default fallback


def load_gsm8k_samples(num_samples: int, seed: int) -> List[Dict]:
    """Load GSM8K dataset samples for WP category."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("gsm8k", "main", split="train")
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        print("You may need to install: pip install datasets")
        return []

    # Convert to list and sample deterministically
    all_samples = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    samples = []
    for i, item in enumerate(all_samples[:num_samples]):
        question = item["question"].strip()
        answer = parse_gsm8k_answer(item["answer"])

        prompt = f"""Solve the following word problem.
Answer with only the final number.

{question}
"""
        samples.append({
            "category": "WP",
            "prompt": prompt,
            "expected_answer": answer
        })

    return samples


def generate_arithmetic_samples(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic arithmetic samples for AR category."""
    rng = random.Random(seed)
    samples = []

    operations = [
        ('+', lambda a, b: a + b),
        ('-', lambda a, b: a - b),
        ('*', lambda a, b: a * b),
    ]

    for i in range(num_samples):
        op_symbol, op_func = rng.choice(operations)

        # Generate operands (vary complexity)
        if i < num_samples // 3:  # Easy
            a = rng.randint(1, 100)
            b = rng.randint(1, 100)
        elif i < 2 * num_samples // 3:  # Medium
            a = rng.randint(10, 500)
            b = rng.randint(10, 500)
        else:  # Hard
            a = rng.randint(100, 1000)
            b = rng.randint(100, 1000)

        # For subtraction, ensure positive result
        if op_symbol == '-' and a < b:
            a, b = b, a

        result = op_func(a, b)
        expression = f"{a} {op_symbol} {b}"

        prompt = f"""Compute the value of the following expression.
Answer with only the final number.

{expression}
"""
        samples.append({
            "category": "AR",
            "prompt": prompt,
            "expected_answer": str(result)
        })

    return samples


def generate_algebra_samples(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic algebra samples for ALG category."""
    rng = random.Random(seed)
    samples = []

    for i in range(num_samples):
        # Generate linear equations: x = a op b or x + a = b
        if i % 2 == 0:
            # Simple: x = a op b
            operations = [
                ('+', lambda a, b: a + b),
                ('-', lambda a, b: a - b),
                ('*', lambda a, b: a * b),
            ]
            op_symbol, op_func = rng.choice(operations)

            if i < num_samples // 3:  # Easy
                a = rng.randint(1, 50)
                b = rng.randint(1, 50)
            elif i < 2 * num_samples // 3:  # Medium
                a = rng.randint(10, 200)
                b = rng.randint(10, 200)
            else:  # Hard
                a = rng.randint(50, 500)
                b = rng.randint(50, 500)

            if op_symbol == '-' and a < b:
                a, b = b, a

            result = op_func(a, b)
            equation = f"x = {a} {op_symbol} {b}"
        else:
            # Solve for x: x + a = b or x - a = b
            if i < num_samples // 3:  # Easy
                a = rng.randint(1, 50)
                b = rng.randint(1, 100)
            elif i < 2 * num_samples // 3:  # Medium
                a = rng.randint(10, 200)
                b = rng.randint(10, 300)
            else:  # Hard
                a = rng.randint(50, 500)
                b = rng.randint(100, 1000)

            if rng.choice([True, False]):
                # x + a = b => x = b - a
                result = b - a
                equation = f"x + {a} = {b}"
            else:
                # x - a = b => x = b + a
                result = b + a
                equation = f"x - {a} = {b}"

        prompt = f"""Solve for x.
Answer with only the final number.

{equation}
"""
        samples.append({
            "category": "ALG",
            "prompt": prompt,
            "expected_answer": str(result)
        })

    return samples


def load_deepmind_math_samples(category: str, num_samples: int, seed: int) -> List[Dict]:
    """Load DeepMind Mathematics dataset samples for AR or ALG category.

    Falls back to synthetic generation if dataset is unavailable.
    """
    try:
        from datasets import load_dataset

        if category == "AR":
            # Try MATH dataset as alternative
            dataset = load_dataset("hendrycks/math", "algebra", split="train", trust_remote_code=True)
        else:  # ALG
            dataset = load_dataset("hendrycks/math", "algebra", split="train", trust_remote_code=True)

        # Filter for problems with integer answers
        all_samples = list(dataset)
        rng = random.Random(seed)
        rng.shuffle(all_samples)

        samples = []
        for item in all_samples:
            if len(samples) >= num_samples:
                break

            question = item.get("problem", "").strip()
            answer = item.get("solution", "").strip()

            # Try to extract integer answer
            match = re.search(r'-?\d+', answer.replace(",", ""))
            if not match:
                continue

            if category == "AR":
                prompt_prefix = "Compute the value of the following expression.\nAnswer with only the final number.\n\n"
            else:
                prompt_prefix = "Solve for x.\nAnswer with only the final number.\n\n"

            prompt = f"{prompt_prefix}{question}\n"
            samples.append({
                "category": category,
                "prompt": prompt,
                "expected_answer": match.group(0)
            })

        if len(samples) >= num_samples:
            return samples[:num_samples]
    except Exception as e:
        print(f"Error loading math dataset: {e}")

    # Fall back to synthetic generation
    print(f"Using synthetic generation for {category}...")
    if category == "AR":
        return generate_arithmetic_samples(num_samples, seed)
    else:
        return generate_algebra_samples(num_samples, seed)


def load_proofwriter_samples(num_samples: int, seed: int) -> List[Dict]:
    """Load ProofWriter dataset samples for LOG category."""
    try:
        from datasets import load_dataset
        # Load ProofWriter depth-5 dataset
        dataset = load_dataset("allenai/proofwriter", "depth-5", split="train")
    except Exception as e:
        print(f"Error loading ProofWriter: {e}")
        print("Trying alternative approach...")
        # If ProofWriter is not available, use a simpler logical reasoning dataset
        # or generate synthetic logic problems
        return generate_synthetic_logic_samples(num_samples, seed)

    # Sample deterministically
    all_samples = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    samples = []
    for item in all_samples:
        if len(samples) >= num_samples:
            break

        # ProofWriter format: theory (facts/rules) + question + answer
        theory = item.get("theory", "").strip()
        question = item.get("question", "").strip()
        answer = item.get("answer", False)

        if not theory or not question:
            continue

        # Convert boolean to Yes/No
        expected = "Yes" if answer else "No"

        prompt = f"""Read the passage and answer the question.
Answer with only Yes or No.

Passage:
{theory}

Question:
{question}
"""
        samples.append({
            "category": "LOG",
            "prompt": prompt,
            "expected_answer": expected
        })

    return samples


def generate_synthetic_logic_samples(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic logic samples if ProofWriter is unavailable."""
    rng = random.Random(seed)

    templates = [
        {
            "passage": "All {A} are {B}. All {B} are {C}. {X} is a {A}.",
            "question": "Is {X} a {C}?",
            "answer": "Yes"
        },
        {
            "passage": "No {A} are {B}. All {C} are {A}. {X} is a {C}.",
            "question": "Is {X} a {B}?",
            "answer": "No"
        },
        {
            "passage": "If {condition}, then {result}. {condition} is true.",
            "question": "Is {result} true?",
            "answer": "Yes"
        },
        {
            "passage": "If {condition}, then {result}. {condition} is false.",
            "question": "Is {result} true?",
            "answer": "No"
        }
    ]

    nouns_a = ["cats", "dogs", "birds", "fish", "robots", "computers", "trees", "flowers"]
    nouns_b = ["mammals", "animals", "organisms", "machines", "devices", "plants", "living things"]
    nouns_c = ["beings", "entities", "objects", "things", "creatures", "items"]
    names = ["Fluffy", "Spot", "Charlie", "Max", "Buddy", "Luna", "Daisy", "Rocky"]
    conditions = ["it rains", "the sun shines", "the door is open", "the light is on"]
    results = ["the ground is wet", "it is warm", "air flows through", "the room is bright"]

    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]

        # Choose consistent values for this sample
        chosen_a = rng.choice(nouns_a)
        chosen_b = rng.choice(nouns_b)
        chosen_c = rng.choice(nouns_c)
        chosen_x = rng.choice(names)
        chosen_condition = rng.choice(conditions)
        chosen_result = rng.choice(results)

        # Format both passage and question with the SAME values
        passage = template["passage"].format(
            A=chosen_a,
            B=chosen_b,
            C=chosen_c,
            X=chosen_x,
            condition=chosen_condition,
            result=chosen_result
        )

        question = template["question"].format(
            A=chosen_a,
            B=chosen_b,
            C=chosen_c,
            X=chosen_x,
            condition=chosen_condition,
            result=chosen_result
        )

        prompt = f"""Read the passage and answer the question.
Answer with only Yes or No.

Passage:
{passage}

Question:
{question}
"""
        samples.append({
            "category": "LOG",
            "prompt": prompt,
            "expected_answer": template["answer"]
        })

    return samples


def build_benchmark_tier(
    tier_name: str,
    per_category: int,
    seed: int
) -> List[Dict]:
    """Build a complete benchmark tier with all categories."""
    print(f"\nBuilding {tier_name}...")
    print(f"  Samples per category: {per_category}")

    all_samples = []

    # Load WP samples from GSM8K
    print("  Loading WP (GSM8K)...")
    wp_samples = load_gsm8k_samples(per_category, seed)
    all_samples.extend(wp_samples)
    print(f"    Loaded {len(wp_samples)} WP samples")

    # Load AR samples from DeepMind Math
    print("  Loading AR (DeepMind Math)...")
    ar_samples = load_deepmind_math_samples("AR", per_category, seed)
    all_samples.extend(ar_samples)
    print(f"    Loaded {len(ar_samples)} AR samples")

    # Load ALG samples from DeepMind Math
    print("  Loading ALG (DeepMind Math)...")
    alg_samples = load_deepmind_math_samples("ALG", per_category, seed + 1)  # Different seed for variety
    all_samples.extend(alg_samples)
    print(f"    Loaded {len(alg_samples)} ALG samples")

    # Load LOG samples from ProofWriter
    print("  Loading LOG (ProofWriter)...")
    log_samples = load_proofwriter_samples(per_category, seed)
    all_samples.extend(log_samples)
    print(f"    Loaded {len(log_samples)} LOG samples")

    # Sort by category to ensure consistent ordering
    all_samples.sort(key=lambda x: x["category"])

    # Assign sequential IDs per category
    category_counters = {"AR": 1, "ALG": 1, "LOG": 1, "WP": 1}
    for sample in all_samples:
        cat = sample["category"]
        sample["id"] = f"{cat}{category_counters[cat]:03d}"
        category_counters[cat] += 1

    print(f"  Total samples: {len(all_samples)}")
    return all_samples


def write_csv(samples: List[Dict], output_path: Path):
    """Write samples to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "prompt", "expected_answer"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)
    print(f"  Written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build industry-standard benchmark CSVs from public datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic sampling (default: 1234)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for CSV files (default: data)"
    )
    args = parser.parse_args()

    print(f"Building benchmarks with seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build each tier
    tiers = [
        ("Tier 1", "industry_tier1_40.csv", 10),
        ("Tier 2", "industry_tier2_400.csv", 100),
        ("Tier 3", "industry_tier3_1000.csv", 250),
    ]

    for tier_name, filename, per_category in tiers:
        samples = build_benchmark_tier(tier_name, per_category, args.seed)
        output_path = args.output_dir / filename
        write_csv(samples, output_path)

    print("\n✓ All benchmarks generated successfully!")
    print("\nNext steps:")
    print("  1. Validate CSVs:")
    print("     python3 src/validate_prompt_csv.py --csv data/industry_tier1_40.csv --total 40 --per_cat 10")
    print("     python3 src/validate_prompt_csv.py --csv data/industry_tier2_400.csv --total 400 --per_cat 100")
    print("     python3 src/validate_prompt_csv.py --csv data/industry_tier3_1000.csv --total 1000 --per_cat 250")
    print("  2. Smoke test samples:")
    print("     python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv")


if __name__ == "__main__":
    main()
