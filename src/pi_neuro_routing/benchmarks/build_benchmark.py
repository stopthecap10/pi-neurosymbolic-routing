#!/usr/bin/env python3
"""
Build industry-standard benchmark CSVs from public datasets.

This script downloads real public datasets and generates deterministic
benchmark CSVs for evaluating mathematical and logical reasoning.

Datasets used:
- WP: GSM8K (grade-school math word problems)
- AR/ALG: DeepMind Mathematics Dataset
- LOG: RuleTaker (logical entailment)
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
import sys


def parse_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K answer field.

    GSM8K answers typically end with #### followed by the final answer.
    Returns None if no valid integer can be extracted.
    """
    if "####" in answer_text:
        # Extract everything after ####
        final_part = answer_text.split("####")[-1].strip()
        # Remove any commas and extract just the number
        final_part = final_part.replace(",", "")
        # Extract the first integer found (with optional negative sign)
        match = re.search(r'-?\d+', final_part)
        if match:
            return match.group(0)

    # Fallback: try to find any integer in the answer
    match = re.search(r'-?\d+', answer_text.replace(",", ""))
    if match:
        return match.group(0)

    return None


def load_gsm8k_samples(num_samples: int, seed: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Load GSM8K dataset samples for WP category.

    Args:
        num_samples: Number of samples to extract
        seed: Random seed for deterministic sampling
        cache_dir: Optional cache directory for datasets

    Returns:
        List of sample dicts with id, category, prompt, expected_answer
    """
    print(f"  Loading WP from GSM8K...")
    try:
        from datasets import load_dataset

        # Use test split for final evaluation; train split for development
        # We use test split to avoid train/test contamination
        dataset = load_dataset(
            "gsm8k",
            "main",
            split="test",
            cache_dir=str(cache_dir) if cache_dir else None
        )

        print(f"    GSM8K test split has {len(dataset)} samples")

    except Exception as e:
        print(f"    ERROR loading GSM8K: {e}")
        print(f"    Make sure you have installed: pip3 install datasets")
        sys.exit(1)

    # Convert to list for deterministic sampling
    all_samples = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    samples = []
    for item in all_samples:
        if len(samples) >= num_samples:
            break

        question = item["question"].strip()
        answer = parse_gsm8k_answer(item["answer"])

        if answer is None:
            # Skip samples without valid integer answers
            continue

        prompt = f"""Solve the following word problem.
Answer with only the final number.

{question}
"""

        samples.append({
            "category": "WP",
            "prompt": prompt,
            "expected_answer": answer
        })

    if len(samples) < num_samples:
        print(f"    WARNING: Only found {len(samples)} valid WP samples (needed {num_samples})")

    print(f"    Loaded {len(samples)} WP samples")
    return samples[:num_samples]


def extract_math_answer(solution: str) -> Optional[str]:
    """Extract integer answer from DeepMind Math solution.

    Returns None if no valid integer found.
    """
    # Remove LaTeX formatting and whitespace
    solution = solution.replace("\\", "").replace(",", "").strip()

    # Try to find an integer (with optional negative sign)
    match = re.search(r'-?\d+', solution)
    if match:
        return match.group(0)

    return None


def generate_synthetic_arithmetic(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic arithmetic problems as fallback.

    These are simple but verifiable arithmetic expressions.
    """
    rng = random.Random(seed)
    samples = []

    operations = [
        ('+', lambda a, b: a + b, "add"),
        ('-', lambda a, b: a - b, "subtract"),
        ('*', lambda a, b: a * b, "multiply"),
    ]

    for i in range(num_samples):
        op_symbol, op_func, op_name = rng.choice(operations)

        # Vary difficulty
        if i < num_samples // 3:  # Easy
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
        elif i < 2 * num_samples // 3:  # Medium
            a = rng.randint(10, 200)
            b = rng.randint(10, 200)
        else:  # Hard
            a = rng.randint(50, 500)
            b = rng.randint(50, 500)

        # Ensure positive results for subtraction
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


def generate_synthetic_algebra(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic algebra problems as fallback.

    These are simple linear equations with guaranteed integer solutions.
    """
    rng = random.Random(seed)
    samples = []

    for i in range(num_samples):
        # Generate different types of equations
        if i % 3 == 0:
            # Type: x = a op b
            operations = [
                ('+', lambda a, b: a + b),
                ('-', lambda a, b: a - b),
                ('*', lambda a, b: a * b),
            ]
            op_symbol, op_func = rng.choice(operations)

            if i < num_samples // 3:
                a = rng.randint(1, 30)
                b = rng.randint(1, 30)
            elif i < 2 * num_samples // 3:
                a = rng.randint(10, 100)
                b = rng.randint(10, 100)
            else:
                a = rng.randint(50, 200)
                b = rng.randint(50, 200)

            if op_symbol == '-' and a < b:
                a, b = b, a

            result = op_func(a, b)
            equation = f"x = {a} {op_symbol} {b}"

        elif i % 3 == 1:
            # Type: x + a = b
            a = rng.randint(1, 100)
            b = rng.randint(a + 1, a + 200)
            result = b - a
            equation = f"x + {a} = {b}"

        else:
            # Type: x - a = b
            a = rng.randint(1, 100)
            b = rng.randint(1, 200)
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


def load_svamp_for_arithmetic(num_samples: int, seed: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Load SVAMP dataset for AR category (arithmetic word problems).

    SVAMP contains simple arithmetic word problems with integer answers.
    """
    print(f"    Trying SVAMP dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "ChilleD/SVAMP",
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None
        )

        rng = random.Random(seed)
        all_samples = list(dataset)
        rng.shuffle(all_samples)

        samples = []
        for item in all_samples:
            if len(samples) >= num_samples:
                break

            # SVAMP has: Body, Question, Equation, Answer
            body = item.get("Body", "").strip()
            question = item.get("Question", "").strip()
            answer_raw = str(item.get("Answer", "")).strip()

            # Try to extract integer
            answer = extract_math_answer(answer_raw)
            if answer is None:
                continue

            # Combine body and question
            full_question = f"{body} {question}".strip()

            prompt = f"""Solve the following word problem.
Answer with only the final number.

{full_question}
"""

            samples.append({
                "category": "AR",
                "prompt": prompt,
                "expected_answer": answer
            })

        return samples[:num_samples]

    except Exception as e:
        print(f"      SVAMP failed: {e}")
        return []


def load_aqua_for_algebra(num_samples: int, seed: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Load AQuA-RAT dataset for ALG category (algebra reasoning).

    AQuA contains algebraic reasoning problems with multiple choice answers.
    """
    print(f"    Trying AQuA-RAT dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "aqua_rat",
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None
        )

        rng = random.Random(seed)
        all_samples = list(dataset)
        rng.shuffle(all_samples)

        samples = []
        for item in all_samples:
            if len(samples) >= num_samples:
                break

            question = item.get("question", "").strip()
            correct = item.get("correct", "").strip()
            options = item.get("options", [])

            # Find the answer value from options
            answer_text = None
            for opt in options:
                if opt.startswith(correct + ")"):
                    answer_text = opt[2:].strip()  # Remove "A)" prefix
                    break

            if not answer_text:
                continue

            # Extract integer
            answer = extract_math_answer(answer_text)
            if answer is None:
                continue

            prompt = f"""Solve the following problem.
Answer with only the final number.

{question}
"""

            samples.append({
                "category": "ALG",
                "prompt": prompt,
                "expected_answer": answer
            })

        return samples[:num_samples]

    except Exception as e:
        print(f"      AQuA-RAT failed: {e}")
        return []


def load_math_samples(
    category: str,
    num_samples: int,
    seed: int,
    cache_dir: Optional[Path] = None
) -> List[Dict]:
    """Load math dataset samples for AR or ALG category.

    Uses real public datasets with fallback to synthetic.

    Args:
        category: Either "AR" (arithmetic) or "ALG" (algebra)
        num_samples: Number of samples to extract
        seed: Random seed for deterministic sampling
        cache_dir: Optional cache directory for datasets

    Returns:
        List of sample dicts with id, category, prompt, expected_answer
    """
    print(f"  Loading {category} from public datasets...")

    samples = []

    if category == "AR":
        # Try SVAMP first (arithmetic word problems)
        samples = load_svamp_for_arithmetic(num_samples, seed, cache_dir)

        if len(samples) >= num_samples:
            print(f"    Loaded {len(samples)} AR samples from SVAMP")
            return samples[:num_samples]
        else:
            print(f"    SVAMP provided {len(samples)} samples, need {num_samples}")

    else:  # ALG
        # Try AQuA-RAT (algebra reasoning)
        samples = load_aqua_for_algebra(num_samples, seed, cache_dir)

        if len(samples) >= num_samples:
            print(f"    Loaded {len(samples)} ALG samples from AQuA-RAT")
            return samples[:num_samples]
        else:
            print(f"    AQuA-RAT provided {len(samples)} samples, need {num_samples}")

    # If we don't have enough samples, fill with synthetic
    if len(samples) < num_samples:
        print(f"    Filling remaining with synthetic {category} generation...")
        remaining = num_samples - len(samples)

        if category == "AR":
            synthetic = generate_synthetic_arithmetic(remaining, seed + 1000)
        else:
            synthetic = generate_synthetic_algebra(remaining, seed + 1000)

        samples.extend(synthetic)
        print(f"    Total: {len(samples[:num_samples])} samples ({len(samples)} real, {len(synthetic)} synthetic)")

    return samples[:num_samples]


def generate_synthetic_logic(num_samples: int, seed: int) -> List[Dict]:
    """Generate synthetic logic problems as fallback.

    These are simple deductive reasoning problems with guaranteed correct answers.
    """
    rng = random.Random(seed)
    samples = []

    # Templates for logic problems
    templates = [
        {
            "passage": "All {A} are {B}. All {B} are {C}. {X} is a {A}.",
            "question": "Is {X} a {C}?",
            "answer": "Yes"
        },
        {
            "passage": "No {A} are {B}. {X} is a {A}.",
            "question": "Is {X} a {B}?",
            "answer": "No"
        },
        {
            "passage": "If {cond}, then {result}. {cond}.",
            "question": "Is {result} true?",
            "answer": "Yes"
        },
        {
            "passage": "If {cond}, then {result}. Not {cond}.",
            "question": "Is {result} true?",
            "answer": "No"
        },
    ]

    # Vocabulary for generating problems
    nouns_a = ["cats", "dogs", "birds", "robots", "computers", "phones", "cars", "bikes"]
    nouns_b = ["mammals", "animals", "machines", "devices", "vehicles", "tools", "objects"]
    nouns_c = ["living things", "entities", "items", "beings", "things"]
    names = ["Fluffy", "Spot", "Charlie", "Max", "Buddy", "Luna", "Daisy", "Rocky"]
    conditions = ["it rains", "the sun shines", "the door is open", "the light is on", "it is warm", "it is cold"]
    results = ["the ground is wet", "it is bright", "air flows", "the room is lit", "ice melts", "water freezes"]

    for i in range(num_samples):
        template = templates[i % len(templates)]

        # Select vocabulary
        a = rng.choice(nouns_a)
        b = rng.choice(nouns_b)
        c = rng.choice(nouns_c)
        x = rng.choice(names)
        cond = rng.choice(conditions)
        result = rng.choice(results)

        # Format passage and question
        passage = template["passage"].format(A=a, B=b, C=c, X=x, cond=cond, result=result)
        question = template["question"].format(A=a, B=b, C=c, X=x, cond=cond, result=result)

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


def load_boolq_for_logic(num_samples: int, seed: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Load BoolQ dataset for LOG category (Yes/No questions).

    BoolQ contains reading comprehension Yes/No questions.
    """
    print(f"    Trying BoolQ dataset...")
    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "google/boolq",
            split="train",
            cache_dir=str(cache_dir) if cache_dir else None
        )

        rng = random.Random(seed)
        all_samples = list(dataset)
        rng.shuffle(all_samples)

        samples = []
        for item in all_samples:
            if len(samples) >= num_samples:
                break

            # BoolQ has: question, passage, answer (bool)
            passage = item.get("passage", "").strip()
            question = item.get("question", "").strip()
            answer_bool = item.get("answer", None)

            if not passage or not question or answer_bool is None:
                continue

            # Truncate very long passages
            if len(passage) > 900:
                passage = passage[:900] + "..."

            # Convert boolean to Yes/No
            expected = "Yes" if answer_bool else "No"

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
                "expected_answer": expected
            })

        return samples[:num_samples]

    except Exception as e:
        print(f"      BoolQ failed: {e}")
        return []


def load_logic_samples(num_samples: int, seed: int, cache_dir: Optional[Path] = None) -> List[Dict]:
    """Load logic/reasoning dataset samples for LOG category.

    Uses BoolQ with fallback to synthetic generation.

    Args:
        num_samples: Number of samples to extract
        seed: Random seed for deterministic sampling
        cache_dir: Optional cache directory for datasets

    Returns:
        List of sample dicts with id, category, prompt, expected_answer
    """
    print(f"  Loading LOG from public datasets...")

    # Try BoolQ (Yes/No reading comprehension)
    samples = load_boolq_for_logic(num_samples, seed, cache_dir)

    if len(samples) >= num_samples:
        print(f"    Loaded {len(samples)} LOG samples from BoolQ")
        return samples[:num_samples]
    else:
        print(f"    BoolQ provided {len(samples)} samples, need {num_samples}")

    # If we don't have enough, fill with synthetic
    if len(samples) < num_samples:
        print(f"    Filling remaining with synthetic LOG generation...")
        remaining = num_samples - len(samples)
        synthetic = generate_synthetic_logic(remaining, seed + 1000)
        samples.extend(synthetic)
        print(f"    Total: {len(samples[:num_samples])} samples ({len(samples)} real, {len(synthetic)} synthetic)")

    return samples[:num_samples]


def build_benchmark_tier(
    tier_name: str,
    per_category: int,
    seed: int,
    cache_dir: Optional[Path] = None
) -> List[Dict]:
    """Build a complete benchmark tier with all categories.

    Args:
        tier_name: Name of tier (for display)
        per_category: Number of samples per category
        seed: Random seed for deterministic sampling
        cache_dir: Optional cache directory for datasets

    Returns:
        List of all samples across all categories
    """
    print(f"\nBuilding {tier_name}...")
    print(f"  Samples per category: {per_category}")
    print(f"  Random seed: {seed}")

    all_samples = []

    # Load samples from each category
    wp_samples = load_gsm8k_samples(per_category, seed, cache_dir)
    ar_samples = load_math_samples("AR", per_category, seed, cache_dir)
    alg_samples = load_math_samples("ALG", per_category, seed + 1, cache_dir)  # Different seed
    log_samples = load_logic_samples(per_category, seed, cache_dir)

    all_samples.extend(ar_samples)
    all_samples.extend(alg_samples)
    all_samples.extend(log_samples)
    all_samples.extend(wp_samples)

    # Sort by category to ensure consistent ordering
    all_samples.sort(key=lambda x: x["category"])

    # Assign sequential IDs per category
    category_counters = {"AR": 1, "ALG": 1, "LOG": 1, "WP": 1}
    for sample in all_samples:
        cat = sample["category"]
        sample["id"] = f"{cat}{category_counters[cat]:03d}"
        category_counters[cat] += 1

    print(f"  Total samples: {len(all_samples)}")

    # Verify we have the right distribution
    from collections import Counter
    dist = Counter(s["category"] for s in all_samples)
    print(f"  Distribution: {dict(dist)}")

    return all_samples


def write_csv(samples: List[Dict], output_path: Path):
    """Write samples to CSV file.

    Args:
        samples: List of sample dicts
        output_path: Path to output CSV file
    """
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
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory for downloaded datasets (default: data/cache)"
    )
    parser.add_argument(
        "--tier",
        choices=["T1", "T2", "T3", "all"],
        default="all",
        help="Which tier to generate (default: all)"
    )
    args = parser.parse_args()

    print(f"Building benchmarks with seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {args.cache_dir}")

    # Create directories if they don't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Define tier configurations
    tiers = {
        "T1": ("Tier 1", "industry_tier1_40.csv", 10),
        "T2": ("Tier 2", "industry_tier2_400.csv", 100),
        "T3": ("Tier 3", "industry_tier3_1000.csv", 250),
    }

    # Determine which tiers to build
    if args.tier == "all":
        tiers_to_build = ["T1", "T2", "T3"]
    else:
        tiers_to_build = [args.tier]

    # Build each tier
    for tier_key in tiers_to_build:
        tier_name, filename, per_category = tiers[tier_key]
        samples = build_benchmark_tier(tier_name, per_category, args.seed, args.cache_dir)
        output_path = args.output_dir / filename
        write_csv(samples, output_path)

    print("\n✓ All benchmarks generated successfully!")
    print("\nNext steps:")
    print("  1. Validate CSVs:")
    if "T1" in tiers_to_build:
        print("     python3 src/validate_prompt_csv.py --csv data/benchmarks/industry_tier1_40.csv --total 40 --per_cat 10")
    if "T2" in tiers_to_build:
        print("     python3 src/validate_prompt_csv.py --csv data/benchmarks/industry_tier2_400.csv --total 400 --per_cat 100")
    if "T3" in tiers_to_build:
        print("     python3 src/validate_prompt_csv.py --csv data/benchmarks/industry_tier3_1000.csv --total 1000 --per_cat 250")
    print("  2. Smoke test samples:")
    print("     python3 src/smoke_test_samples.py --csv data/benchmarks/industry_tier2_400.csv")


if __name__ == "__main__":
    main()
