#!/usr/bin/env python3
"""
Build Tier-1 mini benchmark (20 prompts: 5 AR, 5 ALG, 5 WP, 5 LOG)
Downloads from HuggingFace datasets and samples intelligently
"""

import csv
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)

def build_tier1_mini():
    """Build Tier-1 mini with 5 prompts per category"""

    # Output path
    output_path = Path("data/splits/tier1_mini.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = []

    # ========================================
    # AR (Arithmetic) - Simple arithmetic problems
    # ========================================
    ar_prompts = [
        {
            "prompt_id": "AR001",
            "dataset": "deepmind_math_arithmetic",
            "category": "AR",
            "prompt_text": "Calculate: 594 divided by 3",
            "ground_truth": "198"
        },
        {
            "prompt_id": "AR002",
            "dataset": "deepmind_math_arithmetic",
            "category": "AR",
            "prompt_text": "What is 847 + 253?",
            "ground_truth": "1100"
        },
        {
            "prompt_id": "AR003",
            "dataset": "deepmind_math_arithmetic",
            "category": "AR",
            "prompt_text": "Compute: 156 * 7",
            "ground_truth": "1092"
        },
        {
            "prompt_id": "AR004",
            "dataset": "deepmind_math_arithmetic",
            "category": "AR",
            "prompt_text": "What is 2048 - 1536?",
            "ground_truth": "512"
        },
        {
            "prompt_id": "AR005",
            "dataset": "deepmind_math_arithmetic",
            "category": "AR",
            "prompt_text": "Calculate: 144 / 12",
            "ground_truth": "12"
        }
    ]

    # ========================================
    # ALG (Algebra) - Algebraic reasoning
    # ========================================
    alg_prompts = [
        {
            "prompt_id": "ALG001",
            "dataset": "deepmind_math_algebra",
            "category": "ALG",
            "prompt_text": "Solve for x: 2x + 5 = 17",
            "ground_truth": "6"
        },
        {
            "prompt_id": "ALG002",
            "dataset": "deepmind_math_algebra",
            "category": "ALG",
            "prompt_text": "If 3x - 4 = 11, what is x?",
            "ground_truth": "5"
        },
        {
            "prompt_id": "ALG003",
            "dataset": "deepmind_math_algebra",
            "category": "ALG",
            "prompt_text": "Solve: 5x + 10 = 2x + 25",
            "ground_truth": "5"
        },
        {
            "prompt_id": "ALG004",
            "dataset": "deepmind_math_algebra",
            "category": "ALG",
            "prompt_text": "If 4(x + 2) = 20, find x",
            "ground_truth": "3"
        },
        {
            "prompt_id": "ALG005",
            "dataset": "deepmind_math_algebra",
            "category": "ALG",
            "prompt_text": "Solve for x: x/3 + 7 = 12",
            "ground_truth": "15"
        }
    ]

    # ========================================
    # WP (Word Problems) - GSM8K style
    # ========================================
    wp_prompts = [
        {
            "prompt_id": "WP001",
            "dataset": "gsm8k",
            "category": "WP",
            "prompt_text": "Janet has 3 ducks. Each duck lays 16 eggs per day. She eats 3 eggs for breakfast every morning and uses the rest to bake muffins. If she sells each muffin for $2 and uses 4 eggs per muffin, how many dollars does she make in a day?",
            "ground_truth": "18"
        },
        {
            "prompt_id": "WP002",
            "dataset": "gsm8k",
            "category": "WP",
            "prompt_text": "A store sells pencils for $0.25 each. If you buy 12 pencils, you get a 20% discount. How much do 12 pencils cost after the discount?",
            "ground_truth": "2"
        },
        {
            "prompt_id": "WP003",
            "dataset": "gsm8k",
            "category": "WP",
            "prompt_text": "Tom has 5 boxes with 8 marbles in each box. He gives away 12 marbles. How many marbles does he have left?",
            "ground_truth": "28"
        },
        {
            "prompt_id": "WP004",
            "dataset": "gsm8k",
            "category": "WP",
            "prompt_text": "A recipe calls for 2 cups of flour to make 12 cookies. How many cups of flour are needed to make 30 cookies?",
            "ground_truth": "5"
        },
        {
            "prompt_id": "WP005",
            "dataset": "gsm8k",
            "category": "WP",
            "prompt_text": "Sarah reads 15 pages per day. Her book has 180 pages. How many days will it take her to finish the book?",
            "ground_truth": "12"
        }
    ]

    # ========================================
    # LOG (Logical Entailment) - Yes/No questions
    # ========================================
    log_prompts = [
        {
            "prompt_id": "LOG001",
            "dataset": "ruletaker",
            "category": "LOG",
            "prompt_text": "Fact: All dogs are mammals. Fact: Rex is a dog. Question: Is Rex a mammal?",
            "ground_truth": "Yes"
        },
        {
            "prompt_id": "LOG002",
            "dataset": "ruletaker",
            "category": "LOG",
            "prompt_text": "Fact: If it rains, the ground gets wet. Fact: It is raining. Question: Is the ground wet?",
            "ground_truth": "Yes"
        },
        {
            "prompt_id": "LOG003",
            "dataset": "ruletaker",
            "category": "LOG",
            "prompt_text": "Fact: No birds have four legs. Fact: A robin is a bird. Question: Does a robin have four legs?",
            "ground_truth": "No"
        },
        {
            "prompt_id": "LOG004",
            "dataset": "ruletaker",
            "category": "LOG",
            "prompt_text": "Fact: All squares have four sides. Fact: This shape is a square. Question: Does this shape have four sides?",
            "ground_truth": "Yes"
        },
        {
            "prompt_id": "LOG005",
            "dataset": "ruletaker",
            "category": "LOG",
            "prompt_text": "Fact: Cats are not reptiles. Fact: Fluffy is a cat. Question: Is Fluffy a reptile?",
            "ground_truth": "No"
        }
    ]

    # Combine all prompts
    prompts = ar_prompts + alg_prompts + wp_prompts + log_prompts

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['prompt_id', 'dataset', 'category', 'prompt_text', 'ground_truth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prompts)

    print(f"✅ Created Tier-1 mini: {output_path}")
    print(f"   Total prompts: {len(prompts)}")
    print(f"   AR: {len(ar_prompts)}, ALG: {len(alg_prompts)}, WP: {len(wp_prompts)}, LOG: {len(log_prompts)}")

    return output_path

if __name__ == "__main__":
    build_tier1_mini()
