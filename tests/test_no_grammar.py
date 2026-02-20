#!/usr/bin/env python3
"""Test AR prompts WITHOUT grammar to see if model echoes prompt."""
import requests
import csv

# Load first 2 AR prompts
csv_file = "data/industry_tier2_400.csv"
ar_prompts = []

with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['category'] == 'AR':
            ar_prompts.append(row)
            if len(ar_prompts) >= 2:
                break

# Build prompts same way as safe runner
for row in ar_prompts:
    pid = row['id']
    cat = row['category']
    expected = row['expected_answer']

    base_question = row['prompt'].strip()

    # Remove trailing patterns
    for pattern in ["\nAnswer:", "Answer:", "\nExercise:", "Exercise:"]:
        if base_question.endswith(pattern):
            base_question = base_question[:-len(pattern)].rstrip()

    # Check if instruction already present
    has_instruction = "final number" in base_question.lower()

    if has_instruction:
        prompt = f"{base_question}\nAnswer: "
    else:
        prompt = f"{base_question}\nAnswer with only the final number.\nAnswer: "

    print(f"\n{'='*60}")
    print(f"Testing {pid} (expected: {expected})")
    print(f"{'='*60}")
    print(f"Prompt (last 200 chars):\n{repr(prompt[-200:])}\n")

    # Test WITHOUT grammar
    print("WITHOUT GRAMMAR:")
    try:
        r = requests.post(
            "http://127.0.0.1:8080/completion",
            json={"prompt": prompt, "n_predict": 12, "temperature": 0.0, "grammar": ""},
            timeout=30.0
        )
        j = r.json()
        content = j.get("content", "")
        print(f"  Raw content: {repr(content)}")
        print(f"  Expected: {expected}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

print(f"\n{'='*60}")
print("Test complete - compare with grammar vs without grammar behavior")
print(f"{'='*60}")
