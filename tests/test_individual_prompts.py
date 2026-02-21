#!/usr/bin/env python3
"""Quick test script to verify individual prompts work correctly."""

import requests
import csv
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8080/completion"
CSV_PATH = "data/benchmarks/industry_tier1_40.csv"

# Load grammar
grammar_path = Path("grammars/grammar_phi2_answer_int_strict_final.gbnf")
if grammar_path.exists():
    grammar = grammar_path.read_text()
    print(f"✓ Loaded grammar from {grammar_path}")
    print(f"  Grammar preview: {grammar[:100]}...")
else:
    grammar = None
    print(f"✗ Grammar file not found: {grammar_path}")

# Load first 3 ALG prompts
with open(CSV_PATH, 'r', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    prompts = [row for row in reader if row['id'].startswith('ALG')][:3]

print(f"\n{'='*60}")
print("Testing first 3 ALG prompts")
print(f"{'='*60}\n")

for row in prompts:
    pid = row['id']
    category = row['category']
    expected = row['expected_answer']
    base_question = row['prompt'].strip()

    # Build prompt same way as runner
    has_instruction = "final number" in base_question.lower()
    if has_instruction:
        prompt = f"{base_question}\nAnswer: "
    else:
        instruction = "Answer with only the final number."
        prompt = f"{base_question}\n{instruction}\nAnswer: "

    print(f"{pid} (expected: {expected})")
    print(f"Prompt (last 150 chars): ...{prompt[-150:]}")
    print()

    # Call LLM with grammar
    payload = {
        "prompt": prompt,
        "n_predict": 7,
        "temperature": 0.0
    }
    if grammar:
        payload["grammar"] = grammar

    try:
        response = requests.post(SERVER_URL, json=payload, timeout=30)
        result = response.json()
        content = result.get("content", "")

        print(f"  Raw output: {repr(content)}")
        print(f"  Tokens predicted: {result.get('tokens_predicted', 'N/A')}")

        # Extract number
        import re
        nums = re.findall(r'[-+]?\d+', content.strip())
        extracted = nums[-1] if nums else ""

        print(f"  Extracted: {extracted}")
        print(f"  Match: {'✓' if extracted == expected else '✗'}")
        print()

    except Exception as e:
        print(f"  ERROR: {e}")
        print()

print(f"\n{'='*60}")
print("Test complete. Check if outputs look reasonable.")
print(f"{'='*60}")
