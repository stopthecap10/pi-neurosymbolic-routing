#!/usr/bin/env python3
"""Simple test - just send a basic math problem and see what happens."""

import requests

SERVER_URL = "http://edgeai.local:8080/completion"

# Simple test problem
prompt = """Solve the following problem.
Answer with only the final number.

What is 5 + 3?
Answer: """

print("Testing basic arithmetic: 5 + 3")
print(f"Server: {SERVER_URL}")
print(f"Prompt: {repr(prompt)}")
print()

# Test WITHOUT grammar
print("=" * 60)
print("TEST 1: Without grammar")
print("=" * 60)
try:
    response = requests.post(
        SERVER_URL,
        json={"prompt": prompt, "n_predict": 7, "temperature": 0.0},
        timeout=10
    )
    result = response.json()
    print(f"Raw output: {repr(result.get('content', ''))}")
    print(f"Tokens: {result.get('tokens_predicted', 'N/A')}")
except Exception as e:
    print(f"ERROR: {e}")

print()

# Test WITH grammar
print("=" * 60)
print("TEST 2: With grammar")
print("=" * 60)

try:
    from pathlib import Path
    grammar_file = Path("grammars/grammar_phi2_answer_int_strict_final.gbnf")

    if grammar_file.exists():
        grammar = grammar_file.read_text()
        print(f"Grammar loaded: {len(grammar)} chars")
        print(f"Grammar preview: {grammar[:80]}...")
        print()

        response = requests.post(
            SERVER_URL,
            json={"prompt": prompt, "n_predict": 7, "temperature": 0.0, "grammar": grammar},
            timeout=10
        )
        result = response.json()
        print(f"Raw output: {repr(result.get('content', ''))}")
        print(f"Tokens: {result.get('tokens_predicted', 'N/A')}")
    else:
        print(f"Grammar file not found: {grammar_file}")

except Exception as e:
    print(f"ERROR: {e}")

print()
print("=" * 60)
print("If both tests show '8' or similar correct answer, setup is working.")
print("If you see weird outputs, there's a problem with the server/grammar.")
print("=" * 60)
