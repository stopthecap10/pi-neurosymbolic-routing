#!/usr/bin/env python3
"""Test model WITHOUT grammar - just see raw output."""

import requests

SERVER_URL = "http://edgeai.local:8080/completion"

# Test problems
tests = [
    ("Simple: 5+3", "What is 5 + 3?", "8"),
    ("ALG001", "The radius of a wheel is 22.4 cm. What is the distance covered by the wheel in making 500 resolutions?", "704"),
    ("ALG002", "A and B complete a job in 6 days. A alone can do the job in 24 days. If B works alone, how many days will it take to complete the job?", "8"),
]

for name, question, expected in tests:
    prompt = f"""Solve the following problem.
Answer with only the final number.

{question}
Answer: """

    print("=" * 70)
    print(f"{name} (expected: {expected})")
    print("=" * 70)
    print(f"Question: {question}")
    print()

    try:
        response = requests.post(
            SERVER_URL,
            json={"prompt": prompt, "n_predict": 20, "temperature": 0.0},
            timeout=15
        )
        result = response.json()
        content = result.get('content', '')

        print(f"Raw output: {repr(content[:200])}")
        print(f"Tokens: {result.get('tokens_predicted', 'N/A')}")

        # Try to extract number
        import re
        nums = re.findall(r'[-+]?\d+', content.strip().replace(',', ''))
        extracted = nums[-1] if nums else "NONE"

        print(f"Extracted: {extracted}")
        print(f"Match: {'✓ CORRECT' if extracted == expected else '✗ WRONG'}")

    except Exception as e:
        print(f"ERROR: {e}")

    print()
