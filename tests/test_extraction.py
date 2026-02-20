#!/usr/bin/env python3
"""Test extraction logic improvements"""

import re

INT_RE = re.compile(r"[-+]?\d+")

def extract_last_int(text: str) -> str:
    """Extract last integer from text using regex [-+]?\d+, ignoring continuation markers"""
    if not text:
        return ""

    # Split at common continuation markers to avoid extracting from generated exercises
    for marker in ["\n\nExercise", "\n\n\nExercise", "\nExercise", "\n\n#", "\n\nProblem", "\n\nQuestion"]:
        if marker in text:
            text = text.split(marker)[0]
            break

    # Strip whitespace and remove commas
    cleaned = text.strip().replace(",", "")

    # Remove common answer prefixes that might confuse extraction
    cleaned = cleaned.replace("x = ", "").replace("x=", "")
    cleaned = cleaned.replace("Answer: ", "").replace("Answer:", "")
    cleaned = cleaned.strip()

    # Use INT_RE to find all integers in the text
    nums = INT_RE.findall(cleaned)
    if not nums:
        return ""
    # Return the LAST integer found (before continuation markers)
    return nums[-1]

# Test cases from actual failures
test_cases = [
    ("\n980\n\nExercise 2:", "980", "Should extract before continuation"),
    ("\nx = 980\n", "980", "Should extract after 'x = '"),
    ("\nx = \n", "", "Empty answer - should fail gracefully"),
    ("\nx =  \n\nExercise 3:", "", "Empty with continuation - should fail gracefully"),
    ("\n\nExercise 2:\nSolve the following equation", "", "No answer - should return empty"),
    (" 980", "980", "Simple number with space"),
    ("The answer is 123", "123", "Number in sentence"),
    ("-123", "-123", "Negative number"),
    ("1,234,567", "1234567", "Number with commas"),
]

print("Testing extraction logic:")
print("-" * 80)
for text, expected, description in test_cases:
    result = extract_last_int(text)
    status = "✓" if result == expected else "✗"
    print(f"{status} {description}")
    print(f"  Input:    {repr(text)}")
    print(f"  Expected: {repr(expected)}")
    print(f"  Got:      {repr(result)}")
    if result != expected:
        print(f"  MISMATCH!")
    print()
