"""
Answer extraction logic for different problem categories and modes.

Handles extraction of answers from LLM outputs for:
- Numeric answers (AR, ALG, WP categories)
- Yes/No answers (LOG category)
- Grammar-constrained vs no-grammar outputs
- Prompt echo and continuation marker handling
"""

import re
from typing import Optional

# Regex patterns for extraction
INT_RE = re.compile(r"[-+]?\d+")
NUM_RE = re.compile(r"[-+]?\d+\.?\d*")
YESNO_RE = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)


def is_degenerate_pattern(s: str) -> bool:
    """
    Detect if string has repetitive patterns indicating model runaway.

    Examples of degenerate patterns:
    - 050505 (2-char pattern repeated)
    - 000000 (single char repeated)
    - 170005000 (multi-char patterns)

    Args:
        s: String to check for patterns

    Returns:
        True if degenerate pattern detected
    """
    if len(s) < 6:
        return False

    # Check for repeating single char (000000)
    for i in range(len(s) - 5):
        if len(set(s[i:i+6])) == 1:
            return True

    # Check for repeating 2-char patterns (050505)
    if len(s) >= 6:
        for i in range(len(s) - 5):
            pattern = s[i:i+2]
            if s[i:i+6] == pattern * 3:
                return True

    # Check for repeating 3-char patterns (000500050005)
    if len(s) >= 9:
        for i in range(len(s) - 8):
            pattern = s[i:i+3]
            if s[i:i+9] == pattern * 3:
                return True

    # Check for repeating 4-char patterns (00050005)
    if len(s) >= 8:
        for i in range(len(s) - 7):
            pattern = s[i:i+4]
            if s[i:i+8] == pattern * 2:
                return True

    return False


def strip_continuation_markers(text: str) -> str:
    """
    Remove continuation markers that indicate model generated extra exercises.

    Args:
        text: Raw model output

    Returns:
        Text before first continuation marker
    """
    markers = [
        "\n\nExercise",
        "\n\n\nExercise",
        "\nExercise",
        "\n\n#",
        "\nProblem",
        "\n\nProblem",
        "\nQuestion",
        "\n\nQuestion"
    ]

    for marker in markers:
        if marker in text:
            return text.split(marker)[0]

    return text


def extract_last_int(text: str, strip_continuations: bool = True) -> str:
    """
    Extract last integer from text using regex [-+]?\d+.

    Args:
        text: Raw text to extract from
        strip_continuations: If True, ignore text after continuation markers

    Returns:
        Last integer found as string, or empty string if none found
    """
    if not text:
        return ""

    # Optionally strip continuation markers
    if strip_continuations:
        text = strip_continuation_markers(text)

    # Clean text: strip whitespace, remove commas
    cleaned = text.strip().replace(",", "")

    # Remove common answer prefixes that might confuse extraction
    cleaned = cleaned.replace("x = ", "").replace("x=", "")
    cleaned = cleaned.replace("Answer: ", "").replace("Answer:", "")
    cleaned = cleaned.strip()

    # Find all integers
    nums = INT_RE.findall(cleaned)
    if not nums:
        return ""

    return nums[-1]


def extract_int_safe(text: str) -> str:
    """
    Extract integer with fallback for prompt-echo scenarios.

    This handles cases where the model echoes the prompt instead of answering.
    It tries to extract from after "Answer:" first, then falls back to first
    integer for short outputs.

    Args:
        text: Raw model output

    Returns:
        Extracted integer or empty string
    """
    if not text:
        return ""

    # Primary: extract last integer after "Answer:"
    if "Answer:" in text:
        after_answer = text.split("Answer:", 1)[-1]
        cleaned = after_answer.strip().replace(",", "")
        nums = INT_RE.findall(cleaned)
        if nums:
            return nums[-1]

    # Fallback: if content is short (<80 chars) and contains digits,
    # try first integer. This handles cases where model outputs just
    # a number without "Answer:" prefix.
    if len(text) < 80:
        cleaned = text.strip().replace(",", "")
        nums = INT_RE.findall(cleaned)
        if nums:
            return nums[0]

    return ""


def extract_yesno(text: str, strip_continuations: bool = True) -> str:
    """
    Find last occurrence of Yes or No (case-insensitive).

    Args:
        text: Raw text to extract from
        strip_continuations: If True, ignore text after continuation markers

    Returns:
        "Yes" or "No" (canonical form) or empty string
    """
    if not text:
        return ""

    # Optionally strip continuation markers
    if strip_continuations:
        text = strip_continuation_markers(text)

    # Strip problematic tokens that model might generate
    cleaned = (text
               .replace("<|question_end|>", "")
               .replace("<|question_end|", "")
               .replace("<|endoftext|>", "")
               .replace("<|", "")
               .strip())

    if not cleaned:
        return ""

    # Find last occurrence of yes/no
    t = cleaned.lower()
    yes_pos = t.rfind("yes")
    no_pos = t.rfind("no")

    if yes_pos == -1 and no_pos == -1:
        return ""

    # Return the one that appears last
    return "Yes" if yes_pos > no_pos else "No"


def extract_final_answer(category: str, raw_text: str, mode: str = "grammar") -> str:
    """
    Extract final answer based on category and mode.

    Args:
        category: Problem category (AR, ALG, LOG, WP)
        raw_text: Raw model output
        mode: "grammar" or "nogrammar"

    Returns:
        Extracted answer string
    """
    cat = (category or "").strip().upper()
    t = (raw_text or "").strip()

    if cat == "LOG":
        return extract_yesno(t, strip_continuations=(mode == "nogrammar"))

    # Numeric categories: choose extraction method based on mode
    if mode == "grammar":
        # Grammar mode: simpler extraction, model should follow format
        return extract_last_int(t, strip_continuations=False)
    else:
        # No-grammar mode: need robust extraction with continuation handling
        return extract_last_int(t, strip_continuations=True)


def validate_format(category: str, answer: str) -> bool:
    """
    Validate that answer matches expected format for category.

    Args:
        category: Problem category (AR, ALG, LOG, WP)
        answer: Extracted answer string

    Returns:
        True if format is valid
    """
    a = (answer or "").strip()

    if category.upper() == "LOG":
        return a in ("Yes", "No")

    if not a:
        return False

    # Check if it's a valid integer (possibly negative)
    if a[0] == "-":
        a = a[1:]

    return a.isdigit()


def check_degeneracy(category: str, answer: str) -> bool:
    """
    Check if answer shows signs of model degeneracy/runaway.

    Args:
        category: Problem category
        answer: Extracted answer

    Returns:
        True if answer appears degenerate
    """
    if category.upper() == "LOG":
        return False

    # Check for excessive length
    if len(answer) > 18:
        return True

    # Check for suspicious length (>10 digits)
    if len(answer) > 10:
        return True

    # Check for repetitive patterns
    if is_degenerate_pattern(answer):
        return True

    return False
