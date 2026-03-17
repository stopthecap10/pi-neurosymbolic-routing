#!/usr/bin/env python3
"""
Token Classifier for Grammatical Inference

Maps raw query text to a sequence of abstract token classes
for use with L* and RPNI algorithms.

Alphabet: CMD, NUM, VAR, OP, PAREN, ENT, PROP, RULE, QMARK, NARR, UNIT, DOT
"""

import re
from typing import List, Tuple

# ---------- alphabet ----------
ALPHABET = ("CMD", "NUM", "VAR", "OP", "PAREN", "ENT", "PROP",
            "RULE", "QMARK", "NARR", "UNIT", "DOT")

# ---------- detection patterns (priority-ordered) ----------

# Rule keywords — must match before individual words
RULE_RE = re.compile(
    r'\b(?:if\s+(?:something|someone|some\w+)|then)\b', re.IGNORECASE
)

# Command verbs at start of query or after punctuation
CMD_WORDS = {
    "calculate", "compute", "evaluate", "solve", "find", "simplify",
    "determine", "let", "suppose", "what", "multiply", "add",
    "subtract", "divide",
}

# Number literal (integer, decimal, negative)
NUM_RE = re.compile(r'[-+]?\d+(?:\.\d+)?')

# Single-letter variable (not a common word)
STOP_LETTERS = {"a", "i"}  # common English words that are single letters (not 's' — it's a common variable)
VAR_RE = re.compile(r'\b([a-z])\b')

# Math operators
OP_CHARS = set("+-*/=")
STARSTAR_RE = re.compile(r'\*\*')

# Parentheses
PAREN_CHARS = set("()")

# Named entities — capitalized words (animals, people)
ENT_RE = re.compile(r'\b[A-Z][a-z]+\b')
# Also common RuleTaker entities in lowercase
RULETAKER_ENTITIES = {
    "cat", "dog", "mouse", "rabbit", "squirrel", "tiger", "lion",
    "bear", "bald eagle", "cow",
}

# Properties (RuleTaker adjectives)
PROPERTIES = {
    "green", "red", "blue", "big", "small", "young", "old", "kind",
    "nice", "round", "rough", "cold", "smart", "quiet", "furry",
    "white", "strong", "fast", "slow", "tall", "short",
}

# Units of measurement
UNITS = {
    "miles", "mile", "hours", "hour", "minutes", "minute", "seconds",
    "second", "mph", "km", "kg", "pounds", "pound", "dollars", "dollar",
    "cents", "cent", "feet", "foot", "inches", "inch", "meters", "meter",
    "liters", "liter", "gallons", "gallon", "weeks", "week", "days",
    "day", "months", "month", "years", "year", "percent", "%",
}

MAX_SEQ_LEN = 15


def tokenize(text: str) -> Tuple[str, ...]:
    """Convert raw query text to a tuple of token classes.

    Consecutive NARR tokens are collapsed into one.
    Sequences are truncated to MAX_SEQ_LEN.
    """
    tokens: List[str] = []
    remaining = text.strip()

    while remaining:
        remaining = remaining.lstrip()
        if not remaining:
            break

        tok, consumed = _match_next(remaining)
        if tok is not None:
            # collapse consecutive NARR
            if tok == "NARR" and tokens and tokens[-1] == "NARR":
                pass  # skip duplicate NARR
            else:
                tokens.append(tok)
        remaining = remaining[consumed:]

        if len(tokens) >= MAX_SEQ_LEN:
            break

    return tuple(tokens)


def _match_next(text: str) -> Tuple[str, int]:
    """Match the next token at the start of text. Returns (token_class, chars_consumed)."""

    # 1. Rule keywords ("If something", "then")
    m = RULE_RE.match(text)
    if m:
        return "RULE", m.end()

    # 2. Question mark
    if text[0] == '?':
        return "QMARK", 1

    # 3. Sentence boundary
    if text[0] == '.':
        return "DOT", 1

    # 4. Parentheses
    if text[0] in PAREN_CHARS:
        return "PAREN", 1

    # 5. ** operator (before single *)
    m = STARSTAR_RE.match(text)
    if m:
        return "OP", 2

    # 6. Single-char math operators
    if text[0] in OP_CHARS:
        return "OP", 1

    # 7. Number literal
    m = NUM_RE.match(text)
    if m and (len(text) == len(m.group()) or not text[len(m.group()):][0:1].isalpha()):
        return "NUM", m.end()

    # 7.5. Contractions/possessives ('s, 'd, 't, 're, 've, 'll, n't)
    contraction_m = re.match(r"'(?:s|d|t|re|ve|ll|m)\b", text)
    if contraction_m:
        return None, contraction_m.end()  # skip — part of previous word

    # 8. Word-level matching
    word_m = re.match(r'[a-zA-Z_]\w*', text)
    if word_m:
        word = word_m.group()
        word_lower = word.lower()
        consumed = word_m.end()

        # Check for multi-word phrases
        two_word = re.match(r'(?:what\s+is|divided\s+by|multiplied\s+by|question\s*:)', text, re.IGNORECASE)
        if two_word:
            tw_lower = two_word.group().lower()
            if tw_lower.startswith("question"):
                return "QMARK", two_word.end()
            elif tw_lower.startswith("what"):
                return "CMD", two_word.end()
            else:
                return "OP", two_word.end()

        # Command words
        if word_lower in CMD_WORDS:
            return "CMD", consumed

        # Properties (RuleTaker adjectives)
        if word_lower in PROPERTIES:
            return "PROP", consumed

        # Units
        if word_lower in UNITS:
            return "UNIT", consumed

        # Named entities (capitalized) or known RuleTaker entities
        if word[0].isupper() and len(word) > 1:
            return "ENT", consumed
        if word_lower in RULETAKER_ENTITIES:
            return "ENT", consumed

        # Single-letter variable (not a stopword)
        if len(word) == 1 and word_lower not in STOP_LETTERS:
            return "VAR", consumed

        # Default: narrative word
        return "NARR", consumed

    # 9. Special characters ($, %, comma, colon, etc.) — skip
    if text[0] == '$':
        return "UNIT", 1
    if text[0] == '%':
        return "UNIT", 1

    # Skip unknown character
    return None, 1


def tokens_to_str(tokens: Tuple[str, ...]) -> str:
    """Convert token tuple to a space-separated string."""
    return " ".join(tokens)


def str_to_tokens(s: str) -> Tuple[str, ...]:
    """Convert space-separated string back to token tuple."""
    if not s.strip():
        return ()
    return tuple(s.strip().split())
