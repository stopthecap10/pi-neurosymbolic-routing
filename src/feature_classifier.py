#!/usr/bin/env python3
"""
Feature-based classifier for query routing.

Uses token-set features (bag-of-tokens) derived from the token classifier
to classify queries. This serves as a simple baseline and also provides
a better membership oracle for L* than exact-match lookup.

The classifier uses hand-derived rules based on discriminative token features,
then can be refined by L*/RPNI learned DFAs for the sequential patterns.
"""

from typing import Dict, List, Optional, Tuple
from src.token_classifier import tokenize, ALPHABET


def classify_by_features(tokens: Tuple[str, ...]) -> str:
    """Classify a token sequence using bag-of-token features.

    Rules derived from discriminative analysis of T1 training data:
    - LOG: has PROP or (has ENT + DOT pattern, no NUM-heavy math)
    - ALG: has VAR token
    - AR:  has NUM + OP, no VAR, no ENT-heavy narrative
    - WP:  has ENT/NARR narrative with NUM (fallback)
    """
    token_set = set(tokens)
    token_list = list(tokens)

    # Count occurrences
    counts = {t: 0 for t in ALPHABET}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    has_var = "VAR" in token_set
    has_prop = "PROP" in token_set
    has_num = "NUM" in token_set
    has_op = "OP" in token_set
    has_ent = "ENT" in token_set
    has_narr = "NARR" in token_set
    has_unit = "UNIT" in token_set
    has_rule = "RULE" in token_set
    has_cmd = "CMD" in token_set
    has_qmark = "QMARK" in token_set

    # LOG: Rule keywords or entity-property patterns (no numbers/units — that's WP)
    if has_rule and not has_unit:
        return "LOG"
    if has_prop and has_ent and counts.get("DOT", 0) >= 2 and not has_op and not has_num and not has_unit:
        return "LOG"
    if has_ent and counts.get("DOT", 0) >= 3 and not has_num and not has_op and not has_unit:
        return "LOG"

    # ALG: Has variable tokens BUT not in a narrative context (WP has names that look like vars)
    if has_var and not has_unit and not (has_narr and has_ent and counts.get("NARR", 0) >= 3):
        return "ALG"

    # AR: Math-heavy, no narrative, no entities (or CMD + math)
    if has_cmd and has_num and has_op and not has_narr and not has_ent:
        return "AR"
    if has_cmd and has_num and not has_ent and not has_unit:
        return "AR"
    if has_num and has_op and not has_ent and not has_narr and not has_unit:
        return "AR"
    # "Multiply X and Y" pattern
    if has_cmd and has_num and not has_unit and counts.get("NARR", 0) <= 1:
        return "AR"

    # WP: Narrative with numbers (fallback for remaining)
    if has_narr or has_unit or has_ent:
        return "WP"

    # Final fallback: if it has numbers and operators, probably AR
    if has_num:
        return "AR"

    return "WP"  # default


def create_feature_oracle(target_category: str):
    """Create a membership oracle using feature-based classification.

    This generalizes better than exact-match oracles for L*.
    """
    def oracle(w: Tuple[str, ...]) -> bool:
        if not w:
            return False
        return classify_by_features(w) == target_category

    return oracle
