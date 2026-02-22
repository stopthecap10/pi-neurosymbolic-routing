#!/usr/bin/env python3
"""
V4 Feature Extraction for Calibrated Router

Extracts compact per-attempt features from trial results.
Pure + deterministic — no model calls.

Feature set version: v1 (frozen)
"""

# Feature names in fixed order (must match calibrator training)
FEATURE_NAMES_V1 = [
    "category_is_wp",
    "category_is_log",
    "attempt_idx",
    "action_is_A1",
    "action_is_A2",
    "action_is_A3R",
    "parse_success",
    "timeout_flag",
    "answer_nonempty",
    "output_len_chars",
    "prev_attempt_failed_parse",
    "prev_attempt_timeout",
    "symbolic_parse_success",
]

FEATURE_VERSION = "v1"


def feature_names() -> list:
    """Return ordered feature names for the current version."""
    return list(FEATURE_NAMES_V1)


def extract_router_features(
    category: str,
    attempt_idx: int,
    action: str,
    parse_success: bool,
    timeout_flag: bool,
    answer_raw: str,
    symbolic_parse_success: bool = False,
    prev_attempt_failed_parse: bool = False,
    prev_attempt_timeout: bool = False,
) -> dict:
    """
    Extract features from a single routing attempt result.

    Args:
        category: prompt category (AR, ALG, WP, LOG)
        attempt_idx: 1-based attempt index in the route chain
        action: action ID (A1, A2, A3, A4, A5)
        parse_success: whether the output parsed successfully
        timeout_flag: whether the attempt timed out
        answer_raw: raw LLM output string
        symbolic_parse_success: whether symbolic parsing succeeded
        prev_attempt_failed_parse: whether previous attempt had parse failure
        prev_attempt_timeout: whether previous attempt timed out

    Returns:
        dict with feature names -> float values
    """
    output_len = min(len(answer_raw), 200) if answer_raw else 0

    return {
        "category_is_wp": 1.0 if category == "WP" else 0.0,
        "category_is_log": 1.0 if category == "LOG" else 0.0,
        "attempt_idx": float(attempt_idx),
        "action_is_A1": 1.0 if action == "A1" else 0.0,
        "action_is_A2": 1.0 if action == "A2" else 0.0,
        "action_is_A3R": 1.0 if action == "A3" else 0.0,
        "parse_success": 1.0 if parse_success else 0.0,
        "timeout_flag": 1.0 if timeout_flag else 0.0,
        "answer_nonempty": 1.0 if (answer_raw and answer_raw.strip()) else 0.0,
        "output_len_chars": float(output_len),
        "prev_attempt_failed_parse": 1.0 if prev_attempt_failed_parse else 0.0,
        "prev_attempt_timeout": 1.0 if prev_attempt_timeout else 0.0,
        "symbolic_parse_success": 1.0 if symbolic_parse_success else 0.0,
    }


def feature_vector_from_dict(features: dict) -> list:
    """Convert feature dict to ordered float vector matching FEATURE_NAMES_V1."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES_V1]


def predict_p_correct(feature_vector: list, coefficients: list, intercept: float) -> float:
    """
    Compute P(correct) using logistic regression coefficients.
    Pure Python — no sklearn needed at inference time.

    Args:
        feature_vector: ordered float list matching FEATURE_NAMES_V1
        coefficients: logistic regression weights (same length as feature_vector)
        intercept: logistic regression bias

    Returns:
        float in [0, 1]: predicted probability of correct answer
    """
    import math
    z = intercept + sum(c * x for c, x in zip(coefficients, feature_vector))
    # Clip to avoid overflow
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(-z))
