#!/usr/bin/env python3
"""
Grammatical Inference Router

Uses a learned DFA (from L* or RPNI) to classify incoming queries
into categories, then routes to the appropriate solver via RouterV6.

This replaces the hardcoded category label from the CSV with a
learned classifier, demonstrating that routing patterns can be
automatically inferred rather than manually specified.
"""

import csv
from typing import Dict, Any, Optional, Tuple

from src.token_classifier import tokenize, ALPHABET
from src.dfa import DFA, MultiClassDFA


class RouterGI:
    """Router that uses grammatically-inferred DFAs for classification.

    This is a wrapper that classifies queries using learned DFAs,
    then delegates to the existing routing infrastructure.
    """

    def __init__(
        self,
        multi_dfa: MultiClassDFA,
        priority: Tuple[str, ...] = ("LOG", "ALG", "AR", "WP"),
    ):
        self.multi_dfa = multi_dfa
        self.priority = priority

    def classify(self, prompt_text: str) -> Optional[str]:
        """Classify a query into a category using the learned DFA."""
        tokens = tokenize(prompt_text)
        return self.multi_dfa.classify(tokens)

    def classify_batch(self, csv_path: str) -> Dict[str, Dict[str, Any]]:
        """Classify all prompts in a CSV and return results.

        Returns dict mapping prompt_id to {
            'predicted_category': str,
            'true_category': str,
            'correct': bool,
            'tokens': tuple,
        }
        """
        results = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                tokens = tokenize(row['prompt_text'])
                predicted = self.multi_dfa.classify(tokens)
                true_cat = row['category']
                results[row['prompt_id']] = {
                    'predicted_category': predicted or 'UNKNOWN',
                    'true_category': true_cat,
                    'correct': predicted == true_cat,
                    'tokens': tokens,
                }
        return results


def train_and_evaluate(
    train_csv: str,
    test_csv: str,
    method: str = "lstar",
) -> Dict[str, Any]:
    """Train a GI router on train_csv, evaluate on test_csv.

    Args:
        train_csv: Path to training CSV (e.g., T1 40 prompts).
        test_csv: Path to test CSV (e.g., T2 100 prompts).
        method: "lstar" or "rpni".

    Returns:
        Dict with accuracy, per-category results, and DFA sizes.
    """
    # Load training data
    train_data = []
    with open(train_csv) as f:
        for row in csv.DictReader(f):
            toks = tokenize(row['prompt_text'])
            train_data.append((toks, row['category']))

    # Train
    if method == "rpni":
        from src.rpni import learn_one_vs_rest
        dfas = learn_one_vs_rest(ALPHABET, train_data)
    elif method == "lstar":
        from src.lstar import LStar, ground_truth_equivalence_oracle
        from src.feature_classifier import create_feature_oracle
        dfas = {}
        for cat in ("AR", "ALG", "WP", "LOG"):
            mem_oracle = create_feature_oracle(cat)
            eq_oracle = ground_truth_equivalence_oracle(train_data, cat)
            learner = LStar(ALPHABET, mem_oracle, eq_oracle)
            dfas[cat] = learner.learn()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build multi-class classifier
    mc = MultiClassDFA(dfas, priority=("LOG", "ALG", "AR", "WP"))
    router = RouterGI(mc)

    # Evaluate
    results = router.classify_batch(test_csv)
    total = len(results)
    correct = sum(1 for r in results.values() if r['correct'])

    # Per-category
    per_cat = {}
    for cat in ("AR", "ALG", "WP", "LOG"):
        cat_results = [r for r in results.values() if r['true_category'] == cat]
        cat_correct = sum(1 for r in cat_results if r['correct'])
        per_cat[cat] = {
            'correct': cat_correct,
            'total': len(cat_results),
            'accuracy': cat_correct / len(cat_results) if cat_results else 0,
        }

    return {
        'method': method,
        'total_accuracy': correct / total,
        'correct': correct,
        'total': total,
        'per_category': per_cat,
        'dfa_sizes': {cat: dfa.num_states for cat, dfa in dfas.items()},
        'results': results,
    }


if __name__ == "__main__":
    import json

    for method in ("rpni", "lstar"):
        print(f"\n{'='*40}")
        print(f"Method: {method.upper()}")
        print(f"{'='*40}")

        res = train_and_evaluate(
            train_csv="data/splits/industry_tier1_40_v2.csv",
            test_csv="data/splits/industry_tier2_100.csv",
            method=method,
        )

        print(f"Overall: {res['correct']}/{res['total']} = {res['total_accuracy']:.1%}")
        for cat in ("AR", "ALG", "WP", "LOG"):
            pc = res['per_category'][cat]
            print(f"  {cat}: {pc['correct']}/{pc['total']} = {pc['accuracy']:.0%}")
        print(f"DFA sizes: {res['dfa_sizes']}")
