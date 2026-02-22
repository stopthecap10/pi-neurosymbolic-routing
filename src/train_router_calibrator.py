#!/usr/bin/env python3
"""
Train a logistic regression calibrator from hybrid trial CSVs.

Trains on attempt-level data from WP/LOG categories only
(AR/ALG are deterministic and don't need calibration).

Usage:
    python src/train_router_calibrator.py \
        --in_csv outputs/official/runs/hybrid_v3_1.csv \
        --out_dir outputs/official/calibration

Outputs:
    router_calibrator.json  - coefficients for pure-Python inference
    calibration_metrics.json - Brier, ECE, accuracy
    reliability_bins.csv    - per-bin calibration data
"""

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from router_features import FEATURE_NAMES_V1, FEATURE_VERSION


def load_trials(csv_path: str) -> list:
    """Load trial rows from CSV."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def extract_features_from_trial(row: dict) -> dict:
    """Extract feature dict from a trial CSV row."""
    category = row.get('category', '')
    action = row.get('route_chosen', row.get('final_answer_source', ''))

    # Map route_chosen values to action IDs
    action_map = {
        'llm': 'A1',  # Could be A1 or A2, disambiguate below
        'symbolic': 'A5',
        'sympy': 'A4',
        'repair': 'A3',
    }
    action_id = action_map.get(action, action)

    # Better action detection from route_attempt_sequence if available
    route_seq = row.get('route_attempt_sequence', '')
    escalations = int(row.get('escalations_count', 0))

    # Determine actual action from context
    if action_id == 'A1' or action == 'llm':
        if category == 'WP':
            action_id = 'A2'  # WP primary is A2 in V3.1
        elif category == 'LOG':
            action_id = 'A1'  # LOG primary is A1

    parse_success = bool(int(row.get('parse_success', 0)))
    timeout_flag = bool(int(row.get('timeout_flag', 0)))
    answer_raw = row.get('answer_raw', '')
    symbolic_parse = bool(int(row.get('symbolic_parse_success', 0)))

    output_len = min(len(answer_raw), 200) if answer_raw else 0

    return {
        "category_is_wp": 1.0 if category == "WP" else 0.0,
        "category_is_log": 1.0 if category == "LOG" else 0.0,
        "attempt_idx": float(1 + escalations),
        "action_is_A1": 1.0 if action_id == "A1" else 0.0,
        "action_is_A2": 1.0 if action_id == "A2" else 0.0,
        "action_is_A3R": 1.0 if action_id == "A3" else 0.0,
        "parse_success": 1.0 if parse_success else 0.0,
        "timeout_flag": 1.0 if timeout_flag else 0.0,
        "answer_nonempty": 1.0 if (answer_raw and answer_raw.strip()) else 0.0,
        "output_len_chars": float(output_len),
        "prev_attempt_failed_parse": 1.0 if (escalations > 0 and not parse_success) else 0.0,
        "prev_attempt_timeout": 1.0 if (escalations > 0 and timeout_flag) else 0.0,
        "symbolic_parse_success": 1.0 if symbolic_parse else 0.0,
    }


def feature_vector(features: dict) -> list:
    """Convert feature dict to ordered vector."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES_V1]


def train_logistic_regression(X: list, y: list) -> tuple:
    """
    Train logistic regression using gradient descent.
    Pure Python — no sklearn dependency required.

    Returns (coefficients, intercept, train_metrics)
    """
    n_samples = len(y)
    n_features = len(X[0]) if X else 0

    if n_samples == 0 or n_features == 0:
        return [0.0] * n_features, 0.0, {}

    # Initialize weights
    w = [0.0] * n_features
    b = 0.0

    # Hyperparameters
    lr = 0.01
    n_epochs = 1000
    reg_lambda = 0.01  # L2 regularization

    for epoch in range(n_epochs):
        # Forward pass
        grad_w = [0.0] * n_features
        grad_b = 0.0

        for i in range(n_samples):
            z = b + sum(w[j] * X[i][j] for j in range(n_features))
            z = max(-500, min(500, z))
            p = 1.0 / (1.0 + math.exp(-z))

            error = p - y[i]
            grad_b += error
            for j in range(n_features):
                grad_w[j] += error * X[i][j]

        # Update with L2 regularization
        for j in range(n_features):
            w[j] -= lr * (grad_w[j] / n_samples + reg_lambda * w[j])
        b -= lr * (grad_b / n_samples)

    # Compute train metrics
    predictions = []
    for i in range(n_samples):
        z = b + sum(w[j] * X[i][j] for j in range(n_features))
        z = max(-500, min(500, z))
        predictions.append(1.0 / (1.0 + math.exp(-z)))

    correct = sum(1 for i in range(n_samples) if (predictions[i] >= 0.5) == bool(y[i]))
    accuracy = correct / n_samples if n_samples > 0 else 0
    brier = sum((predictions[i] - y[i]) ** 2 for i in range(n_samples)) / n_samples

    metrics = {
        "train_accuracy": accuracy,
        "train_brier_score": brier,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_epochs": n_epochs,
        "learning_rate": lr,
        "reg_lambda": reg_lambda,
    }

    return w, b, metrics


def compute_calibration_metrics(predictions: list, labels: list, n_bins: int = 10) -> tuple:
    """Compute ECE and reliability bins."""
    n = len(predictions)
    if n == 0:
        return 0.0, []

    # Create bins
    bins = []
    for i in range(n_bins):
        bin_low = i / n_bins
        bin_high = (i + 1) / n_bins
        bin_preds = []
        bin_labels = []
        for j in range(n):
            if bin_low <= predictions[j] < bin_high or (i == n_bins - 1 and predictions[j] == 1.0):
                bin_preds.append(predictions[j])
                bin_labels.append(labels[j])

        if bin_preds:
            avg_conf = sum(bin_preds) / len(bin_preds)
            emp_acc = sum(bin_labels) / len(bin_labels)
            bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "count": len(bin_preds),
                "avg_confidence": round(avg_conf, 4),
                "empirical_accuracy": round(emp_acc, 4),
            })
        else:
            bins.append({
                "bin_low": bin_low,
                "bin_high": bin_high,
                "count": 0,
                "avg_confidence": 0.0,
                "empirical_accuracy": 0.0,
            })

    # ECE = weighted average |confidence - accuracy| per bin
    ece = 0.0
    for b in bins:
        if b["count"] > 0:
            ece += (b["count"] / n) * abs(b["avg_confidence"] - b["empirical_accuracy"])

    return ece, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, nargs="+",
                    help="Input trial CSV(s) to train on")
    ap.add_argument("--out_dir", default="outputs/official/calibration",
                    help="Output directory for calibrator files")
    ap.add_argument("--categories", nargs="*", default=["WP", "LOG"],
                    help="Categories to include in training (default: WP LOG)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all trial rows
    all_rows = []
    for csv_path in args.in_csv:
        rows = load_trials(csv_path)
        all_rows.extend(rows)
        print(f"Loaded {len(rows)} rows from {csv_path}")

    # Filter to target categories
    filtered = [r for r in all_rows if r.get('category', '') in args.categories]
    print(f"Filtered to {len(filtered)} rows for categories: {args.categories}")

    if not filtered:
        print("ERROR: No rows after filtering. Cannot train.")
        sys.exit(1)

    # Extract features and labels
    X = []
    y = []
    for row in filtered:
        features = extract_features_from_trial(row)
        vec = feature_vector(features)
        label = int(row.get('correct', 0))
        X.append(vec)
        y.append(label)

    print(f"Training set: {len(X)} samples, {len(FEATURE_NAMES_V1)} features")
    print(f"  Positive (correct): {sum(y)}")
    print(f"  Negative (incorrect): {len(y) - sum(y)}")

    # Train
    print("\nTraining logistic regression...")
    coefficients, intercept, train_metrics = train_logistic_regression(X, y)

    # Generate predictions for calibration metrics
    predictions = []
    for vec in X:
        z = intercept + sum(c * x for c, x in zip(coefficients, vec))
        z = max(-500, min(500, z))
        predictions.append(1.0 / (1.0 + math.exp(-z)))

    # Calibration metrics
    brier = sum((predictions[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
    mean_pred = sum(predictions) / len(predictions)
    ece, reliability_bins = compute_calibration_metrics(predictions, y)

    print(f"\nCalibration metrics:")
    print(f"  Brier score: {brier:.4f}")
    print(f"  ECE (10-bin): {ece:.4f}")
    print(f"  Mean predicted prob: {mean_pred:.4f}")
    print(f"  Train accuracy: {train_metrics['train_accuracy']:.4f}")

    # Save calibrator JSON
    calibrator = {
        "model_type": "logistic_regression",
        "version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES_V1),
        "coefficients": [round(c, 8) for c in coefficients],
        "intercept": round(intercept, 8),
        "train_run_names": [os.path.basename(p) for p in args.in_csv],
        "train_split": "t1",
        "train_row_count": len(X),
        "train_categories": args.categories,
        "created_at": datetime.now().isoformat(),
        "notes": "Trained on WP/LOG rows only. AR/ALG are deterministic.",
    }

    cal_path = os.path.join(args.out_dir, "router_calibrator.json")
    with open(cal_path, 'w') as f:
        json.dump(calibrator, f, indent=2)
    print(f"\nSaved calibrator: {cal_path}")

    # Save calibration metrics
    metrics = {
        "n_samples": len(X),
        "brier_score": round(brier, 6),
        "ece_10bin": round(ece, 6),
        "accuracy": round(train_metrics['train_accuracy'], 6),
        "mean_predicted_prob": round(mean_pred, 6),
        "positive_rate": round(sum(y) / len(y), 6),
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now().isoformat(),
    }

    metrics_path = os.path.join(args.out_dir, "calibration_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # Save reliability bins
    bins_path = os.path.join(args.out_dir, "reliability_bins.csv")
    with open(bins_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["bin_low", "bin_high", "count", "avg_confidence", "empirical_accuracy"])
        writer.writeheader()
        writer.writerows(reliability_bins)
    print(f"Saved reliability bins: {bins_path}")

    # Print coefficient summary
    print(f"\nCoefficient summary:")
    for name, coef in zip(FEATURE_NAMES_V1, coefficients):
        print(f"  {name:30s} {coef:+.6f}")
    print(f"  {'intercept':30s} {intercept:+.6f}")


if __name__ == "__main__":
    main()
