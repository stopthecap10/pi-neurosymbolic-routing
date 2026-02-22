#!/usr/bin/env python3
"""
Tau Sweep for V4 Calibrated Router

Replays V4 trial data with different tau values to find the optimal
accuracy-cost tradeoff WITHOUT re-running the LLM.

Requires a V4 trial CSV that includes p_correct_pre_escalation values.
For each tau, simulates which prompts would have been escalated.

Usage:
    python src/sweep_tau_v4.py \
        --in_csv outputs/official/runs/hybrid_v4.csv \
        --out_dir outputs/official/calibration

Outputs:
    tau_sweep.csv         - per-tau metrics
    tau_sweep_summary.md  - summary with recommendations
"""

import argparse
import csv
import os
import sys
from datetime import datetime


def load_trials(csv_path: str) -> list:
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def simulate_tau(trials: list, tau: float) -> dict:
    """
    Simulate V4 decisions at a given tau threshold.

    For WP trials with p_correct_pre_escalation:
    - If p >= tau: would accept A2 (no escalation)
    - If p < tau: would escalate to A3R (use actual A3R result if available)

    For AR/ALG: always deterministic (no change)
    For LOG: always A1 (no fallback in V4-min)
    """
    total = 0
    correct = 0
    timeouts = 0
    escalations = 0
    action_counts = {"A1": 0, "A2": 0, "A3R": 0, "A4": 0, "A5": 0}
    latency_sum = 0.0
    energy_sum = 0.0
    energy_count = 0

    for row in trials:
        total += 1
        category = row.get('category', '')
        p_pre = float(row.get('p_correct_pre_escalation', -1))
        was_escalated = int(row.get('repair_attempted', 0))

        if category in ('AR', 'ALG'):
            # Deterministic — always same result
            correct += int(row.get('correct', 0))
            timeouts += int(row.get('timeout_flag', 0))
            latency_sum += float(row.get('total_latency_ms', 0))
            source = row.get('route_chosen', '')
            if source == 'symbolic':
                action_counts['A5'] += 1
            elif source == 'sympy':
                action_counts['A4'] += 1

        elif category == 'LOG':
            # A1 only
            correct += int(row.get('correct', 0))
            timeouts += int(row.get('timeout_flag', 0))
            latency_sum += float(row.get('total_latency_ms', 0))
            action_counts['A1'] += 1

        elif category == 'WP':
            # Simulate tau decision
            would_escalate = (p_pre >= 0 and p_pre < tau)

            if would_escalate and was_escalated:
                # Tau says escalate, and we have A3R data → use full result
                correct += int(row.get('correct', 0))
                timeouts += int(row.get('timeout_flag', 0))
                latency_sum += float(row.get('total_latency_ms', 0))
                escalations += 1
                action_counts['A3R'] += 1
            elif would_escalate and not was_escalated:
                # Tau says escalate but we don't have A3R data
                # (A2 parsed ok so it wasn't escalated)
                # Use A2 result since we can't know what A3R would do
                correct += int(row.get('correct', 0))
                timeouts += int(row.get('timeout_flag', 0))
                latency_sum += float(row.get('total_latency_ms', 0))
                action_counts['A2'] += 1
            elif not would_escalate and was_escalated:
                # Tau says accept A2, but original run escalated
                # We don't have A2-only result cleanly, approximate:
                # Use whatever the outcome was (conservative)
                correct += int(row.get('correct', 0))
                timeouts += int(row.get('timeout_flag', 0))
                latency_sum += float(row.get('total_latency_ms', 0))
                action_counts['A2'] += 1
            else:
                # Tau says accept, and it wasn't escalated
                correct += int(row.get('correct', 0))
                timeouts += int(row.get('timeout_flag', 0))
                latency_sum += float(row.get('total_latency_ms', 0))
                action_counts['A2'] += 1

        # Energy
        e = row.get('energy_per_prompt_mwh', 'NA')
        if e != 'NA':
            try:
                energy_sum += float(e)
                energy_count += 1
            except ValueError:
                pass

    accuracy = correct / total if total > 0 else 0
    median_latency = latency_sum / total if total > 0 else 0  # mean as proxy
    median_energy = energy_sum / energy_count if energy_count > 0 else 0

    return {
        "tau": tau,
        "overall_accuracy": round(accuracy, 4),
        "mean_latency_ms": round(median_latency, 1),
        "mean_energy_mwh": round(median_energy, 2),
        "timeouts": timeouts,
        "escalations": escalations,
        "pct_A1": round(action_counts['A1'] / total * 100, 1) if total > 0 else 0,
        "pct_A2": round(action_counts['A2'] / total * 100, 1) if total > 0 else 0,
        "pct_A3R": round(action_counts['A3R'] / total * 100, 1) if total > 0 else 0,
        "pct_A4": round(action_counts['A4'] / total * 100, 1) if total > 0 else 0,
        "pct_A5": round(action_counts['A5'] / total * 100, 1) if total > 0 else 0,
        "n_trials": total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="V4 trial CSV with p_correct fields")
    ap.add_argument("--out_dir", default="outputs/official/calibration",
                    help="Output directory")
    ap.add_argument("--tau_min", type=float, default=0.1)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--tau_step", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    trials = load_trials(args.in_csv)
    print(f"Loaded {len(trials)} trials from {args.in_csv}")

    # Generate tau values
    tau_values = []
    tau = args.tau_min
    while tau <= args.tau_max + 0.001:
        tau_values.append(round(tau, 3))
        tau += args.tau_step

    print(f"Sweeping {len(tau_values)} tau values: {tau_values[0]} to {tau_values[-1]}")

    # Run sweep
    results = []
    for tau in tau_values:
        r = simulate_tau(trials, tau)
        results.append(r)

    # Write sweep CSV
    sweep_path = os.path.join(args.out_dir, "tau_sweep.csv")
    fieldnames = ["tau", "overall_accuracy", "mean_latency_ms", "mean_energy_mwh",
                  "timeouts", "escalations", "pct_A1", "pct_A2", "pct_A3R",
                  "pct_A4", "pct_A5", "n_trials"]

    with open(sweep_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {sweep_path}")

    # Find best tau values
    best_acc = max(results, key=lambda r: r['overall_accuracy'])
    best_energy = min(results, key=lambda r: r['mean_energy_mwh']) if any(r['mean_energy_mwh'] > 0 for r in results) else results[0]
    # Best balanced: highest accuracy with fewest escalations
    best_balanced = max(results, key=lambda r: r['overall_accuracy'] - 0.01 * r['escalations'])

    # Write summary
    summary_path = os.path.join(args.out_dir, "tau_sweep_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# Tau Sweep Summary\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Source**: {args.in_csv}\n")
        f.write(f"**Trials**: {len(trials)}\n\n")
        f.write(f"## Best Tau Values\n\n")
        f.write(f"- **Best accuracy**: tau={best_acc['tau']} ({best_acc['overall_accuracy']*100:.1f}%, {best_acc['escalations']} escalations)\n")
        f.write(f"- **Best energy**: tau={best_energy['tau']} (energy={best_energy['mean_energy_mwh']:.2f} mWh, {best_energy['overall_accuracy']*100:.1f}%)\n")
        f.write(f"- **Best balanced**: tau={best_balanced['tau']} ({best_balanced['overall_accuracy']*100:.1f}%, {best_balanced['escalations']} escalations)\n\n")
        f.write(f"## Full Sweep\n\n")
        f.write("| tau | accuracy | latency_ms | energy_mwh | escalations | %A3R |\n")
        f.write("|-----|----------|-----------|------------|-------------|------|\n")
        for r in results:
            f.write(f"| {r['tau']:.2f} | {r['overall_accuracy']*100:.1f}% | {r['mean_latency_ms']:.0f} | {r['mean_energy_mwh']:.2f} | {r['escalations']} | {r['pct_A3R']:.1f}% |\n")
        f.write(f"\n## Recommendation\n\n")
        f.write(f"Recommended poster tau: **{best_balanced['tau']}** — ")
        f.write(f"achieves {best_balanced['overall_accuracy']*100:.1f}% accuracy with {best_balanced['escalations']} escalations.\n")
        f.write(f"Higher tau increases WP escalation cost but may improve WP recovery rate.\n")

    print(f"Saved: {summary_path}")

    # Print summary table
    print(f"\nTau Sweep Results:")
    print(f"{'tau':>6} {'accuracy':>9} {'latency':>10} {'escalations':>12} {'%A3R':>6}")
    for r in results:
        print(f"{r['tau']:6.2f} {r['overall_accuracy']*100:8.1f}% {r['mean_latency_ms']:9.0f}ms {r['escalations']:11d} {r['pct_A3R']:5.1f}%")

    print(f"\nBest accuracy: tau={best_acc['tau']} ({best_acc['overall_accuracy']*100:.1f}%)")
    print(f"Best balanced: tau={best_balanced['tau']} ({best_balanced['overall_accuracy']*100:.1f}%)")


if __name__ == "__main__":
    main()
