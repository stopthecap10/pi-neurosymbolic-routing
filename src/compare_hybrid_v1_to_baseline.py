#!/usr/bin/env python3
"""
Compare Hybrid V1 to Canonical Baseline
Generates comparison report showing improvements
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

def load_trials(csv_path):
    """Load trials from CSV"""
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def calc_metrics(trials):
    """Calculate metrics for a set of trials"""
    if not trials:
        return None

    total = len(trials)
    correct = sum(int(t['correct']) for t in trials)

    # Latency (exclude timeouts)
    latencies = [
        float(t['total_latency_ms'])
        for t in trials
        if not int(t.get('timeout_flag', 0))
    ]
    median_lat = sorted(latencies)[len(latencies)//2] if latencies else 0

    # Energy
    energy_vals = []
    for t in trials:
        e = t.get('energy_per_prompt_mwh', 'NA')
        if e != 'NA' and e != '':
            try:
                energy_vals.append(float(e))
            except ValueError:
                pass
    median_energy = sorted(energy_vals)[len(energy_vals)//2] if energy_vals else None

    # Timeouts and parse failures
    timeouts = sum(int(t.get('timeout_flag', 0)) for t in trials)
    parse_fails = sum(1 - int(t.get('parse_success', 1)) for t in trials)

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'median_latency_ms': median_lat,
        'median_energy_mwh': median_energy,
        'timeouts': timeouts,
        'parse_fails': parse_fails
    }

def calc_category_metrics(trials):
    """Calculate per-category metrics"""
    by_cat = defaultdict(list)
    for t in trials:
        by_cat[t['category']].append(t)

    return {cat: calc_metrics(cat_trials) for cat, cat_trials in by_cat.items()}

def main():
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs"

    # Load baseline (default to v1_a1_nogrammar, can change)
    baseline_name = "v1_a1_nogrammar"
    baseline_path = output_dir / f"{baseline_name}.csv"
    hybrid_path = output_dir / "hybrid_v1.csv"

    print("="*70)
    print("HYBRID V1 vs BASELINE COMPARISON")
    print("="*70)
    print()

    # Check files exist
    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_path}")
        print("\nRun the baseline matrix first:")
        print("  bash scripts/run_baseline_matrix.sh")
        sys.exit(1)

    if not hybrid_path.exists():
        print(f"ERROR: Hybrid V1 file not found: {hybrid_path}")
        print("\nRun Hybrid V1 first:")
        print("  bash scripts/run_hybrid_v1.sh")
        sys.exit(1)

    # Load trials
    baseline_trials = load_trials(baseline_path)
    hybrid_trials = load_trials(hybrid_path)

    print(f"Loaded {baseline_name}: {len(baseline_trials)} trials")
    print(f"Loaded hybrid_v1: {len(hybrid_trials)} trials")
    print()

    # Calculate overall metrics
    baseline_metrics = calc_metrics(baseline_trials)
    hybrid_metrics = calc_metrics(hybrid_trials)

    # Calculate category metrics
    baseline_cat = calc_category_metrics(baseline_trials)
    hybrid_cat = calc_category_metrics(hybrid_trials)

    # Print overall comparison
    print("="*70)
    print("OVERALL RESULTS")
    print("="*70)
    print()
    print(f"{'Metric':<25} {'Baseline':>15} {'Hybrid V1':>15} {'Delta':>12}")
    print("-"*70)

    # Accuracy
    print(f"{'Accuracy':<25} "
          f"{baseline_metrics['accuracy']*100:>14.1f}% "
          f"{hybrid_metrics['accuracy']*100:>14.1f}% "
          f"{(hybrid_metrics['accuracy'] - baseline_metrics['accuracy'])*100:>+11.1f} pp")

    # Latency
    lat_pct = ((hybrid_metrics['median_latency_ms'] / baseline_metrics['median_latency_ms']) - 1) * 100 if baseline_metrics['median_latency_ms'] > 0 else 0
    print(f"{'Median Latency (ms)':<25} "
          f"{baseline_metrics['median_latency_ms']:>14.0f} "
          f"{hybrid_metrics['median_latency_ms']:>14.0f} "
          f"{lat_pct:>+11.1f}%")

    # Energy
    if baseline_metrics['median_energy_mwh'] and hybrid_metrics['median_energy_mwh']:
        energy_pct = ((hybrid_metrics['median_energy_mwh'] / baseline_metrics['median_energy_mwh']) - 1) * 100
        print(f"{'Median Energy (mWh)':<25} "
              f"{baseline_metrics['median_energy_mwh']:>14.2f} "
              f"{hybrid_metrics['median_energy_mwh']:>14.2f} "
              f"{energy_pct:>+11.1f}%")
    else:
        print(f"{'Median Energy (mWh)':<25} {'NA':>15} {'NA':>15} {'NA':>12}")

    # Timeouts
    print(f"{'Timeouts':<25} "
          f"{baseline_metrics['timeouts']:>15} "
          f"{hybrid_metrics['timeouts']:>15} "
          f"{hybrid_metrics['timeouts'] - baseline_metrics['timeouts']:>+12}")

    # Parse failures
    print(f"{'Parse Failures':<25} "
          f"{baseline_metrics['parse_fails']:>15} "
          f"{hybrid_metrics['parse_fails']:>15} "
          f"{hybrid_metrics['parse_fails'] - baseline_metrics['parse_fails']:>+12}")

    print()

    # Per-category comparison
    print("="*70)
    print("PER-CATEGORY RESULTS")
    print("="*70)
    print()

    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        cat_names = {
            'AR': 'Arithmetic',
            'ALG': 'Algebra',
            'WP': 'Word Problems',
            'LOG': 'Logical Entailment'
        }

        if cat not in baseline_cat or cat not in hybrid_cat:
            continue

        b = baseline_cat[cat]
        h = hybrid_cat[cat]

        print(f"{cat_names[cat]} ({cat})")
        print("-"*70)

        acc_delta = (h['accuracy'] - b['accuracy']) * 100
        print(f"  Accuracy:  {b['accuracy']*100:5.1f}% → {h['accuracy']*100:5.1f}% ({acc_delta:+.1f} pp)")

        if b['median_latency_ms'] > 0:
            lat_delta = ((h['median_latency_ms'] / b['median_latency_ms']) - 1) * 100
            print(f"  Latency:   {b['median_latency_ms']:5.0f}ms → {h['median_latency_ms']:5.0f}ms ({lat_delta:+.1f}%)")

        print()

    # Routing insights (if available)
    if 'route_chosen' in hybrid_trials[0]:
        print("="*70)
        print("ROUTING INSIGHTS")
        print("="*70)
        print()

        # Count routes per category
        route_counts = defaultdict(lambda: defaultdict(int))
        for t in hybrid_trials:
            cat = t['category']
            route = t.get('route_chosen', 'unknown')
            route_counts[cat][route] += 1

        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            if cat in route_counts:
                print(f"{cat}:")
                for route, count in sorted(route_counts[cat].items()):
                    pct = count / sum(route_counts[cat].values()) * 100
                    print(f"  {route}: {count} ({pct:.0f}%)")
                print()

        # Escalations
        if 'escalations_count' in hybrid_trials[0]:
            total_escalations = sum(int(t.get('escalations_count', 0)) for t in hybrid_trials)
            trials_with_escalation = sum(1 for t in hybrid_trials if int(t.get('escalations_count', 0)) > 0)
            print(f"Total escalations: {total_escalations}")
            print(f"Trials needing fallback: {trials_with_escalation}/{len(hybrid_trials)}")
            print()

    # Key findings
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)
    print()

    acc_improvement = (hybrid_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
    if acc_improvement > 0:
        print(f"✅ Hybrid V1 improved accuracy by {acc_improvement:.1f} percentage points")
    elif acc_improvement < 0:
        print(f"⚠️  Hybrid V1 decreased accuracy by {abs(acc_improvement):.1f} percentage points")
    else:
        print(f"➖ No change in accuracy")

    if hybrid_metrics['median_energy_mwh'] and baseline_metrics['median_energy_mwh']:
        energy_pct = ((hybrid_metrics['median_energy_mwh'] / baseline_metrics['median_energy_mwh']) - 1) * 100
        if energy_pct < -5:
            print(f"✅ Hybrid V1 reduced energy by {abs(energy_pct):.1f}%")
        elif energy_pct > 5:
            print(f"⚠️  Hybrid V1 increased energy by {energy_pct:.1f}%")
        else:
            print(f"➖ No significant change in energy")

    print()
    print("Next steps:")
    if acc_improvement > 0 or (hybrid_metrics['median_energy_mwh'] and energy_pct < -5):
        print("  ✅ Hybrid V1 shows improvement - proceed to Hybrid V2")
    else:
        print("  ⚠️  Review routing decisions and debug Hybrid V1")
        print("  - Check if A5 is working correctly for AR")
        print("  - Consider implementing A4 for ALG")
        print("  - Review fallback logic")

if __name__ == "__main__":
    main()
