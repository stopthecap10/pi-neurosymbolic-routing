#!/usr/bin/env python3
"""
Compare Hybrid V1 to Hybrid V2
Focus on ALG improvement from A4 symbolic solver
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

    v1_path = output_dir / "hybrid_v1.csv"
    v2_path = output_dir / "hybrid_v2.csv"

    print("=" * 70)
    print("HYBRID V1 vs V2 COMPARISON")
    print("=" * 70)
    print()

    # Check files exist
    if not v1_path.exists():
        print(f"ERROR: V1 file not found: {v1_path}")
        sys.exit(1)

    if not v2_path.exists():
        print(f"ERROR: V2 file not found: {v2_path}")
        print("\nRun V2 first:")
        print("  bash scripts/run_hybrid_v2.sh")
        sys.exit(1)

    # Load trials
    v1_trials = load_trials(v1_path)
    v2_trials = load_trials(v2_path)

    print(f"Loaded V1: {len(v1_trials)} trials")
    print(f"Loaded V2: {len(v2_trials)} trials")
    print()

    # Calculate overall metrics
    v1_metrics = calc_metrics(v1_trials)
    v2_metrics = calc_metrics(v2_trials)

    # Calculate category metrics
    v1_cat = calc_category_metrics(v1_trials)
    v2_cat = calc_category_metrics(v2_trials)

    # Print overall comparison
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'V1':>15} {'V2':>15} {'Delta':>12}")
    print("-" * 70)

    # Accuracy
    print(f"{'Accuracy':<25} "
          f"{v1_metrics['accuracy']*100:>14.1f}% "
          f"{v2_metrics['accuracy']*100:>14.1f}% "
          f"{(v2_metrics['accuracy'] - v1_metrics['accuracy'])*100:>+11.1f} pp")

    # Latency
    if v1_metrics['median_latency_ms'] > 0:
        lat_pct = ((v2_metrics['median_latency_ms'] / v1_metrics['median_latency_ms']) - 1) * 100
        print(f"{'Median Latency (ms)':<25} "
              f"{v1_metrics['median_latency_ms']:>14.0f} "
              f"{v2_metrics['median_latency_ms']:>14.0f} "
              f"{lat_pct:>+11.1f}%")

    # Energy
    if v1_metrics['median_energy_mwh'] and v2_metrics['median_energy_mwh']:
        energy_pct = ((v2_metrics['median_energy_mwh'] / v1_metrics['median_energy_mwh']) - 1) * 100
        print(f"{'Median Energy (mWh)':<25} "
              f"{v1_metrics['median_energy_mwh']:>14.2f} "
              f"{v2_metrics['median_energy_mwh']:>14.2f} "
              f"{energy_pct:>+11.1f}%")
    else:
        print(f"{'Median Energy (mWh)':<25} {'NA':>15} {'NA':>15} {'NA':>12}")

    # Timeouts
    print(f"{'Timeouts':<25} "
          f"{v1_metrics['timeouts']:>15} "
          f"{v2_metrics['timeouts']:>15} "
          f"{v2_metrics['timeouts'] - v1_metrics['timeouts']:>+12}")

    # Parse failures
    print(f"{'Parse Failures':<25} "
          f"{v1_metrics['parse_fails']:>15} "
          f"{v2_metrics['parse_fails']:>15} "
          f"{v2_metrics['parse_fails'] - v1_metrics['parse_fails']:>+12}")

    print()

    # Per-category comparison - FOCUS ON ALG
    print("=" * 70)
    print("PER-CATEGORY RESULTS (FOCUS: ALG IMPROVEMENT)")
    print("=" * 70)
    print()

    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        cat_names = {
            'AR': 'Arithmetic',
            'ALG': 'Algebra',
            'WP': 'Word Problems',
            'LOG': 'Logical Entailment'
        }

        if cat not in v1_cat or cat not in v2_cat:
            continue

        v1 = v1_cat[cat]
        v2 = v2_cat[cat]

        print(f"{cat_names[cat]} ({cat})")
        print("-" * 70)

        acc_delta = (v2['accuracy'] - v1['accuracy']) * 100
        indicator = ""
        if cat == 'ALG':
            indicator = "  ← KEY IMPROVEMENT TARGET" if acc_delta > 0 else "  ⚠️  NO IMPROVEMENT"

        print(f"  Accuracy:  {v1['accuracy']*100:5.1f}% → {v2['accuracy']*100:5.1f}% ({acc_delta:+.1f} pp){indicator}")

        if v1['median_latency_ms'] > 0:
            lat_delta = ((v2['median_latency_ms'] / v1['median_latency_ms']) - 1) * 100
            print(f"  Latency:   {v1['median_latency_ms']:5.0f}ms → {v2['median_latency_ms']:5.0f}ms ({lat_delta:+.1f}%)")

        print()

    # V2-specific insights (symbolic stats)
    if 'symbolic_parse_success' in v2_trials[0]:
        print("=" * 70)
        print("V2 SYMBOLIC EXECUTION STATS")
        print("=" * 70)
        print()

        # Count by category
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            cat_trials = [t for t in v2_trials if t['category'] == cat]
            if not cat_trials:
                continue

            sym_parse = sum(int(t.get('symbolic_parse_success', 0)) for t in cat_trials)
            sympy_solve = sum(int(t.get('sympy_solve_success', 0)) for t in cat_trials)
            total_cat = len(cat_trials)

            print(f"{cat}:")
            print(f"  Symbolic parse success: {sym_parse}/{total_cat} ({100*sym_parse/total_cat:.0f}%)")
            print(f"  SymPy solve success:    {sympy_solve}/{total_cat} ({100*sympy_solve/total_cat:.0f}%)")
            print()

    # Escalation comparison
    v1_escalations = sum(int(t.get('escalations_count', 0)) for t in v1_trials)
    v2_escalations = sum(int(t.get('escalations_count', 0)) for t in v2_trials)

    print("=" * 70)
    print("ESCALATION COMPARISON")
    print("=" * 70)
    print()
    print(f"V1 total escalations: {v1_escalations}")
    print(f"V2 total escalations: {v2_escalations}")
    print(f"Delta: {v2_escalations - v1_escalations:+d}")
    print()

    # Key findings
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()

    acc_improvement = (v2_metrics['accuracy'] - v1_metrics['accuracy']) * 100
    alg_improvement = (v2_cat['ALG']['accuracy'] - v1_cat['ALG']['accuracy']) * 100 if 'ALG' in v1_cat and 'ALG' in v2_cat else 0

    if alg_improvement > 0:
        print(f"✅ ALG improved by {alg_improvement:.1f} pp (A4 symbolic solver working)")
    elif alg_improvement == 0:
        print(f"⚠️  ALG showed NO improvement (A4 may be falling back to A1/A2)")
    else:
        print(f"❌ ALG regressed by {abs(alg_improvement):.1f} pp (A4 implementation issue)")

    if acc_improvement > 0:
        print(f"✅ Overall accuracy improved by {acc_improvement:.1f} pp")
    else:
        print(f"➖ Overall accuracy unchanged ({v2_metrics['accuracy']*100:.1f}%)")

    # AR stability check
    ar_delta = (v2_cat['AR']['accuracy'] - v1_cat['AR']['accuracy']) * 100 if 'AR' in v1_cat and 'AR' in v2_cat else 0
    if ar_delta == 0 and v2_cat.get('AR', {}).get('accuracy', 0) == 1.0:
        print(f"✅ AR remains stable at 100% (A5 symbolic preserved)")
    else:
        print(f"⚠️  AR changed (delta: {ar_delta:+.1f} pp)")

    # LOG stability check
    log_delta = (v2_cat['LOG']['accuracy'] - v1_cat['LOG']['accuracy']) * 100 if 'LOG' in v1_cat and 'LOG' in v2_cat else 0
    if log_delta == 0 and v2_cat.get('LOG', {}).get('accuracy', 0) == 1.0:
        print(f"✅ LOG remains stable at 100%")
    else:
        print(f"⚠️  LOG changed (delta: {log_delta:+.1f} pp)")

    print()
    print("V2 Acceptance Decision:")
    if alg_improvement > 0 and ar_delta >= 0 and log_delta >= 0:
        print("  ✅ PASS - V2 shows improvement, proceed to V3")
    elif alg_improvement == 0:
        print("  ⚠️  INVESTIGATE - Debug A4 extraction before V3")
        print("  - Check symbolic_parse_success on ALG trials")
        print("  - Review A4 equation extraction logic")
    else:
        print("  ❌ FAIL - Fix V2 before proceeding")
        print("  - ALG regressed or AR/LOG unstable")

    print()
    print("Output saved to: (create manually)")
    print("  outputs/v1_vs_v2_summary.md")

if __name__ == "__main__":
    main()
