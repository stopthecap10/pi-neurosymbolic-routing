#!/usr/bin/env python3
"""
Compare Hybrid V1 to Canonical Baseline
Generates v1_vs_baseline.md with overall + per-category comparison
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics


def load_trials(csv_path):
    if not Path(csv_path).exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)
    with open(csv_path, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def calc_metrics(trials):
    if not trials:
        return None
    total = len(trials)
    correct = sum(int(t['correct']) for t in trials)

    latencies = [
        float(t['total_latency_ms'])
        for t in trials
        if not int(t.get('timeout_flag', 0))
    ]
    median_lat = statistics.median(latencies) if latencies else 0

    energy_vals = []
    for t in trials:
        e = t.get('energy_per_prompt_mwh', 'NA')
        if e not in ('NA', ''):
            try:
                energy_vals.append(float(e))
            except ValueError:
                pass
    median_energy = statistics.median(energy_vals) if energy_vals else None

    timeouts = sum(int(t.get('timeout_flag', 0)) for t in trials)
    parse_fails = sum(1 - int(t.get('parse_success', 1)) for t in trials)

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'median_latency_ms': median_lat,
        'median_energy_mwh': median_energy,
        'timeouts': timeouts,
        'timeout_rate': timeouts / total if total > 0 else 0,
        'parse_fails': parse_fails,
        'parse_fail_rate': parse_fails / total if total > 0 else 0,
    }


def calc_category_metrics(trials):
    by_cat = defaultdict(list)
    for t in trials:
        by_cat[t['category']].append(t)
    return {cat: calc_metrics(cat_trials) for cat, cat_trials in by_cat.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/official")
    ap.add_argument("--baseline", default="v1_a1_nogrammar",
                    help="Baseline config name (default: auto-detect best from decisions)")
    args = ap.parse_args()

    output_dir = Path(args.out_dir)
    baseline_name = args.baseline
    baseline_path = output_dir / f"{baseline_name}.csv"
    hybrid_path = output_dir / "hybrid_v1.csv"

    if not baseline_path.exists():
        print(f"ERROR: Baseline not found: {baseline_path}")
        sys.exit(1)
    if not hybrid_path.exists():
        print(f"ERROR: Hybrid V1 not found: {hybrid_path}")
        sys.exit(1)

    baseline_trials = load_trials(baseline_path)
    hybrid_trials = load_trials(hybrid_path)

    print(f"Loaded {baseline_name}: {len(baseline_trials)} trials")
    print(f"Loaded hybrid_v1: {len(hybrid_trials)} trials")

    b = calc_metrics(baseline_trials)
    h = calc_metrics(hybrid_trials)
    b_cat = calc_category_metrics(baseline_trials)
    h_cat = calc_category_metrics(hybrid_trials)

    # ======================================================================
    # Write v1_vs_baseline.md
    # ======================================================================
    report_path = output_dir / "v1_vs_baseline.md"
    with open(report_path, 'w') as f:
        f.write("# Hybrid V1 vs Baseline Comparison — Official T1\n\n")
        f.write(f"- **Baseline**: `{baseline_name}` ({len(baseline_trials)} trials)\n")
        f.write(f"- **Hybrid V1**: deterministic routing ({len(hybrid_trials)} trials)\n\n")

        # Overall table
        f.write("## Overall Results\n\n")
        f.write("| Metric | Baseline | Hybrid V1 | Delta |\n")
        f.write("|--------|----------|-----------|-------|\n")

        acc_delta = (h['accuracy'] - b['accuracy']) * 100
        f.write(f"| Accuracy | {b['accuracy']*100:.1f}% ({b['correct']}/{b['total']}) "
                f"| {h['accuracy']*100:.1f}% ({h['correct']}/{h['total']}) "
                f"| {acc_delta:+.1f} pp |\n")

        if b['median_latency_ms'] > 0:
            lat_pct = ((h['median_latency_ms'] / b['median_latency_ms']) - 1) * 100
        else:
            lat_pct = 0
        f.write(f"| Median Latency | {b['median_latency_ms']:.0f} ms "
                f"| {h['median_latency_ms']:.0f} ms "
                f"| {lat_pct:+.1f}% |\n")

        if b['median_energy_mwh'] and h['median_energy_mwh']:
            e_pct = ((h['median_energy_mwh'] / b['median_energy_mwh']) - 1) * 100
            f.write(f"| Median Energy | {b['median_energy_mwh']:.2f} mWh "
                    f"| {h['median_energy_mwh']:.2f} mWh "
                    f"| {e_pct:+.1f}% |\n")
        else:
            f.write("| Median Energy | NA | NA | NA |\n")

        f.write(f"| Timeouts | {b['timeouts']} ({b['timeout_rate']*100:.1f}%) "
                f"| {h['timeouts']} ({h['timeout_rate']*100:.1f}%) "
                f"| {h['timeouts'] - b['timeouts']:+d} |\n")
        f.write(f"| Parse Failures | {b['parse_fails']} ({b['parse_fail_rate']*100:.1f}%) "
                f"| {h['parse_fails']} ({h['parse_fail_rate']*100:.1f}%) "
                f"| {h['parse_fails'] - b['parse_fails']:+d} |\n")

        # Per-category table
        f.write("\n## Per-Category Comparison\n\n")
        f.write("| Category | Baseline Acc | Hybrid V1 Acc | Delta | Baseline Lat | Hybrid V1 Lat | Lat Delta |\n")
        f.write("|----------|-------------|--------------|-------|-------------|--------------|----------|\n")

        cat_names = {'AR': 'Arithmetic', 'ALG': 'Algebra', 'WP': 'Word Problems', 'LOG': 'Logic'}
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            bc = b_cat.get(cat)
            hc = h_cat.get(cat)
            if not bc or not hc:
                continue
            cat_acc_d = (hc['accuracy'] - bc['accuracy']) * 100
            if bc['median_latency_ms'] > 0:
                cat_lat_d = ((hc['median_latency_ms'] / bc['median_latency_ms']) - 1) * 100
            else:
                cat_lat_d = 0
            f.write(f"| {cat} ({cat_names[cat]}) "
                    f"| {bc['accuracy']*100:.1f}% ({bc['correct']}/{bc['total']}) "
                    f"| {hc['accuracy']*100:.1f}% ({hc['correct']}/{hc['total']}) "
                    f"| {cat_acc_d:+.1f} pp "
                    f"| {bc['median_latency_ms']:.0f} ms "
                    f"| {hc['median_latency_ms']:.0f} ms "
                    f"| {cat_lat_d:+.1f}% |\n")

        # Route usage
        if hybrid_trials and 'route_chosen' in hybrid_trials[0]:
            f.write("\n## Route Usage (Hybrid V1)\n\n")
            f.write("| Category | Route | Count | % |\n")
            f.write("|----------|-------|-------|---|\n")

            route_counts = defaultdict(lambda: defaultdict(int))
            for t in hybrid_trials:
                route_counts[t['category']][t.get('route_chosen', '?')] += 1

            for cat in ['AR', 'ALG', 'WP', 'LOG']:
                if cat not in route_counts:
                    continue
                cat_total = sum(route_counts[cat].values())
                for route, count in sorted(route_counts[cat].items()):
                    pct = count / cat_total * 100
                    f.write(f"| {cat} | {route} | {count} | {pct:.0f}% |\n")

        # Escalations
        if hybrid_trials and 'escalations_count' in hybrid_trials[0]:
            total_esc = sum(int(t.get('escalations_count', 0)) for t in hybrid_trials)
            trials_esc = sum(1 for t in hybrid_trials if int(t.get('escalations_count', 0)) > 0)
            f.write(f"\n## Escalation Summary\n\n")
            f.write(f"- Total escalations: {total_esc}\n")
            f.write(f"- Trials needing fallback: {trials_esc}/{len(hybrid_trials)} "
                    f"({trials_esc/len(hybrid_trials)*100:.1f}%)\n")

        # Key findings
        f.write("\n## Key Findings\n\n")
        if acc_delta > 0:
            f.write(f"- Hybrid V1 improved accuracy by **{acc_delta:.1f} pp**\n")
        elif acc_delta < 0:
            f.write(f"- Hybrid V1 decreased accuracy by **{abs(acc_delta):.1f} pp**\n")
        else:
            f.write("- No change in accuracy\n")

        if b['median_energy_mwh'] and h['median_energy_mwh']:
            e_pct = ((h['median_energy_mwh'] / b['median_energy_mwh']) - 1) * 100
            if e_pct < -5:
                f.write(f"- Hybrid V1 reduced energy by **{abs(e_pct):.1f}%**\n")
            elif e_pct > 5:
                f.write(f"- Hybrid V1 increased energy by **{e_pct:.1f}%**\n")

        if lat_pct < -10:
            f.write(f"- Hybrid V1 reduced latency by **{abs(lat_pct):.1f}%**\n")
        elif lat_pct > 10:
            f.write(f"- Hybrid V1 increased latency by **{lat_pct:.1f}%**\n")

    print(f"\nComparison written to: {report_path}")

    # Also print to console
    with open(report_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
