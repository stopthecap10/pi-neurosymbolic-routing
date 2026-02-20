#!/usr/bin/env python3
"""
V1 Baseline Matrix Summary Generator
Analyzes 2×2 factorial results and generates summary reports
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
import statistics

def load_trials(csv_path):
    """Load trials from CSV"""
    trials = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            trials.append(row)
    return trials

def compute_metrics(trials):
    """Compute summary metrics for a set of trials"""
    if not trials:
        return None

    total = len(trials)
    correct = sum(int(t['correct']) for t in trials)
    timeouts = sum(int(t['timeout_flag']) for t in trials)
    parse_fails = sum(1 - int(t['parse_success']) for t in trials)

    # Latency (exclude timeouts)
    latencies = [float(t['total_latency_ms']) for t in trials if not int(t['timeout_flag'])]
    median_lat = statistics.median(latencies) if latencies else 0

    # Energy (if available)
    energy_values = []
    for t in trials:
        e = t.get('energy_per_prompt_mwh', 'NA')
        if e != 'NA' and e != '':
            try:
                energy_values.append(float(e))
            except ValueError:
                pass
    median_energy = statistics.median(energy_values) if energy_values else None

    return {
        'total': total,
        'correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'timeouts': timeouts,
        'timeout_rate': timeouts / total if total > 0 else 0,
        'parse_fails': parse_fails,
        'parse_fail_rate': parse_fails / total if total > 0 else 0,
        'median_latency_ms': median_lat,
        'median_energy_mwh': median_energy,
        'energy_available': len(energy_values) > 0
    }

def main():
    # Input files
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "outputs"

    configs = {
        'v1_a1_grammar': output_dir / 'v1_a1_grammar.csv',
        'v1_a1_nogrammar': output_dir / 'v1_a1_nogrammar.csv',
        'v1_a2_grammar': output_dir / 'v1_a2_grammar.csv',
        'v1_a2_nogrammar': output_dir / 'v1_a2_nogrammar.csv',
    }

    # Check all files exist
    missing = [name for name, path in configs.items() if not path.exists()]
    if missing:
        print(f"ERROR: Missing baseline files: {missing}")
        print("\nRun the baseline matrix first:")
        print("  bash scripts/run_baseline_matrix.sh")
        sys.exit(1)

    # Load all trials
    all_trials = {}
    for name, path in configs.items():
        all_trials[name] = load_trials(path)
        print(f"Loaded {name}: {len(all_trials[name])} trials")

    print("\nGenerating summary reports...")

    # ========================================================================
    # OVERALL METRICS BY SYSTEM
    # ========================================================================

    system_metrics = {}
    for name, trials in all_trials.items():
        system_metrics[name] = compute_metrics(trials)

    # ========================================================================
    # BY-CATEGORY METRICS
    # ========================================================================

    category_metrics = defaultdict(lambda: defaultdict(list))
    for name, trials in all_trials.items():
        for t in trials:
            cat = t['category']
            category_metrics[name][cat].append(t)

    category_results = {}
    for name in configs.keys():
        category_results[name] = {}
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            cat_trials = category_metrics[name].get(cat, [])
            category_results[name][cat] = compute_metrics(cat_trials)

    # ========================================================================
    # FACTOR ROLLUPS
    # ========================================================================

    # A1 vs A2 (collapse over grammar)
    a1_trials = all_trials['v1_a1_grammar'] + all_trials['v1_a1_nogrammar']
    a2_trials = all_trials['v1_a2_grammar'] + all_trials['v1_a2_nogrammar']

    a1_metrics = compute_metrics(a1_trials)
    a2_metrics = compute_metrics(a2_trials)

    # Grammar vs No-Grammar (collapse over action)
    grammar_trials = all_trials['v1_a1_grammar'] + all_trials['v1_a2_grammar']
    nogrammar_trials = all_trials['v1_a1_nogrammar'] + all_trials['v1_a2_nogrammar']

    grammar_metrics = compute_metrics(grammar_trials)
    nogrammar_metrics = compute_metrics(nogrammar_trials)

    # ========================================================================
    # GENERATE SUMMARY MARKDOWN
    # ========================================================================

    summary_md = output_dir / 't1_baseline_summary.md'
    with open(summary_md, 'w') as f:
        f.write("# V1 Baseline Matrix Summary - Tier 1 Mini\n\n")
        f.write("**Generated**: Automatically from baseline CSV files\n\n")
        f.write("## Overall Results by System\n\n")
        f.write("| System | Accuracy | Median Latency | Median Energy | Timeout Rate | Parse Fail Rate |\n")
        f.write("|--------|----------|----------------|---------------|--------------|------------------|\n")

        for name in ['v1_a1_grammar', 'v1_a1_nogrammar', 'v1_a2_grammar', 'v1_a2_nogrammar']:
            m = system_metrics[name]
            energy_str = f"{m['median_energy_mwh']:.2f} mWh" if m['energy_available'] else "NA"
            f.write(f"| {name} | {m['accuracy']*100:.1f}% ({m['correct']}/{m['total']}) | "
                   f"{m['median_latency_ms']:.0f} ms | {energy_str} | "
                   f"{m['timeout_rate']*100:.1f}% | {m['parse_fail_rate']*100:.1f}% |\n")

        f.write("\n## Factor Main Effects\n\n")
        f.write("### Action Effect (A1 vs A2)\n\n")
        f.write("| Factor | Accuracy | Median Latency | Timeout Rate |\n")
        f.write("|--------|----------|----------------|---------------|\n")
        f.write(f"| **A1 (12 tok)** | {a1_metrics['accuracy']*100:.1f}% | {a1_metrics['median_latency_ms']:.0f} ms | {a1_metrics['timeout_rate']*100:.1f}% |\n")
        f.write(f"| **A2 (30 tok)** | {a2_metrics['accuracy']*100:.1f}% | {a2_metrics['median_latency_ms']:.0f} ms | {a2_metrics['timeout_rate']*100:.1f}% |\n")

        f.write("\n### Grammar Effect\n\n")
        f.write("| Factor | Accuracy | Median Latency | Parse Fail Rate |\n")
        f.write("|--------|----------|----------------|------------------|\n")
        f.write(f"| **Grammar** | {grammar_metrics['accuracy']*100:.1f}% | {grammar_metrics['median_latency_ms']:.0f} ms | {grammar_metrics['parse_fail_rate']*100:.1f}% |\n")
        f.write(f"| **No-Grammar** | {nogrammar_metrics['accuracy']*100:.1f}% | {nogrammar_metrics['median_latency_ms']:.0f} ms | {nogrammar_metrics['parse_fail_rate']*100:.1f}% |\n")

        f.write("\n## Results by Category\n\n")

        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            cat_names = {
                'AR': 'Arithmetic',
                'ALG': 'Algebra',
                'WP': 'Word Problems',
                'LOG': 'Logical Entailment'
            }
            f.write(f"### {cat_names[cat]} ({cat})\n\n")
            f.write("| System | Accuracy | Median Latency |\n")
            f.write("|--------|----------|----------------|\n")

            for name in ['v1_a1_grammar', 'v1_a1_nogrammar', 'v1_a2_grammar', 'v1_a2_nogrammar']:
                m = category_results[name][cat]
                if m:
                    f.write(f"| {name} | {m['accuracy']*100:.1f}% ({m['correct']}/{m['total']}) | {m['median_latency_ms']:.0f} ms |\n")
            f.write("\n")

        f.write("## Key Findings\n\n")
        f.write("### Best Overall System\n\n")

        best_system = max(system_metrics.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"- **{best_system[0]}**: {best_system[1]['accuracy']*100:.1f}% accuracy\n\n")

        f.write("### Main Effects\n\n")
        acc_diff = (a2_metrics['accuracy'] - a1_metrics['accuracy']) * 100
        f.write(f"- **A1 vs A2**: A2 is {acc_diff:+.1f} percentage points vs A1\n")

        grammar_diff = (grammar_metrics['accuracy'] - nogrammar_metrics['accuracy']) * 100
        f.write(f"- **Grammar vs No-Grammar**: Grammar is {grammar_diff:+.1f} percentage points vs no-grammar\n\n")

        f.write("### Category Performance\n\n")
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            best_for_cat = max(
                [(name, category_results[name][cat]) for name in configs.keys()],
                key=lambda x: x[1]['accuracy'] if x[1] else 0
            )
            cat_names = {'AR': 'AR (Arithmetic)', 'ALG': 'ALG (Algebra)', 'WP': 'WP (Word Problems)', 'LOG': 'LOG (Logic)'}
            f.write(f"- **{cat_names[cat]}**: Best = {best_for_cat[0]} at {best_for_cat[1]['accuracy']*100:.1f}%\n")

        f.write("\n---\n\n")
        f.write("**Next Step**: Review this summary and create `outputs/v1_decisions_for_hybrid.md` to define routing rules.\n")

    print(f"✅ Summary written to: {summary_md}")

    # ========================================================================
    # GENERATE BY-CATEGORY CSV
    # ========================================================================

    category_csv = output_dir / 't1_baseline_by_category.csv'
    with open(category_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'system', 'category', 'accuracy', 'correct', 'total',
            'median_latency_ms', 'timeout_rate', 'parse_fail_rate'
        ])
        writer.writeheader()

        for name in ['v1_a1_grammar', 'v1_a1_nogrammar', 'v1_a2_grammar', 'v1_a2_nogrammar']:
            for cat in ['AR', 'ALG', 'WP', 'LOG']:
                m = category_results[name][cat]
                if m:
                    writer.writerow({
                        'system': name,
                        'category': cat,
                        'accuracy': f"{m['accuracy']:.3f}",
                        'correct': m['correct'],
                        'total': m['total'],
                        'median_latency_ms': f"{m['median_latency_ms']:.1f}",
                        'timeout_rate': f"{m['timeout_rate']:.3f}",
                        'parse_fail_rate': f"{m['parse_fail_rate']:.3f}"
                    })

    print(f"✅ Category breakdown written to: {category_csv}")
    print("\n" + "="*60)
    print("SUMMARY GENERATION COMPLETE")
    print("="*60)
    print(f"\nReview outputs:")
    print(f"  1. {summary_md}")
    print(f"  2. {category_csv}")
    print("\nNext: Create routing decisions file:")
    print("  outputs/v1_decisions_for_hybrid.md")

if __name__ == "__main__":
    main()
