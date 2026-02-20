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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="outputs/official", help="Directory with baseline CSVs")
    args = ap.parse_args()

    output_dir = Path(args.out_dir)

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
        f.write("# V1 Baseline Matrix Summary - Official Tier 1\n\n")
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

    # ========================================================================
    # GENERATE v1_decisions_for_hybrid.md
    # ========================================================================

    decisions_md = output_dir / 'v1_decisions_for_hybrid.md'
    with open(decisions_md, 'w') as f:
        f.write("# V1 Routing Decisions for Hybrid System\n\n")
        f.write("**Auto-generated from T1 baseline matrix results.**\n\n")

        # Pick best baseline for main comparison
        best_name, best_m = max(system_metrics.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"## Main Comparison Baseline\n\n")
        f.write(f"- **Selected**: `{best_name}` ({best_m['accuracy']*100:.1f}% accuracy, "
                f"{best_m['median_latency_ms']:.0f} ms median latency)\n\n")

        # Per-category decisions
        f.write("## Per-Category Routing Decisions\n\n")
        f.write("| Category | Primary Action | Grammar | Fallback | Rationale |\n")
        f.write("|----------|---------------|---------|----------|----------|\n")

        # For each category, find best baseline config and derive decisions
        cat_decisions = {}
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            best_for_cat = max(
                [(name, category_results[name][cat]) for name in configs.keys()],
                key=lambda x: x[1]['accuracy'] if x[1] else 0
            )
            cat_best_name = best_for_cat[0]
            cat_best_m = best_for_cat[1]

            # Derive action, grammar from best baseline name
            uses_a1 = 'a1' in cat_best_name
            uses_grammar = 'grammar' in cat_best_name and 'nogrammar' not in cat_best_name

            if cat == 'AR':
                action = 'A5'
                grammar = False
                fallback = 'A1 → A2'
                rationale = (f"Symbolic direct (A5) for speed; baseline best={cat_best_name} "
                            f"({cat_best_m['accuracy']*100:.0f}%)")
            elif cat == 'ALG':
                action = 'A1' if uses_a1 else 'A2'
                grammar = uses_grammar
                fallback = 'A2' if action == 'A1' else 'None'
                rationale = f"Neural; baseline best={cat_best_name} ({cat_best_m['accuracy']*100:.0f}%)"
            elif cat == 'WP':
                action = 'A1' if uses_a1 else 'A2'
                grammar = uses_grammar
                fallback = 'A2' if action == 'A1' else 'None'
                rationale = f"Neural; baseline best={cat_best_name} ({cat_best_m['accuracy']*100:.0f}%)"
            else:  # LOG
                action = 'A1'
                grammar = uses_grammar
                fallback = 'A2'
                rationale = f"Fast yes/no; baseline best={cat_best_name} ({cat_best_m['accuracy']*100:.0f}%)"

            cat_decisions[cat] = {
                'action': action,
                'grammar_enabled': grammar,
                'fallback': fallback,
            }

            f.write(f"| {cat} | {action} | {'Yes' if grammar else 'No'} | {fallback} | {rationale} |\n")

        f.write("\n## Routing Map (YAML format for hybrid runner)\n\n")
        f.write("```yaml\n")
        f.write("category_routes:\n")
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            d = cat_decisions[cat]
            f.write(f"  {cat}:\n")
            f.write(f"    action: {d['action']}\n")
            f.write(f"    grammar_enabled: {str(d['grammar_enabled']).lower()}\n")
        f.write("max_escalations: 1\n")
        f.write("```\n\n")

        f.write("## Grammar Policy\n\n")
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            d = cat_decisions[cat]
            g_str = "grammar-constrained" if d['grammar_enabled'] else "no grammar"
            f.write(f"- **{cat}**: {g_str}\n")

        f.write("\n## Fallback Order\n\n")
        f.write("- **AR**: A5 (symbolic) → A1 (neural 12-tok) → A2 (neural 30-tok)\n")
        f.write("- **ALG**: A1 (neural) → A2 (extended neural)\n")
        f.write("- **WP**: A1 (neural) → A2 (extended neural)\n")
        f.write("- **LOG**: A1 (strict yes/no) → A2 (extended)\n")

    print(f"✅ Decisions written to: {decisions_md}")

    print("\n" + "="*60)
    print("SUMMARY GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  1. {summary_md}")
    print(f"  2. {category_csv}")
    print(f"  3. {decisions_md}")
    print(f"\nNext: Run Hybrid V1:")
    print(f"  python3 src/run_hybrid_v1.py --config configs/run_tier1.yaml \\")
    print(f"    --csv data/splits/industry_tier1_40.csv --split_role official \\")
    print(f"    --out_trials {output_dir}/hybrid_v1.csv")

if __name__ == "__main__":
    main()
