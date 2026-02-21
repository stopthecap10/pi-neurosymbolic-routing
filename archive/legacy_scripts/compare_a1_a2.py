#!/usr/bin/env python3
"""
Generate A1 vs A2 Comparison Summary
Analyzes controlled experiment results
"""

import pandas as pd
import sys
from pathlib import Path

def load_trials(*csv_paths):
    """Load and combine trial CSVs"""
    dfs = []
    for path in csv_paths:
        if Path(path).exists():
            dfs.append(pd.read_csv(path))

    if not dfs:
        print("ERROR: No trial CSVs found")
        sys.exit(1)

    return pd.concat(dfs, ignore_index=True)

def compute_metrics(df):
    """Compute metrics for each action type"""
    metrics = []

    for (category, action), group in df.groupby(['category', 'action_type']):
        correct = pd.to_numeric(group['correct'], errors='coerce')
        timeout = pd.to_numeric(group['timeout_flag'], errors='coerce')
        parse_success = pd.to_numeric(group['parse_success'], errors='coerce')
        latency = pd.to_numeric(group['latency_ms_total'], errors='coerce')
        energy = pd.to_numeric(group['energy_per_prompt_mwh'], errors='coerce')

        # Filter valid latencies and energy
        valid_latency = latency[(timeout == 0) & latency.notna()]
        valid_energy = energy[energy.notna() & (energy > 0)]

        metrics.append({
            'category': category,
            'action': action,
            'n_trials': len(group),
            'accuracy': correct.mean(),
            'timeout_rate': timeout.mean(),
            'parse_fail_rate': 1 - parse_success.mean(),
            'median_latency_ms': valid_latency.median() if len(valid_latency) > 0 else float('nan'),
            'p25_latency_ms': valid_latency.quantile(0.25) if len(valid_latency) > 0 else float('nan'),
            'p75_latency_ms': valid_latency.quantile(0.75) if len(valid_latency) > 0 else float('nan'),
            'median_energy_mwh': valid_energy.median() if len(valid_energy) > 0 else float('nan'),
        })

    return pd.DataFrame(metrics)

def make_decision(alg_metrics, wp_metrics, threshold_pp=3.0):
    """Make routing decisions based on results with quality checks"""
    decisions = {}

    # ALG decision
    alg_a1 = alg_metrics[alg_metrics['action'] == 'A1_short'].iloc[0] if len(alg_metrics[alg_metrics['action'] == 'A1_short']) > 0 else None
    alg_a2 = alg_metrics[alg_metrics['action'] == 'A2_extended'].iloc[0] if len(alg_metrics[alg_metrics['action'] == 'A2_extended']) > 0 else None

    if alg_a1 is not None and alg_a2 is not None:
        acc_diff = (alg_a2['accuracy'] - alg_a1['accuracy']) * 100
        timeout_diff = (alg_a2['timeout_rate'] - alg_a1['timeout_rate']) * 100
        parse_diff = (alg_a2['parse_fail_rate'] - alg_a1['parse_fail_rate']) * 100
        energy_diff = alg_a2['median_energy_mwh'] - alg_a1['median_energy_mwh']
        latency_diff = alg_a2['median_latency_ms'] - alg_a1['median_latency_ms']

        # Quality check: A2 shouldn't massively worsen reliability
        quality_ok = (timeout_diff < 10) and (parse_diff < 10)

        if acc_diff >= threshold_pp and quality_ok:
            decisions['ALG'] = {
                'route': 'A2_extended',
                'reason': f'A2 improves accuracy by {acc_diff:.1f}pp without quality degradation',
                'metrics': f'Δenergy={energy_diff:.1f}mWh, Δlatency={latency_diff:.0f}ms',
                'alt': 'A4_extract_solve (SymPy) if both weak'
            }
        elif alg_a1['accuracy'] < 0.5 and alg_a2['accuracy'] < 0.5:
            decisions['ALG'] = {
                'route': 'A4_extract_solve',
                'reason': f'Both A1 ({alg_a1["accuracy"]:.1%}) and A2 ({alg_a2["accuracy"]:.1%}) weak - use SymPy',
                'metrics': f'A2 costs +{energy_diff:.1f}mWh but doesn\'t solve the problem',
                'alt': 'A2_extended as fallback'
            }
        else:
            decisions['ALG'] = {
                'route': 'A1_short',
                'reason': f'A2 improvement ({acc_diff:.1f}pp) below threshold ({threshold_pp}pp)',
                'metrics': f'Δenergy={energy_diff:.1f}mWh not justified',
                'alt': 'A2_extended as escalation'
            }

    # WP decision
    wp_a1 = wp_metrics[wp_metrics['action'] == 'A1_short'].iloc[0] if len(wp_metrics[wp_metrics['action'] == 'A1_short']) > 0 else None
    wp_a2 = wp_metrics[wp_metrics['action'] == 'A2_extended'].iloc[0] if len(wp_metrics[wp_metrics['action'] == 'A2_extended']) > 0 else None

    if wp_a1 is not None and wp_a2 is not None:
        acc_diff = (wp_a2['accuracy'] - wp_a1['accuracy']) * 100
        timeout_diff = (wp_a2['timeout_rate'] - wp_a1['timeout_rate']) * 100
        parse_diff = (wp_a2['parse_fail_rate'] - wp_a1['parse_fail_rate']) * 100
        energy_diff = wp_a2['median_energy_mwh'] - wp_a1['median_energy_mwh']
        latency_diff = wp_a2['median_latency_ms'] - wp_a1['median_latency_ms']

        quality_ok = (timeout_diff < 10) and (parse_diff < 10)

        if acc_diff >= threshold_pp and quality_ok:
            decisions['WP'] = {
                'route': 'A2_extended',
                'reason': f'A2 improves accuracy by {acc_diff:.1f}pp without quality degradation',
                'metrics': f'Δenergy={energy_diff:.1f}mWh, Δlatency={latency_diff:.0f}ms',
                'alt': 'A1_short + A2 escalation'
            }
        else:
            decisions['WP'] = {
                'route': 'A1_short',
                'reason': f'A2 improvement ({acc_diff:.1f}pp) below threshold or quality degraded',
                'metrics': f'Δenergy={energy_diff:.1f}mWh, Δtimeout={timeout_diff:.1f}pp',
                'alt': 'A2_extended as escalation only'
            }

    return decisions

def write_summary(metrics_df, decisions, output_path):
    """Write markdown summary"""

    with open(output_path, 'w') as f:
        f.write("# A1 vs A2 Action Comparison Summary\n\n")
        f.write("**Controlled Experiment:** Only token budget varies (A1=12, A2=30)\n")
        f.write("**Everything else frozen:** temperature=0.0, seed=42, timeout=20s, same prompts, same model\n\n")
        f.write("---\n\n")

        # Metrics table
        f.write("## Results by Category and Action\n\n")
        f.write("| Category | Action | N | Accuracy | Timeout | Parse Fail | Median Latency (ms) | Median Energy (mWh) |\n")
        f.write("|----------|--------|---|----------|---------|------------|---------------------|-----------------|\n")

        for _, row in metrics_df.iterrows():
            f.write(f"| {row['category']} | {row['action']} | {row['n_trials']} | "
                   f"{row['accuracy']:.1%} | {row['timeout_rate']:.1%} | "
                   f"{row['parse_fail_rate']:.1%} | {row['median_latency_ms']:.0f} | "
                   f"{row['median_energy_mwh']:.2f} |\n")

        f.write("\n")

        # Decisions
        f.write("## Routing Decisions (≥3pp improvement threshold)\n\n")

        for category, decision in decisions.items():
            f.write(f"### {category}\n\n")
            f.write(f"**Route:** `{decision['route']}`\n\n")
            f.write(f"**Reason:** {decision['reason']}\n\n")
            f.write(f"**Metrics:** {decision['metrics']}\n\n")
            f.write(f"**Alternative:** {decision['alt']}\n\n")

        # Next steps
        f.write("## Next Steps for V2\n\n")
        f.write("Use these routing decisions in V2 implementation:\n\n")

        for category, decision in decisions.items():
            f.write(f"- **{category}:** `{decision['route']}`\n")

        f.write("\n")
        f.write("**Fallback policy:**\n")
        f.write("- If primary route fails → escalate to A2_extended\n")
        f.write("- If A2_extended fails → mark as final error\n")

def main():
    # Load all trials
    print("Loading trial CSVs...")

    csv_files = [
        "outputs/alg_a1.csv",
        "outputs/alg_a2.csv",
        "outputs/wp_a1.csv",
        "outputs/wp_a2.csv",
    ]

    df = load_trials(*csv_files)
    print(f"Loaded {len(df)} trials\n")

    # Compute metrics
    print("Computing metrics...")
    metrics_df = compute_metrics(df)

    # Split by category
    alg_metrics = metrics_df[metrics_df['category'] == 'ALG']
    wp_metrics = metrics_df[metrics_df['category'] == 'WP']

    # Make decisions
    print("Making routing decisions...\n")
    decisions = make_decision(alg_metrics, wp_metrics, threshold_pp=3.0)

    # Write markdown summary
    output_md = "outputs/a1_a2_comparison_summary.md"
    write_summary(metrics_df, decisions, output_md)
    print(f"✅ Summary written to: {output_md}\n")

    # Write CSV for easy import
    output_csv = "outputs/a1_a2_comparison_by_category.csv"
    metrics_df.to_csv(output_csv, index=False)
    print(f"✅ CSV written to: {output_csv}\n")

    # Print decisions
    print("📊 ROUTING DECISIONS:\n")
    for category, decision in decisions.items():
        print(f"{category}: {decision['route']}")
        print(f"  → {decision['reason']}\n")

if __name__ == "__main__":
    main()
