#!/usr/bin/env python3
"""
Generate Quick Summary from Trials CSV
Validates data quality and produces human-readable report
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def validate_columns(df, required_cols):
    """Check required columns exist"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ ERROR: Missing required columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    print(f"✓ All required columns present")

def compute_energy_delta(df):
    """Compute energy delta if missing"""
    if 'energy_delta_mwh' not in df.columns:
        if 'energy_start_mwh' in df.columns and 'energy_end_mwh' in df.columns:
            df['energy_delta_mwh'] = pd.to_numeric(df['energy_end_mwh'], errors='coerce') - \
                                     pd.to_numeric(df['energy_start_mwh'], errors='coerce')
            print("✓ Computed energy_delta_mwh from start/end")
        else:
            df['energy_delta_mwh'] = float('nan')
            print("⚠️  No energy delta available")

    # Compute energy in Joules
    if 'energy_delta_j' not in df.columns:
        df['energy_delta_j'] = pd.to_numeric(df['energy_delta_mwh'], errors='coerce') * 3.6

    return df

def check_data_quality(df):
    """Flag suspicious data"""
    warnings = []

    # Check for negative energy
    if 'energy_delta_mwh' in df.columns:
        neg_energy = df[pd.to_numeric(df['energy_delta_mwh'], errors='coerce') < 0]
        if len(neg_energy) > 0:
            warnings.append(f"⚠️  {len(neg_energy)} rows with negative energy_delta_mwh")

    # Check for zero energy on long runs
    if 'energy_delta_mwh' in df.columns:
        zero_energy = df[pd.to_numeric(df['energy_delta_mwh'], errors='coerce') == 0]
        if len(zero_energy) > 0:
            warnings.append(f"⚠️  {len(zero_energy)} rows with zero energy (meter refresh issue?)")

    # Check for impossible latency
    if 'latency_ms_total' in df.columns:
        bad_lat = df[pd.to_numeric(df['latency_ms_total'], errors='coerce') <= 0]
        if len(bad_lat) > 0:
            warnings.append(f"⚠️  {len(bad_lat)} rows with latency <= 0")

    # Check for small sample sizes
    system_counts = df.groupby('system').size()
    for system, count in system_counts.items():
        if count < 10:
            warnings.append(f"⚠️  System '{system}' has only {count} samples (too small for stats)")

    return warnings

def compute_system_metrics(df):
    """Aggregate by system"""
    metrics = []

    for system in df['system'].unique():
        sys_df = df[df['system'] == system]

        # Convert to numeric for calculations
        correct = pd.to_numeric(sys_df['correct'], errors='coerce')
        parse_success = pd.to_numeric(sys_df['parse_success'], errors='coerce')
        timeout_flag = pd.to_numeric(sys_df['timeout_flag'], errors='coerce')
        latency = pd.to_numeric(sys_df['latency_ms_total'], errors='coerce')
        energy = pd.to_numeric(sys_df.get('energy_per_prompt_mwh', sys_df.get('energy_delta_mwh', [])), errors='coerce')

        # Filter valid latencies (non-timeout, non-null)
        valid_latency = latency[(timeout_flag == 0) & latency.notna()]
        valid_energy = energy[energy.notna() & (energy > 0)]

        metrics.append({
            'system': system,
            'n_runs': len(sys_df),
            'n_unique_prompts': sys_df['prompt_id'].nunique(),
            'accuracy': correct.mean(),
            'parse_failure_rate': 1 - parse_success.mean(),
            'timeout_rate': timeout_flag.mean(),
            'median_latency_ms': valid_latency.median() if len(valid_latency) > 0 else float('nan'),
            'p25_latency_ms': valid_latency.quantile(0.25) if len(valid_latency) > 0 else float('nan'),
            'p75_latency_ms': valid_latency.quantile(0.75) if len(valid_latency) > 0 else float('nan'),
            'median_energy_mwh': valid_energy.median() if len(valid_energy) > 0 else float('nan'),
            'p25_energy_mwh': valid_energy.quantile(0.25) if len(valid_energy) > 0 else float('nan'),
            'p75_energy_mwh': valid_energy.quantile(0.75) if len(valid_energy) > 0 else float('nan'),
        })

    return pd.DataFrame(metrics)

def compute_category_metrics(df):
    """Aggregate by system × category"""
    metrics = []

    for system in df['system'].unique():
        for category in df['category'].unique():
            cat_df = df[(df['system'] == system) & (df['category'] == category)]

            if len(cat_df) == 0:
                continue

            correct = pd.to_numeric(cat_df['correct'], errors='coerce')
            timeout_flag = pd.to_numeric(cat_df['timeout_flag'], errors='coerce')

            metrics.append({
                'system': system,
                'category': category,
                'n_runs': len(cat_df),
                'accuracy': correct.mean(),
                'timeout_rate': timeout_flag.mean(),
            })

    return pd.DataFrame(metrics)

def write_summary(df, system_metrics, category_metrics, warnings, output_path):
    """Write human-readable markdown report"""

    with open(output_path, 'w') as f:
        f.write("# Tier-1 Mini Quick Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Run Coverage
        f.write("## 1. Run Coverage\n\n")
        f.write(f"- **Total rows:** {len(df)}\n")
        f.write(f"- **Systems:** {', '.join(df['system'].unique())}\n")
        f.write(f"- **Categories:** {', '.join(df['category'].unique())}\n")
        f.write(f"- **Unique prompts:** {df['prompt_id'].nunique()}\n\n")

        # System-Level Metrics
        f.write("## 2. System-Level Metrics\n\n")
        f.write("| System | N | Accuracy | Timeout Rate | Parse Fail Rate | Median Latency (ms) | Median Energy (mWh) |\n")
        f.write("|--------|---|----------|--------------|-----------------|---------------------|---------------------|\n")

        for _, row in system_metrics.iterrows():
            f.write(f"| {row['system']} | {row['n_runs']} | "
                   f"{row['accuracy']:.1%} | {row['timeout_rate']:.1%} | "
                   f"{row['parse_failure_rate']:.1%} | "
                   f"{row['median_latency_ms']:.0f} | "
                   f"{row['median_energy_mwh']:.2f} |\n")

        f.write("\n")

        # Category-Level Metrics
        f.write("## 3. Category-Level Metrics\n\n")
        f.write("| System | Category | N | Accuracy | Timeout Rate |\n")
        f.write("|--------|----------|---|----------|-------------|\n")

        for _, row in category_metrics.iterrows():
            f.write(f"| {row['system']} | {row['category']} | {row['n_runs']} | "
                   f"{row['accuracy']:.1%} | {row['timeout_rate']:.1%} |\n")

        f.write("\n")

        # Data Quality Warnings
        f.write("## 4. Data Quality Warnings\n\n")
        if warnings:
            for warning in warnings:
                f.write(f"- {warning}\n")
        else:
            f.write("✅ No data quality issues detected.\n")

        f.write("\n")

        # Takeaways
        f.write("## 5. Key Takeaways\n\n")

        # Compare systems if multiple
        if len(system_metrics) > 1:
            sys_sorted = system_metrics.sort_values('accuracy', ascending=False)
            best = sys_sorted.iloc[0]
            worst = sys_sorted.iloc[-1]

            f.write(f"- **Best system:** {best['system']} ({best['accuracy']:.1%} accuracy)\n")
            f.write(f"- **Worst system:** {worst['system']} ({worst['accuracy']:.1%} accuracy)\n")

            # Latency comparison
            if best['median_latency_ms'] < worst['median_latency_ms']:
                f.write(f"- **Latency:** {best['system']} is faster ({best['median_latency_ms']:.0f}ms vs {worst['median_latency_ms']:.0f}ms)\n")
            else:
                f.write(f"- **Latency:** {worst['system']} is faster ({worst['median_latency_ms']:.0f}ms vs {best['median_latency_ms']:.0f}ms)\n")

        # Category insights
        f.write("\n**Category Performance:**\n\n")
        cat_pivot = category_metrics.pivot(index='category', columns='system', values='accuracy')

        for category in cat_pivot.index:
            cat_accs = cat_pivot.loc[category]
            best_sys = cat_accs.idxmax()
            best_acc = cat_accs.max()
            f.write(f"- **{category}:** Best = {best_sys} ({best_acc:.1%})\n")

        f.write("\n")

        # Next Actions
        f.write("## 6. Next Actions\n\n")
        f.write("- [ ] Review category-specific failures\n")
        f.write("- [ ] Investigate timeout patterns\n")
        f.write("- [ ] Decide on baseline system for V2-V4 development\n")
        f.write("- [ ] Scale to full Tier-1 (80 prompts) if results look stable\n")

    print(f"\n✅ Summary written to: {output_path}")

def main():
    # Input/output paths
    trials_csv = "outputs/tier1_mini_grammar.csv"
    output_md = "outputs/tier1_quick_summary.md"

    # Check if multiple CSVs exist and combine them
    grammar_csv = Path("outputs/tier1_mini_grammar.csv")
    nogrammar_csv = Path("outputs/tier1_mini_nogrammar.csv")

    dfs = []
    if grammar_csv.exists():
        dfs.append(pd.read_csv(grammar_csv))
        print(f"✓ Loaded {grammar_csv}")
    if nogrammar_csv.exists():
        dfs.append(pd.read_csv(nogrammar_csv))
        print(f"✓ Loaded {nogrammar_csv}")

    if not dfs:
        print("❌ ERROR: No trial CSVs found in outputs/")
        sys.exit(1)

    # Combine all trials
    df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Combined {len(df)} total trials\n")

    # Required columns
    required = ['system', 'category', 'correct', 'parse_success', 'timeout_flag',
                'latency_ms_total', 'prompt_id']

    # Validate
    validate_columns(df, required)

    # Compute energy if needed
    df = compute_energy_delta(df)

    # Check data quality
    warnings = check_data_quality(df)

    # Compute metrics
    print("\nComputing metrics...")
    system_metrics = compute_system_metrics(df)
    category_metrics = compute_category_metrics(df)

    # Write summary
    write_summary(df, system_metrics, category_metrics, warnings, output_md)

    print(f"\n📊 SUMMARY:")
    print(system_metrics.to_string(index=False))

if __name__ == "__main__":
    main()
