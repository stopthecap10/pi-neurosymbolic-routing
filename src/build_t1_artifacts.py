#!/usr/bin/env python3
"""
T1 Artifact Pipeline
Generates all analysis artifacts from trial CSVs.

RULES:
- Raw trial CSVs in outputs/official/runs/ are IMMUTABLE (read-only inputs).
- This script does NOT modify any runner/router/scoring code.
- Only reads existing CSVs and generates derived artifacts.

Usage:
    python src/build_t1_artifacts.py --runs-dir outputs/official/runs/
    python src/build_t1_artifacts.py --runs-dir outputs/official/ --no-figures
    python src/build_t1_artifacts.py --figures-only
    python src/build_t1_artifacts.py --systems v1_a1_grammar hybrid_v1 hybrid_v2
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from pi_neuro_routing.analysis.artifacts import (
    CATEGORIES,
    CATEGORY_NAMES,
    discover_runs,
    load_trials_df,
    detect_system_type,
    validate_run,
    compute_prompt_medians,
    compute_category_summary,
    compute_system_metrics,
    extract_failure_cases,
    compute_error_taxonomy,
    build_system_comparison_table,
    build_prompt_outcome_matrix,
    format_markdown_summary,
    format_comparison_markdown,
    format_latex_table,
    generate_key_findings,
    find_best_case_studies,
    generate_figures,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="T1 Artifact Pipeline - generate all analysis artifacts from trial CSVs"
    )
    ap.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("outputs/official/runs"),
        help="Directory containing raw trial CSVs (default: outputs/official/runs/). "
             "Also scans outputs/official/ for flat CSVs.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/official"),
        help="Root output directory (default: outputs/official/)",
    )
    ap.add_argument(
        "--systems",
        nargs="*",
        default=None,
        help="Specific systems to process (default: all discovered)",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Canonical baseline for comparison (default: auto-detect best accuracy)",
    )
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures (skip summaries/comparisons)",
    )
    ap.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation (useful on headless Pi)",
    )
    ap.add_argument(
        "--expected-repeats",
        type=int,
        default=None,
        help="Expected number of repeats per prompt for QA (default: auto-detect)",
    )
    ap.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=None,
        help="Dataset CSV with ground truth for QA validation "
             "(e.g. data/splits/industry_tier1_40_v2.csv)",
    )
    return ap.parse_args()


# ── Helpers ────────────────────────────────────────────────

def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _auto_detect_baseline(system_metrics):
    """Pick the best-accuracy baseline system."""
    baselines = {k: v for k, v in system_metrics.items() if k.startswith("v1_a")}
    if not baselines:
        return next(iter(system_metrics))
    return max(baselines, key=lambda k: baselines[k]["accuracy"])


def _build_run_manifest(system_name, df, csv_path, qa_report):
    """Build JSON manifest for a run file with provenance and frozen config snapshot."""
    # Config snapshot from first row
    config_cols = ["config_version", "prompt_template_version", "parser_version",
                   "timeout_sec", "max_tokens_budget", "grammar_enabled",
                   "model_name", "quantization", "temperature", "top_p", "seed"]
    first_row = df.iloc[0]
    config_snapshot = {}
    for col in config_cols:
        val = first_row.get(col, "unknown")
        if pd.isna(val) or val == "":
            val = "unknown"
        config_snapshot[col] = str(val)

    # Config hash for reproducibility
    config_str = "|".join(config_snapshot.get(c, "") for c in config_cols)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    return {
        "system": system_name,
        "run_id": str(first_row.get("run_id", "")),
        "source_csv": str(csv_path),
        "source_csv_hash_sha256": _file_sha256(csv_path),
        "generated_at": datetime.now().isoformat(),
        "row_count": len(df),
        "detected_columns": list(df.columns),
        "config_hash": config_hash,
        "config_snapshot": config_snapshot,
        "system_type": detect_system_type(df),
        "qa_passed": qa_report["passed"],
        "qa_notes": qa_report["notes"],
        "metrics_snapshot": {
            "accuracy": float(df["correct"].mean()),
            "n_timeouts": int(df["timeout_flag"].sum()),
            "n_parse_fails": int((1 - df["parse_success"]).sum()),
        },
    }


def _build_infra_report(df, system_name):
    """Build infrastructure stability metrics."""
    total = len(df)
    timeouts = int(df["timeout_flag"].sum())
    parse_fails = int((1 - df["parse_success"]).sum())
    missing_raw = int(df["answer_raw"].isna().sum() + (df["answer_raw"].astype(str).str.strip() == "").sum()) if "answer_raw" in df.columns else 0

    # Successful latencies
    successful = df[df["timeout_flag"] == 0]["total_latency_ms"].dropna()
    median_lat = float(successful.median()) if len(successful) > 0 else np.nan

    # Long-tail latencies
    long_20s = int((successful > 20000).sum()) if len(successful) > 0 else 0
    long_40s = int((successful > 40000).sum()) if len(successful) > 0 else 0

    return {
        "system": system_name,
        "total_trials": total,
        "timeout_count": timeouts,
        "timeout_rate_pct": round(timeouts / total * 100, 1) if total > 0 else 0,
        "parse_fail_count": parse_fails,
        "parse_fail_rate_pct": round(parse_fails / total * 100, 1) if total > 0 else 0,
        "missing_answer_raw": missing_raw,
        "median_latency_ms_successful": round(median_lat, 1) if not np.isnan(median_lat) else "NA",
        "latency_gt_20s_count": long_20s,
        "latency_gt_40s_count": long_40s,
    }


def _format_infra_markdown(infra_report):
    """Format infra report as markdown."""
    r = infra_report
    lines = [
        f"# Infrastructure Stability Report: {r['system']}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Trials | {r['total_trials']} |",
        f"| Timeout Rate | {r['timeout_rate_pct']}% ({r['timeout_count']}) |",
        f"| Parse Fail Rate | {r['parse_fail_rate_pct']}% ({r['parse_fail_count']}) |",
        f"| Missing answer_raw | {r['missing_answer_raw']} |",
        f"| Median Latency (successful) | {r['median_latency_ms_successful']} ms |",
        f"| Latencies > 20s | {r['latency_gt_20s_count']} |",
        f"| Latencies > 40s | {r['latency_gt_40s_count']} |",
        "",
    ]

    if r["timeout_count"] == 0 and r["parse_fail_count"] == 0:
        lines.append("**Status**: Clean run - zero infrastructure errors.")
    elif r["timeout_rate_pct"] > 10:
        lines.append(f"**WARNING**: High timeout rate ({r['timeout_rate_pct']}%). Check server stability.")
    elif r["parse_fail_rate_pct"] > 20:
        lines.append(f"**WARNING**: High parse fail rate ({r['parse_fail_rate_pct']}%). Check parser/grammar.")

    lines += ["", "---", ""]
    return "\n".join(lines)


def _build_prompt_characteristics(all_trials, prompt_medians_all):
    """Build prompt characteristics table with difficulty/length analysis."""
    # Collect from first available system for prompt text info
    rows = []
    first_system = next(iter(all_trials.keys()))
    first_df = all_trials[first_system]

    # Get unique prompts
    seen = set()
    for _, trial_row in first_df.iterrows():
        pid = trial_row["prompt_id"]
        if pid in seen:
            continue
        seen.add(pid)

        row = {
            "prompt_id": pid,
            "category": trial_row["category"],
        }

        # Prompt characteristics
        raw = str(trial_row.get("answer_raw", ""))
        gt = str(trial_row.get("ground_truth", ""))
        row["ground_truth"] = gt
        row["ground_truth_len"] = len(gt)

        # Per-system correctness and timeout
        for sys_name, pm in prompt_medians_all.items():
            match = pm[pm["prompt_id"] == pid]
            if len(match) > 0:
                row[f"{sys_name}_correct"] = int(match["majority_correct"].iloc[0])
                row[f"{sys_name}_timeout_rate"] = float(match["timeout_rate"].iloc[0])
            else:
                row[f"{sys_name}_correct"] = ""
                row[f"{sys_name}_timeout_rate"] = ""

        rows.append(row)

    return pd.DataFrame(rows).sort_values("prompt_id")


def _build_route_usage(all_trials):
    """Build route usage tables for hybrid systems."""
    route_by_system = []
    route_by_category = []

    for sys_name, df in all_trials.items():
        if "route_chosen" not in df.columns:
            continue
        if df["route_chosen"].astype(str).str.strip().eq("").all():
            continue

        # Route usage by system (overall)
        route_counts = df["route_chosen"].value_counts()
        for route, count in route_counts.items():
            if str(route).strip() == "":
                continue
            route_by_system.append({
                "system": sys_name,
                "route": route,
                "count": int(count),
                "pct": round(count / len(df) * 100, 1),
            })

        # Route usage by category
        for cat in CATEGORIES:
            cat_df = df[df["category"] == cat]
            if len(cat_df) == 0:
                continue
            cat_route_counts = cat_df["route_chosen"].value_counts()
            for route, count in cat_route_counts.items():
                if str(route).strip() == "":
                    continue
                route_by_category.append({
                    "system": sys_name,
                    "category": cat,
                    "route": route,
                    "count": int(count),
                    "pct": round(count / len(cat_df) * 100, 1),
                })

    return pd.DataFrame(route_by_system), pd.DataFrame(route_by_category)


def _build_route_examples(all_trials, prompt_medians_all):
    """Find example prompts showing routing decisions for poster."""
    rows = []
    for sys_name, df in all_trials.items():
        if "route_chosen" not in df.columns:
            continue
        if df["route_chosen"].astype(str).str.strip().eq("").all():
            continue

        # Get one example per route per category
        seen_combos = set()
        for _, trial_row in df.iterrows():
            route = str(trial_row.get("route_chosen", "")).strip()
            cat = trial_row["category"]
            combo = (sys_name, cat, route)
            if combo in seen_combos or route == "":
                continue
            seen_combos.add(combo)

            rows.append({
                "system": sys_name,
                "prompt_id": trial_row["prompt_id"],
                "category": cat,
                "route_chosen": route,
                "correct": int(trial_row["correct"]),
                "ground_truth": trial_row["ground_truth"],
                "answer_parsed": trial_row.get("answer_parsed", ""),
                "decision_reason": trial_row.get("decision_reason", ""),
            })

    return pd.DataFrame(rows)


def _generate_index(output_dir, run_paths, qa_reports, system_metrics_all):
    """Generate master INDEX.md file."""
    lines = [
        "# Artifact Pipeline Index",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Discovered Systems",
        "",
        "| System | Source CSV | QA Status | Accuracy |",
        "|--------|-----------|-----------|----------|",
    ]

    for sys_name in sorted(run_paths.keys()):
        csv_name = run_paths[sys_name].name
        qa_status = "PASS" if qa_reports.get(sys_name, {}).get("passed", False) else "FAIL"
        m = system_metrics_all.get(sys_name, {})
        acc = f"{m.get('accuracy', 0)*100:.1f}%" if m else "N/A"
        lines.append(f"| {sys_name} | {csv_name} | {qa_status} | {acc} |")

    lines += [
        "",
        "## Key Artifacts",
        "",
        "### Per-System",
        "",
    ]
    for sys_name in sorted(run_paths.keys()):
        lines.append(f"- **{sys_name}**")
        lines.append(f"  - Summary: `summaries/summary_{sys_name}.md`")
        lines.append(f"  - Category: `summaries/category_{sys_name}.csv`")
        lines.append(f"  - Infra: `summaries/infra_{sys_name}.md`")
        lines.append(f"  - Failures: `failcases/failures_{sys_name}.csv`")
        lines.append(f"  - Manifest: `manifests/manifest_{sys_name}.json`")

    lines += [
        "",
        "### Cross-System",
        "",
        "- Comparison table: `comparisons/system_comparison.csv`",
        "- Error taxonomy: `comparisons/error_taxonomy.csv`",
        "- Prompt outcome matrix: `comparisons/prompt_outcome_matrix.csv`",
        "- Full comparison report: `comparisons/full_comparison.md`",
        "- Infra stability: `comparisons/infra_stability.csv`",
        "- Route usage (by system): `comparisons/route_usage_by_system.csv`",
        "- Route usage (by category): `comparisons/route_usage_by_category.csv`",
        "",
        "### Poster Assets",
        "",
        "- LaTeX table: `tables/comparison_table.tex`",
        "- CSV table: `tables/comparison_table.csv`",
        "- Key findings: `poster/key_findings.md`",
        "- Case studies (wins): `poster/case_studies_wins.csv`",
        "- Case studies (failures): `poster/case_studies_failures.csv`",
        "- Prompt characteristics: `tables/prompt_characteristics.csv`",
        "",
        "### Figures",
        "",
        "- `figures/accuracy_vs_energy.png` / `.pdf`",
        "- `figures/accuracy_vs_latency.png` / `.pdf`",
        "- `figures/error_taxonomy.png` / `.pdf`",
        "- `figures/category_heatmap.png` / `.pdf`",
        "- `figures/latency_boxplots.png` / `.pdf`",
        "- `figures/energy_boxplots.png` / `.pdf`",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────

def main():
    args = parse_args()

    # ── Step 0: Create output directory structure ──────────
    subdirs = [
        "summaries", "comparisons", "failcases", "traces",
        "calibration", "figures", "tables", "manifests", "poster",
    ]
    for sd in subdirs:
        (args.output_dir / sd).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Discover and load runs ────────────────────
    print("=" * 60)
    print("T1 ARTIFACT PIPELINE")
    print("=" * 60)
    print(f"\nDiscovering trial CSVs in {args.runs_dir}...")

    run_paths = discover_runs(args.runs_dir)

    # Also scan output_dir itself for flat CSVs (backward compat)
    if args.runs_dir != args.output_dir:
        flat_paths = discover_runs(args.output_dir)
        for k, v in flat_paths.items():
            if k not in run_paths:
                run_paths[k] = v

    if args.systems:
        run_paths = {k: v for k, v in run_paths.items() if k in args.systems}

    if not run_paths:
        print(f"ERROR: No trial CSVs found in {args.runs_dir}")
        print("  Expected naming: {system}_{YYYYMMDD}_{HHMMSS}.csv or {system}.csv")
        sys.exit(1)

    print(f"Found {len(run_paths)} systems:")
    for name, path in sorted(run_paths.items()):
        print(f"  {name} -> {path.name}")

    all_trials = {}
    for system_name, csv_path in sorted(run_paths.items()):
        print(f"\n  Loading {system_name}...")
        df = load_trials_df(csv_path)
        all_trials[system_name] = df
        sys_type = detect_system_type(df)
        print(f"    {len(df)} rows, type={sys_type}")

    # ── Step 2: QA Validation ─────────────────────────────
    print("\n" + "-" * 40)
    print("QA VALIDATION")
    print("-" * 40)

    ground_truth_map = None
    if args.ground_truth_csv and args.ground_truth_csv.exists():
        gt_df = pd.read_csv(args.ground_truth_csv, dtype=str)
        # Try common column names
        gt_col = "ground_truth" if "ground_truth" in gt_df.columns else "expected_answer"
        if gt_col in gt_df.columns and "prompt_id" in gt_df.columns:
            ground_truth_map = dict(zip(gt_df["prompt_id"], gt_df[gt_col]))
            print(f"  Loaded ground truth from {args.ground_truth_csv.name} ({len(ground_truth_map)} entries)")

    qa_reports = {}
    for system_name, df in all_trials.items():
        qa = validate_run(
            df, system_name,
            expected_repeats=args.expected_repeats,
            ground_truth_map=ground_truth_map,
        )
        qa_reports[system_name] = qa
        status = "PASS" if qa["passed"] else "FAIL"
        print(f"  {system_name}: {status} ({qa['row_count']} rows)")
        for note in qa["notes"]:
            print(f"    - {note}")

    # Write QA reports
    for system_name, qa in qa_reports.items():
        qa_path = args.output_dir / "calibration" / f"qa_{system_name}.json"
        with open(qa_path, "w") as f:
            json.dump(qa, f, indent=2, default=str)

    if args.figures_only:
        _generate_figures_stage(args, all_trials)
        _print_done(args.output_dir)
        return

    # ── Step 3: Per-Run Artifacts ─────────────────────────
    print("\n" + "-" * 40)
    print("PER-RUN ARTIFACTS")
    print("-" * 40)

    prompt_medians_all = {}
    category_summaries_all = {}
    system_metrics_all = {}

    # Filter to only QA-passed systems
    passed_systems = {name: df for name, df in all_trials.items()
                      if qa_reports.get(name, {}).get("passed", False)}
    if not passed_systems:
        print("\n  WARNING: No systems passed QA. Nothing to process.")
        _print_done(args.output_dir)
        return

    skipped = set(all_trials.keys()) - set(passed_systems.keys())
    if skipped:
        print(f"\n  Skipping QA-failed systems: {', '.join(sorted(skipped))}")

    for system_name, df in passed_systems.items():
        csv_path = run_paths[system_name]
        print(f"\n  Processing {system_name}...")

        # 3a. Prompt medians
        pm = compute_prompt_medians(df)
        prompt_medians_all[system_name] = pm
        pm.to_csv(
            args.output_dir / "traces" / f"prompt_medians_{system_name}.csv",
            index=False,
        )
        print(f"    -> traces/prompt_medians_{system_name}.csv")

        # 3b. Category summary
        cs = compute_category_summary(df)
        category_summaries_all[system_name] = cs
        cs.to_csv(
            args.output_dir / "summaries" / f"category_{system_name}.csv",
            index=False,
        )
        print(f"    -> summaries/category_{system_name}.csv")

        # 3c. System metrics
        sm = compute_system_metrics(df)
        system_metrics_all[system_name] = sm

        # 3d. Markdown summary
        failures = extract_failure_cases(df)
        md = format_markdown_summary(
            system_name, sm, cs, qa_reports[system_name], len(failures),
        )
        md_path = args.output_dir / "summaries" / f"summary_{system_name}.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"    -> summaries/summary_{system_name}.md")

        # 3e. Failure cases
        if len(failures) > 0:
            failures.to_csv(
                args.output_dir / "failcases" / f"failures_{system_name}.csv",
                index=False,
            )
            print(f"    -> failcases/failures_{system_name}.csv ({len(failures)} failures)")

        # 3f. Run manifest (per run file, with provenance)
        manifest = _build_run_manifest(system_name, df, csv_path, qa_reports[system_name])
        manifest_path = args.output_dir / "manifests" / f"manifest_{system_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"    -> manifests/manifest_{system_name}.json")

        # 3g. Infra stability report
        infra = _build_infra_report(df, system_name)
        infra_md = _format_infra_markdown(infra)
        infra_path = args.output_dir / "summaries" / f"infra_{system_name}.md"
        with open(infra_path, "w") as f:
            f.write(infra_md)
        print(f"    -> summaries/infra_{system_name}.md")

    # ── Step 4: Cross-System Comparisons ──────────────────
    print("\n" + "-" * 40)
    print("CROSS-SYSTEM COMPARISONS")
    print("-" * 40)

    # 4a. System comparison table
    comp_table = build_system_comparison_table(system_metrics_all)
    comp_table.to_csv(
        args.output_dir / "comparisons" / "system_comparison.csv",
        index=False,
    )
    print("  -> comparisons/system_comparison.csv")

    # 4b. Error taxonomy
    error_tax = compute_error_taxonomy(passed_systems)
    error_tax.to_csv(
        args.output_dir / "comparisons" / "error_taxonomy.csv",
        index=False,
    )
    print("  -> comparisons/error_taxonomy.csv")

    # 4c. Prompt outcome matrix
    outcome = build_prompt_outcome_matrix(prompt_medians_all)
    outcome.to_csv(
        args.output_dir / "comparisons" / "prompt_outcome_matrix.csv",
        index=False,
    )
    print("  -> comparisons/prompt_outcome_matrix.csv")

    # 4d. Comparison markdown
    comp_md = format_comparison_markdown(comp_table, error_tax, outcome)
    with open(args.output_dir / "comparisons" / "full_comparison.md", "w") as f:
        f.write(comp_md)
    print("  -> comparisons/full_comparison.md")

    # 4e. Infra stability CSV (cross-system)
    infra_rows = []
    for system_name, df in passed_systems.items():
        infra_rows.append(_build_infra_report(df, system_name))
    pd.DataFrame(infra_rows).to_csv(
        args.output_dir / "comparisons" / "infra_stability.csv",
        index=False,
    )
    print("  -> comparisons/infra_stability.csv")

    # 4f. Route usage tables (hybrid systems)
    route_by_system, route_by_category = _build_route_usage(passed_systems)
    if len(route_by_system) > 0:
        route_by_system.to_csv(
            args.output_dir / "comparisons" / "route_usage_by_system.csv",
            index=False,
        )
        print("  -> comparisons/route_usage_by_system.csv")
    if len(route_by_category) > 0:
        route_by_category.to_csv(
            args.output_dir / "comparisons" / "route_usage_by_category.csv",
            index=False,
        )
        print("  -> comparisons/route_usage_by_category.csv")

    # ── Step 5: Poster Assets ─────────────────────────────
    print("\n" + "-" * 40)
    print("POSTER ASSETS")
    print("-" * 40)

    # 5a. LaTeX table
    latex = format_latex_table(comp_table)
    with open(args.output_dir / "tables" / "comparison_table.tex", "w") as f:
        f.write(latex)
    print("  -> tables/comparison_table.tex")

    # 5b. Clean CSV for poster tools
    comp_table.to_csv(
        args.output_dir / "tables" / "comparison_table.csv",
        index=False,
    )
    print("  -> tables/comparison_table.csv")

    # 5c. Key findings
    findings = generate_key_findings(system_metrics_all, category_summaries_all)
    with open(args.output_dir / "poster" / "key_findings.md", "w") as f:
        f.write("# Key Findings\n\n")
        for finding in findings:
            f.write(f"- {finding}\n")
    print("  -> poster/key_findings.md")

    # 5d. Case studies - split into wins and failures
    baseline_name = args.baseline or _auto_detect_baseline(system_metrics_all)
    case_wins = find_best_case_studies(prompt_medians_all, baseline_name)
    if len(case_wins) > 0:
        case_wins.to_csv(
            args.output_dir / "poster" / "case_studies_wins.csv",
            index=False,
        )
        print(f"  -> poster/case_studies_wins.csv ({len(case_wins)} examples)")

    # Failure cases: all systems wrong
    _write_failure_case_studies(prompt_medians_all, args.output_dir)

    # 5e. Route examples for poster
    route_examples = _build_route_examples(passed_systems, prompt_medians_all)
    if len(route_examples) > 0:
        route_examples.to_csv(
            args.output_dir / "poster" / "route_examples.csv",
            index=False,
        )
        print(f"  -> poster/route_examples.csv ({len(route_examples)} examples)")

    # 5f. Prompt characteristics table
    prompt_chars = _build_prompt_characteristics(passed_systems, prompt_medians_all)
    prompt_chars.to_csv(
        args.output_dir / "tables" / "prompt_characteristics.csv",
        index=False,
    )
    print("  -> tables/prompt_characteristics.csv")

    # ── Step 6: Figures ───────────────────────────────────
    if not args.no_figures:
        _generate_figures_stage(
            args, passed_systems, system_metrics_all,
            category_summaries_all, error_tax,
        )
    else:
        print("\nSkipping figures (--no-figures)")

    # ── Step 7: Master Index ──────────────────────────────
    index_md = _generate_index(args.output_dir, run_paths, qa_reports, system_metrics_all)
    with open(args.output_dir / "INDEX.md", "w") as f:
        f.write(index_md)
    print("\n  -> INDEX.md")

    _print_done(args.output_dir)


def _write_failure_case_studies(prompt_medians_all, output_dir):
    """Write failure case studies: prompts where all systems got wrong."""
    if not prompt_medians_all:
        return

    # Collect all prompt_ids
    all_pids = set()
    for pm in prompt_medians_all.values():
        all_pids.update(pm["prompt_id"].tolist())

    rows = []
    for pid in sorted(all_pids):
        all_wrong = True
        info = {"prompt_id": pid, "category": "", "ground_truth": ""}
        for sys_name, pm in prompt_medians_all.items():
            match = pm[pm["prompt_id"] == pid]
            if len(match) > 0:
                info["category"] = match["category"].iloc[0]
                info["ground_truth"] = match["ground_truth"].iloc[0]
                correct = int(match["majority_correct"].iloc[0])
                info[f"{sys_name}_correct"] = correct
                info[f"{sys_name}_answer"] = match["majority_answer"].iloc[0]
                if correct == 1:
                    all_wrong = False
            else:
                info[f"{sys_name}_correct"] = ""
                info[f"{sys_name}_answer"] = ""

        if all_wrong:
            info["case_type"] = "ALL_SYSTEMS_WRONG"
            rows.append(info)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "poster" / "case_studies_failures.csv", index=False)
        print(f"  -> poster/case_studies_failures.csv ({len(rows)} hard cases)")


def _generate_figures_stage(args, all_trials, system_metrics=None,
                            category_summaries=None, error_taxonomy=None):
    """Figure generation with graceful matplotlib import handling."""
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("\nWARNING: matplotlib not installed, skipping figures")
        print("  Install with: pip install 'pi-neuro-routing[viz]'")
        return

    print("\n" + "-" * 40)
    print("FIGURES")
    print("-" * 40)

    # Recompute metrics if not provided (for --figures-only mode)
    if system_metrics is None:
        system_metrics = {name: compute_system_metrics(df)
                         for name, df in all_trials.items()}
    if category_summaries is None:
        category_summaries = {name: compute_category_summary(df)
                              for name, df in all_trials.items()}
    if error_taxonomy is None:
        error_taxonomy = compute_error_taxonomy(all_trials)

    fig_dir = args.output_dir / "figures"
    generated = generate_figures(
        system_metrics, category_summaries, all_trials,
        error_taxonomy, fig_dir,
    )
    for p in generated:
        print(f"  -> figures/{p.name}")

    if not generated:
        print("  (no figures generated - possibly missing data)")


def _print_done(output_dir):
    """Print completion summary."""
    print("\n" + "=" * 60)
    print("ARTIFACT GENERATION COMPLETE")
    print("=" * 60)

    for subdir in ["summaries", "comparisons", "failcases", "traces",
                   "calibration", "figures", "tables", "manifests", "poster"]:
        d = output_dir / subdir
        if d.exists():
            files = list(d.iterdir())
            if files:
                print(f"\n  {subdir}/ ({len(files)} files)")
                for f in sorted(files)[:5]:
                    print(f"    {f.name}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")

    index_path = output_dir / "INDEX.md"
    if index_path.exists():
        print(f"\n  Master index: {index_path}")

    print()


if __name__ == "__main__":
    main()
