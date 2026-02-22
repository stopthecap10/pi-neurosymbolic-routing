#!/usr/bin/env python3
"""
Artifact computation functions for T1 pipeline.
All functions are pure: data in, data out. No CLI logic, no argparse.
"""

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────

CATEGORIES = ["AR", "ALG", "WP", "LOG"]
CATEGORY_NAMES = {
    "AR": "Arithmetic",
    "ALG": "Algebra",
    "WP": "Word Problems",
    "LOG": "Logical Entailment",
}
VALID_ERROR_CODES = {"E0", "E1", "E2", "E3", "E5", "E7", "E8"}

REQUIRED_COLUMNS = [
    "run_id", "prompt_id", "category", "correct", "parse_success",
    "total_latency_ms", "timeout_flag", "error_code", "ground_truth",
    "answer_parsed",
]
HYBRID_EXTRA_COLUMNS = [
    "route_chosen", "route_attempt_sequence", "escalations_count",
    "decision_reason", "final_answer_source", "router_version",
]
HYBRID_V2_EXTRA_COLUMNS = [
    "reasoning_mode", "symbolic_parse_success", "sympy_solve_success",
]

# Timestamp pattern in filenames: system_YYYYMMDD_HHMMSS.csv
_TIMESTAMP_RE = re.compile(r'^(.+)_(\d{8}_\d{6})\.csv$')


# ── Data Loading ───────────────────────────────────────────

def discover_runs(runs_dir: Path) -> Dict[str, Path]:
    """
    Scan runs_dir for trial CSVs.
    Returns dict mapping system_name -> Path to latest CSV for that system.
    Handles both timestamped (system_YYYYMMDD_HHMMSS.csv) and flat (system.csv) names.
    """
    candidates: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for f in sorted(runs_dir.glob("*.csv")):
        m = _TIMESTAMP_RE.match(f.name)
        if m:
            system_name = m.group(1)
            timestamp = m.group(2)
            candidates[system_name].append((timestamp, f))
        else:
            # Non-timestamped: use stem directly
            candidates[f.stem].append(("000000_000000", f))

    result = {}
    for system_name, entries in candidates.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        result[system_name] = entries[0][1]

    return result


def load_trials_df(csv_path: Path) -> pd.DataFrame:
    """
    Load a trial CSV into a DataFrame with proper type coercion.
    Handles 'NA' strings in energy columns, fills missing hybrid columns with defaults.
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Type coercion for numeric columns
    int_cols = ["correct", "parse_success", "timeout_flag", "grammar_enabled",
                "escalations_count", "symbolic_parse_success", "sympy_solve_success"]
    float_cols = ["total_latency_ms", "energy_start_mwh", "energy_end_mwh",
                  "energy_delta_mwh", "energy_per_prompt_mwh"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col in float_cols:
        if col in df.columns:
            # Replace 'NA' string with NaN before numeric conversion
            df[col] = df[col].replace("NA", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing hybrid columns with defaults
    for col in int_cols:
        if col not in df.columns:
            df[col] = 0
    for col in ["route_chosen", "route_attempt_sequence", "decision_reason",
                "final_answer_source", "router_version", "reasoning_mode"]:
        if col not in df.columns:
            df[col] = ""

    return df


def detect_system_type(df: pd.DataFrame, system_name: str = "") -> str:
    """Detect whether a DataFrame is baseline, hybrid_v1, hybrid_v2, or probe."""
    # Name-based detection first (most reliable)
    name_lower = system_name.lower()
    if "probe" in name_lower:
        return "probe"
    if name_lower.startswith("v1_a") and "hybrid" not in name_lower:
        return "baseline"

    # Column-based detection fallback
    cols = set(df.columns)
    has_sympy = "symbolic_parse_success" in cols or "sympy_solve_success" in cols
    has_route = "route_chosen" in cols and df["route_chosen"].notna().any() and (df["route_chosen"] != "").any()

    if has_sympy and has_route:
        return "hybrid_v2"
    if has_route:
        return "hybrid_v1"
    return "baseline"


# ── QA Validation ──────────────────────────────────────────

def validate_run(
    df: pd.DataFrame,
    system_name: str,
    expected_prompts: int = 40,
    expected_repeats: Optional[int] = None,
    ground_truth_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validate a trial CSV. Returns a QA report dict.
    Never raises — returns pass/fail status with notes.
    """
    notes = []
    passed = True

    # 1. Required columns
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        notes.append(f"Missing required columns: {missing_cols}")
        passed = False

    # 2. Row count
    if expected_repeats is not None:
        expected_rows = expected_prompts * expected_repeats
    else:
        # Auto-detect repeats
        if "prompt_id" in df.columns:
            repeat_counts = df.groupby("prompt_id").size()
            unique_repeats = repeat_counts.unique()
            if len(unique_repeats) == 1:
                expected_repeats = int(unique_repeats[0])
                expected_rows = expected_prompts * expected_repeats
                notes.append(f"Auto-detected {expected_repeats} repeats per prompt")
            else:
                expected_rows = None
                notes.append(f"Inconsistent repeat counts: {sorted(unique_repeats.tolist())}")
                passed = False
        else:
            expected_rows = None

    if expected_rows is not None and len(df) != expected_rows:
        notes.append(f"Row count mismatch: got {len(df)}, expected {expected_rows}")
        passed = False

    # 3. Valid error codes
    if "error_code" in df.columns:
        invalid_codes = set(df["error_code"].dropna().unique()) - VALID_ERROR_CODES
        if invalid_codes:
            notes.append(f"Invalid error codes: {invalid_codes}")
            passed = False

    # 4. Parse consistency
    parse_inconsistencies = 0
    if "parse_success" in df.columns and "answer_parsed" in df.columns:
        for _, row in df.iterrows():
            if int(row.get("parse_success", 0)) == 1:
                parsed = str(row.get("answer_parsed", "")).strip()
                if parsed == "" or parsed == "nan":
                    parse_inconsistencies += 1
        if parse_inconsistencies > 0:
            notes.append(f"Parse inconsistencies: {parse_inconsistencies} rows with parse_success=1 but empty answer")

    # 5. Category coverage
    category_counts = {}
    if "category" in df.columns:
        category_counts = df["category"].value_counts().to_dict()
        missing_cats = set(CATEGORIES) - set(category_counts.keys())
        if missing_cats:
            notes.append(f"Missing categories: {missing_cats}")
            passed = False

    # 6. Duplicate check
    duplicate_rows = 0
    if "prompt_id" in df.columns and "repeat_idx" in df.columns:
        dupes = df.duplicated(subset=["prompt_id", "repeat_idx"], keep=False)
        duplicate_rows = int(dupes.sum() - df.drop_duplicates(subset=["prompt_id", "repeat_idx"]).shape[0])
        # More precisely: count actual duplicates
        duplicate_rows = int(df.duplicated(subset=["prompt_id", "repeat_idx"]).sum())
        if duplicate_rows > 0:
            notes.append(f"Duplicate (prompt_id, repeat_idx) pairs: {duplicate_rows}")

    # 7. Ground truth check
    gt_mismatches = 0
    if ground_truth_map and "prompt_id" in df.columns and "ground_truth" in df.columns:
        for _, row in df.iterrows():
            pid = str(row.get("prompt_id", ""))
            gt = str(row.get("ground_truth", "")).strip()
            expected_gt = ground_truth_map.get(pid)
            if expected_gt is not None and gt != expected_gt:
                gt_mismatches += 1
        if gt_mismatches > 0:
            notes.append(f"Ground truth mismatches: {gt_mismatches}")
            passed = False

    if not notes:
        notes.append("All checks passed")

    return {
        "system": system_name,
        "passed": passed,
        "row_count": len(df),
        "expected_rows": expected_rows,
        "missing_columns": missing_cols,
        "invalid_error_codes": list(set(df["error_code"].dropna().unique()) - VALID_ERROR_CODES) if "error_code" in df.columns else [],
        "parse_inconsistencies": parse_inconsistencies,
        "ground_truth_mismatches": gt_mismatches,
        "duplicate_rows": duplicate_rows,
        "category_counts": category_counts,
        "notes": notes,
    }


# ── Per-Run Aggregation ───────────────────────────────────

def compute_prompt_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across repeats to get per-prompt medians.
    Returns DataFrame with one row per prompt_id.
    """
    rows = []
    for pid, group in df.groupby("prompt_id"):
        # Latency: median excluding timeouts
        non_timeout = group[group["timeout_flag"] == 0]
        med_lat = float(non_timeout["total_latency_ms"].median()) if len(non_timeout) > 0 else np.nan

        # Energy: median excluding NaN
        energy = group["energy_per_prompt_mwh"].dropna()
        med_energy = float(energy.median()) if len(energy) > 0 else np.nan

        # Majority vote correctness
        correct_counts = group["correct"].value_counts()
        majority_correct = 1 if correct_counts.get(1, 0) > correct_counts.get(0, 0) else 0

        # Majority answer (deterministic tie-break: sorted)
        answer_counts = group["answer_parsed"].astype(str).value_counts()
        # Filter out empty/nan
        answer_counts = answer_counts[~answer_counts.index.isin(["", "nan"])]
        if len(answer_counts) > 0:
            # Deterministic tie-break: highest count, then alphabetical
            max_count = answer_counts.max()
            top_answers = sorted(answer_counts[answer_counts == max_count].index)
            majority_answer = top_answers[0]
        else:
            majority_answer = ""

        # Error code distribution
        error_dist = group["error_code"].value_counts().to_dict()

        rows.append({
            "prompt_id": pid,
            "category": group["category"].iloc[0],
            "ground_truth": group["ground_truth"].iloc[0],
            "system": group["system"].iloc[0] if "system" in group.columns else "",
            "run_id": group["run_id"].iloc[0] if "run_id" in group.columns else "",
            "n_repeats": len(group),
            "median_latency_ms": med_lat,
            "median_energy_mwh": med_energy,
            "majority_correct": majority_correct,
            "majority_answer": majority_answer,
            "timeout_rate": float(group["timeout_flag"].mean()),
            "parse_fail_rate": float(1 - group["parse_success"].mean()),
            "error_codes": json.dumps(error_dist),
        })

    return pd.DataFrame(rows)


def compute_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-category summary metrics.
    Returns DataFrame with one row per category + an 'ALL' row.
    """
    rows = []

    for cat in CATEGORIES + ["ALL"]:
        subset = df if cat == "ALL" else df[df["category"] == cat]
        if len(subset) == 0:
            continue

        non_timeout = subset[subset["timeout_flag"] == 0]
        energy = subset["energy_per_prompt_mwh"].dropna()

        rows.append({
            "category": cat,
            "n_trials": len(subset),
            "n_correct": int(subset["correct"].sum()),
            "accuracy": float(subset["correct"].mean()),
            "median_latency_ms": float(non_timeout["total_latency_ms"].median()) if len(non_timeout) > 0 else np.nan,
            "median_energy_mwh": float(energy.median()) if len(energy) > 0 else np.nan,
            "timeout_rate": float(subset["timeout_flag"].mean()),
            "parse_fail_rate": float(1 - subset["parse_success"].mean()),
        })

    return pd.DataFrame(rows)


def compute_system_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Overall system metrics. Replicates pattern from summarize_v1_matrix.py.
    """
    total = len(df)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0, "timeouts": 0,
                "timeout_rate": 0, "parse_fails": 0, "parse_fail_rate": 0,
                "median_latency_ms": 0, "median_energy_mwh": None,
                "energy_available": False}

    correct = int(df["correct"].sum())
    timeouts = int(df["timeout_flag"].sum())
    parse_fails = int((1 - df["parse_success"]).sum())

    non_timeout = df[df["timeout_flag"] == 0]
    median_lat = float(non_timeout["total_latency_ms"].median()) if len(non_timeout) > 0 else 0

    energy = df["energy_per_prompt_mwh"].dropna()
    median_energy = float(energy.median()) if len(energy) > 0 else None

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "timeouts": timeouts,
        "timeout_rate": timeouts / total,
        "parse_fails": parse_fails,
        "parse_fail_rate": parse_fails / total,
        "median_latency_ms": median_lat,
        "median_energy_mwh": median_energy,
        "energy_available": len(energy) > 0,
    }


# ── Failure Analysis ──────────────────────────────────────

def extract_failure_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all trials with error_code != 'E0'."""
    failures = df[df["error_code"] != "E0"].copy()
    if len(failures) == 0:
        return pd.DataFrame()

    # Select relevant columns
    keep_cols = ["prompt_id", "category", "repeat_idx", "error_code",
                 "answer_parsed", "ground_truth", "total_latency_ms",
                 "timeout_flag", "parse_success"]

    # Add answer_raw (truncated)
    if "answer_raw" in failures.columns:
        failures["answer_raw_preview"] = failures["answer_raw"].astype(str).str[:200]
        keep_cols.append("answer_raw_preview")

    # Add hybrid columns if present
    for col in ["route_chosen", "escalations_count", "final_answer_source"]:
        if col in failures.columns:
            keep_cols.append(col)

    available_cols = [c for c in keep_cols if c in failures.columns]
    failures = failures[available_cols].sort_values(
        ["error_code", "category", "prompt_id"]
    )
    return failures


def compute_error_taxonomy(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Cross-system error code distribution.
    Rows: error codes. Columns: system names. Values: counts.
    """
    all_codes = sorted(VALID_ERROR_CODES)
    rows = []
    for code in all_codes:
        row = {"error_code": code}
        for system_name, df in dfs.items():
            if "error_code" in df.columns:
                row[system_name] = int((df["error_code"] == code).sum())
            else:
                row[system_name] = 0
        rows.append(row)
    return pd.DataFrame(rows)


# ── Cross-System Comparisons ──────────────────────────────

def build_system_comparison_table(
    system_metrics: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """All systems side-by-side."""
    rows = []
    for system_name, m in system_metrics.items():
        energy_str = f"{m['median_energy_mwh']:.2f}" if m.get("median_energy_mwh") is not None else "NA"
        rows.append({
            "system": system_name,
            "accuracy_pct": round(m["accuracy"] * 100, 1),
            "correct": m["correct"],
            "total": m["total"],
            "median_latency_ms": round(m["median_latency_ms"], 1),
            "median_energy_mwh": energy_str,
            "timeout_rate_pct": round(m["timeout_rate"] * 100, 1),
            "parse_fail_rate_pct": round(m["parse_fail_rate"] * 100, 1),
        })
    return pd.DataFrame(rows)


def build_prompt_outcome_matrix(
    prompt_medians: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Per-prompt correctness across all systems.
    Rows: 40 prompt_ids. Columns: prompt_id, category, ground_truth, then one per system.
    """
    # Collect all prompt_ids
    all_pids = set()
    for pm in prompt_medians.values():
        all_pids.update(pm["prompt_id"].tolist())

    # Build base from first available system
    first_pm = next(iter(prompt_medians.values()))
    base = first_pm[["prompt_id", "category", "ground_truth"]].copy()

    # Add system columns
    for system_name, pm in prompt_medians.items():
        pid_to_correct = dict(zip(pm["prompt_id"], pm["majority_correct"]))
        base[system_name] = base["prompt_id"].map(pid_to_correct).fillna(0).astype(int)

    # Add pids missing from base
    missing_pids = all_pids - set(base["prompt_id"])
    if missing_pids:
        for pid in sorted(missing_pids):
            row = {"prompt_id": pid, "category": "", "ground_truth": ""}
            for system_name, pm in prompt_medians.items():
                match = pm[pm["prompt_id"] == pid]
                if len(match) > 0:
                    row["category"] = match["category"].iloc[0]
                    row["ground_truth"] = match["ground_truth"].iloc[0]
                    row[system_name] = int(match["majority_correct"].iloc[0])
                else:
                    row[system_name] = 0
            base = pd.concat([base, pd.DataFrame([row])], ignore_index=True)

    return base.sort_values("prompt_id").reset_index(drop=True)


# ── Formatting / Output ───────────────────────────────────

def format_markdown_summary(
    system_name: str,
    system_metrics: Dict[str, Any],
    category_summary: pd.DataFrame,
    qa_report: Dict[str, Any],
    failure_count: int,
) -> str:
    """Generate a human-readable markdown report for a single run."""
    m = system_metrics
    energy_str = f"{m['median_energy_mwh']:.2f} mWh" if m.get("median_energy_mwh") is not None else "NA"

    lines = [
        f"# Run Summary: {system_name}",
        f"",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## QA Status",
        f"",
        f"- **Passed**: {'Yes' if qa_report['passed'] else 'NO'}",
        f"- **Row count**: {qa_report['row_count']}",
    ]
    for note in qa_report["notes"]:
        lines.append(f"- {note}")

    lines += [
        f"",
        f"## Overall Metrics",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {m['accuracy']*100:.1f}% ({m['correct']}/{m['total']}) |",
        f"| Median Latency | {m['median_latency_ms']:.1f} ms |",
        f"| Median Energy | {energy_str} |",
        f"| Timeout Rate | {m['timeout_rate']*100:.1f}% ({m['timeouts']}) |",
        f"| Parse Fail Rate | {m['parse_fail_rate']*100:.1f}% ({m['parse_fails']}) |",
        f"| Failure Cases | {failure_count} |",
        f"",
        f"## Results by Category",
        f"",
        f"| Category | Accuracy | Median Latency | Timeout Rate | Parse Fail Rate |",
        f"|----------|----------|----------------|--------------|-----------------|",
    ]

    for _, row in category_summary.iterrows():
        cat = row["category"]
        if cat == "ALL":
            continue
        cat_name = CATEGORY_NAMES.get(cat, cat)
        energy_cat = f"{row['median_energy_mwh']:.2f}" if pd.notna(row["median_energy_mwh"]) else "NA"
        lat = f"{row['median_latency_ms']:.1f}" if pd.notna(row["median_latency_ms"]) else "NA"
        lines.append(
            f"| {cat_name} ({cat}) | {row['accuracy']*100:.1f}% ({row['n_correct']}/{row['n_trials']}) "
            f"| {lat} ms | {row['timeout_rate']*100:.1f}% | {row['parse_fail_rate']*100:.1f}% |"
        )

    lines += ["", "---", ""]
    return "\n".join(lines)


def format_comparison_markdown(
    comparison_table: pd.DataFrame,
    error_taxonomy: pd.DataFrame,
    outcome_matrix: pd.DataFrame,
) -> str:
    """Generate cross-system comparison markdown."""
    lines = [
        "# Cross-System Comparison Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## System Comparison",
        "",
        "| System | Accuracy | Median Latency | Median Energy | Timeouts | Parse Fails |",
        "|--------|----------|----------------|---------------|----------|-------------|",
    ]

    for _, row in comparison_table.iterrows():
        lines.append(
            f"| {row['system']} | {row['accuracy_pct']}% ({row['correct']}/{row['total']}) "
            f"| {row['median_latency_ms']} ms | {row['median_energy_mwh']} mWh "
            f"| {row['timeout_rate_pct']}% | {row['parse_fail_rate_pct']}% |"
        )

    lines += [
        "",
        "## Error Taxonomy",
        "",
    ]

    # Build error taxonomy table
    system_cols = [c for c in error_taxonomy.columns if c != "error_code"]
    header = "| Error Code | " + " | ".join(system_cols) + " |"
    sep = "|------------|" + "|".join(["------" for _ in system_cols]) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in error_taxonomy.iterrows():
        vals = " | ".join(str(row[c]) for c in system_cols)
        lines.append(f"| {row['error_code']} | {vals} |")

    lines += [
        "",
        "## Prompt Outcome Matrix",
        "",
    ]

    # Build outcome matrix table
    system_cols = [c for c in outcome_matrix.columns
                   if c not in ("prompt_id", "category", "ground_truth")]
    header = "| Prompt | Category | " + " | ".join(system_cols) + " |"
    sep = "|--------|----------|" + "|".join(["---" for _ in system_cols]) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in outcome_matrix.iterrows():
        vals = " | ".join(str(row[c]) for c in system_cols)
        lines.append(f"| {row['prompt_id']} | {row['category']} | {vals} |")

    lines += ["", "---", ""]
    return "\n".join(lines)


def format_latex_table(comparison_table: pd.DataFrame) -> str:
    """Generate a LaTeX tabular for poster use with booktabs formatting."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{System Comparison -- Tier 1 Official Results}",
        r"\label{tab:system-comparison}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"System & Accuracy (\%) & Latency (ms) & Energy (mWh) & Timeouts (\%) & Parse Fails (\%) \\",
        r"\midrule",
    ]

    for _, row in comparison_table.iterrows():
        name = row["system"].replace("_", r"\_")
        lines.append(
            f"  {name} & {row['accuracy_pct']} & {row['median_latency_ms']} "
            f"& {row['median_energy_mwh']} & {row['timeout_rate_pct']} "
            f"& {row['parse_fail_rate_pct']} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def generate_key_findings(
    system_metrics: Dict[str, Dict[str, Any]],
    category_summaries: Dict[str, pd.DataFrame],
) -> List[str]:
    """Auto-generate bullet-point findings."""
    findings = []

    if not system_metrics:
        return ["No systems available for analysis."]

    # 1. Best overall system
    best_name = max(system_metrics, key=lambda k: system_metrics[k]["accuracy"])
    best_m = system_metrics[best_name]
    findings.append(
        f"Best overall system: **{best_name}** at {best_m['accuracy']*100:.1f}% accuracy "
        f"({best_m['correct']}/{best_m['total']})"
    )

    # 2. Best system per category
    for cat in CATEGORIES:
        best_cat_name = None
        best_cat_acc = -1
        for name, cs in category_summaries.items():
            cat_row = cs[cs["category"] == cat]
            if len(cat_row) > 0:
                acc = float(cat_row["accuracy"].iloc[0])
                if acc > best_cat_acc:
                    best_cat_acc = acc
                    best_cat_name = name
        if best_cat_name:
            findings.append(
                f"Best for {CATEGORY_NAMES[cat]} ({cat}): **{best_cat_name}** "
                f"at {best_cat_acc*100:.1f}%"
            )

    # 3. Hybrid vs best baseline delta
    baselines = {k: v for k, v in system_metrics.items() if k.startswith("v1_a")}
    hybrids = {k: v for k, v in system_metrics.items() if "hybrid" in k}

    if baselines and hybrids:
        best_baseline = max(baselines, key=lambda k: baselines[k]["accuracy"])
        best_hybrid = max(hybrids, key=lambda k: hybrids[k]["accuracy"])
        delta = (hybrids[best_hybrid]["accuracy"] - baselines[best_baseline]["accuracy"]) * 100
        findings.append(
            f"Best hybrid ({best_hybrid}) vs best baseline ({best_baseline}): "
            f"{delta:+.1f} percentage points accuracy"
        )

        # 4. Energy comparison
        bl_energy = baselines[best_baseline].get("median_energy_mwh")
        hy_energy = hybrids[best_hybrid].get("median_energy_mwh")
        if bl_energy is not None and hy_energy is not None:
            energy_delta = hy_energy - bl_energy
            findings.append(
                f"Energy: {best_hybrid} uses {hy_energy:.2f} mWh vs "
                f"{best_baseline} at {bl_energy:.2f} mWh ({energy_delta:+.2f} mWh)"
            )

        # 5. Latency comparison
        bl_lat = baselines[best_baseline]["median_latency_ms"]
        hy_lat = hybrids[best_hybrid]["median_latency_ms"]
        findings.append(
            f"Latency: {best_hybrid} at {hy_lat:.0f} ms vs "
            f"{best_baseline} at {bl_lat:.0f} ms"
        )

    # 6. Categories where hybrid helps most
    if baselines and hybrids:
        best_bl_cs = category_summaries.get(best_baseline)
        best_hy_cs = category_summaries.get(best_hybrid)
        if best_bl_cs is not None and best_hy_cs is not None:
            cat_deltas = []
            for cat in CATEGORIES:
                bl_row = best_bl_cs[best_bl_cs["category"] == cat]
                hy_row = best_hy_cs[best_hy_cs["category"] == cat]
                if len(bl_row) > 0 and len(hy_row) > 0:
                    d = float(hy_row["accuracy"].iloc[0]) - float(bl_row["accuracy"].iloc[0])
                    cat_deltas.append((cat, d * 100))
            if cat_deltas:
                cat_deltas.sort(key=lambda x: -x[1])
                best_cat, best_delta = cat_deltas[0]
                if best_delta > 0:
                    findings.append(
                        f"Largest category improvement: {CATEGORY_NAMES[best_cat]} ({best_cat}) "
                        f"at {best_delta:+.1f} pp"
                    )

    return findings


def find_best_case_studies(
    prompt_medians: Dict[str, pd.DataFrame],
    baseline_name: str = "v1_a1_nogrammar",
) -> pd.DataFrame:
    """
    Find examples where baseline got wrong but a hybrid got right.
    Returns DataFrame of candidate case studies.
    """
    if baseline_name not in prompt_medians:
        return pd.DataFrame()

    baseline_pm = prompt_medians[baseline_name]
    hybrid_systems = {k: v for k, v in prompt_medians.items() if "hybrid" in k}

    if not hybrid_systems:
        return pd.DataFrame()

    rows = []
    for _, bl_row in baseline_pm.iterrows():
        pid = bl_row["prompt_id"]
        if bl_row["majority_correct"] == 1:
            continue  # Baseline got it right, not interesting

        for hy_name, hy_pm in hybrid_systems.items():
            hy_match = hy_pm[hy_pm["prompt_id"] == pid]
            if len(hy_match) > 0 and int(hy_match["majority_correct"].iloc[0]) == 1:
                rows.append({
                    "prompt_id": pid,
                    "category": bl_row["category"],
                    "ground_truth": bl_row["ground_truth"],
                    "baseline_system": baseline_name,
                    "baseline_answer": bl_row["majority_answer"],
                    "hybrid_system": hy_name,
                    "hybrid_answer": hy_match["majority_answer"].iloc[0],
                    "case_type": "BASELINE_WRONG__HYBRID_CORRECT",
                })

    return pd.DataFrame(rows)


# ── Figure Generation ─────────────────────────────────────

# Poster style config
POSTER_STYLE = {
    "figure.figsize": (8, 5),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

SYSTEM_COLORS = {
    "v1_a1_nogrammar": "#1f77b4",
    "v1_a1_grammar": "#ff7f0e",
    "v1_a2_nogrammar": "#2ca02c",
    "v1_a2_grammar": "#d62728",
    "hybrid_v1": "#9467bd",
    "hybrid_v2": "#8c564b",
    "hybrid_v3": "#e377c2",
    "hybrid_v4": "#7f7f7f",
    "hybrid_v5": "#bcbd22",
}


def _get_color(system_name: str) -> str:
    """Get a consistent color for a system."""
    return SYSTEM_COLORS.get(system_name, "#333333")


def generate_figures(
    system_metrics: Dict[str, Dict[str, Any]],
    category_summaries: Dict[str, pd.DataFrame],
    all_trials: Dict[str, pd.DataFrame],
    error_taxonomy: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Generate all 6 figure types. Returns list of generated file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(POSTER_STYLE)

    generated = []
    generated.append(_fig_accuracy_vs_energy(system_metrics, output_dir, plt))
    generated.append(_fig_accuracy_vs_latency(system_metrics, output_dir, plt))
    generated.append(_fig_error_taxonomy_bar(error_taxonomy, output_dir, plt))
    generated.append(_fig_category_heatmap(category_summaries, output_dir, plt))
    generated.append(_fig_latency_boxplots(all_trials, output_dir, plt))
    generated.append(_fig_energy_boxplots(all_trials, output_dir, plt))

    return [p for p in generated if p is not None]


def _fig_accuracy_vs_energy(system_metrics, output_dir, plt) -> Optional[Path]:
    """Scatter plot: x=median_energy, y=accuracy, one point per system."""
    systems_with_energy = {
        k: v for k, v in system_metrics.items()
        if v.get("median_energy_mwh") is not None
    }
    if not systems_with_energy:
        return None

    fig, ax = plt.subplots()
    for name, m in systems_with_energy.items():
        ax.scatter(m["median_energy_mwh"], m["accuracy"] * 100,
                   color=_get_color(name), s=100, label=name, zorder=5)
    ax.set_xlabel("Median Energy (mWh)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Energy per System")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = output_dir / "accuracy_vs_energy.png"
    fig.savefig(path)
    fig.savefig(output_dir / "accuracy_vs_energy.pdf")
    plt.close(fig)
    return path


def _fig_accuracy_vs_latency(system_metrics, output_dir, plt) -> Optional[Path]:
    """Scatter plot: x=median_latency, y=accuracy, one point per system."""
    fig, ax = plt.subplots()
    for name, m in system_metrics.items():
        ax.scatter(m["median_latency_ms"], m["accuracy"] * 100,
                   color=_get_color(name), s=100, label=name, zorder=5)
    ax.set_xlabel("Median Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Latency per System")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    path = output_dir / "accuracy_vs_latency.png"
    fig.savefig(path)
    fig.savefig(output_dir / "accuracy_vs_latency.pdf")
    plt.close(fig)
    return path


def _fig_error_taxonomy_bar(error_taxonomy, output_dir, plt) -> Optional[Path]:
    """Grouped bar chart: x=error_code, y=count, grouped by system."""
    system_cols = [c for c in error_taxonomy.columns if c != "error_code"]
    if not system_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(error_taxonomy))
    width = 0.8 / len(system_cols)

    for i, sys_name in enumerate(system_cols):
        offset = (i - len(system_cols) / 2 + 0.5) * width
        ax.bar(x + offset, error_taxonomy[sys_name].values,
               width, label=sys_name, color=_get_color(sys_name))

    ax.set_xlabel("Error Code")
    ax.set_ylabel("Count")
    ax.set_title("Error Taxonomy by System")
    ax.set_xticks(x)
    ax.set_xticklabels(error_taxonomy["error_code"].values)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "error_taxonomy.png"
    fig.savefig(path)
    fig.savefig(output_dir / "error_taxonomy.pdf")
    plt.close(fig)
    return path


def _fig_category_heatmap(category_summaries, output_dir, plt) -> Optional[Path]:
    """Heatmap: rows=systems, cols=categories, values=accuracy."""
    if not category_summaries:
        return None

    systems = sorted(category_summaries.keys())
    matrix = []
    for sys_name in systems:
        cs = category_summaries[sys_name]
        row = []
        for cat in CATEGORIES:
            cat_row = cs[cs["category"] == cat]
            if len(cat_row) > 0:
                row.append(float(cat_row["accuracy"].iloc[0]) * 100)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(systems) * 0.6 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([CATEGORY_NAMES[c] for c in CATEGORIES])
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)

    # Annotate cells
    for i in range(len(systems)):
        for j in range(len(CATEGORIES)):
            val = matrix[i, j]
            color = "white" if val < 30 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    ax.set_title("Accuracy by System and Category (%)")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    path = output_dir / "category_heatmap.png"
    fig.savefig(path)
    fig.savefig(output_dir / "category_heatmap.pdf")
    plt.close(fig)
    return path


def _fig_latency_boxplots(all_trials, output_dir, plt) -> Optional[Path]:
    """Box plots: one box per system, y=total_latency_ms (excl timeouts)."""
    data = []
    labels = []
    for name in sorted(all_trials.keys()):
        df = all_trials[name]
        non_timeout = df[df["timeout_flag"] == 0]["total_latency_ms"].dropna()
        if len(non_timeout) > 0:
            data.append(non_timeout.values)
            labels.append(name)

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_get_color(labels[i]))
        patch.set_alpha(0.7)

    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution by System (excl. timeouts)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "latency_boxplots.png"
    fig.savefig(path)
    fig.savefig(output_dir / "latency_boxplots.pdf")
    plt.close(fig)
    return path


def _fig_energy_boxplots(all_trials, output_dir, plt) -> Optional[Path]:
    """Box plots: one box per system, y=energy_per_prompt_mwh (excl NaN)."""
    data = []
    labels = []
    for name in sorted(all_trials.keys()):
        df = all_trials[name]
        energy = df["energy_per_prompt_mwh"].dropna()
        if len(energy) > 0:
            data.append(energy.values)
            labels.append(name)

    if not data:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_get_color(labels[i]))
        patch.set_alpha(0.7)

    ax.set_ylabel("Energy (mWh)")
    ax.set_title("Energy Distribution by System")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    path = output_dir / "energy_boxplots.png"
    fig.savefig(path)
    fig.savefig(output_dir / "energy_boxplots.pdf")
    plt.close(fig)
    return path
