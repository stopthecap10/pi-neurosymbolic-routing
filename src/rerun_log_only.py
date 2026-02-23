#!/usr/bin/env python3
"""Re-run LOG prompts through A6 symbolic solver and INSERT into existing CSV.

The LOG rows were lost from a prior failed patch. This script:
1. Runs A6 on all LOG prompts × 3 repeats
2. Builds complete trial rows matching the V5 runner format
3. Copies metadata (run_id, config, energy, etc.) from an existing row
4. Appends the new LOG rows to the CSV
"""

import csv
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, 'src')
from a6_logic_engine import solve_logic


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'data/splits/industry_tier2_100.csv'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'outputs/official/runs/t2_v5_qwen_300tok.csv'

    # Load LOG prompts from dataset
    log_prompts = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['category'] == 'LOG':
                log_prompts.append(r)

    # Read existing CSV to get fieldnames and a reference row for metadata
    with open(out_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)

    # Check if LOG rows already exist
    log_count = sum(1 for r in existing if r.get('category') == 'LOG')
    if log_count > 0:
        print(f"WARNING: {log_count} LOG rows already exist in CSV. Will PATCH them.")
        mode = 'patch'
    else:
        print(f"No LOG rows in CSV. Will INSERT new rows.")
        mode = 'insert'

    # Grab metadata from first existing row (run_id, config, etc.)
    ref = existing[0]

    print(f"Re-running {len(log_prompts)} LOG prompts × 3 repeats through A6")
    print()

    # Run A6 on each prompt × 3
    new_trials = []
    ok = 0
    total = 0
    for p in log_prompts:
        pid = p['prompt_id']
        expected = p['ground_truth'].strip()
        for rep in range(1, 4):
            total += 1
            result = solve_logic(p['prompt_text'])
            ans = result['answer'] if result['parse_success'] else ''
            parse_ok = result['parse_success'] and result['answer'] in ('Yes', 'No')
            correct = (ans == expected)
            if correct:
                ok += 1
            mark = 'OK' if correct else 'X '
            print(f"{mark} {pid} #{rep}/3 ans={ans} exp={expected} lat={result['latency_ms']:.1f}ms")

            # Build a complete trial row matching V5 runner format
            raw_trace = result.get('trace', '')[:200]
            trial = {
                # --- Metadata (copied from existing rows) ---
                "run_id": ref.get('run_id', ''),
                "timestamp": datetime.now().isoformat(),
                "prompt_id": pid,
                "dataset": ref.get('dataset', ''),
                "category": "LOG",
                "split": ref.get('split', ''),
                "system": ref.get('system', ''),
                "repeat_idx": str(rep),
                "prompt_template_version": ref.get('prompt_template_version', ''),
                "parser_version": ref.get('parser_version', ''),

                # --- Routing ---
                "route_chosen": "logic_symbolic",
                "route_attempt_sequence": "A6",
                "escalations_count": "0",
                "decision_reason": "LOG->A6_symbolic",
                "final_answer_source": "logic_symbolic",
                "reasoning_mode": "symbolic",

                # --- Symbolic flags ---
                "symbolic_parse_success": '1' if result['parse_success'] else '0',
                "sympy_solve_success": "0",

                # --- Repair fields ---
                "repair_attempted": "0",
                "repair_success": "0",
                "repair_trigger_reason": "",
                "previous_raw_len": "0",
                "previous_action_id": "",

                # --- Answer ---
                "answer_raw": f"[A6: {result['answer']}] {raw_trace}".replace("\n", "\\n")[:500],
                "answer_parsed": ans,
                "parse_success": '1' if parse_ok else '0',
                "ground_truth": expected,
                "correct": '1' if correct else '0',

                # --- Latency ---
                "total_latency_ms": f"{result['latency_ms']:.3f}",
                "timeout_flag": "0",
                "timeout_policy_version": ref.get('timeout_policy_version', ''),
                "action_timeout_sec_used": "0",
                "timeout_reason": "none",

                # --- A6-specific ---
                "a6_parse_success": '1' if result['parse_success'] else '0',
                "a6_n_facts": str(result.get('n_facts', 0)),
                "a6_n_rules": str(result.get('n_rules', 0)),
                "a6_n_derived": str(result.get('n_derived', 0)),
                "a6_rule_fired": result.get('rule_fired', ''),
                "a6_pattern": result.get('pattern', ''),

                # --- Energy (same as rest of run) ---
                "energy_start_mwh": ref.get('energy_start_mwh', 'NA'),
                "energy_end_mwh": ref.get('energy_end_mwh', 'NA'),
                "energy_delta_mwh": ref.get('energy_delta_mwh', 'NA'),
                "energy_per_prompt_mwh": ref.get('energy_per_prompt_mwh', 'NA'),

                # --- Error ---
                "error_code": "E0" if correct else "E2",

                # --- Config (copied from existing rows) ---
                "model_name": ref.get('model_name', ''),
                "quantization": ref.get('quantization', ''),
                "temperature": ref.get('temperature', '0.0'),
                "top_p": ref.get('top_p', '1.0'),
                "top_k": ref.get('top_k', '1'),
                "seed": ref.get('seed', '42'),
                "timeout_sec": ref.get('timeout_sec', ''),
                "config_version": ref.get('config_version', ''),
                "router_version": ref.get('router_version', ''),
            }

            # Only keep fields that exist in the CSV fieldnames
            trial_filtered = {k: v for k, v in trial.items() if k in fieldnames}
            new_trials.append(trial_filtered)

    print(f"\nLOG accuracy: {ok}/{total} ({100*ok/total:.1f}%)")

    if mode == 'insert':
        # Append LOG rows to existing data
        all_rows = existing + new_trials
    else:
        # Patch mode: replace existing LOG rows
        non_log = [r for r in existing if r.get('category') != 'LOG']
        all_rows = non_log + new_trials

    # Write back
    print(f"\nWriting {len(all_rows)} total rows to {out_path}...")
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Inserted {len(new_trials)} LOG rows")

    # Final accuracy
    total_ok = sum(1 for r in all_rows if r.get('correct') == '1')
    total_all = len(all_rows)
    print(f"Overall accuracy: {total_ok}/{total_all} ({100*total_ok/total_all:.1f}%)")

    # Per-category breakdown
    from collections import Counter
    cat_ok = Counter()
    cat_total = Counter()
    for r in all_rows:
        cat = r.get('category', '?')
        cat_total[cat] += 1
        if r.get('correct') == '1':
            cat_ok[cat] += 1
    print("\nPer-category accuracy:")
    for cat in sorted(cat_total.keys()):
        n = cat_total[cat]
        c = cat_ok[cat]
        print(f"  {cat}: {c}/{n} ({100*c/n:.1f}%)")


if __name__ == '__main__':
    main()
