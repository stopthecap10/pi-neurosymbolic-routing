#!/usr/bin/env python3
"""Re-run only LOG prompts through A6 symbolic solver and patch into existing CSV.

Patches ALL fields that would differ if A6 had succeeded in the original run:
routing fields, answer fields, latency, symbolic flags, A6-specific fields, error code.
"""

import csv
import sys
import time

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

    print(f"Re-running {len(log_prompts)} LOG prompts × 3 repeats through A6")
    print()

    # Run A6 on each prompt × 3, storing full result
    new_rows = []
    ok = 0
    total = 0
    for p in log_prompts:
        pid = p['prompt_id']
        expected = p['ground_truth'].strip()
        for rep in range(1, 4):
            total += 1
            result = solve_logic(p['prompt_text'])
            ans = result['answer'] if result['parse_success'] else ''
            parse_success = result['parse_success'] and result['answer'] in ('Yes', 'No')
            correct = (ans == expected)
            if correct:
                ok += 1
            mark = 'OK' if correct else 'X '
            print(f"{mark} {pid} #{rep}/3 ans={ans} exp={expected} lat={result['latency_ms']:.1f}ms")
            new_rows.append({
                'prompt_id': pid,
                'repeat': rep,
                'answer': ans,
                'expected': expected,
                'correct': correct,
                'parse_success': parse_success,
                'latency_ms': result['latency_ms'],
                # Full A6 result for patching
                'a6_result': result,
            })

    print(f"\nLOG accuracy: {ok}/{total} ({100*ok/total:.1f}%)")

    # Now patch the existing trials CSV
    print(f"\nPatching {out_path}...")
    # Read existing CSV
    with open(out_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        existing = list(reader)

    # Build lookup of new LOG results keyed by (prompt_id, repeat_idx)
    new_lookup = {}
    for nr in new_rows:
        key = (nr['prompt_id'], str(nr['repeat']))
        new_lookup[key] = nr

    patched = 0
    for row in existing:
        if row.get('category') != 'LOG':
            continue
        key = (row['prompt_id'], row.get('repeat_idx', row.get('repeat_num', row.get('repeat', ''))))
        if key not in new_lookup:
            continue

        nr = new_lookup[key]
        a6 = nr['a6_result']

        # --- Answer fields ---
        raw_trace = a6['trace'][:200] if a6.get('trace') else ''
        row['answer_raw'] = f"[A6: {a6['answer']}] {raw_trace}".replace("\n", "\\n")[:500]
        row['answer_parsed'] = nr['answer']
        row['parse_success'] = '1' if nr['parse_success'] else '0'
        row['correct'] = '1' if nr['correct'] else '0'
        row['error_code'] = 'E0' if nr['correct'] else 'E2'

        # --- Routing fields ---
        row['route_chosen'] = 'logic_symbolic'
        row['route_attempt_sequence'] = 'A6'
        row['escalations_count'] = '0'
        row['decision_reason'] = 'LOG->A6_symbolic'
        row['final_answer_source'] = 'logic_symbolic'
        row['reasoning_mode'] = 'symbolic'

        # --- Symbolic flags ---
        row['symbolic_parse_success'] = '1' if a6['parse_success'] else '0'
        row['sympy_solve_success'] = '0'

        # --- Repair fields (not used for A6 direct solve) ---
        if 'repair_attempted' in fieldnames:
            row['repair_attempted'] = '0'
        if 'repair_success' in fieldnames:
            row['repair_success'] = '0'
        if 'repair_trigger_reason' in fieldnames:
            row['repair_trigger_reason'] = ''
        if 'previous_raw_len' in fieldnames:
            row['previous_raw_len'] = '0'
        if 'previous_action_id' in fieldnames:
            row['previous_action_id'] = ''

        # --- Latency ---
        row['total_latency_ms'] = f"{nr['latency_ms']:.3f}"
        row['timeout_flag'] = '0'
        if 'timeout_reason' in fieldnames:
            row['timeout_reason'] = 'none'
        if 'action_timeout_sec_used' in fieldnames:
            row['action_timeout_sec_used'] = '0'

        # --- A6-specific fields ---
        if 'a6_parse_success' in fieldnames:
            row['a6_parse_success'] = '1' if a6['parse_success'] else '0'
        if 'a6_n_facts' in fieldnames:
            row['a6_n_facts'] = str(a6.get('n_facts', 0))
        if 'a6_n_rules' in fieldnames:
            row['a6_n_rules'] = str(a6.get('n_rules', 0))
        if 'a6_n_derived' in fieldnames:
            row['a6_n_derived'] = str(a6.get('n_derived', 0))
        if 'a6_rule_fired' in fieldnames:
            row['a6_rule_fired'] = a6.get('rule_fired', '')
        if 'a6_pattern' in fieldnames:
            row['a6_pattern'] = a6.get('pattern', '')

        patched += 1

    # Write back
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    print(f"Patched {patched} LOG rows in {out_path}")

    # Recount accuracy
    total_ok = sum(1 for r in existing if r.get('correct') == '1')
    total_all = len(existing)
    print(f"New overall accuracy: {total_ok}/{total_all} ({100*total_ok/total_all:.1f}%)")

    # Show per-category breakdown
    from collections import Counter
    cat_ok = Counter()
    cat_total = Counter()
    for r in existing:
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
