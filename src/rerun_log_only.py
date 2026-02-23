#!/usr/bin/env python3
"""Re-run only LOG prompts through A6 symbolic solver and patch into existing CSV."""

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

    # Run A6 on each prompt × 3
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
            correct = (ans == expected)
            if correct:
                ok += 1
            mark = 'OK' if correct else 'X '
            print(f"{mark} {pid} #{rep}/3 ans={ans} exp={expected} lat={result['latency_ms']:.0f}ms")
            new_rows.append({
                'prompt_id': pid,
                'repeat': rep,
                'answer': ans,
                'expected': expected,
                'correct': correct,
                'latency_ms': result['latency_ms'],
            })

    print(f"\nLOG accuracy: {ok}/{total} ({100*ok/total:.1f}%)")

    # Now patch the existing trials CSV
    if len(sys.argv) > 2 or True:
        print(f"\nPatching {out_path}...")
        # Read existing CSV
        with open(out_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing = list(reader)

        # Build lookup of new LOG results keyed by (prompt_id, repeat_num)
        new_lookup = {}
        for nr in new_rows:
            key = (nr['prompt_id'], str(nr['repeat']))
            new_lookup[key] = nr

        patched = 0
        for row in existing:
            if row.get('category') != 'LOG':
                continue
            key = (row['prompt_id'], row.get('repeat_num', row.get('repeat', '')))
            if key in new_lookup:
                nr = new_lookup[key]
                row['answer_parsed'] = nr['answer']
                row['correct'] = '1' if nr['correct'] else '0'
                row['error_code'] = 'E0' if nr['correct'] else 'E2'
                row['latency_ms'] = f"{nr['latency_ms']:.2f}"
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

if __name__ == '__main__':
    main()
