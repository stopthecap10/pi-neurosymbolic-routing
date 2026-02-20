#!/usr/bin/env python3
"""
V1 Baseline Matrix Runner
Runs 2x2 factorial: (A1/A2) × (grammar/no-grammar)
"""

import argparse
import csv
import time
import re
import sys
from datetime import datetime
import requests
import yaml

INT_RE = re.compile(r"[-+]?\d+")

# FROZEN ACTION BUDGETS
ACTION_BUDGETS = {
    "A1": 12,  # Short decode
    "A2": 30,  # Extended decode
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_last_int(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip().replace(",", "")
    nums = INT_RE.findall(cleaned)
    if not nums:
        return ""
    result = nums[-1].lstrip('+')
    # Normalize through int() to strip leading zeros: "007" -> "7"
    try:
        return str(int(result))
    except ValueError:
        return result

def extract_yesno(text: str) -> str:
    """Extract Yes/No from text using word-boundary matching"""
    if not text:
        return ""
    t = text.lower().strip()
    yes_matches = list(re.finditer(r'\byes\b', t))
    no_matches = list(re.finditer(r'\bno\b', t))
    if not yes_matches and not no_matches:
        return ""
    yes_pos = yes_matches[-1].start() if yes_matches else -1
    no_pos = no_matches[-1].start() if no_matches else -1
    return "Yes" if yes_pos > no_pos else "No"

def determine_error_code(category: str, pred: str, expected: str, timed_out: bool, parse_success: bool) -> str:
    if timed_out:
        return "E7"
    if not parse_success or pred == "":
        return "E8"
    if pred == expected:
        return "E0"
    if category == "AR":
        return "E1"
    if category == "ALG":
        return "E3"
    if category == "LOG":
        return "E2"
    if category == "WP":
        return "E5"
    return "E8"

def run_inference(prompt_text, server_url, timeout_sec, n_pred, use_grammar, grammar_file):
    """Run single inference - FROZEN params"""

    request = {
        "prompt": prompt_text,
        "n_predict": int(n_pred),
        "temperature": 0.0,  # FROZEN
        "top_p": 1.0,        # FROZEN
        "seed": 42,          # FROZEN
    }

    if use_grammar and grammar_file:
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        request["grammar"] = grammar_content

    t0 = time.time()
    content = ""
    timed_out = False

    try:
        r = requests.post(
            server_url,
            json=request,
            timeout=(10.0, float(timeout_sec)),
        )
        j = r.json()
        content = j.get("content", "") or ""
    except requests.exceptions.Timeout:
        timed_out = True
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        content = ""

    latency_ms = (time.time() - t0) * 1000

    if latency_ms >= timeout_sec * 1000:
        timed_out = True

    return content, latency_ms, timed_out

OFFICIAL_PROVENANCE_COLS = ['dataset_name', 'dataset_source', 'source_type', 'source_record_id', 'field_map_version']

def verify_official_split(csv_path: str, prompts: list):
    """Guardrail: verify CSV is a valid official split (Section 2)."""
    errors = []
    import os
    fname = os.path.basename(csv_path)
    if not fname.startswith('industry_tier'):
        errors.append(f"Filename '{fname}' is not an official split (expected industry_tier*.csv)")
    if prompts:
        cols = set(prompts[0].keys())
        missing = [c for c in OFFICIAL_PROVENANCE_COLS if c not in cols]
        if missing:
            errors.append(f"Missing provenance columns: {missing}")
        bad = [r['prompt_id'] for r in prompts if r.get('source_type') != 'dataset_raw']
        if bad:
            errors.append(f"{len(bad)} rows have source_type != dataset_raw: {bad[:5]}")
    if errors:
        print("ERROR: Official split validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"[guardrail] Official split verified: {csv_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config YAML")
    ap.add_argument("--csv", required=True, help="Prompts CSV")
    ap.add_argument("--action", required=True, choices=["A1", "A2"], help="Action (A1=12tok, A2=30tok)")
    ap.add_argument("--grammar", action="store_true", help="Enable grammar")
    ap.add_argument("--out_trials", required=True, help="Output CSV")
    ap.add_argument("--split_role", choices=["official", "dev"], default="dev",
                    help="Set to 'official' to enforce provenance guardrails")
    args = ap.parse_args()

    # Load config
    config = load_config(args.config)

    # Load prompts
    with open(args.csv, 'r', encoding='utf-8') as f:
        prompts = list(csv.DictReader(f))

    # Official-mode guardrail
    if args.split_role == "official":
        verify_official_split(args.csv, prompts)

    # Get token budget
    n_pred_base = ACTION_BUDGETS[args.action]

    # System name
    system_name = f"v1_{args.action.lower()}_{'grammar' if args.grammar else 'nogrammar'}"

    print(f"Running {system_name}")
    print(f"Action: {args.action} (max_tokens={n_pred_base})")
    print(f"Grammar: {'enabled' if args.grammar else 'disabled'}")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print()

    # Energy measurement
    print("=" * 60)
    print("ENERGY MEASUREMENT")
    print("=" * 60)
    print("📊 Check your USB power meter now.")
    print()
    start_mwh_str = input("Enter STARTING mWh reading (just the number): ").strip()

    try:
        start_mwh = float(start_mwh_str)
        print(f"✓ Recorded starting: {start_mwh} mWh")
    except ValueError:
        print("⚠️  Invalid input, energy will be marked as NA")
        start_mwh = None

    print("=" * 60)
    print()

    # Generate run_id
    run_id = f"{system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run trials
    trials = []

    for prompt_row in prompts:
        prompt_id = prompt_row['prompt_id']
        dataset = prompt_row.get('dataset_name', prompt_row.get('dataset', ''))
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']

        is_log = (category == "LOG")

        # FROZEN prompt template
        # Official CSVs (fm_v1.1+) bake the full instruction into LOG prompt_text.
        # Fall back to appending for legacy CSVs that don't.
        if is_log and base_prompt.rstrip().endswith("Answer:"):
            prompt_text = base_prompt
        elif is_log:
            prompt_text = f"{base_prompt}\nAnswer with only Yes or No.\nAnswer:"
        else:
            prompt_text = f"{base_prompt}\nAnswer with only the final number.\nAnswer:"

        # Determine n_pred and grammar file per category
        if is_log:
            n_pred = 6  # LOG always uses 6 tokens
        else:
            n_pred = n_pred_base  # Use action budget

        grammar_file = None
        if args.grammar:
            if is_log:
                grammar_file = config['grammar_yesno']
            else:
                grammar_file = config['grammar_num']

        for repeat_idx in range(1, config['repeats'] + 1):
            # Run inference
            content, latency_ms, timed_out = run_inference(
                prompt_text=prompt_text,
                server_url=config['server_url'],
                timeout_sec=config['timeout_sec'],
                n_pred=n_pred,
                use_grammar=args.grammar,
                grammar_file=grammar_file
            )

            # Parse answer
            if is_log:
                answer_parsed = extract_yesno(content)
            else:
                answer_parsed = extract_last_int(content)

            parse_success = (answer_parsed != "")
            correct = (answer_parsed == ground_truth)

            error_code = determine_error_code(
                category=category,
                pred=answer_parsed,
                expected=ground_truth,
                timed_out=timed_out,
                parse_success=parse_success
            )

            # Build trial record
            trial = {
                # Identity
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "dataset": dataset,
                "category": category,
                "split": os.path.splitext(os.path.basename(args.csv))[0],
                "system": system_name,
                "repeat_idx": repeat_idx,

                # Action/Grammar tracking
                "action_id": args.action,
                "grammar_enabled": int(args.grammar),
                "grammar_version": "G1-relaxed" if args.grammar else "none",
                "max_tokens_budget": n_pred,
                "prompt_template_version": "v1.0",

                # Output
                "answer_raw": content.replace("\n", "\\n")[:200],
                "answer_parsed": answer_parsed,
                "parse_success": int(parse_success),
                "ground_truth": ground_truth,
                "correct": int(correct),

                # Runtime
                "total_latency_ms": f"{latency_ms:.3f}",
                "timeout_flag": int(timed_out),

                # Energy (placeholder)
                "energy_start_mwh": "NA",
                "energy_end_mwh": "NA",
                "energy_delta_mwh": "NA",
                "energy_per_prompt_mwh": "NA",

                # Error
                "error_code": error_code,

                # Reproducibility (all frozen)
                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_sec": config['timeout_sec'],
                "config_version": config['config_version'],
            }

            trials.append(trial)

            # Print progress
            status = "✓" if correct else ("T" if timed_out else "✗")
            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"lat={latency_ms:.0f}ms ans={answer_parsed} exp={ground_truth} err={error_code}")

            # Small delay
            if repeat_idx < config['repeats']:
                time.sleep(0.1)

    # Energy end reading
    print()
    print("=" * 60)
    print("ENERGY MEASUREMENT")
    print("=" * 60)
    print("📊 Check your USB power meter now.")
    print()
    end_mwh_str = input("Enter ENDING mWh reading (just the number): ").strip()

    try:
        end_mwh = float(end_mwh_str)
        print(f"✓ Recorded ending: {end_mwh} mWh")

        if start_mwh is not None:
            delta_mwh = end_mwh - start_mwh
            num_prompts = len(set(t['prompt_id'] for t in trials))
            energy_per_prompt = delta_mwh / num_prompts if num_prompts > 0 else 0

            print(f"✓ Total energy: {delta_mwh:.2f} mWh")
            print(f"✓ Energy per prompt: {energy_per_prompt:.2f} mWh")

            # Update all trials
            for trial in trials:
                trial['energy_start_mwh'] = f"{start_mwh:.2f}"
                trial['energy_end_mwh'] = f"{end_mwh:.2f}"
                trial['energy_delta_mwh'] = f"{delta_mwh:.2f}"
                trial['energy_per_prompt_mwh'] = f"{energy_per_prompt:.2f}"
    except ValueError:
        print("⚠️  Invalid input, energy marked as NA")

    print("=" * 60)
    print()

    # Write trials CSV
    fieldnames = [
        "run_id", "timestamp", "prompt_id", "dataset", "category", "split", "system", "repeat_idx",
        "action_id", "grammar_enabled", "grammar_version", "max_tokens_budget", "prompt_template_version",
        "answer_raw", "answer_parsed", "parse_success", "ground_truth", "correct",
        "total_latency_ms", "timeout_flag",
        "energy_start_mwh", "energy_end_mwh", "energy_delta_mwh", "energy_per_prompt_mwh",
        "error_code",
        "model_name", "quantization", "temperature", "top_p", "seed", "timeout_sec", "config_version"
    ]

    with open(args.out_trials, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    # Summary
    total = len(trials)
    correct_count = sum(t['correct'] for t in trials)
    timeout_count = sum(t['timeout_flag'] for t in trials)
    parse_fail_count = sum(1 - t['parse_success'] for t in trials)

    latencies = [float(t['total_latency_ms']) for t in trials if not t['timeout_flag']]
    if latencies:
        latencies.sort()
        median_lat = latencies[len(latencies)//2]
    else:
        median_lat = 0

    print(f"📊 SUMMARY:")
    print(f"  Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"  Timeouts: {timeout_count}/{total}")
    print(f"  Parse failures: {parse_fail_count}/{total}")
    print(f"  Median latency: {median_lat:.0f}ms")
    print(f"\n✅ Saved to: {args.out_trials}")

if __name__ == "__main__":
    main()
