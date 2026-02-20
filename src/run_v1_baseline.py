#!/usr/bin/env python3
"""
V1 Baseline Runner with Complete Trial Logging
Captures all required fields for ISEF project
"""

import argparse
import csv
import time
import re
import sys
import uuid
from pathlib import Path
from datetime import datetime
import requests
import yaml

INT_RE = re.compile(r"[-+]?\d+")

def load_config(config_path):
    """Load frozen config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_last_int(text: str) -> str:
    """Extract last integer from text"""
    if not text:
        return ""
    cleaned = text.strip().replace(",", "")
    nums = INT_RE.findall(cleaned)
    if not nums:
        return ""
    # Remove leading + sign if present
    result = nums[-1].lstrip('+')
    return result

def extract_yesno(text: str) -> str:
    """Extract Yes/No from text"""
    if not text:
        return ""
    t = text.lower().strip()
    yes_pos = t.rfind("yes")
    no_pos = t.rfind("no")
    if yes_pos == -1 and no_pos == -1:
        return ""
    return "Yes" if yes_pos > no_pos else "No"

def determine_error_code(category: str, pred: str, expected: str, timed_out: bool, parse_success: bool) -> str:
    """Determine error code"""
    if timed_out:
        return "E7"
    if not parse_success or pred == "":
        return "E8"
    if pred == expected:
        return "E0"
    # Wrong answer codes
    if category == "AR":
        return "E1"
    if category == "ALG":
        return "E3"
    if category == "LOG":
        return "E2"
    if category == "WP":
        return "E5"
    return "E8"

def run_inference(prompt_text, server_url, timeout_sec, is_log, use_grammar, n_pred, grammar_file=None):
    """Run single inference and return result"""

    # Build request
    request = {
        "prompt": prompt_text,
        "n_predict": int(n_pred),
        "temperature": 0.0,
    }

    if use_grammar and grammar_file:
        # Load grammar content
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        request["grammar"] = grammar_content

    # Time the inference
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

    # Check if actually timed out
    if latency_ms >= timeout_sec * 1000:
        timed_out = True

    return content, latency_ms, timed_out

OFFICIAL_PROVENANCE_COLS = ['dataset_name', 'dataset_source', 'source_type', 'source_record_id', 'field_map_version']

def verify_official_split(csv_path: str, prompts: list):
    """Guardrail: verify CSV is a valid official split (Section 2)."""
    errors = []
    # Check filename
    import os
    fname = os.path.basename(csv_path)
    if not fname.startswith('industry_tier'):
        errors.append(f"Filename '{fname}' is not an official split (expected industry_tier*.csv)")
    # Check provenance columns
    if prompts:
        cols = set(prompts[0].keys())
        missing = [c for c in OFFICIAL_PROVENANCE_COLS if c not in cols]
        if missing:
            errors.append(f"Missing provenance columns: {missing}")
        # Check source_type
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
    ap.add_argument("--config", required=True, help="Path to frozen config YAML")
    ap.add_argument("--csv", required=True, help="Path to prompts CSV")
    ap.add_argument("--system", required=True, help="System name (phi_grammar or phi_nogrammar)")
    ap.add_argument("--use_grammar", action="store_true", help="Use grammar constraints")
    ap.add_argument("--out_trials", required=True, help="Output trials CSV path")
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

    print(f"Running V1 Baseline: {args.system}")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print(f"Use grammar: {args.use_grammar}")
    print()

    # Prepare trials output
    trials = []

    # Generate run_id
    run_id = f"{args.system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Prompt for starting energy reading
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

    for prompt_row in prompts:
        prompt_id = prompt_row['prompt_id']
        dataset = prompt_row.get('dataset_name', prompt_row.get('dataset', ''))
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']

        is_log = (category == "LOG")

        # Format prompt with instruction
        # Official CSVs (fm_v1.1+) bake the full instruction into LOG prompt_text.
        # Fall back to appending for legacy CSVs that don't.
        if is_log and base_prompt.rstrip().endswith("Answer:"):
            prompt_text = base_prompt
        elif is_log:
            prompt_text = f"{base_prompt}\nAnswer with only Yes or No.\nAnswer:"
        else:
            prompt_text = f"{base_prompt}\nAnswer with only the final number.\nAnswer:"

        # Determine n_pred and grammar file
        if args.use_grammar:
            n_pred = 12  # Grammar mode - match no-grammar token count for fair comparison
            if is_log:
                grammar_file = config['grammar_yesno']
            else:
                grammar_file = config['grammar_num']
        else:
            if is_log:
                n_pred = config['n_pred_log']
            else:
                n_pred = config['n_pred_num']
            grammar_file = None

        # Run repeats
        for repeat_idx in range(1, config['repeats'] + 1):
            # Run inference
            content, latency_ms, timed_out = run_inference(
                prompt_text=prompt_text,
                server_url=config['server_url'],
                timeout_sec=config['timeout_sec'],
                is_log=is_log,
                use_grammar=args.use_grammar,
                n_pred=n_pred,
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
                "system": args.system,
                "repeat_idx": repeat_idx,

                # Output
                "answer_raw": content.replace("\n", "\\n")[:200],  # Truncate for CSV
                "answer_parsed": answer_parsed,
                "parse_success": int(parse_success),
                "ground_truth": ground_truth,
                "correct": int(correct),

                # Runtime
                "latency_ms_total": f"{latency_ms:.3f}",
                "timeout_flag": int(timed_out),

                # Energy (placeholders - filled after run completes)
                "energy_start_mwh": "NA",
                "energy_end_mwh": "NA",
                "energy_delta_mwh": "NA",
                "energy_delta_j": "NA",
                "energy_per_prompt_mwh": "NA",
                "energy_method": config['energy_method'],

                # Error
                "error_code": error_code,

                # Reproducibility
                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": config['temperature'],
                "top_p": config['top_p'],
                "max_tokens": n_pred,
                "timeout_sec": config['timeout_sec'],
                "seed": config['seed'],
                "config_version": config['config_version'],
            }

            trials.append(trial)

            # Print progress
            status = "✓" if correct else ("T" if timed_out else "✗")
            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"lat={latency_ms:.0f}ms ans={answer_parsed} exp={ground_truth} err={error_code}")

            # Small delay between repeats
            if repeat_idx < config['repeats']:
                time.sleep(0.1)

    # Prompt for ending energy reading
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

            # Validation check
            if delta_mwh < 0:
                print(f"⚠️  WARNING: Negative energy delta ({delta_mwh:.2f} mWh)")
                print("⚠️  This indicates measurement error - please verify meter readings")

            num_prompts = len(set(t['prompt_id'] for t in trials))
            energy_per_prompt = delta_mwh / num_prompts if num_prompts > 0 else 0
            energy_delta_j = delta_mwh * 3.6  # Convert mWh to J

            print(f"✓ Total energy: {delta_mwh:.2f} mWh ({energy_delta_j:.2f} J)")
            print(f"✓ Energy per prompt: {energy_per_prompt:.2f} mWh")

            # Update all trials with energy (full audit trail)
            for trial in trials:
                trial['energy_start_mwh'] = f"{start_mwh:.2f}"
                trial['energy_end_mwh'] = f"{end_mwh:.2f}"
                trial['energy_delta_mwh'] = f"{delta_mwh:.2f}"
                trial['energy_delta_j'] = f"{energy_delta_j:.2f}"
                trial['energy_per_prompt_mwh'] = f"{energy_per_prompt:.2f}"
        else:
            print("⚠️  No starting energy recorded, marking as NA")
            # Mark all as NA
            for trial in trials:
                trial['energy_start_mwh'] = "NA"
                trial['energy_end_mwh'] = "NA"
                trial['energy_delta_mwh'] = "NA"
                trial['energy_delta_j'] = "NA"
                trial['energy_per_prompt_mwh'] = "NA"
    except ValueError:
        print("⚠️  Invalid input, energy marked as NA")
        for trial in trials:
            trial['energy_start_mwh'] = "NA"
            trial['energy_end_mwh'] = "NA"
            trial['energy_delta_mwh'] = "NA"
            trial['energy_delta_j'] = "NA"
            trial['energy_per_prompt_mwh'] = "NA"

    print("=" * 60)
    print()

    # Quick preview summary
    total = len(trials)
    correct_count = sum(t['correct'] for t in trials)
    timeout_count = sum(t['timeout_flag'] for t in trials)
    parse_fail_count = sum(1 - t['parse_success'] for t in trials)

    print("📊 PREVIEW SUMMARY:")
    print(f"  Total runs: {total}")
    print(f"  Correct: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"  Timeouts: {timeout_count}/{total}")
    print(f"  Parse failures: {parse_fail_count}/{total}")
    print()

    # Ask if user wants to save
    save_choice = input("Save this run? (y=save, n=discard, a=archive): ").strip().lower()

    if save_choice == 'n':
        print("❌ Run discarded (not saved)")
        return
    elif save_choice == 'a':
        # Save to archive
        archive_path = args.out_trials.replace('.csv', '_ARCHIVED.csv')
        print(f"📦 Archiving to: {archive_path}")
        args.out_trials = archive_path
    else:
        print("✅ Saving run...")

    # Write trials CSV
    fieldnames = [
        "run_id", "timestamp", "prompt_id", "dataset", "category", "system", "repeat_idx",
        "answer_raw", "answer_parsed", "parse_success", "ground_truth", "correct",
        "latency_ms_total", "timeout_flag",
        "energy_start_mwh", "energy_end_mwh", "energy_delta_mwh", "energy_delta_j", "energy_per_prompt_mwh", "energy_method",
        "error_code",
        "model_name", "quantization", "temperature", "top_p", "max_tokens", "timeout_sec", "seed", "config_version"
    ]

    with open(args.out_trials, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)

    print(f"\n✅ Trials saved to: {args.out_trials}")

    # Log to runs manifest
    manifest_path = "outputs/runs_manifest.csv"
    manifest_exists = Path(manifest_path).exists()

    # Calculate summary stats for manifest
    num_prompts = len(set(t['prompt_id'] for t in trials))
    num_inferences = len(trials)
    accuracy = correct_count / total if total > 0 else 0

    latencies = [float(t['latency_ms_total']) for t in trials if not t['timeout_flag']]
    median_lat = 0
    if latencies:
        latencies.sort()
        median_lat = latencies[len(latencies)//2]

    # Get energy delta (all trials have same values)
    energy_delta = trials[0]['energy_delta_mwh'] if trials else "NA"

    # Determine status
    status = "ARCHIVED" if "_ARCHIVED" in args.out_trials else "COMPLETED"

    manifest_row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "system": args.system,
        "dataset_csv": args.csv,
        "num_prompts": num_prompts,
        "num_inferences": num_inferences,
        "config_file": args.config,
        "grammar_mode": int(args.use_grammar),
        "trials_csv": args.out_trials,
        "accuracy": f"{accuracy:.3f}",
        "median_latency_ms": f"{median_lat:.1f}",
        "energy_delta_mwh": energy_delta,
        "status": status,
        "notes": ""
    }

    with open(manifest_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ["run_id", "timestamp", "system", "dataset_csv", "num_prompts", "num_inferences",
                      "config_file", "grammar_mode", "trials_csv", "accuracy", "median_latency_ms",
                      "energy_delta_mwh", "status", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not manifest_exists:
            writer.writeheader()
        writer.writerow(manifest_row)

    print(f"✅ Run logged to: {manifest_path}")

    # Detailed summary
    print(f"\n📈 FINAL SUMMARY:")
    print(f"  Median latency: {median_lat:.0f}ms")

if __name__ == "__main__":
    main()
