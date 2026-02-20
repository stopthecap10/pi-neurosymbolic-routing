#!/usr/bin/env python3
"""
V1 Baseline Matrix Runner
Runs 2x2 factorial: (A1/A2) × (grammar/no-grammar)

Parser version: P2_numeric_robust
Prompt template version: PT2_frozen
"""

import argparse
import csv
import os
import time
import re
import sys
from datetime import datetime
import requests
import yaml

# ============================================================
# PARSER: P2_numeric_robust
# ============================================================
PARSER_VERSION = "P2_numeric_robust"

# Matches signed integers, decimals, and simple fractions
# Examples: 12, -4, 3.0, -0.5, -5/2
NUM_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?")


def parse_numeric_robust(text: str) -> str:
    """
    Extract the LAST valid numeric token from text.

    Supports:
    - Integers: 12, -4, +7
    - Decimals: 3.0, -0.5
    - Simple fractions: -5/2, 3/4

    Normalization:
    - If value is exactly integral (e.g., 3.0, 6/2), return as integer string
    - Preserves sign correctly
    - Returns "" if no valid numeric token found
    """
    if not text:
        return ""

    # Clean up common model artifacts
    cleaned = text.strip().replace(",", "")

    # Find all numeric tokens
    tokens = NUM_TOKEN_RE.findall(cleaned)
    if not tokens:
        return ""

    # Take the last token
    raw_token = tokens[-1].lstrip('+')

    try:
        # Handle fractions: "5/2" -> 2.5
        if '/' in raw_token:
            parts = raw_token.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0:
                    return ""
                value = num / den
            else:
                return ""
        else:
            value = float(raw_token)

        # Normalize: if exactly integral, return as int string
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        else:
            # Return decimal as-is (will likely be E8 for integer-expected tasks)
            return str(value)
    except (ValueError, ZeroDivisionError):
        return ""


def extract_yesno(text: str) -> str:
    """Extract Yes/No from text using word-boundary matching."""
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


# ============================================================
# PROMPT TEMPLATES (FROZEN)
# ============================================================
PROMPT_TEMPLATE_VERSION = "PT2_frozen"

# AR / ALG / WP wrapper
NUMERIC_WRAPPER = "{source_question}\nReturn only the final numeric answer.\nAnswer:"

# LOG wrapper (already baked into official CSVs, but fallback for legacy)
LOG_WRAPPER = "{source_question}\nReturn only Yes or No.\nAnswer:"


def build_prompt(base_prompt: str, category: str) -> str:
    """Build the full prompt with frozen wrapper."""
    is_log = (category == "LOG")

    # Official CSVs (fm_v1.1+) bake the full instruction into LOG prompt_text.
    # Detect this and don't double-wrap.
    if is_log and base_prompt.rstrip().endswith("Answer:"):
        return base_prompt
    elif is_log:
        return LOG_WRAPPER.format(source_question=base_prompt)
    else:
        return NUMERIC_WRAPPER.format(source_question=base_prompt)


# ============================================================
# FROZEN ACTION BUDGETS
# ============================================================
ACTION_BUDGETS = {
    "A1": 12,  # Short decode
    "A2": 30,  # Extended decode
}


# ============================================================
# ERROR CODES
# ============================================================
def determine_error_code(category: str, pred: str, expected: str,
                         timed_out: bool, parse_success: bool) -> str:
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


# ============================================================
# INFERENCE
# ============================================================
def run_inference(prompt_text, server_url, timeout_sec, n_pred, use_grammar, grammar_file):
    """Run single inference - FROZEN params."""
    request = {
        "prompt": prompt_text,
        "n_predict": int(n_pred),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }

    if use_grammar and grammar_file:
        try:
            with open(grammar_file, 'r') as f:
                request["grammar"] = f.read()
        except Exception as e:
            print(f"WARNING: Grammar file {grammar_file} failed to load: {e}",
                  file=sys.stderr)

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


# ============================================================
# GUARDRAILS
# ============================================================
OFFICIAL_PROVENANCE_COLS = [
    'dataset_name', 'dataset_source', 'source_type',
    'source_record_id', 'field_map_version'
]


def verify_official_split(csv_path: str, prompts: list):
    errors = []
    fname = os.path.basename(csv_path)
    if not fname.startswith('industry_tier'):
        errors.append(f"Filename '{fname}' is not an official split")
    if prompts:
        cols = set(prompts[0].keys())
        missing = [c for c in OFFICIAL_PROVENANCE_COLS if c not in cols]
        if missing:
            errors.append(f"Missing provenance columns: {missing}")
        bad = [r['prompt_id'] for r in prompts
               if r.get('source_type') != 'dataset_raw']
        if bad:
            errors.append(f"{len(bad)} rows have source_type != dataset_raw")
    if errors:
        print("ERROR: Official split validation FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print(f"[guardrail] Official split verified: {csv_path}")


def qa_check_trials(trials: list):
    """Post-run QA checks. Fails the run if critical issues found."""
    errors = []

    # Check >20% missing answer_raw
    missing_raw = sum(1 for t in trials if not t.get('answer_raw'))
    if len(trials) > 0 and missing_raw / len(trials) > 0.20:
        errors.append(f"{missing_raw}/{len(trials)} trials ({100*missing_raw/len(trials):.0f}%) "
                      f"have missing answer_raw (>20% threshold)")

    # Check parser_version populated
    blank_parser = sum(1 for t in trials if not t.get('parser_version'))
    if blank_parser > 0:
        errors.append(f"{blank_parser} trials have blank parser_version")

    # Check prompt_template_version populated
    blank_pt = sum(1 for t in trials if not t.get('prompt_template_version'))
    if blank_pt > 0:
        errors.append(f"{blank_pt} trials have blank prompt_template_version")

    if errors:
        print("\nQA CHECK FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("[QA] All checks passed.")


# ============================================================
# DEBUG MODE
# ============================================================
def run_debug_mode(args, config, prompts, debug_ids):
    """Run debug mode for specific prompt IDs."""
    debug_prompts = [p for p in prompts if p['prompt_id'] in debug_ids]
    if not debug_prompts:
        print(f"ERROR: None of {debug_ids} found in CSV")
        sys.exit(1)

    print(f"DEBUG MODE: Running {len(debug_prompts)} prompts × 1 repeat\n")

    lines = []
    for prompt_row in debug_prompts:
        prompt_id = prompt_row['prompt_id']
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']
        is_log = (category == "LOG")

        prompt_text = build_prompt(base_prompt, category)
        n_pred = 6 if is_log else ACTION_BUDGETS[args.action]

        grammar_file = None
        if args.grammar:
            grammar_file = config.get('grammar_yesno' if is_log else 'grammar_num')

        content, latency_ms, timed_out = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=config['timeout_sec'],
            n_pred=n_pred,
            use_grammar=args.grammar,
            grammar_file=grammar_file
        )

        if is_log:
            answer_parsed = extract_yesno(content)
        else:
            answer_parsed = parse_numeric_robust(content)

        parse_success = (answer_parsed != "")
        error_code = determine_error_code(
            category, answer_parsed, ground_truth, timed_out, parse_success
        )

        block = [
            "=" * 70,
            f"PROMPT: {prompt_id}",
            "=" * 70,
            f"  category:              {category}",
            f"  dataset_name:          {prompt_row.get('dataset_name', 'N/A')}",
            f"  source_record_id:      {prompt_row.get('source_record_id', 'N/A')}",
            f"  grammar_enabled:       {args.grammar}",
            f"  max_tokens_budget:     {n_pred}",
            f"  timeout_sec:           {config['timeout_sec']}",
            f"  prompt_template_ver:   {PROMPT_TEMPLATE_VERSION}",
            f"  parser_version:        {PARSER_VERSION}",
            "",
            f"  FULL prompt_text sent to model:",
            f"  ---",
        ]
        for line in prompt_text.split('\n'):
            block.append(f"  | {line}")
        block += [
            f"  ---",
            "",
            f"  FULL answer_raw returned:",
            f"  ---",
            f"  | {repr(content)}",
            f"  ---",
            "",
            f"  answer_parsed:         {repr(answer_parsed)}",
            f"  ground_truth:          {repr(ground_truth)}",
            f"  parse_success:         {parse_success}",
            f"  error_code:            {error_code}",
            f"  total_latency_ms:      {latency_ms:.1f}",
            f"  timed_out:             {timed_out}",
            f"  correct:               {answer_parsed == ground_truth}",
            "",
        ]

        for line in block:
            print(line)
        lines.extend(block)

    # Save debug output
    debug_path = os.path.join(os.path.dirname(args.out_trials), "debug_failures_t1.txt")
    os.makedirs(os.path.dirname(debug_path), exist_ok=True)
    with open(debug_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nDebug output saved to: {debug_path}")


# ============================================================
# SMOKE TEST MODE
# ============================================================
def run_smoke_test(args, config, prompts):
    """Run smoke test: 1 prompt per category × 1 repeat × current config."""
    # Pick first prompt per category
    seen_cats = set()
    smoke_prompts = []
    for p in prompts:
        cat = p['category']
        if cat not in seen_cats:
            seen_cats.add(cat)
            smoke_prompts.append(p)

    print(f"SMOKE TEST: {len(smoke_prompts)} prompts × 1 repeat\n")

    trials = []
    for prompt_row in smoke_prompts:
        prompt_id = prompt_row['prompt_id']
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']
        is_log = (category == "LOG")

        prompt_text = build_prompt(base_prompt, category)
        n_pred = 6 if is_log else ACTION_BUDGETS[args.action]

        grammar_file = None
        if args.grammar:
            grammar_file = config.get('grammar_yesno' if is_log else 'grammar_num')

        content, latency_ms, timed_out = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=config['timeout_sec'],
            n_pred=n_pred,
            use_grammar=args.grammar,
            grammar_file=grammar_file
        )

        if is_log:
            answer_parsed = extract_yesno(content)
        else:
            answer_parsed = parse_numeric_robust(content)

        parse_success = (answer_parsed != "")
        correct = (answer_parsed == ground_truth)
        error_code = determine_error_code(
            category, answer_parsed, ground_truth, timed_out, parse_success
        )

        trial = {
            "prompt_id": prompt_id,
            "category": category,
            "action": args.action,
            "grammar": int(args.grammar),
            "answer_raw": content.replace("\n", "\\n")[:200],
            "answer_parsed": answer_parsed,
            "ground_truth": ground_truth,
            "correct": int(correct),
            "parse_success": int(parse_success),
            "timeout_flag": int(timed_out),
            "error_code": error_code,
            "total_latency_ms": f"{latency_ms:.1f}",
            "parser_version": PARSER_VERSION,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        }
        trials.append(trial)

        status = "OK" if correct else "FAIL"
        print(f"  [{status}] {prompt_id} {category}: "
              f"ans={repr(answer_parsed)} exp={ground_truth} "
              f"raw={repr(content[:60])} err={error_code}")

    # Summary
    total = len(trials)
    correct_count = sum(t['correct'] for t in trials)
    timeout_count = sum(t['timeout_flag'] for t in trials)
    parse_fail_count = sum(1 - t['parse_success'] for t in trials)

    print(f"\nSMOKE SUMMARY: {correct_count}/{total} correct, "
          f"{timeout_count} timeouts, {parse_fail_count} parse failures")

    return trials


# ============================================================
# MAIN
# ============================================================
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Config YAML")
    ap.add_argument("--csv", required=True, help="Prompts CSV")
    ap.add_argument("--action", required=True, choices=["A1", "A2"])
    ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--out_trials", required=True, help="Output CSV")
    ap.add_argument("--split_role", choices=["official", "dev"], default="dev")
    ap.add_argument("--debug_prompt_ids", nargs="+", default=None,
                    help="Debug mode: run only these prompt IDs with full diagnostics")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 1 prompt per category, 1 repeat")
    args = ap.parse_args()

    config = load_config(args.config)

    with open(args.csv, 'r', encoding='utf-8') as f:
        prompts = list(csv.DictReader(f))

    if args.split_role == "official":
        verify_official_split(args.csv, prompts)

    # Debug mode
    if args.debug_prompt_ids:
        run_debug_mode(args, config, prompts, args.debug_prompt_ids)
        return

    # Smoke test mode
    if args.smoke:
        smoke_trials = run_smoke_test(args, config, prompts)
        # Save smoke CSV
        smoke_csv = args.out_trials.replace('.csv', '_smoke.csv')
        if smoke_trials:
            with open(smoke_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=smoke_trials[0].keys())
                writer.writeheader()
                writer.writerows(smoke_trials)
            print(f"Smoke CSV saved to: {smoke_csv}")
        return

    # ========================================================
    # FULL RUN
    # ========================================================
    n_pred_base = ACTION_BUDGETS[args.action]
    system_name = f"v1_{args.action.lower()}_{'grammar' if args.grammar else 'nogrammar'}"

    print(f"Running {system_name}")
    print(f"Action: {args.action} (max_tokens={n_pred_base})")
    print(f"Grammar: {'enabled' if args.grammar else 'disabled'}")
    print(f"Parser: {PARSER_VERSION}")
    print(f"Prompt template: {PROMPT_TEMPLATE_VERSION}")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print()

    # Energy measurement
    print("=" * 60)
    print("ENERGY MEASUREMENT")
    print("=" * 60)
    start_mwh_str = input("Enter STARTING mWh reading (or 'skip'): ").strip()

    start_mwh = None
    if start_mwh_str.lower() != 'skip':
        try:
            start_mwh = float(start_mwh_str)
            print(f"Recorded starting: {start_mwh} mWh")
        except ValueError:
            print("Invalid input, energy will be marked as NA")
    print("=" * 60)
    print()

    run_id = f"{system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trials = []

    for prompt_row in prompts:
        prompt_id = prompt_row['prompt_id']
        dataset = prompt_row.get('dataset_name', prompt_row.get('dataset', ''))
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']
        is_log = (category == "LOG")

        prompt_text = build_prompt(base_prompt, category)
        n_pred = 6 if is_log else n_pred_base

        grammar_file = None
        if args.grammar:
            grammar_file = config.get('grammar_yesno' if is_log else 'grammar_num')

        for repeat_idx in range(1, config['repeats'] + 1):
            content, latency_ms, timed_out = run_inference(
                prompt_text=prompt_text,
                server_url=config['server_url'],
                timeout_sec=config['timeout_sec'],
                n_pred=n_pred,
                use_grammar=args.grammar,
                grammar_file=grammar_file
            )

            if is_log:
                answer_parsed = extract_yesno(content)
            else:
                answer_parsed = parse_numeric_robust(content)

            parse_success = (answer_parsed != "")
            correct = (answer_parsed == ground_truth)

            error_code = determine_error_code(
                category, answer_parsed, ground_truth, timed_out, parse_success
            )

            trial = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "dataset": dataset,
                "category": category,
                "split": os.path.splitext(os.path.basename(args.csv))[0],
                "system": system_name,
                "repeat_idx": repeat_idx,

                "action_id": args.action,
                "grammar_enabled": int(args.grammar),
                "grammar_version": "G1-relaxed" if args.grammar else "none",
                "max_tokens_budget": n_pred,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "parser_version": PARSER_VERSION,

                "answer_raw": content.replace("\n", "\\n")[:500],
                "answer_parsed": answer_parsed,
                "parse_success": int(parse_success),
                "ground_truth": ground_truth,
                "correct": int(correct),

                "total_latency_ms": f"{latency_ms:.3f}",
                "timeout_flag": int(timed_out),

                "energy_start_mwh": "NA",
                "energy_end_mwh": "NA",
                "energy_delta_mwh": "NA",
                "energy_per_prompt_mwh": "NA",

                "error_code": error_code,

                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_sec": config['timeout_sec'],
                "config_version": config['config_version'],
            }

            trials.append(trial)

            status = "+" if correct else ("T" if timed_out else "X")
            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"lat={latency_ms:.0f}ms ans={answer_parsed} exp={ground_truth} "
                  f"err={error_code} raw={repr(content[:80])}")

            if repeat_idx < config['repeats']:
                time.sleep(0.1)

    # Energy end reading
    print()
    print("=" * 60)
    end_mwh_str = input("Enter ENDING mWh reading (or 'skip'): ").strip()

    if end_mwh_str.lower() != 'skip':
        try:
            end_mwh = float(end_mwh_str)
            if start_mwh is not None:
                delta_mwh = end_mwh - start_mwh
                num_prompts = len(set(t['prompt_id'] for t in trials))
                energy_per_prompt = delta_mwh / num_prompts if num_prompts > 0 else 0

                print(f"Total energy: {delta_mwh:.2f} mWh")
                print(f"Energy per prompt: {energy_per_prompt:.2f} mWh")

                for trial in trials:
                    trial['energy_start_mwh'] = f"{start_mwh:.2f}"
                    trial['energy_end_mwh'] = f"{end_mwh:.2f}"
                    trial['energy_delta_mwh'] = f"{delta_mwh:.2f}"
                    trial['energy_per_prompt_mwh'] = f"{energy_per_prompt:.2f}"
        except ValueError:
            print("Invalid input, energy marked as NA")
    print("=" * 60)
    print()

    # QA checks
    qa_check_trials(trials)

    # Write trials CSV
    fieldnames = [
        "run_id", "timestamp", "prompt_id", "dataset", "category", "split",
        "system", "repeat_idx",
        "action_id", "grammar_enabled", "grammar_version", "max_tokens_budget",
        "prompt_template_version", "parser_version",
        "answer_raw", "answer_parsed", "parse_success", "ground_truth", "correct",
        "total_latency_ms", "timeout_flag",
        "energy_start_mwh", "energy_end_mwh", "energy_delta_mwh",
        "energy_per_prompt_mwh",
        "error_code",
        "model_name", "quantization", "temperature", "top_p", "seed",
        "timeout_sec", "config_version"
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
        median_lat = latencies[len(latencies) // 2]
    else:
        median_lat = 0

    print(f"SUMMARY:")
    print(f"  Accuracy: {correct_count}/{total} ({100 * correct_count / total:.1f}%)")
    print(f"  Timeouts: {timeout_count}/{total}")
    print(f"  Parse failures: {parse_fail_count}/{total}")
    print(f"  Median latency: {median_lat:.0f}ms")
    print(f"  Parser: {PARSER_VERSION}")
    print(f"  Prompt template: {PROMPT_TEMPLATE_VERSION}")
    print(f"\nSaved to: {args.out_trials}")


if __name__ == "__main__":
    main()
