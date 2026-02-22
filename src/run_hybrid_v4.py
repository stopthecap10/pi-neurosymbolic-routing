#!/usr/bin/env python3
"""
Hybrid V4 Runner - Calibrated Selective Compute
AR/ALG: deterministic (A5/A4), no calibration
WP: A2 -> p_correct check -> A3R if p < tau
LOG: A1, p_correct logged (no fallback in V4-min)
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

from router_v4 import RouterV4

PARSER_VERSION = "P3_numeric_context"
PROMPT_TEMPLATE_VERSION = "PT3_phi_chat"

SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_routing_decisions(decisions_path=None):
    """V4 routing decisions (same primary routes as V3.1)."""
    return {
        'category_routes': {
            'AR': {'action': 'A5', 'grammar_enabled': False},
            'ALG': {'action': 'A4', 'grammar_enabled': False},
            'WP': {'action': 'A2', 'grammar_enabled': False},
            'LOG': {'action': 'A1', 'grammar_enabled': False},
        },
        'max_escalations': 2,
    }


OFFICIAL_PROVENANCE_COLS = ['dataset_name', 'dataset_source', 'source_type', 'source_record_id', 'field_map_version']


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
    ap.add_argument("--decisions", help="Routing decisions file (optional)")
    ap.add_argument("--out_trials", required=True, help="Output CSV")
    ap.add_argument("--split_role", choices=["official", "dev"], default="dev")
    ap.add_argument("--api_mode", choices=["chat", "completion"], default="chat")
    ap.add_argument("--probe", action="store_true",
                    help="Quick probe: 1 prompt per category x 1 repeat")
    ap.add_argument("--calibrator_json", default=None,
                    help="Path to router_calibrator.json")
    ap.add_argument("--tau", type=float, default=0.70,
                    help="Confidence threshold for escalation (default: 0.70)")
    args = ap.parse_args()

    config = load_config(args.config)
    config['api_mode'] = args.api_mode
    routing_decisions = load_routing_decisions(args.decisions)

    # Load prompts
    with open(args.csv, 'r', encoding='utf-8') as f:
        prompts = list(csv.DictReader(f))

    if args.split_role == "official":
        verify_official_split(args.csv, prompts)

    if args.probe:
        seen_cats = set()
        probe_prompts = []
        for p in prompts:
            cat = p['category']
            if cat not in seen_cats:
                seen_cats.add(cat)
                probe_prompts.append(p)
        prompts = probe_prompts
        config['repeats'] = 1

    # Initialize V4 router
    router = RouterV4(config, routing_decisions,
                      calibrator_path=args.calibrator_json,
                      tau=args.tau)

    system_name = "hybrid_v4"

    print(f"Running {system_name}")
    print(f"Router: V4 calibrated selective compute (tau={args.tau})")
    print(f"Calibrator: {args.calibrator_json or 'NONE (uncalibrated mode)'}")
    if args.probe:
        print(f"MODE: PROBE (1 per category x 1 repeat)")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print()
    print("Category routing:")
    for cat, route_info in routing_decisions['category_routes'].items():
        action = route_info['action']
        print(f"  {cat:4} -> {action}")
    print()
    print("V4 policy:")
    print("  AR:  A5 (deterministic, no calibration)")
    print("  ALG: A4 (deterministic, no calibration)")
    print(f"  WP:  A2 -> p_correct check -> A3R if p < {args.tau}")
    print(f"  LOG: A1 (p_correct logged, no fallback)")
    print()

    # Energy measurement
    print("=" * 60)
    print("ENERGY MEASUREMENT")
    print("=" * 60)
    print("Check your USB power meter now.")
    print()
    start_mwh_str = input("Enter STARTING mWh reading (just the number): ").strip()
    start_mwh_clean = re.sub(r'[^\d.\-+]', '', start_mwh_str)

    try:
        start_mwh = float(start_mwh_clean)
        print(f"Recorded starting: {start_mwh} mWh (raw input: '{start_mwh_str}')")
    except ValueError:
        print(f"Invalid input (raw: '{start_mwh_str}', cleaned: '{start_mwh_clean}'), energy will be marked as NA")
        start_mwh = None

    print("=" * 60)
    print()

    run_id = f"{system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Track WP A3R stats
    wp_a3_attempts = 0
    wp_a3_successes = 0

    trials = []

    for prompt_row in prompts:
        prompt_id = prompt_row['prompt_id']
        dataset = prompt_row.get('dataset_name', prompt_row.get('dataset', ''))
        category = prompt_row['category']
        base_prompt = prompt_row['prompt_text']
        ground_truth = prompt_row['ground_truth']

        is_log = (category == "LOG")

        # Build Phi chat template
        question = base_prompt
        lines_q = question.rstrip().split('\n')
        while lines_q and lines_q[-1].strip() in ("Answer:", ""):
            lines_q.pop()
        while lines_q and lines_q[-1].strip().startswith("Return only"):
            lines_q.pop()
        question = '\n'.join(lines_q).strip()

        system_msg = SYSTEM_MSG_YESNO if is_log else SYSTEM_MSG_NUMERIC
        prompt_text = f"<|system|>{system_msg}<|end|><|user|>{question}<|end|><|assistant|>"

        for repeat_idx in range(1, config['repeats'] + 1):
            result = router.route(
                prompt_id=prompt_id,
                category=category,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
            )

            # Track WP repair stats
            if category == "WP" and result.get('repair_attempted', False):
                wp_a3_attempts += 1
                if result.get('repair_success', False):
                    wp_a3_successes += 1

            # Build step latency string
            step_latencies = result.get('step_latencies', [])
            if len(step_latencies) > 1:
                parts = [f"{a}={lat:.0f}ms" for a, lat in step_latencies]
                step_lat_str = f" ({'+'.join(parts)})"
            else:
                step_lat_str = ""

            trial = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "dataset": dataset,
                "category": category,
                "split": os.path.splitext(os.path.basename(args.csv))[0],
                "system": system_name,
                "repeat_idx": repeat_idx,
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "parser_version": PARSER_VERSION,
                "route_chosen": result['final_source'],
                "route_attempt_sequence": result['route_attempt_sequence'],
                "escalations_count": result['escalations_count'],
                "decision_reason": result['decision_reason'],
                "final_answer_source": result['final_source'],
                "reasoning_mode": result['reasoning_mode'],
                "symbolic_parse_success": int(result['symbolic_parse_success']),
                "sympy_solve_success": int(result['sympy_solve_success']),
                "repair_attempted": int(result.get('repair_attempted', False)),
                "repair_success": int(result.get('repair_success', False)),
                "repair_trigger_reason": result.get('repair_trigger_reason', ''),
                "previous_raw_len": result.get('previous_raw_len', 0),
                "previous_action_id": result.get('previous_action_id', ''),
                "answer_raw": result['answer_raw'].replace("\n", "\\n")[:500],
                "answer_parsed": result['answer_final'],
                "parse_success": int(result['parse_success']),
                "ground_truth": ground_truth,
                "correct": int(result['correct']),
                "total_latency_ms": f"{result['total_latency_ms']:.3f}",
                "timeout_flag": int(result['timeout_flag']),
                "timeout_policy_version": result.get('timeout_policy_version', ''),
                "action_timeout_sec_used": result.get('action_timeout_sec_used', ''),
                "timeout_reason": result.get('timeout_reason', ''),
                # V4 calibration fields
                "tau": f"{result.get('tau', args.tau):.4f}",
                "p_correct": f"{result.get('p_correct', -1.0):.6f}",
                "p_correct_pre_escalation": f"{result.get('p_correct_pre_escalation', -1.0):.6f}",
                "p_correct_post_escalation": f"{result.get('p_correct_post_escalation', -1.0):.6f}",
                "calibrator_version": result.get('calibrator_version', 'none'),
                "calibrator_loaded": int(result.get('calibrator_loaded', False)),
                "feature_extraction_ms": f"{result.get('feature_extraction_ms', 0.0):.3f}",
                "calibrator_ms": f"{result.get('calibrator_ms', 0.0):.3f}",
                "routing_logic_ms": f"{result.get('routing_logic_ms', 0.0):.3f}",
                # Energy (placeholder)
                "energy_start_mwh": "NA",
                "energy_end_mwh": "NA",
                "energy_delta_mwh": "NA",
                "energy_per_prompt_mwh": "NA",
                # Error
                "error_code": result['error_code'],
                # Reproducibility
                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
                "timeout_sec": config['timeout_sec'],
                "config_version": config['config_version'],
                "router_version": "v4.0",
            }

            trials.append(trial)

            # Progress output
            status = "OK" if result['correct'] else ("T" if result['timeout_flag'] else "X")
            route_display = " -> ".join(result['route_sequence'])
            p_str = f" p={result.get('p_correct', -1):.2f}" if result.get('p_correct', -1) >= 0 else ""

            mode_indicator = ""
            if result['symbolic_parse_success']:
                mode_indicator += "S"
            if result['sympy_solve_success']:
                mode_indicator += "P"
            if result.get('repair_attempted', False):
                mode_indicator += "R+" if result.get('repair_success', False) else "R"
            mode_str = f" [{mode_indicator}]" if mode_indicator else ""

            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"route={route_display}{mode_str} "
                  f"mode={result['reasoning_mode']}{p_str} "
                  f"lat={result['total_latency_ms']:.0f}ms{step_lat_str} "
                  f"ans={result['answer_final']} exp={ground_truth} "
                  f"err={result['error_code']} {result.get('decision_reason', '')}")

            if repeat_idx < config['repeats']:
                time.sleep(0.1)

    # Energy end reading
    print()
    print("=" * 60)
    print("ENERGY MEASUREMENT")
    print("=" * 60)
    print("Check your USB power meter now.")
    print()
    end_mwh_str = input("Enter ENDING mWh reading (just the number): ").strip()
    end_mwh_clean = re.sub(r'[^\d.\-+]', '', end_mwh_str)

    try:
        end_mwh = float(end_mwh_clean)
        print(f"Recorded ending: {end_mwh} mWh (raw input: '{end_mwh_str}')")

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
        print(f"Invalid input (raw: '{end_mwh_str}', cleaned: '{end_mwh_clean}'), energy marked as NA")

    print("=" * 60)
    print()

    # Write trials CSV
    fieldnames = [
        "run_id", "timestamp", "prompt_id", "dataset", "category", "split", "system", "repeat_idx",
        "prompt_template_version", "parser_version",
        "route_chosen", "route_attempt_sequence", "escalations_count", "decision_reason", "final_answer_source",
        "reasoning_mode", "symbolic_parse_success", "sympy_solve_success",
        "repair_attempted", "repair_success", "repair_trigger_reason", "previous_raw_len", "previous_action_id",
        "answer_raw", "answer_parsed", "parse_success", "ground_truth", "correct",
        "total_latency_ms", "timeout_flag",
        "timeout_policy_version", "action_timeout_sec_used", "timeout_reason",
        "tau", "p_correct", "p_correct_pre_escalation", "p_correct_post_escalation",
        "calibrator_version", "calibrator_loaded",
        "feature_extraction_ms", "calibrator_ms", "routing_logic_ms",
        "energy_start_mwh", "energy_end_mwh", "energy_delta_mwh", "energy_per_prompt_mwh",
        "error_code",
        "model_name", "quantization", "temperature", "top_p", "top_k", "seed", "timeout_sec", "config_version", "router_version"
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

    escalation_counts = [int(t['escalations_count']) for t in trials]
    total_escalations = sum(escalation_counts)
    prompts_with_escalation = sum(1 for e in escalation_counts if e > 0)

    symbolic_parse_count = sum(t['symbolic_parse_success'] for t in trials)
    sympy_solve_count = sum(t['sympy_solve_success'] for t in trials)

    all_latencies = sorted(float(t['total_latency_ms']) for t in trials)
    nontimeout_latencies = sorted(float(t['total_latency_ms']) for t in trials if not t['timeout_flag'])
    median_all = all_latencies[len(all_latencies)//2] if all_latencies else 0
    median_nontimeout = nontimeout_latencies[len(nontimeout_latencies)//2] if nontimeout_latencies else 0

    print(f"SUMMARY:")
    print(f"  Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"  Timeouts: {timeout_count}/{total}")
    print(f"  Parse failures: {parse_fail_count}/{total}")
    print(f"  Median latency (all trials): {median_all:.0f}ms")
    print(f"  Median latency (non-timeout): {median_nontimeout:.0f}ms")
    print(f"  Total escalations: {total_escalations}")
    print(f"  Prompts needing fallback: {prompts_with_escalation}/{len(prompts)}")
    print(f"  Symbolic parse success: {symbolic_parse_count}/{total}")
    print(f"  SymPy solve success: {sympy_solve_count}/{total}")
    print(f"  Tau: {args.tau}")
    print(f"  Calibrator: {'loaded' if router.calibrator_loaded else 'NOT loaded'}")

    # WP A3 repair stats
    wp_a3_rate = (100 * wp_a3_successes / wp_a3_attempts) if wp_a3_attempts > 0 else 0
    print(f"\n  WP A3 repair stats:")
    print(f"    wp_a3_attempts: {wp_a3_attempts}")
    print(f"    wp_a3_successes: {wp_a3_successes}")
    print(f"    wp_a3_success_rate: {wp_a3_rate:.1f}%")

    # Per-category accuracy
    print(f"\n  Per-category:")
    from collections import Counter
    cat_correct = Counter()
    cat_total = Counter()
    route_counts = Counter()
    for t in trials:
        cat = t['category']
        cat_total[cat] += 1
        cat_correct[cat] += t['correct']
        route_counts[t['route_chosen']] += 1
    for cat in sorted(cat_total.keys()):
        c, n = cat_correct[cat], cat_total[cat]
        print(f"    {cat:4}: {c}/{n} correct ({100*c/n:.1f}%)")

    print(f"\n  Route usage:")
    for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
        print(f"    {route}: {count}")

    print(f"\nSaved to: {args.out_trials}")


if __name__ == "__main__":
    main()
