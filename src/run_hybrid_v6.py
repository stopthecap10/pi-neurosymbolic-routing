#!/usr/bin/env python3
"""
Hybrid V6 Runner — All 4 categories symbolic
AR:  A5 (arithmetic eval)
ALG: A4 (SymPy solve)
LOG: A6 (forward-chain logic)
WP:  A7 (word problem solver) -> A2 -> A3 fallback
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

from router_v6 import RouterV6

PARSER_VERSION = "P3_numeric_context"
PROMPT_TEMPLATE_VERSION = "PT3_phi_chat"

SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_routing_decisions(decisions_path=None):
    """V6 routing decisions: all 4 categories route to symbolic primary."""
    return {
        'category_routes': {
            'AR': {'action': 'A5', 'grammar_enabled': False},
            'ALG': {'action': 'A4', 'grammar_enabled': False},
            'WP': {'action': 'A7', 'grammar_enabled': False},
            'LOG': {'action': 'A6', 'grammar_enabled': False},
        },
        'max_escalations': 3,
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

    # Initialize V6 router
    router = RouterV6(config, routing_decisions)
    system_name = "hybrid_v6"

    print(f"Running {system_name}")
    print(f"Router: V6 (all 4 categories symbolic)")
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
    print("V6 policy (neurosymbolic: symbolic compute + LLM verify):")
    print("  AR:  A5 (symbolic arithmetic)")
    print("  ALG: A4 (SymPy solve)")
    print("  WP:  A7 (symbolic solver) + LLM verify (logged only) -> A2 -> A3 fallback")
    print("  LOG: A6 (symbolic forward-chain) -> A1 fallback")
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

    # Track stats
    a6_attempts = 0
    a6_successes = 0
    a6_fallbacks = 0
    a7_attempts = 0
    a7_successes = 0
    a7_fallbacks = 0
    llm_verify_agree = 0
    llm_verify_disagree = 0
    llm_verify_error = 0

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

            # Track A6 stats
            if category == "LOG":
                a6_attempts += 1
                route_seq = result.get('route_sequence', [])
                if 'A6' in route_seq:
                    if result['correct']:
                        a6_successes += 1
                    if len(route_seq) > 1:
                        a6_fallbacks += 1

            # Track A7 stats
            if category == "WP":
                a7_attempts += 1
                route_seq = result.get('route_sequence', [])
                if 'A7' in route_seq:
                    if result['correct']:
                        a7_successes += 1
                    if len(route_seq) > 1:
                        a7_fallbacks += 1

                # Track LLM verification stats
                for entry in result.get('route_log', []):
                    r = entry.get('result', {})
                    if r.get('llm_agrees') is True:
                        llm_verify_agree += 1
                    elif r.get('llm_agrees') is False:
                        llm_verify_disagree += 1
                    elif r.get('llm_verify_error', ''):
                        llm_verify_error += 1

            # Build step latency string
            step_latencies = result.get('step_latencies', [])
            if len(step_latencies) > 1:
                parts = [f"{a}={lat:.0f}ms" for a, lat in step_latencies]
                step_lat_str = f" ({'+'.join(parts)})"
            else:
                step_lat_str = ""

            # Get A6-specific fields
            a6_fields = {}
            a7_fields = {}
            for entry in result.get('route_log', []):
                r = entry.get('result', {})
                if r.get('a6_parse_success') is not None:
                    a6_fields = {
                        'a6_parse_success': int(r.get('a6_parse_success', False)),
                        'a6_n_facts': r.get('a6_n_facts', 0),
                        'a6_n_rules': r.get('a6_n_rules', 0),
                        'a6_n_derived': r.get('a6_n_derived', 0),
                        'a6_rule_fired': r.get('a6_rule_fired', ''),
                        'a6_pattern': r.get('a6_pattern', ''),
                    }
                if r.get('a7_parse_success') is not None:
                    a7_fields = {
                        'a7_parse_success': int(r.get('a7_parse_success', False)),
                        'a7_strategy': r.get('a7_strategy', ''),
                        'a7_n_steps': r.get('a7_n_steps', 0),
                        'a7_steps': r.get('a7_steps', ''),
                        # LLM verification (logged only, does not affect answer)
                        'llm_agrees': r.get('llm_agrees', ''),
                        'llm_verify_raw': r.get('llm_verify_raw', ''),
                        'llm_verify_latency_ms': f"{r.get('llm_verify_latency_ms', 0):.0f}",
                        'llm_verify_error': r.get('llm_verify_error', ''),
                    }

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
                # A6-specific fields
                "a6_parse_success": a6_fields.get('a6_parse_success', ''),
                "a6_n_facts": a6_fields.get('a6_n_facts', ''),
                "a6_n_rules": a6_fields.get('a6_n_rules', ''),
                "a6_n_derived": a6_fields.get('a6_n_derived', ''),
                "a6_rule_fired": a6_fields.get('a6_rule_fired', ''),
                "a6_pattern": a6_fields.get('a6_pattern', ''),
                # A7-specific fields
                "a7_parse_success": a7_fields.get('a7_parse_success', ''),
                "a7_strategy": a7_fields.get('a7_strategy', ''),
                "a7_n_steps": a7_fields.get('a7_n_steps', ''),
                "a7_steps": a7_fields.get('a7_steps', ''),
                # LLM verification (logged only)
                "llm_agrees": a7_fields.get('llm_agrees', ''),
                "llm_verify_raw": a7_fields.get('llm_verify_raw', ''),
                "llm_verify_latency_ms": a7_fields.get('llm_verify_latency_ms', ''),
                "llm_verify_error": a7_fields.get('llm_verify_error', ''),
                # Energy
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
                "router_version": "v6.0",
            }

            trials.append(trial)

            # Progress output
            status = "OK" if result['correct'] else ("T" if result['timeout_flag'] else "X")
            route_display = " -> ".join(result['route_sequence'])

            mode_indicator = ""
            if result['symbolic_parse_success']:
                mode_indicator += "S"
            if result['sympy_solve_success']:
                mode_indicator += "P"
            if result.get('repair_attempted', False):
                mode_indicator += "R+" if result.get('repair_success', False) else "R"
            if a6_fields.get('a6_parse_success'):
                mode_indicator += "L"
            if a7_fields.get('a7_parse_success'):
                mode_indicator += "W"  # Word problem symbolic
                # Show LLM verification result
                llm_v = a7_fields.get('llm_agrees', '')
                if llm_v is True:
                    mode_indicator += "✓"  # LLM agreed
                elif llm_v is False:
                    mode_indicator += "✗"  # LLM disagreed
            mode_str = f" [{mode_indicator}]" if mode_indicator else ""

            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"route={route_display}{mode_str} "
                  f"mode={result['reasoning_mode']} "
                  f"lat={result['total_latency_ms']:.0f}ms{step_lat_str} "
                  f"ans={result['answer_final']} exp={ground_truth} "
                  f"err={result['error_code']}")

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
        "a6_parse_success", "a6_n_facts", "a6_n_rules", "a6_n_derived", "a6_rule_fired", "a6_pattern",
        "a7_parse_success", "a7_strategy", "a7_n_steps", "a7_steps",
        "llm_agrees", "llm_verify_raw", "llm_verify_latency_ms", "llm_verify_error",
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

    # A6 stats
    a6_rate = (100 * a6_successes / a6_attempts) if a6_attempts > 0 else 0
    print(f"\n  A6 logic solver stats:")
    print(f"    a6_attempts: {a6_attempts}")
    print(f"    a6_correct: {a6_successes}")
    print(f"    a6_accuracy: {a6_rate:.1f}%")
    print(f"    a6_fallbacks_to_A1: {a6_fallbacks}")

    # A7 stats
    a7_rate = (100 * a7_successes / a7_attempts) if a7_attempts > 0 else 0
    print(f"\n  A7 WP solver stats:")
    print(f"    a7_attempts: {a7_attempts}")
    print(f"    a7_correct: {a7_successes}")
    print(f"    a7_accuracy: {a7_rate:.1f}%")
    print(f"    a7_fallbacks_to_A2: {a7_fallbacks}")

    # LLM verification stats
    llm_verify_total = llm_verify_agree + llm_verify_disagree + llm_verify_error
    print(f"\n  LLM verification stats (logged only, does NOT affect answers):")
    print(f"    llm_verify_total: {llm_verify_total}")
    print(f"    llm_agrees: {llm_verify_agree}")
    print(f"    llm_disagrees: {llm_verify_disagree}")
    print(f"    llm_verify_errors: {llm_verify_error}")
    if llm_verify_total > 0:
        agree_pct = 100 * llm_verify_agree / llm_verify_total
        print(f"    agreement_rate: {agree_pct:.1f}%")

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
