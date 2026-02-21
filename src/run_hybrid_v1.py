#!/usr/bin/env python3
"""
Hybrid V1 Runner - Deterministic Rule-Based Router
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml

from router_v1 import RouterV1

PARSER_VERSION = "P2_numeric_robust"
PROMPT_TEMPLATE_VERSION = "PT3_phi_chat"

# System messages for Phi chat format
SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."

def load_config(config_path):
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_routing_decisions(decisions_path=None):
    """
    Load routing decisions

    For now, returns hardcoded initial routing map.
    Later, can load from v1_decisions_for_hybrid.md
    """
    # Default routing map (can be overridden)
    return {
        'category_routes': {
            'AR': {
                'action': 'A5',  # Direct symbolic computation
                'grammar_enabled': False
            },
            'ALG': {
                'action': 'A1',  # Try A1 first (A4 not implemented yet)
                'grammar_enabled': False
            },
            'WP': {
                'action': 'A1',  # Start with A1, fallback to A2
                'grammar_enabled': False
            },
            'LOG': {
                'action': 'A1',  # Fast yes/no
                'grammar_enabled': False
            }
        },
        'max_escalations': 1,
        'fallback_rules': {
            'timeout': 'escalate_to_A2',
            'parse_fail': 'escalate_to_A2',
            'symbolic_fail': 'fallback_to_A1'
        }
    }

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
    ap.add_argument("--decisions", help="Routing decisions file (optional)")
    ap.add_argument("--out_trials", required=True, help="Output CSV")
    ap.add_argument("--split_role", choices=["official", "dev"], default="dev",
                    help="Set to 'official' to enforce provenance guardrails")
    args = ap.parse_args()

    # Load config and routing decisions
    config = load_config(args.config)
    routing_decisions = load_routing_decisions(args.decisions)

    # Load prompts
    with open(args.csv, 'r', encoding='utf-8') as f:
        prompts = list(csv.DictReader(f))

    # Official-mode guardrail
    if args.split_role == "official":
        verify_official_split(args.csv, prompts)

    # Initialize router
    router = RouterV1(config, routing_decisions)

    system_name = "hybrid_v1"

    print(f"Running {system_name}")
    print(f"Router: Deterministic rule-based")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print()
    print("Category routing:")
    for cat, route_info in routing_decisions['category_routes'].items():
        print(f"  {cat:4} → {route_info['action']} (grammar: {route_info['grammar_enabled']})")
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

        # Build Phi chat template prompt (PT3_phi_chat)
        # Strip any existing instruction suffix from CSV
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
            # Route through Hybrid V1
            result = router.route(
                prompt_id=prompt_id,
                category=category,
                prompt_text=prompt_text,
                ground_truth=ground_truth
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

                # Version tracking
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "parser_version": PARSER_VERSION,

                # Routing info
                "route_chosen": result['final_source'],
                "route_attempt_sequence": result['route_attempt_sequence'],
                "escalations_count": result['escalations_count'],
                "decision_reason": result['decision_reason'],
                "final_answer_source": result['final_source'],

                # Output
                "answer_raw": result['answer_raw'].replace("\n", "\\n")[:500],
                "answer_parsed": result['answer_final'],
                "parse_success": int(result['parse_success']),
                "ground_truth": ground_truth,
                "correct": int(result['correct']),

                # Runtime
                "total_latency_ms": f"{result['total_latency_ms']:.3f}",
                "timeout_flag": int(result['timeout_flag']),

                # Energy (placeholder)
                "energy_start_mwh": "NA",
                "energy_end_mwh": "NA",
                "energy_delta_mwh": "NA",
                "energy_per_prompt_mwh": "NA",

                # Error
                "error_code": result['error_code'],

                # Reproducibility (all frozen)
                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_sec": config['timeout_sec'],
                "config_version": config['config_version'],
                "router_version": "v1.0",
            }

            trials.append(trial)

            # Print progress
            status = "✓" if result['correct'] else ("T" if result['timeout_flag'] else "✗")
            route_display = " → ".join(result['route_sequence'])
            print(f"{status} {prompt_id} {category} #{repeat_idx}/{config['repeats']} "
                  f"route={route_display} lat={result['total_latency_ms']:.0f}ms "
                  f"ans={result['answer_final']} exp={ground_truth} err={result['error_code']}")

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
        "prompt_template_version", "parser_version",
        "route_chosen", "route_attempt_sequence", "escalations_count", "decision_reason", "final_answer_source",
        "answer_raw", "answer_parsed", "parse_success", "ground_truth", "correct",
        "total_latency_ms", "timeout_flag",
        "energy_start_mwh", "energy_end_mwh", "energy_delta_mwh", "energy_per_prompt_mwh",
        "error_code",
        "model_name", "quantization", "temperature", "top_p", "seed", "timeout_sec", "config_version", "router_version"
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

    # Escalation stats
    escalation_counts = [int(t['escalations_count']) for t in trials]
    total_escalations = sum(escalation_counts)
    prompts_with_escalation = sum(1 for e in escalation_counts if e > 0)

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
    print(f"  Total escalations: {total_escalations}")
    print(f"  Prompts needing fallback: {prompts_with_escalation}/{len(prompts)}")
    print(f"\n✅ Saved to: {args.out_trials}")

if __name__ == "__main__":
    main()
