#!/usr/bin/env python3
"""
L* learned routing -> V5 solver, END-TO-END, at a HIGH WP token budget (512 + CoT).

This is the experiment that fills the gap in Table II: the existing end-to-end
results (`run_gi_experiments.py`, Experiment 5) build RouterV5 with the default
30-token WP budget, which caps WP accuracy at ~12%. Here we run the *same*
DFA-classification -> V5-dispatch pipeline but with the 512-token chain-of-thought
WP configuration used by the best hand-coded system (V5, 512tok).

Pipeline per prompt:
  1. tokenize(prompt) -> token-class sequence
  2. MultiClassDFA.classify(seq) -> predicted category (L*-learned DFAs)
  3. RouterV5 dispatches to the symbolic solver (AR/ALG/LOG) or to A2 CoT (WP, 512 tok)

DFA source:
  - If --dfa <path> is given, the learned MultiClassDFA is loaded from JSON
    (e.g. outputs/gi_results/lstar_slm_dfas.json from a prior SLM-oracle run).
  - Otherwise the DFAs are learned locally with the feature oracle, which gives
    identical 100% routing on T2 to the SLM oracle (Table V) and needs no SLM.
    Routing is therefore identical; only WP *solving* needs the llama.cpp server.

Usage (on the Pi, with llama.cpp running):
  python3 -m src.run_lstar_v5_512 --server http://127.0.0.1:8080 \
      --out_trials outputs/lstar_v5_512_trials.csv

  # Reuse the exact SLM-oracle DFAs from the main run:
  python3 -m src.run_lstar_v5_512 --server http://127.0.0.1:8080 \
      --dfa outputs/gi_results/lstar_slm_dfas.json \
      --out_trials outputs/lstar_v5_512_trials.csv

  # Quick check: WP only, 1 repeat
  python3 -m src.run_lstar_v5_512 --server http://127.0.0.1:8080 --wp_only --repeats 1
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

# RouterV5 uses bare imports (from router_v3 import ...), so src/ must be on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from token_classifier import tokenize, ALPHABET
from dfa import MultiClassDFA
from feature_classifier import create_feature_oracle
from lstar import LStar, ground_truth_equivalence_oracle
from router_v5 import RouterV5

T1_CSV = "data/splits/industry_tier1_40_v2.csv"
T2_CSV = "data/splits/industry_tier2_100.csv"
PRIORITY = ("LOG", "ALG", "AR", "WP")

SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."


def read_mwh(label):
    """Prompt for a USB-power-meter reading; returns float or None."""
    try:
        s = input(f"Enter {label} mWh reading (or Enter to skip): ").strip()
    except EOFError:
        return None
    if not s:
        return None
    cleaned = re.sub(r'[^\d.\-+]', '', s)
    try:
        v = float(cleaned)
        print(f"Recorded {label}: {v} mWh")
        return v
    except ValueError:
        print(f"Invalid input '{s}' -> NA")
        return None


def build_phi_prompt(base_prompt, category):
    """Build the Phi chat template (identical to run_hybrid_v5.py / Experiment 5)."""
    question = base_prompt
    lines_q = question.rstrip().split('\n')
    while lines_q and lines_q[-1].strip() in ("Answer:", ""):
        lines_q.pop()
    while lines_q and lines_q[-1].strip().startswith("Return only"):
        lines_q.pop()
    question = '\n'.join(lines_q).strip()
    system_msg = SYSTEM_MSG_YESNO if category == "LOG" else SYSTEM_MSG_NUMERIC
    return f"<|system|>{system_msg}<|end|><|user|>{question}<|end|><|assistant|>"


def load_or_learn_dfas(dfa_path):
    """Load a saved MultiClassDFA, or learn one locally with the feature oracle."""
    if dfa_path:
        if not os.path.exists(dfa_path):
            print(f"ERROR: --dfa path not found: {dfa_path}")
            sys.exit(1)
        print(f"Loading learned DFAs from {dfa_path}")
        return MultiClassDFA.from_json(dfa_path)

    print("No --dfa given: learning L* DFAs locally with the feature oracle")
    print("(identical T2 routing to the SLM oracle; no SLM needed for routing).")
    train_data = []
    with open(T1_CSV) as f:
        for row in csv.DictReader(f):
            train_data.append((tokenize(row['prompt_text']), row['category']))

    dfas = {}
    for cat in ("AR", "ALG", "WP", "LOG"):
        mem_oracle = create_feature_oracle(cat)
        eq_oracle = ground_truth_equivalence_oracle(train_data, cat)
        dfas[cat] = LStar(ALPHABET, mem_oracle, eq_oracle).learn()
        print(f"  {cat}: learned DFA with {dfas[cat].num_states} states")
    return MultiClassDFA(dfas, priority=PRIORITY)


def main():
    ap = argparse.ArgumentParser(description="L* routing -> V5 solver @ 512-token CoT")
    ap.add_argument("--server", default="http://127.0.0.1:8080",
                    help="llama.cpp server URL")
    ap.add_argument("--dfa", default=None,
                    help="Path to a saved MultiClassDFA JSON (e.g. lstar_slm_dfas.json). "
                         "If omitted, DFAs are learned locally with the feature oracle.")
    ap.add_argument("--wp_tokens", type=int, default=512,
                    help="WP A2 token budget (default: 512)")
    ap.add_argument("--no_cot", action="store_true",
                    help="Disable chain-of-thought for WP (default: CoT on)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Repeats per prompt (default: 3, matching the paper)")
    ap.add_argument("--wp_only", action="store_true",
                    help="Run only WP prompts (quick WP check)")
    ap.add_argument("--out_trials", default="outputs/lstar_v5_512_trials.csv",
                    help="Per-trial output CSV")
    ap.add_argument("--out_json", default=None,
                    help="Summary JSON output (default: alongside out_trials)")
    args = ap.parse_args()

    wp_cot = not args.no_cot
    os.makedirs(os.path.dirname(os.path.abspath(args.out_trials)), exist_ok=True)
    out_json = args.out_json or os.path.splitext(args.out_trials)[0] + "_summary.json"

    # --- Build the learned router classifier ---
    dfa_mc = load_or_learn_dfas(args.dfa)

    # --- V5 router with the high WP budget ---
    v5_config = {
        'model_name': 'Phi-4-Mini',
        'quantization': 'Q6_K',
        'server_url': f"{args.server}/completion",
        'timeout_sec': 20,
        'temperature': 0.0,
        'top_p': 1.0,
        'top_k': 1,
        'seed': 42,
        'repeats': args.repeats,
        'api_mode': 'chat',
        'config_version': 'v1.0',
        'n_pred_num': 12,
        'n_pred_log': 6,
    }
    v5_routing_decisions = {
        'category_routes': {
            'AR': {'action': 'A5', 'grammar_enabled': False},
            'ALG': {'action': 'A4', 'grammar_enabled': False},
            'WP': {'action': 'A2', 'grammar_enabled': False},
            'LOG': {'action': 'A6', 'grammar_enabled': False},
        },
        'max_escalations': 2,
    }
    router = RouterV5(v5_config, v5_routing_decisions,
                      wp_token_budget=args.wp_tokens, wp_cot=wp_cot)

    # --- Load T2 ---
    t2_prompts = []
    with open(T2_CSV) as f:
        for row in csv.DictReader(f):
            if args.wp_only and row['category'] != 'WP':
                continue
            t2_prompts.append(row)

    system_name = f"lstar_v5_wp{args.wp_tokens}{'_cot' if wp_cot else ''}"
    print()
    print("=" * 60)
    print(f"L* routing -> V5 solver  |  WP budget={args.wp_tokens}  CoT={wp_cot}")
    print(f"DFA source: {args.dfa or 'feature-oracle (learned locally)'}")
    print(f"Prompts: {len(t2_prompts)}  Repeats: {args.repeats}")
    print("=" * 60)

    start_mwh = read_mwh("STARTING")

    run_id = f"{system_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trials = []
    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    routing_correct = 0
    trial_num = 0
    total_trials = len(t2_prompts) * args.repeats

    for row in t2_prompts:
        prompt_id = row['prompt_id']
        true_cat = row['category']
        ground_truth = row['ground_truth']

        toks = tokenize(row['prompt_text'])
        predicted_cat = dfa_mc.classify(toks) or 'WP'  # default fallback
        if predicted_cat == true_cat:
            routing_correct += 1

        prompt_text = build_phi_prompt(row['prompt_text'], predicted_cat)

        for repeat in range(1, args.repeats + 1):
            trial_num += 1
            result = router.route(
                prompt_id=prompt_id,
                category=predicted_cat,
                prompt_text=prompt_text,
                ground_truth=ground_truth,
            )

            by_cat[true_cat]['total'] += 1
            if result['correct']:
                by_cat[true_cat]['correct'] += 1

            trials.append({
                'run_id': run_id,
                'prompt_id': prompt_id,
                'true_category': true_cat,
                'predicted_category': predicted_cat,
                'category_match': int(predicted_cat == true_cat),
                'repeat': repeat,
                'correct': int(result['correct']),
                'answer_parsed': result['answer_final'],
                'ground_truth': ground_truth,
                'parse_success': int(result['parse_success']),
                'total_latency_ms': f"{result['total_latency_ms']:.3f}",
                'timeout_flag': int(result['timeout_flag']),
                'route_attempt_sequence': result['route_attempt_sequence'],
                'error_code': result['error_code'],
            })

            status = "OK" if result['correct'] else ("T" if result['timeout_flag'] else "X")
            match = "=" if predicted_cat == true_cat else f"!={predicted_cat}"
            print(f"[{trial_num}/{total_trials}] {status} {prompt_id} {true_cat}{match} "
                  f"r{repeat} lat={result['total_latency_ms']:.0f}ms "
                  f"ans={result['answer_final']} exp={ground_truth} err={result['error_code']}")
            time.sleep(0.1)

    end_mwh = read_mwh("ENDING")

    # --- Aggregate ---
    total = len(trials)
    correct = sum(t['correct'] for t in trials)
    latencies = sorted(float(t['total_latency_ms']) for t in trials)
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    median_lat = latencies[len(latencies) // 2] if latencies else 0
    n_prompts = len(t2_prompts)

    energy_total = energy_per_prompt = None
    if start_mwh is not None and end_mwh is not None:
        energy_total = end_mwh - start_mwh
        energy_per_prompt = energy_total / n_prompts if n_prompts else None

    for cat in by_cat:
        bc = by_cat[cat]
        bc['accuracy'] = bc['correct'] / bc['total'] if bc['total'] else 0

    # --- Write per-trial CSV ---
    fieldnames = list(trials[0].keys()) if trials else []
    with open(args.out_trials, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(trials)

    summary = {
        'system': system_name,
        'run_id': run_id,
        'wp_tokens': args.wp_tokens,
        'wp_cot': wp_cot,
        'dfa_source': args.dfa or 'feature_oracle_local',
        'repeats': args.repeats,
        'total_accuracy': correct / total if total else 0,
        'correct': correct,
        'total': total,
        'routing_accuracy': routing_correct / n_prompts if n_prompts else 0,
        'routing_correct': routing_correct,
        'per_category': dict(by_cat),
        'avg_latency_ms': avg_lat,
        'median_latency_ms': median_lat,
        'energy_start_mwh': start_mwh,
        'energy_end_mwh': end_mwh,
        'energy_total_mwh': energy_total,
        'energy_per_prompt_mwh': energy_per_prompt,
    }
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("SUMMARY: L* routing -> V5 @ %d tokens%s" % (args.wp_tokens, " (CoT)" if wp_cot else ""))
    print("=" * 60)
    print(f"  Routing accuracy: {routing_correct}/{n_prompts} = "
          f"{100*routing_correct/n_prompts:.1f}%" if n_prompts else "  (no prompts)")
    print(f"  Task accuracy:    {correct}/{total} = {100*correct/total:.1f}%")
    for cat in ('AR', 'ALG', 'WP', 'LOG'):
        if cat in by_cat:
            bc = by_cat[cat]
            print(f"    {cat:4}: {bc['correct']}/{bc['total']} = {100*bc['accuracy']:.1f}%")
    print(f"  Mean latency:   {avg_lat:.0f} ms")
    print(f"  Median latency: {median_lat:.0f} ms")
    if energy_per_prompt is not None:
        print(f"  Energy: {energy_total:.2f} mWh total, {energy_per_prompt:.2f} mWh/prompt")
    print(f"\nTrials -> {args.out_trials}")
    print(f"Summary -> {out_json}")


if __name__ == "__main__":
    main()
