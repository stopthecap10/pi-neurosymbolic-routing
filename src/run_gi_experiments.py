#!/usr/bin/env python3
"""
Run all grammatical inference experiments on the Pi.

This script:
  1. Trains L* with SLM oracle (novel contribution)
  2. Trains L* with feature oracle (comparison)
  3. Trains RPNI (passive baseline)
  4. Evaluates all three on T2 (100 prompts)
  5. Runs tool-calling agent baseline on T2
  6. Saves all results to outputs/gi_results/

Usage (on the Pi, with llama.cpp running):
  python3 -m src.run_gi_experiments --server http://127.0.0.1:8080

Without Pi (feature oracle + RPNI only):
  python3 -m src.run_gi_experiments --no-slm
"""

import argparse
import csv
import json
import os
import time
import requests
from collections import defaultdict
from datetime import datetime

from src.token_classifier import tokenize, ALPHABET
from src.dfa import DFA, MultiClassDFA
from src.rpni import learn_one_vs_rest as rpni_learn
from src.lstar import LStar, ground_truth_equivalence_oracle
from src.feature_classifier import create_feature_oracle, classify_by_features


# ---- SLM-based membership oracle ----

class SLMOracle:
    """Membership oracle that ACTUALLY queries Phi-4-mini on the Pi.

    For each prompt, asks the SLM: "Is this a [category] problem? Yes/No."

    Approach:
    1. Pre-query the SLM for all known prompts (training data)
    2. Cache results keyed by token sequence
    3. For synthetic L* queries (not real prompts), use feature classifier
       as fallback (these are intermediate L* exploration strings)
    """

    CATEGORY_DESCRIPTIONS = {
        "AR": "a pure arithmetic calculation with only numbers and operators",
        "ALG": "an algebra problem with variables to solve for",
        "WP": "a word problem requiring multi-step reasoning",
        "LOG": "a formal logic problem with facts, rules, and a yes/no question",
    }

    def __init__(self, target_category: str, server_url: str,
                 token_to_text: dict, cache_path: str = None):
        self.target_category = target_category
        self.server_url = server_url.rstrip('/')
        self.desc = self.CATEGORY_DESCRIPTIONS[target_category]
        self.token_to_text = token_to_text  # token_key -> original text
        self.cache = {}
        self.cache_path = cache_path
        self.slm_query_count = 0  # actual SLM calls
        self.feature_fallback_count = 0  # feature classifier fallbacks
        self.cache_hits = 0

        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    def __call__(self, token_seq: tuple) -> bool:
        """Membership oracle for L*."""
        key = " ".join(token_seq)

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        # If we have the original text, query the actual SLM
        if key in self.token_to_text:
            result = self._query_slm(self.token_to_text[key])
            self.slm_query_count += 1
        else:
            # Synthetic L* sequence — no original text exists
            # Fall back to feature classifier
            result = classify_by_features(token_seq) == self.target_category
            self.feature_fallback_count += 1

        self.cache[key] = result

        # Periodically save cache
        if self.cache_path and (self.slm_query_count + self.feature_fallback_count) % 20 == 0:
            self.save_cache()

        return result

    def _query_slm(self, text: str) -> bool:
        """Actually query Phi-4-mini: 'Is this problem [category]?'"""
        prompt = (
            f"Is the following problem {self.desc}?\n\n"
            f"{text}\n\n"
            f"Answer Yes or No."
        )

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Answer with one word: Yes or No."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 6,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                },
                timeout=(10.0, 120),
            )
            answer = response.json()["choices"][0]["message"]["content"].strip().lower()
            result = answer.startswith("yes")
            return result
        except Exception as e:
            print(f"    SLM oracle error: {e}")
            # On error, fall back to feature classifier
            return classify_by_features(tokenize(text)) == self.target_category

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)


# ---- Main experiment runner ----

def run_experiments(args):
    out_dir = "outputs/gi_results"
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    train_data = []
    train_texts = {}
    with open("data/splits/industry_tier1_40_v2.csv") as f:
        for row in csv.DictReader(f):
            toks = tokenize(row['prompt_text'])
            train_data.append((toks, row['category']))
            train_texts[" ".join(toks)] = row['prompt_text']

    test_data = []
    with open("data/splits/industry_tier2_100.csv") as f:
        for row in csv.DictReader(f):
            toks = tokenize(row['prompt_text'])
            test_data.append((toks, row['category'], row['prompt_id']))

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- Experiment 1: RPNI (passive, no oracle) ----
    print("\n" + "="*50)
    print("Experiment 1: RPNI (passive grammatical inference)")
    print("="*50)

    rpni_start_mwh = None
    try:
        s = input("Enter STARTING mWh for RPNI experiment (or Enter to skip): ").strip()
        if s:
            import re as re_mod
            rpni_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            print(f"Recorded: {rpni_start_mwh} mWh")
    except (ValueError, EOFError):
        pass

    t0 = time.time()
    rpni_dfas = rpni_learn(ALPHABET, train_data)
    rpni_time = time.time() - t0
    rpni_mc = MultiClassDFA(rpni_dfas, priority=("LOG", "ALG", "AR", "WP"))

    rpni_results = evaluate(rpni_mc, test_data, "RPNI")
    rpni_results['train_time_s'] = rpni_time
    rpni_results['dfa_sizes'] = {cat: dfa.num_states for cat, dfa in rpni_dfas.items()}

    rpni_end_mwh = None
    rpni_energy = None
    try:
        s = input("Enter ENDING mWh for RPNI experiment (or Enter to skip): ").strip()
        if s and rpni_start_mwh is not None:
            import re as re_mod
            rpni_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            rpni_energy = rpni_end_mwh - rpni_start_mwh
            print(f"Energy: {rpni_energy:.2f} mWh total")
    except (ValueError, EOFError):
        pass

    rpni_results['energy_start_mwh'] = rpni_start_mwh
    rpni_results['energy_end_mwh'] = rpni_end_mwh
    rpni_results['energy_total_mwh'] = rpni_energy
    rpni_results['energy_per_prompt_mwh'] = rpni_energy / len(test_data) if rpni_energy is not None else None
    results['rpni'] = rpni_results

    rpni_mc.to_json(os.path.join(out_dir, "rpni_dfas.json"))

    # ---- Experiment 2: L* with feature oracle ----
    print("\n" + "="*50)
    print("Experiment 2: L* with feature-based oracle")
    print("="*50)

    lstar_feat_start_mwh = None
    try:
        s = input("Enter STARTING mWh for L* feature experiment (or Enter to skip): ").strip()
        if s:
            import re as re_mod
            lstar_feat_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            print(f"Recorded: {lstar_feat_start_mwh} mWh")
    except (ValueError, EOFError):
        pass

    t0 = time.time()
    lstar_feat_dfas = {}
    for cat in ("AR", "ALG", "WP", "LOG"):
        mem_oracle = create_feature_oracle(cat)
        eq_oracle = ground_truth_equivalence_oracle(train_data, cat)
        learner = LStar(ALPHABET, mem_oracle, eq_oracle)
        lstar_feat_dfas[cat] = learner.learn()
        print(f"  {cat}: learned DFA with {lstar_feat_dfas[cat].num_states} states")
    lstar_feat_time = time.time() - t0

    lstar_feat_mc = MultiClassDFA(lstar_feat_dfas, priority=("LOG", "ALG", "AR", "WP"))
    lstar_feat_results = evaluate(lstar_feat_mc, test_data, "L* (feature)")
    lstar_feat_results['train_time_s'] = lstar_feat_time
    lstar_feat_results['dfa_sizes'] = {cat: dfa.num_states for cat, dfa in lstar_feat_dfas.items()}

    lstar_feat_end_mwh = None
    lstar_feat_energy = None
    try:
        s = input("Enter ENDING mWh for L* feature experiment (or Enter to skip): ").strip()
        if s and lstar_feat_start_mwh is not None:
            import re as re_mod
            lstar_feat_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            lstar_feat_energy = lstar_feat_end_mwh - lstar_feat_start_mwh
            print(f"Energy: {lstar_feat_energy:.2f} mWh total")
    except (ValueError, EOFError):
        pass

    lstar_feat_results['energy_start_mwh'] = lstar_feat_start_mwh
    lstar_feat_results['energy_end_mwh'] = lstar_feat_end_mwh
    lstar_feat_results['energy_total_mwh'] = lstar_feat_energy
    lstar_feat_results['energy_per_prompt_mwh'] = lstar_feat_energy / len(test_data) if lstar_feat_energy is not None else None
    results['lstar_feature'] = lstar_feat_results

    lstar_feat_mc.to_json(os.path.join(out_dir, "lstar_feature_dfas.json"))

    # ---- Experiment 3: L* with SLM oracle (if Pi available) ----
    if not args.no_slm:
        print("\n" + "="*50)
        print("Experiment 3: L* with SLM oracle (NOVEL)")
        print("="*50)

        # Test SLM connectivity
        try:
            r = requests.get(f"{args.server}/health", timeout=5)
            print(f"  SLM server status: {r.status_code}")
        except Exception as e:
            print(f"  WARNING: Cannot reach SLM at {args.server}: {e}")
            print(f"  Skipping SLM experiments.")
            args.no_slm = True

    if not args.no_slm:
        cache_dir = os.path.join(out_dir, "oracle_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Build token_sequence -> original_text mapping for all known prompts
        token_to_text = {}
        for csv_path in ["data/splits/industry_tier1_40_v2.csv",
                         "data/splits/industry_tier2_100.csv"]:
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    toks = tokenize(row['prompt_text'])
                    key = " ".join(toks)
                    token_to_text[key] = row['prompt_text']
        print(f"  Loaded {len(token_to_text)} prompts for SLM oracle")

        # Prompt for energy reading before SLM oracle queries
        slm_start_mwh = None
        try:
            s = input("Enter STARTING mWh for L* SLM oracle (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                slm_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {slm_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        t0 = time.time()
        lstar_slm_dfas = {}
        total_slm_queries = 0
        total_fallbacks = 0

        for cat in ("AR", "ALG", "WP", "LOG"):
            print(f"\n  Learning DFA for {cat}...")
            cache_path = os.path.join(cache_dir, f"slm_oracle_{cat}.json")
            slm_oracle = SLMOracle(cat, args.server, token_to_text, cache_path)

            eq_oracle = ground_truth_equivalence_oracle(train_data, cat)
            learner = LStar(ALPHABET, slm_oracle, eq_oracle)
            lstar_slm_dfas[cat] = learner.learn()
            slm_oracle.save_cache()

            total_slm_queries += slm_oracle.slm_query_count
            total_fallbacks += slm_oracle.feature_fallback_count
            print(f"  {cat}: {lstar_slm_dfas[cat].num_states} states, "
                  f"{slm_oracle.slm_query_count} SLM queries, "
                  f"{slm_oracle.feature_fallback_count} feature fallbacks, "
                  f"{slm_oracle.cache_hits} cache hits")

        lstar_slm_time = time.time() - t0

        # Prompt for ending energy
        slm_end_mwh = None
        slm_energy_per_query = None
        try:
            s = input("Enter ENDING mWh for L* SLM oracle (or Enter to skip): ").strip()
            if s and slm_start_mwh is not None:
                import re as re_mod
                slm_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                delta = slm_end_mwh - slm_start_mwh
                slm_energy_per_query = delta / total_slm_queries if total_slm_queries else 0
                print(f"Energy: {delta:.2f} mWh total, "
                      f"{slm_energy_per_query:.4f} mWh/query over {total_slm_queries} SLM queries")
        except (ValueError, EOFError):
            pass

        lstar_slm_mc = MultiClassDFA(lstar_slm_dfas, priority=("LOG", "ALG", "AR", "WP"))
        lstar_slm_results = evaluate(lstar_slm_mc, test_data, "L* (SLM)")
        lstar_slm_results['train_time_s'] = lstar_slm_time
        lstar_slm_results['total_slm_queries'] = total_slm_queries
        lstar_slm_results['total_feature_fallbacks'] = total_fallbacks
        lstar_slm_results['dfa_sizes'] = {cat: dfa.num_states for cat, dfa in lstar_slm_dfas.items()}
        slm_energy_total = (slm_end_mwh - slm_start_mwh) if (slm_end_mwh is not None and slm_start_mwh is not None) else None
        lstar_slm_results['energy_start_mwh'] = slm_start_mwh
        lstar_slm_results['energy_end_mwh'] = slm_end_mwh
        lstar_slm_results['energy_total_mwh'] = slm_energy_total
        lstar_slm_results['energy_per_query_mwh'] = slm_energy_per_query
        lstar_slm_results['energy_per_prompt_mwh'] = slm_energy_total / len(test_data) if slm_energy_total is not None else None
        results['lstar_slm'] = lstar_slm_results

        lstar_slm_mc.to_json(os.path.join(out_dir, "lstar_slm_dfas.json"))

    # ---- Experiment 4: Tool-calling agent (if Pi available) ----
    if not args.no_slm:
        print("\n" + "="*50)
        print("Experiment 4: Tool-calling agent baseline")
        print("="*50)

        # Prompt for energy measurement
        start_mwh = None
        try:
            start_mwh_str = input("Enter STARTING mWh reading from USB power meter (or press Enter to skip): ").strip()
            if start_mwh_str:
                import re as re_mod
                start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', start_mwh_str))
                print(f"Recorded starting: {start_mwh} mWh")
        except (ValueError, EOFError):
            print("Skipping energy measurement.")

        from src.agent_baseline import run_benchmark
        agent_results = run_benchmark(
            "data/splits/industry_tier2_100.csv",
            server_url=args.server,
            max_tokens=50,
            repeats=3,
        )

        # Compute latency stats
        latencies = [r['latency_ms'] for r in agent_results['results']]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        median_latency = sorted(latencies)[len(latencies)//2] if latencies else 0

        # Prompt for ending energy
        end_mwh = None
        energy_per_prompt = None
        try:
            end_mwh_str = input("Enter ENDING mWh reading (or press Enter to skip): ").strip()
            if end_mwh_str and start_mwh is not None:
                import re as re_mod
                end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', end_mwh_str))
                delta = end_mwh - start_mwh
                num_unique_prompts = len(test_data)  # 100 unique prompts, consistent with V1-V5
                energy_per_prompt = delta / num_unique_prompts if num_unique_prompts else 0
                print(f"Energy: {delta:.2f} mWh total, {energy_per_prompt:.2f} mWh/prompt (over {num_unique_prompts} unique prompts)")
        except (ValueError, EOFError):
            pass

        results['agent'] = {
            'total_accuracy': agent_results['total_accuracy'],
            'correct': agent_results['correct'],
            'total': agent_results['total'],
            'repeats': agent_results['repeats'],
            'per_category': agent_results['per_category'],
            'tool_usage': agent_results['tool_usage'],
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'energy_start_mwh': start_mwh,
            'energy_end_mwh': end_mwh,
            'energy_per_prompt_mwh': energy_per_prompt,
        }

    # ---- Experiment 5: Feature classifier (hand-coded baseline) ----
    print("\n" + "="*50)
    print("Experiment 5: Feature classifier (hand-coded)")
    print("="*50)

    feat_start_mwh = None
    try:
        s = input("Enter STARTING mWh for feature classifier experiment (or Enter to skip): ").strip()
        if s:
            import re as re_mod
            feat_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            print(f"Recorded: {feat_start_mwh} mWh")
    except (ValueError, EOFError):
        pass

    feat_correct = 0
    feat_by_cat = defaultdict(lambda: [0, 0])
    feat_latencies = []
    for toks, cat, pid in test_data:
        t0 = time.time()
        pred = classify_by_features(toks)
        feat_latencies.append((time.time() - t0) * 1000)
        feat_by_cat[cat][1] += 1
        if pred == cat:
            feat_correct += 1
            feat_by_cat[cat][0] += 1

    feat_avg_latency = sum(feat_latencies) / len(feat_latencies) if feat_latencies else 0
    feat_median_latency = sorted(feat_latencies)[len(feat_latencies)//2] if feat_latencies else 0

    feat_end_mwh = None
    feat_energy = None
    try:
        s = input("Enter ENDING mWh for feature classifier experiment (or Enter to skip): ").strip()
        if s and feat_start_mwh is not None:
            import re as re_mod
            feat_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
            feat_energy = feat_end_mwh - feat_start_mwh
            print(f"Energy: {feat_energy:.2f} mWh total")
    except (ValueError, EOFError):
        pass

    results['feature'] = {
        'total_accuracy': feat_correct / len(test_data),
        'correct': feat_correct,
        'total': len(test_data),
        'per_category': {cat: {'correct': v[0], 'total': v[1],
                               'accuracy': v[0]/v[1] if v[1] else 0}
                         for cat, v in feat_by_cat.items()},
        'avg_latency_ms': feat_avg_latency,
        'median_latency_ms': feat_median_latency,
        'energy_start_mwh': feat_start_mwh,
        'energy_end_mwh': feat_end_mwh,
        'energy_total_mwh': feat_energy,
        'energy_per_prompt_mwh': feat_energy / len(test_data) if feat_energy is not None else None,
    }
    print(f"  Overall: {feat_correct}/{len(test_data)} = {feat_correct/len(test_data):.1%}")
    print(f"  Avg classification latency: {feat_avg_latency:.3f} ms (median: {feat_median_latency:.3f} ms)")

    # ---- Save all results ----
    results_path = os.path.join(out_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    # ---- Print summary table ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'System':<25s} {'AR':>5s} {'ALG':>5s} {'WP':>5s} {'LOG':>5s} {'All':>6s} {'Lat(ms)':>8s} {'mWh':>8s}")
    print("-" * 80)
    for name, key in [("RPNI (passive)", "rpni"),
                       ("L* (feature oracle)", "lstar_feature"),
                       ("L* (SLM oracle)", "lstar_slm"),
                       ("Tool-calling agent", "agent"),
                       ("Feature classifier", "feature")]:
        if key not in results:
            continue
        r = results[key]
        pc = r['per_category']
        parts = []
        for cat in ('AR', 'ALG', 'WP', 'LOG'):
            if cat in pc:
                acc = pc[cat].get('accuracy', pc[cat].get('correct', 0) / max(pc[cat].get('total', 1), 1))
                parts.append(f"{acc:5.0%}")
            else:
                parts.append("  N/A")
        lat = r.get('avg_latency_ms')
        lat_str = f"{lat:8.3f}" if lat is not None else "     N/A"
        epp = r.get('energy_per_prompt_mwh')
        epp_str = f"{epp:8.3f}" if epp is not None else "     N/A"
        print(f"{name:<25s} {'  '.join(parts)}  {r['total_accuracy']:5.1%}  {lat_str}  {epp_str}")

    if 'rpni' in results:
        print(f"\nDFA sizes (states):")
        for key_name, key in [("RPNI", "rpni"), ("L* feature", "lstar_feature"), ("L* SLM", "lstar_slm")]:
            if key in results and 'dfa_sizes' in results[key]:
                sizes = results[key]['dfa_sizes']
                print(f"  {key_name}: " + ", ".join(f"{c}={sizes[c]}" for c in ('AR','ALG','WP','LOG')))


def evaluate(mc, test_data, name):
    """Evaluate a MultiClassDFA on test data."""
    correct = 0
    by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
    latencies = []

    for toks, cat, pid in test_data:
        t0 = time.time()
        pred = mc.classify(toks) or 'UNKNOWN'
        latencies.append((time.time() - t0) * 1000)  # ms
        by_cat[cat]['total'] += 1
        if pred == cat:
            correct += 1
            by_cat[cat]['correct'] += 1

    total = len(test_data)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    median_latency = sorted(latencies)[len(latencies)//2] if latencies else 0

    for cat in by_cat:
        by_cat[cat]['accuracy'] = by_cat[cat]['correct'] / by_cat[cat]['total']

    print(f"  {name}: {correct}/{total} = {correct/total:.1%}")
    print(f"  Avg classification latency: {avg_latency:.3f} ms (median: {median_latency:.3f} ms)")
    for cat in ('AR', 'ALG', 'WP', 'LOG'):
        if cat in by_cat:
            bc = by_cat[cat]
            print(f"    {cat}: {bc['correct']}/{bc['total']} = {bc['accuracy']:.0%}")

    return {
        'total_accuracy': correct / total,
        'correct': correct,
        'total': total,
        'per_category': dict(by_cat),
        'avg_latency_ms': avg_latency,
        'median_latency_ms': median_latency,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GI experiments")
    parser.add_argument("--server", default="http://127.0.0.1:8080",
                        help="llama.cpp server URL")
    parser.add_argument("--no-slm", action="store_true",
                        help="Skip SLM-dependent experiments (run locally)")
    args = parser.parse_args()
    run_experiments(args)
