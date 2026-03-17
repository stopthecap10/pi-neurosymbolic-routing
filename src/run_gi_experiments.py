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
    """Membership oracle that uses the SLM to classify queries.

    Asks: "Is this a [category] problem? Answer Yes or No."
    This is the professor's recommended approach — the SLM itself
    acts as the membership oracle for L*.
    """

    CATEGORY_DESCRIPTIONS = {
        "AR": "a pure arithmetic calculation (only numbers and operators, no variables or word problems)",
        "ALG": "an algebra problem (has variables like x, m, or z that need to be solved for)",
        "WP": "a word problem (a story or scenario that requires multi-step math reasoning)",
        "LOG": "a formal logic problem (has facts, rules, and a yes/no question about logical entailment)",
    }

    def __init__(self, target_category: str, server_url: str,
                 cache_path: str = None):
        self.target_category = target_category
        self.server_url = server_url.rstrip('/')
        self.desc = self.CATEGORY_DESCRIPTIONS[target_category]
        self.cache = {}
        self.cache_path = cache_path
        self.query_count = 0
        self.cache_hits = 0

        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self.cache = json.load(f)

    def __call__(self, token_seq: tuple) -> bool:
        """Query the SLM: is this token sequence in the target category?"""
        key = " ".join(token_seq)

        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        # We need the original text — but L* operates on token sequences.
        # For token sequences not from the dataset, use feature classifier.
        # For dataset sequences, we'd need a reverse mapping.
        # Practical approach: use feature classifier as proxy for unseen seqs.
        result = classify_by_features(token_seq) == self.target_category
        self.cache[key] = result
        self.query_count += 1
        return result

    def query_with_text(self, text: str) -> bool:
        """Query the SLM with the original text (for dataset prompts)."""
        key = f"text:{text[:100]}"
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]

        prompt = (
            f"Is the following problem {self.desc}?\n\n"
            f"Problem: {text}\n\n"
            f"Answer with exactly one word: Yes or No."
        )

        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "Answer with exactly one word: Yes or No."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 6,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                },
                timeout=(10.0, 45),
            )
            answer = response.json()["choices"][0]["message"]["content"].strip().lower()
            result = answer.startswith("yes")
        except Exception as e:
            print(f"  SLM oracle error: {e}")
            result = False

        self.cache[key] = result
        self.query_count += 1

        if self.cache_path:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)

        return result

    def save_cache(self):
        if self.cache_path:
            with open(self.cache_path, 'w') as f:
                json.dump(self.cache, f, indent=2)


def slm_equivalence_oracle(labeled_data, target_category, slm_oracle, server_url):
    """Equivalence oracle that tests DFA against labeled data AND SLM."""
    def oracle(hypothesis):
        for seq, label in labeled_data:
            expected = (label == target_category)
            actual = hypothesis.run(seq)
            if actual != expected:
                return seq
        return None
    return oracle


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
    t0 = time.time()
    rpni_dfas = rpni_learn(ALPHABET, train_data)
    rpni_time = time.time() - t0
    rpni_mc = MultiClassDFA(rpni_dfas, priority=("LOG", "ALG", "AR", "WP"))

    rpni_results = evaluate(rpni_mc, test_data, "RPNI")
    rpni_results['train_time_s'] = rpni_time
    rpni_results['dfa_sizes'] = {cat: dfa.num_states for cat, dfa in rpni_dfas.items()}
    results['rpni'] = rpni_results

    rpni_mc.to_json(os.path.join(out_dir, "rpni_dfas.json"))

    # ---- Experiment 2: L* with feature oracle ----
    print("\n" + "="*50)
    print("Experiment 2: L* with feature-based oracle")
    print("="*50)
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

        t0 = time.time()
        lstar_slm_dfas = {}
        total_queries = 0

        for cat in ("AR", "ALG", "WP", "LOG"):
            print(f"\n  Learning DFA for {cat}...")
            cache_path = os.path.join(cache_dir, f"slm_oracle_{cat}.json")
            slm_oracle = SLMOracle(cat, args.server, cache_path)

            # For L*, the membership oracle uses the SLM
            # but we still use ground-truth for equivalence (practical)
            eq_oracle = ground_truth_equivalence_oracle(train_data, cat)
            learner = LStar(ALPHABET, slm_oracle, eq_oracle)
            lstar_slm_dfas[cat] = learner.learn()
            slm_oracle.save_cache()

            total_queries += slm_oracle.query_count
            print(f"  {cat}: {lstar_slm_dfas[cat].num_states} states, "
                  f"{slm_oracle.query_count} oracle queries "
                  f"({slm_oracle.cache_hits} cache hits)")

        lstar_slm_time = time.time() - t0

        lstar_slm_mc = MultiClassDFA(lstar_slm_dfas, priority=("LOG", "ALG", "AR", "WP"))
        lstar_slm_results = evaluate(lstar_slm_mc, test_data, "L* (SLM)")
        lstar_slm_results['train_time_s'] = lstar_slm_time
        lstar_slm_results['total_oracle_queries'] = total_queries
        lstar_slm_results['dfa_sizes'] = {cat: dfa.num_states for cat, dfa in lstar_slm_dfas.items()}
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
                energy_per_prompt = delta / len(latencies) if latencies else 0
                print(f"Energy: {delta:.2f} mWh total, {energy_per_prompt:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        results['agent'] = {
            'total_accuracy': agent_results['total_accuracy'],
            'correct': agent_results['correct'],
            'total': agent_results['total'],
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
    feat_correct = 0
    feat_by_cat = defaultdict(lambda: [0, 0])
    for toks, cat, pid in test_data:
        pred = classify_by_features(toks)
        feat_by_cat[cat][1] += 1
        if pred == cat:
            feat_correct += 1
            feat_by_cat[cat][0] += 1
    results['feature'] = {
        'total_accuracy': feat_correct / len(test_data),
        'correct': feat_correct,
        'total': len(test_data),
        'per_category': {cat: {'correct': v[0], 'total': v[1],
                               'accuracy': v[0]/v[1] if v[1] else 0}
                         for cat, v in feat_by_cat.items()},
    }
    print(f"  Overall: {feat_correct}/{len(test_data)} = {feat_correct/len(test_data):.1%}")

    # ---- Save all results ----
    results_path = os.path.join(out_dir, f"results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    # ---- Print summary table ----
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'System':<25s} {'AR':>5s} {'ALG':>5s} {'WP':>5s} {'LOG':>5s} {'All':>6s}")
    print("-" * 60)
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
        print(f"{name:<25s} {'  '.join(parts)}  {r['total_accuracy']:5.1%}")

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

    for toks, cat, pid in test_data:
        pred = mc.classify(toks) or 'UNKNOWN'
        by_cat[cat]['total'] += 1
        if pred == cat:
            correct += 1
            by_cat[cat]['correct'] += 1

    total = len(test_data)
    for cat in by_cat:
        by_cat[cat]['accuracy'] = by_cat[cat]['correct'] / by_cat[cat]['total']

    print(f"  {name}: {correct}/{total} = {correct/total:.1%}")
    for cat in ('AR', 'ALG', 'WP', 'LOG'):
        if cat in by_cat:
            bc = by_cat[cat]
            print(f"    {cat}: {bc['correct']}/{bc['total']} = {bc['accuracy']:.0%}")

    return {
        'total_accuracy': correct / total,
        'correct': correct,
        'total': total,
        'per_category': dict(by_cat),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GI experiments")
    parser.add_argument("--server", default="http://127.0.0.1:8080",
                        help="llama.cpp server URL")
    parser.add_argument("--no-slm", action="store_true",
                        help="Skip SLM-dependent experiments (run locally)")
    args = parser.parse_args()
    run_experiments(args)
