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
    if not args.agents_only:
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
    else:
        rpni_mc = None

    # ---- Experiment 2: L* with feature oracle ----
    if not args.agents_only:
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
    else:
        lstar_feat_mc = None

    # ---- Experiment 3: L* with SLM oracle (if Pi available) ----
    if not args.no_slm and not args.agents_only:
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

    if not args.no_slm and not args.agents_only:
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
    if not args.no_slm and not args.agents_only:
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

    # ---- Experiment 4b: ReAct agent ----
    if not args.no_slm:
        print("\n" + "="*50)
        print("Experiment 4b: ReAct agent baseline")
        print("="*50)

        react_start_mwh = None
        try:
            s = input("Enter STARTING mWh for ReAct experiment (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                react_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {react_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        from src.react_agent import run_benchmark as run_react
        react_results = run_react(
            "data/splits/industry_tier2_100.csv",
            server_url=args.server,
            max_tokens=300,
            repeats=3,
        )

        latencies = [r['latency_ms'] for r in react_results['results']]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        median_latency = sorted(latencies)[len(latencies)//2] if latencies else 0

        react_end_mwh = None
        react_energy = None
        react_energy_per_prompt = None
        try:
            s = input("Enter ENDING mWh for ReAct experiment (or Enter to skip): ").strip()
            if s and react_start_mwh is not None:
                import re as re_mod
                react_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                react_energy = react_end_mwh - react_start_mwh
                react_energy_per_prompt = react_energy / len(test_data)
                print(f"Energy: {react_energy:.2f} mWh total, {react_energy_per_prompt:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        results['react'] = {
            'total_accuracy': react_results['total_accuracy'],
            'correct': react_results['correct'],
            'total': react_results['total'],
            'repeats': react_results['repeats'],
            'per_category': react_results['per_category'],
            'avg_slm_calls': react_results.get('avg_slm_calls'),
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'energy_start_mwh': react_start_mwh,
            'energy_end_mwh': react_end_mwh,
            'energy_per_prompt_mwh': react_energy_per_prompt,
        }

    # ---- Experiment 4c: Program-of-Thought agent ----
    if not args.no_slm:
        print("\n" + "="*50)
        print("Experiment 4c: Program-of-Thought agent baseline")
        print("="*50)

        pot_start_mwh = None
        try:
            s = input("Enter STARTING mWh for PoT experiment (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                pot_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {pot_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        from src.pot_agent import run_benchmark as run_pot
        pot_results = run_pot(
            "data/splits/industry_tier2_100.csv",
            server_url=args.server,
            max_tokens=300,
            repeats=3,
        )

        latencies = [r['latency_ms'] for r in pot_results['results']]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        median_latency = sorted(latencies)[len(latencies)//2] if latencies else 0

        pot_end_mwh = None
        pot_energy = None
        pot_energy_per_prompt = None
        try:
            s = input("Enter ENDING mWh for PoT experiment (or Enter to skip): ").strip()
            if s and pot_start_mwh is not None:
                import re as re_mod
                pot_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                pot_energy = pot_end_mwh - pot_start_mwh
                pot_energy_per_prompt = pot_energy / len(test_data)
                print(f"Energy: {pot_energy:.2f} mWh total, {pot_energy_per_prompt:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        results['pot'] = {
            'total_accuracy': pot_results['total_accuracy'],
            'correct': pot_results['correct'],
            'total': pot_results['total'],
            'repeats': pot_results['repeats'],
            'per_category': pot_results['per_category'],
            'avg_slm_calls': pot_results.get('avg_slm_calls'),
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'energy_start_mwh': pot_start_mwh,
            'energy_end_mwh': pot_end_mwh,
            'energy_per_prompt_mwh': pot_energy_per_prompt,
        }

    # ---- Experiment 4d: Plan-and-Solve agent ----
    if not args.no_slm:
        print("\n" + "="*50)
        print("Experiment 4d: Plan-and-Solve agent baseline")
        print("="*50)

        ps_start_mwh = None
        try:
            s = input("Enter STARTING mWh for Plan-and-Solve experiment (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                ps_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {ps_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        from src.plan_and_solve_agent import run_benchmark as run_ps
        ps_results = run_ps(
            "data/splits/industry_tier2_100.csv",
            server_url=args.server,
            max_tokens=400,
            repeats=3,
        )

        latencies = [r['latency_ms'] for r in ps_results['results']]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        median_latency = sorted(latencies)[len(latencies)//2] if latencies else 0

        ps_end_mwh = None
        ps_energy = None
        ps_energy_per_prompt = None
        try:
            s = input("Enter ENDING mWh for Plan-and-Solve experiment (or Enter to skip): ").strip()
            if s and ps_start_mwh is not None:
                import re as re_mod
                ps_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                ps_energy = ps_end_mwh - ps_start_mwh
                ps_energy_per_prompt = ps_energy / len(test_data)
                print(f"Energy: {ps_energy:.2f} mWh total, {ps_energy_per_prompt:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        results['plan_and_solve'] = {
            'total_accuracy': ps_results['total_accuracy'],
            'correct': ps_results['correct'],
            'total': ps_results['total'],
            'repeats': ps_results['repeats'],
            'per_category': ps_results['per_category'],
            'avg_slm_calls': ps_results.get('avg_slm_calls'),
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'energy_start_mwh': ps_start_mwh,
            'energy_end_mwh': ps_end_mwh,
            'energy_per_prompt_mwh': ps_energy_per_prompt,
        }

    # ---- Experiments 5 & 6: End-to-end with learned routing → V5 solver ----
    if not args.no_slm and not args.agents_only:
        # RouterV5 uses bare imports (from router_v3 import ...), so we need
        # src/ on sys.path for those to resolve when run as python3 -m src.run_gi_experiments
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from router_v5 import RouterV5

        SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
        SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."

        v5_routing_decisions = {
            'category_routes': {
                'AR': {'action': 'A5', 'grammar_enabled': False},
                'ALG': {'action': 'A4', 'grammar_enabled': False},
                'WP': {'action': 'A2', 'grammar_enabled': False},
                'LOG': {'action': 'A6', 'grammar_enabled': False},
            },
            'max_escalations': 2,
        }

        v5_config = {
            'model_name': 'Phi-4-Mini',
            'quantization': 'Q6_K',
            'server_url': f"{args.server}/completion",
            'timeout_sec': 20,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 1,
            'seed': 42,
            'repeats': 3,
            'api_mode': 'chat',
            'config_version': 'v1.0',
            'n_pred_num': 12,
            'n_pred_log': 6,
        }

        v5_router = RouterV5(v5_config, v5_routing_decisions)

        # Load T2 prompts with full info
        t2_prompts = []
        with open("data/splits/industry_tier2_100.csv") as f:
            for row in csv.DictReader(f):
                t2_prompts.append(row)

        def build_phi_prompt(base_prompt, category):
            """Build Phi chat template from base prompt (same as run_hybrid_v5.py)."""
            question = base_prompt
            lines_q = question.rstrip().split('\n')
            while lines_q and lines_q[-1].strip() in ("Answer:", ""):
                lines_q.pop()
            while lines_q and lines_q[-1].strip().startswith("Return only"):
                lines_q.pop()
            question = '\n'.join(lines_q).strip()
            system_msg = SYSTEM_MSG_YESNO if category == "LOG" else SYSTEM_MSG_NUMERIC
            return f"<|system|>{system_msg}<|end|><|user|>{question}<|end|><|assistant|>"

        def run_e2e_experiment(exp_name, dfa_mc):
            """Run end-to-end experiment: DFA classification → V5 solver pipeline."""
            print(f"\n  Running 3 repeats per prompt (prompt-wise)...")

            trials = []
            by_cat = defaultdict(lambda: {'correct': 0, 'total': 0})
            routing_correct = 0
            trial_num = 0
            total_trials = len(t2_prompts) * 3

            for row in t2_prompts:
                prompt_id = row['prompt_id']
                true_cat = row['category']
                ground_truth = row['ground_truth']

                # DFA classification
                toks = tokenize(row['prompt_text'])
                predicted_cat = dfa_mc.classify(toks) or 'WP'  # fallback to WP

                if predicted_cat == true_cat:
                    routing_correct += 1

                # Build prompt using PREDICTED category (determines system msg)
                prompt_text = build_phi_prompt(row['prompt_text'], predicted_cat)

                for repeat in range(1, 4):
                    trial_num += 1
                    result = v5_router.route(
                        prompt_id=prompt_id,
                        category=predicted_cat,
                        prompt_text=prompt_text,
                        ground_truth=ground_truth,
                    )

                    trials.append({
                        'prompt_id': prompt_id,
                        'true_category': true_cat,
                        'predicted_category': predicted_cat,
                        'category_match': predicted_cat == true_cat,
                        'repeat': repeat,
                        'correct': result['correct'],
                        'answer': result['answer_final'],
                        'ground_truth': ground_truth,
                        'latency_ms': result['total_latency_ms'],
                        'route_sequence': result['route_attempt_sequence'],
                        'error_code': result['error_code'],
                    })

                    by_cat[true_cat]['total'] += 1
                    if result['correct']:
                        by_cat[true_cat]['correct'] += 1

                    status = "OK" if result['correct'] else "X"
                    cat_match = "=" if predicted_cat == true_cat else f"!={predicted_cat}"
                    print(f"  [{trial_num}/{total_trials}] {prompt_id} r{repeat} "
                          f"{true_cat}{cat_match} "
                          f"correct={result['correct']} "
                          f"lat={result['total_latency_ms']:.0f}ms")

                    time.sleep(0.1)

            total = len(trials)
            correct = sum(1 for t in trials if t['correct'])
            latencies = [t['latency_ms'] for t in trials]
            avg_lat = sum(latencies) / len(latencies) if latencies else 0
            median_lat = sorted(latencies)[len(latencies)//2] if latencies else 0

            for cat in by_cat:
                bc = by_cat[cat]
                bc['accuracy'] = bc['correct'] / bc['total'] if bc['total'] else 0

            return {
                'total_accuracy': correct / total if total else 0,
                'correct': correct,
                'total': total,
                'repeats': 3,
                'routing_accuracy': routing_correct / len(t2_prompts),
                'routing_correct': routing_correct,
                'per_category': dict(by_cat),
                'avg_latency_ms': avg_lat,
                'median_latency_ms': median_lat,
                'trials': trials,
            }

        # ---- Experiment 5: L* routing → V5 solver (end-to-end) ----
        print("\n" + "="*50)
        print("Experiment 5: L* learned routing -> V5 solver (END-TO-END)")
        print("="*50)

        # Use L* feature DFAs (100% routing accuracy, or SLM DFAs if available)
        e2e_lstar_mc = lstar_slm_mc if 'lstar_slm' in results else lstar_feat_mc
        e2e_lstar_label = "L* SLM" if 'lstar_slm' in results else "L* feature"
        print(f"  Using {e2e_lstar_label} DFAs for classification")

        lstar_e2e_start_mwh = None
        try:
            s = input(f"Enter STARTING mWh for {e2e_lstar_label} end-to-end (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                lstar_e2e_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {lstar_e2e_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        lstar_e2e_results = run_e2e_experiment(e2e_lstar_label, e2e_lstar_mc)

        lstar_e2e_end_mwh = None
        lstar_e2e_energy = None
        try:
            s = input(f"Enter ENDING mWh for {e2e_lstar_label} end-to-end (or Enter to skip): ").strip()
            if s and lstar_e2e_start_mwh is not None:
                import re as re_mod
                lstar_e2e_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                lstar_e2e_energy = lstar_e2e_end_mwh - lstar_e2e_start_mwh
                num_unique = len(t2_prompts)
                print(f"Energy: {lstar_e2e_energy:.2f} mWh total, {lstar_e2e_energy/num_unique:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        lstar_e2e_results['energy_start_mwh'] = lstar_e2e_start_mwh
        lstar_e2e_results['energy_end_mwh'] = lstar_e2e_end_mwh
        lstar_e2e_results['energy_total_mwh'] = lstar_e2e_energy
        lstar_e2e_results['energy_per_prompt_mwh'] = lstar_e2e_energy / len(t2_prompts) if lstar_e2e_energy is not None else None
        lstar_e2e_results['dfa_source'] = e2e_lstar_label
        results['lstar_e2e'] = lstar_e2e_results

        # ---- Experiment 6: RPNI routing → V5 solver (end-to-end) ----
        print("\n" + "="*50)
        print("Experiment 6: RPNI learned routing -> V5 solver (END-TO-END)")
        print("="*50)

        rpni_e2e_start_mwh = None
        try:
            s = input("Enter STARTING mWh for RPNI end-to-end (or Enter to skip): ").strip()
            if s:
                import re as re_mod
                rpni_e2e_start_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                print(f"Recorded: {rpni_e2e_start_mwh} mWh")
        except (ValueError, EOFError):
            pass

        rpni_e2e_results = run_e2e_experiment("RPNI", rpni_mc)

        rpni_e2e_end_mwh = None
        rpni_e2e_energy = None
        try:
            s = input("Enter ENDING mWh for RPNI end-to-end (or Enter to skip): ").strip()
            if s and rpni_e2e_start_mwh is not None:
                import re as re_mod
                rpni_e2e_end_mwh = float(re_mod.sub(r'[^\d.\-+]', '', s))
                rpni_e2e_energy = rpni_e2e_end_mwh - rpni_e2e_start_mwh
                num_unique = len(t2_prompts)
                print(f"Energy: {rpni_e2e_energy:.2f} mWh total, {rpni_e2e_energy/num_unique:.2f} mWh/prompt")
        except (ValueError, EOFError):
            pass

        rpni_e2e_results['energy_start_mwh'] = rpni_e2e_start_mwh
        rpni_e2e_results['energy_end_mwh'] = rpni_e2e_end_mwh
        rpni_e2e_results['energy_total_mwh'] = rpni_e2e_energy
        rpni_e2e_results['energy_per_prompt_mwh'] = rpni_e2e_energy / len(t2_prompts) if rpni_e2e_energy is not None else None
        results['rpni_e2e'] = rpni_e2e_results

    # ---- Experiment 7: Feature classifier (hand-coded baseline) ----
    if not args.agents_only:
        print("\n" + "="*50)
        print("Experiment 7: Feature classifier (hand-coded)")
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
    print("\n  --- Routing Classification Only ---")
    for name, key in [("RPNI (classify)", "rpni"),
                       ("L* feature (classify)", "lstar_feature"),
                       ("L* SLM (classify)", "lstar_slm"),
                       ("Feature clf (classify)", "feature")]:
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

    # End-to-end comparison (the main paper table)
    e2e_systems = [("L* routing -> V5", "lstar_e2e"),
                   ("RPNI routing -> V5", "rpni_e2e"),
                   ("Tool-calling agent", "agent"),
                   ("ReAct agent", "react"),
                   ("Program-of-Thought", "pot"),
                   ("Plan-and-Solve", "plan_and_solve")]
    if any(k in results for _, k in e2e_systems):
        print("\n  --- End-to-End Task Accuracy (3 repeats) ---")
        print(f"{'System':<25s} {'AR':>5s} {'ALG':>5s} {'WP':>5s} {'LOG':>5s} {'All':>6s} {'Lat(ms)':>8s} {'mWh':>8s}")
        print("-" * 80)
        for name, key in e2e_systems:
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
            lat_str = f"{lat:8.1f}" if lat is not None else "     N/A"
            epp = r.get('energy_per_prompt_mwh')
            epp_str = f"{epp:8.2f}" if epp is not None else "     N/A"
            print(f"{name:<25s} {'  '.join(parts)}  {r['total_accuracy']:5.1%}  {lat_str}  {epp_str}")

        # Show routing accuracy for learned systems
        for name, key in [("L* routing", "lstar_e2e"), ("RPNI routing", "rpni_e2e")]:
            if key in results and 'routing_accuracy' in results[key]:
                print(f"  {name} classification accuracy: {results[key]['routing_accuracy']:.1%}")

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
    parser.add_argument("--agents-only", action="store_true",
                        help="Skip experiments 1-7, run only new agent baselines (4b, 4c, 4d)")
    args = parser.parse_args()
    run_experiments(args)
