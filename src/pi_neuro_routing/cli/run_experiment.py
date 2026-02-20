#!/usr/bin/env python3
"""
Unified CLI for running Pi Neurosymbolic Routing experiments.

This replaces the multiple runner scripts with a single unified interface.

Usage examples:
    # Baseline with grammar
    python -m pi_neuro_routing.cli.run_experiment --mode baseline --grammar \\
        --csv data/benchmarks/industry_tier2_400.csv --out results/baseline_grammar.csv

    # Baseline without grammar
    python -m pi_neuro_routing.cli.run_experiment --mode baseline --no-grammar \\
        --csv data/benchmarks/industry_tier2_400.csv --out results/baseline_nogrammar.csv

    # Both grammar and no-grammar
    python -m pi_neuro_routing.cli.run_experiment --mode baseline --both \\
        --csv data/benchmarks/industry_tier2_400.csv --out results/baseline_both.csv

    # Hybrid v1
    python -m pi_neuro_routing.cli.run_experiment --mode hybrid-v1 \\
        --csv data/benchmarks/industry_tier2_400.csv --out results/hybrid_v1.csv

    # Hybrid v2
    python -m pi_neuro_routing.cli.run_experiment --mode hybrid-v2 \\
        --csv data/benchmarks/industry_tier2_400.csv --out results/hybrid_v2.csv
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict

from ..core import (
    Phi2Runner,
    create_baseline_config,
    create_hybrid_v1_config,
    create_hybrid_v2_config,
)


def load_prompts_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load prompts from CSV file."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def run_baseline(args: argparse.Namespace) -> None:
    """Run baseline experiment."""
    rows = load_prompts_csv(Path(args.csv))

    # Determine which variants to run
    run_grammar = args.grammar or args.both
    run_nogrammar = args.no_grammar or args.both

    if not (run_grammar or run_nogrammar):
        print("Error: Must specify --grammar, --no-grammar, or --both", file=sys.stderr)
        sys.exit(1)

    all_summaries = []
    all_trials = []

    # Run grammar variant
    if run_grammar:
        print(f"Running baseline with grammar constraints...")
        config = create_baseline_config(
            use_grammar=True,
            timeout_s=args.timeout_s,
            repeats=args.repeats,
            verbose=args.verbose,
            debug=args.debug,
        )
        runner = Phi2Runner(config)
        summaries, trials = runner.run_batch(rows)
        all_summaries.extend(summaries)
        all_trials.extend(trials)

    # Run no-grammar variant
    if run_nogrammar:
        print(f"Running baseline without grammar constraints...")
        config = create_baseline_config(
            use_grammar=False,
            timeout_s=args.timeout_s,
            repeats=args.repeats,
            verbose=args.verbose,
            debug=args.debug,
        )
        runner = Phi2Runner(config)
        summaries, trials = runner.run_batch(rows)
        all_summaries.extend(summaries)
        all_trials.extend(trials)

    # Write results
    out_path = Path(args.out)
    trials_path = Path(args.trials_out) if args.trials_out else None

    Phi2Runner.write_results(all_summaries, all_trials, out_path, trials_path)
    print(f"\nResults written to: {out_path}")
    if trials_path:
        print(f"Trials written to: {trials_path}")


def run_hybrid_v1(args: argparse.Namespace) -> None:
    """Run hybrid v1 experiment (SymPy routing)."""
    print("Error: Hybrid v1 not yet implemented in refactored structure", file=sys.stderr)
    print("Use original src/run_hybrid_v1.py for now", file=sys.stderr)
    sys.exit(1)


def run_hybrid_v2(args: argparse.Namespace) -> None:
    """Run hybrid v2 experiment (LLM + SymPy verify)."""
    print("Error: Hybrid v2 not yet implemented in refactored structure", file=sys.stderr)
    print("Use original src/run_hybrid_v2.py for now", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Pi Neurosymbolic Routing experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid-v1", "hybrid-v2"],
        default="baseline",
        help="Experiment mode to run",
    )

    # Grammar options (for baseline)
    grammar_group = parser.add_mutually_exclusive_group()
    grammar_group.add_argument(
        "--grammar",
        action="store_true",
        help="Use grammar constraints (baseline only)",
    )
    grammar_group.add_argument(
        "--no-grammar",
        dest="no_grammar",
        action="store_true",
        help="Do not use grammar constraints (baseline only)",
    )
    grammar_group.add_argument(
        "--both",
        action="store_true",
        help="Run both grammar and no-grammar variants (baseline only)",
    )

    # Input/output
    parser.add_argument(
        "--csv",
        required=True,
        help="Input prompts CSV file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output summary CSV file",
    )
    parser.add_argument(
        "--trials_out",
        default=None,
        help="Output trials CSV file (optional)",
    )

    # Server configuration
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8080/completion",
        help="llama.cpp server URL",
    )
    parser.add_argument(
        "--timeout_s",
        type=float,
        default=20.0,
        help="Request timeout in seconds",
    )

    # Inference parameters
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of trials per prompt",
    )
    parser.add_argument(
        "--warmup_per_prompt",
        type=int,
        default=0,
        help="Number of warmup inferences per prompt",
    )
    parser.add_argument(
        "--n_pred_num",
        type=int,
        default=32,
        help="Max tokens for numeric answers",
    )
    parser.add_argument(
        "--n_pred_log",
        type=int,
        default=8,
        help="Max tokens for yes/no answers",
    )

    # Grammar files (for baseline)
    parser.add_argument(
        "--num_grammar_file",
        default="grammars/final/int_strict_final.gbnf",
        help="Numeric grammar file",
    )
    parser.add_argument(
        "--yesno_grammar_file",
        default="grammars/final/yesno_strict_final.gbnf",
        help="Yes/no grammar file",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-prompt/per-trial progress",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print header every N prompts (verbose only)",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Also print prompt text snippets (verbose only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    # Dispatch to appropriate runner
    try:
        if args.mode == "baseline":
            run_baseline(args)
        elif args.mode == "hybrid-v1":
            run_hybrid_v1(args)
        elif args.mode == "hybrid-v2":
            run_hybrid_v2(args)
    except KeyboardInterrupt:
        print("\n\nCTRL-C: stopping entire run.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
