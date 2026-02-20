"""
Core runner logic for Phi-2 experiments.

Provides base classes and common functionality for running experiments
with the Phi-2 model via llama.cpp server.
"""

import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import RunConfig
from .extraction import extract_final_answer, check_degeneracy, validate_format


@dataclass
class TrialResult:
    """Result from a single trial."""
    id: str
    category: str
    variant: str
    expected_answer: str
    trial_index: int
    final_output: str
    correct: int
    err: str
    latency_wall_s: float
    latency_compute_s: Optional[float]
    raw_content: str = ""


@dataclass
class SummaryResult:
    """Summary result across multiple trials."""
    id: str
    category: str
    variant: str
    expected_answer: str
    final_output: str
    correct: int
    err: str
    latency_wall_median_s: float
    latency_compute_median_s: Optional[float]


def error_taxonomy(
    exc: Optional[BaseException],
    category: str,
    output: str,
    expected: str,
    mode: str,
    degenerate: bool = False,
) -> str:
    """
    Classify errors using standard error taxonomy.

    Error codes:
        E0: Correct answer
        E1: AR computation error
        E2: LOG inference error
        E3: ALG manipulation error
        E4: Hallucinated step (rare in final-answer-only baseline)
        E5: Partial reasoning (default for WP wrong)
        E6: Instruction following failure (mostly no-grammar)
        E7: Timeout or degeneracy
        E8: Parse failure / malformed output

    Args:
        exc: Exception if one occurred
        category: Problem category (AR, ALG, LOG, WP)
        output: Extracted output
        expected: Expected answer
        mode: "grammar" or "nogrammar"
        degenerate: Whether answer shows degeneracy

    Returns:
        Error code string (E0-E8)
    """
    cat = (category or "").strip().upper()
    got = (output or "").strip()
    exp = (expected or "").strip()
    m = (mode or "").strip().lower()

    # Timeout or degeneracy
    if exc is not None:
        if isinstance(exc, (requests.exceptions.ReadTimeout, requests.exceptions.Timeout)):
            return "E7"
        # Server/connect/HTTP issues: treat as malformed
        return "E8"

    if degenerate:
        return "E7"

    # No exception: check output presence + format
    if not got:
        return "E8"

    # Format validation
    if cat == "LOG":
        if got not in ("Yes", "No"):
            return "E6" if m == "nogrammar" else "E8"
    else:
        # Numeric: check if valid integer
        g = got[1:] if got.startswith("-") else got
        if not g.isdigit():
            return "E6" if m == "nogrammar" else "E8"

    # Well-formed: check correctness
    if got == exp:
        return "E0"

    # Wrong but well-formed: category-specific error
    if cat == "AR":
        return "E1"
    if cat == "ALG":
        return "E3"
    if cat == "LOG":
        return "E2"
    if cat == "WP":
        return "E5"

    return "E5"


def is_correct(expected: str, got: str) -> int:
    """Check if answer is correct (returns 1 or 0)."""
    return 1 if (expected or "").strip() == (got or "").strip() else 0


def log(msg: str, verbose: bool) -> None:
    """Print log message if verbose mode enabled."""
    if verbose:
        print(msg, flush=True)


class Phi2Runner:
    """Base class for running Phi-2 experiments."""

    def __init__(self, config: RunConfig):
        """
        Initialize runner with configuration.

        Args:
            config: Run configuration
        """
        self.config = config

        # Load grammars if using grammar mode
        self.num_grammar_text = ""
        self.yesno_grammar_text = ""
        if config.use_grammar:
            self.num_grammar_text = config.grammar.get_num_grammar()
            self.yesno_grammar_text = config.grammar.get_yesno_grammar()

    def pick_grammar(self, category: str) -> str:
        """Select grammar based on category."""
        if not self.config.use_grammar:
            return ""
        return (
            self.yesno_grammar_text
            if category.upper() == "LOG"
            else self.num_grammar_text
        )

    def post_completion(
        self,
        prompt: str,
        category: str,
    ) -> Tuple[str, Optional[float]]:
        """
        Send completion request to llama.cpp server.

        Args:
            prompt: Input prompt
            category: Problem category (for selecting grammar and n_predict)

        Returns:
            Tuple of (content, compute_time_s)

        Raises:
            requests.exceptions.Timeout: On timeout
            requests.exceptions.RequestException: On other HTTP errors
        """
        n_predict = self.config.inference.get_n_predict(category)
        grammar = self.pick_grammar(category)

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "n_predict": int(n_predict),
            "temperature": float(self.config.inference.temperature),
            "grammar": grammar,
        }

        # Separate connect/read timeouts
        timeout = (
            self.config.server.connect_timeout_s,
            float(self.config.server.timeout_s),
        )

        r = requests.post(self.config.server.url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        content = data.get("content", "")

        # Extract compute time if available
        compute_s = None
        timings = data.get("timings")
        if isinstance(timings, dict):
            pm = timings.get("predicted_ms")
            if isinstance(pm, (int, float)):
                compute_s = float(pm) / 1000.0

        return content, compute_s

    def run_single_trial(
        self,
        prompt_id: str,
        category: str,
        prompt: str,
        expected: str,
        variant: str,
        trial_index: int,
    ) -> TrialResult:
        """
        Run a single trial for one prompt.

        Args:
            prompt_id: Prompt ID
            category: Problem category
            prompt: Prompt text
            expected: Expected answer
            variant: Variant name
            trial_index: Trial number (1-indexed)

        Returns:
            TrialResult with outcomes
        """
        mode = "grammar" if self.config.use_grammar else "nogrammar"

        start = time.time()
        out_text = ""
        raw_content = ""
        compute_s = None
        exc: Optional[BaseException] = None

        try:
            content, compute_s = self.post_completion(prompt, category)
            raw_content = content
            out_text = extract_final_answer(category, content, mode)
        except BaseException as e:
            exc = e
            out_text = ""
            compute_s = None

        wall = time.time() - start

        # Check for degeneracy
        degenerate = check_degeneracy(category, out_text)

        # Classify error
        err = error_taxonomy(exc, category, out_text, expected, mode, degenerate)

        # Determine correctness
        ok = is_correct(expected, out_text)

        if self.config.verbose:
            log(
                f"  trial {trial_index}/{self.config.inference.repeats} "
                f"id={prompt_id} cat={category} "
                f"wall={wall:.3f}s out='{out_text}' ok={ok} err={err}",
                True
            )

        return TrialResult(
            id=prompt_id,
            category=category,
            variant=variant,
            expected_answer=expected,
            trial_index=trial_index,
            final_output=out_text,
            correct=ok,
            err=err,
            latency_wall_s=wall,
            latency_compute_s=compute_s,
            raw_content=raw_content,
        )

    def run_prompt(
        self,
        prompt_id: str,
        category: str,
        prompt: str,
        expected: str,
        variant: str,
    ) -> Tuple[SummaryResult, List[TrialResult]]:
        """
        Run multiple trials for a single prompt.

        Args:
            prompt_id: Prompt ID
            category: Problem category
            prompt: Prompt text
            expected: Expected answer
            variant: Variant name

        Returns:
            Tuple of (summary_result, trial_results)
        """
        # Warmup runs (not recorded)
        for _ in range(self.config.inference.warmup_per_prompt):
            try:
                self.post_completion(prompt, category)
            except Exception:
                pass

        # Recorded trials
        trials: List[TrialResult] = []
        for t in range(1, self.config.inference.repeats + 1):
            trial = self.run_single_trial(
                prompt_id, category, prompt, expected, variant, t
            )
            trials.append(trial)

        # Compute summary
        wall_times = [t.latency_wall_s for t in trials]
        compute_times = [
            t.latency_compute_s for t in trials if t.latency_compute_s is not None
        ]

        wall_med = float(statistics.median(wall_times))
        compute_med = (
            float(statistics.median(compute_times)) if compute_times else None
        )

        # Choose output from median wall latency trial
        med_idx = sorted(range(len(wall_times)), key=lambda i: wall_times[i])[
            len(wall_times) // 2
        ]
        final_out = trials[med_idx].final_output
        final_err = trials[med_idx].err
        final_ok = trials[med_idx].correct

        summary = SummaryResult(
            id=prompt_id,
            category=category,
            variant=variant,
            expected_answer=expected,
            final_output=final_out,
            correct=final_ok,
            err=final_err,
            latency_wall_median_s=wall_med,
            latency_compute_median_s=compute_med,
        )

        return summary, trials

    def run_batch(
        self,
        rows: List[Dict[str, str]],
        variant: Optional[str] = None,
    ) -> Tuple[List[SummaryResult], List[TrialResult]]:
        """
        Run a batch of prompts.

        Args:
            rows: List of prompt dicts with keys: id, category, prompt, expected_answer
            variant: Optional variant name override

        Returns:
            Tuple of (summaries, all_trials)
        """
        if variant is None:
            variant = self.config.variant_name

        summaries: List[SummaryResult] = []
        all_trials: List[TrialResult] = []

        for idx, row in enumerate(rows, 1):
            prompt_id = row["id"].strip()
            category = row["category"].strip()

            # Build prompt with Answer: suffix
            base_prompt = row["prompt"].rstrip()
            # Remove existing Answer: suffix if present
            for suffix in ["\nAnswer:", "Answer:"]:
                if base_prompt.endswith(suffix):
                    base_prompt = base_prompt[:-len(suffix)].rstrip()

            prompt = base_prompt + "\nAnswer: "
            expected = row["expected_answer"].strip()

            # Progress logging
            if self.config.verbose and (
                self.config.print_every > 0
                and (idx % self.config.print_every == 0 or idx == 1)
            ):
                log(
                    f"[{variant}] starting prompt {idx}/{len(rows)} "
                    f"id={prompt_id} cat={category}",
                    True
                )
                if self.config.show_prompt:
                    p = prompt.replace("\n", "\\n")
                    log(f"  prompt={p[:240]}", True)

            summary, trials = self.run_prompt(
                prompt_id, category, prompt, expected, variant
            )
            summaries.append(summary)
            all_trials.extend(trials)

        return summaries, all_trials

    @staticmethod
    def write_results(
        summaries: List[SummaryResult],
        trials: List[TrialResult],
        summary_path: Path,
        trials_path: Optional[Path] = None,
    ) -> None:
        """
        Write results to CSV files.

        Args:
            summaries: Summary results
            trials: Trial results
            summary_path: Path for summary CSV
            trials_path: Optional path for trials CSV
        """
        # Write summary
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "category", "variant", "expected_answer",
                "final_output", "correct", "err",
                "latency_wall_median_s", "latency_compute_median_s"
            ])
            for r in summaries:
                writer.writerow([
                    r.id, r.category, r.variant, r.expected_answer,
                    r.final_output, r.correct, r.err,
                    f"{r.latency_wall_median_s:.6f}",
                    (f"{r.latency_compute_median_s:.6f}"
                     if r.latency_compute_median_s is not None else "")
                ])

        # Write trials if path provided
        if trials_path:
            with open(trials_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id", "category", "variant", "expected_answer",
                    "trial_index", "final_output", "correct", "err",
                    "latency_wall_s", "latency_compute_s"
                ])
                for r in trials:
                    writer.writerow([
                        r.id, r.category, r.variant, r.expected_answer,
                        r.trial_index, r.final_output, r.correct, r.err,
                        f"{r.latency_wall_s:.6f}",
                        (f"{r.latency_compute_s:.6f}"
                         if r.latency_compute_s is not None else "")
                    ])
