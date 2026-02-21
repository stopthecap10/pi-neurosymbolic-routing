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
PROMPT_TEMPLATE_VERSION = "PT3_phi_chat"

# System message for Phi chat format
SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."


def build_prompt(base_prompt: str, category: str) -> str:
    """Build the full prompt using Phi-4 chat template format.

    Format: <|system|>{system}<|end|><|user|>{question}<|end|><|assistant|>
    """
    is_log = (category == "LOG")

    # Strip any existing instruction suffix from official CSVs
    # (e.g. "...\nReturn only Yes or No.\nAnswer:")
    question = base_prompt
    for suffix in ["\nReturn only Yes or No.\nAnswer:",
                   "\nReturn only the final numeric answer.\nAnswer:"]:
        if question.rstrip().endswith(suffix.strip()):
            question = question[:question.rfind(suffix.split('\n')[1].strip().split()[0]) - 1]
            break
    # Simpler approach: strip trailing "Answer:" and instruction lines
    lines = question.rstrip().split('\n')
    while lines and lines[-1].strip() in ("Answer:", ""):
        lines.pop()
    while lines and lines[-1].strip().startswith("Return only"):
        lines.pop()
    question = '\n'.join(lines).strip()

    if is_log:
        system_msg = SYSTEM_MSG_YESNO
    else:
        system_msg = SYSTEM_MSG_NUMERIC

    return f"<|system|>{system_msg}<|end|><|user|>{question}<|end|><|assistant|>"


# ============================================================
# FROZEN ACTION BUDGETS
# ============================================================
ACTION_BUDGETS = {
    "A1": 12,  # Short decode
    "A2": 30,  # Extended decode
}

# Client-side timeouts per action (seconds)
# Must be long enough for Pi 4 prefill + decode
ACTION_TIMEOUTS = {
    "A1": 45,
    "A2": 60,
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
def check_server_health(server_url, timeout=10):
    """Check if llama.cpp server is healthy before starting run."""
    # Derive health URL from completion URL
    base_url = server_url.rsplit('/', 1)[0]
    health_url = f"{base_url}/health"

    try:
        r = requests.get(health_url, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            status = data.get("status", "unknown")
            if status == "ok":
                print(f"[health] Server healthy: {health_url}")
                return True
            else:
                print(f"[health] Server status: {status}")
                return status != "error"
        else:
            print(f"[health] Server returned HTTP {r.status_code}")
            return False
    except Exception as e:
        print(f"[health] Server unreachable: {e}")
        return False


def run_warmup(server_url, timeout_sec):
    """Send a trivial warmup prompt to prime KV cache and model loading."""
    print("[warmup] Sending warmup prompt: '2+2='")
    warmup_req = {
        "prompt": "2+2=",
        "n_predict": 4,
        "temperature": 0.0,
        "seed": 42,
    }
    t0 = time.time()
    try:
        r = requests.post(server_url, json=warmup_req, timeout=(10.0, float(timeout_sec)))
        j = r.json()
        content = j.get("content", "")
        latency_ms = (time.time() - t0) * 1000
        print(f"[warmup] Response: {repr(content[:40])} in {latency_ms:.0f}ms")
        return True
    except Exception as e:
        print(f"[warmup] FAILED: {e}")
        return False


def run_inference(prompt_text, server_url, timeout_sec, n_pred, use_grammar, grammar_file):
    """Run single inference - FROZEN params."""
    request = {
        "prompt": prompt_text,
        "n_predict": int(n_pred),
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stop": ["\n", "<|end|>", "<|endoftext|>"],
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
    http_status = None
    exception_type = None
    tokens_predicted = None
    stop_type = None

    try:
        r = requests.post(
            server_url,
            json=request,
            timeout=(10.0, float(timeout_sec)),
        )
        http_status = r.status_code
        j = r.json()
        content = j.get("content", "") or ""
        tokens_predicted = j.get("tokens_predicted", None)
        stop_type = j.get("stop_type", None)
    except requests.exceptions.Timeout:
        timed_out = True
        exception_type = "Timeout"
    except requests.exceptions.ConnectionError:
        exception_type = "ConnectionError"
        content = ""
    except Exception as e:
        exception_type = type(e).__name__
        print(f"ERROR: {e}", file=sys.stderr)
        content = ""

    latency_ms = (time.time() - t0) * 1000

    if latency_ms >= timeout_sec * 1000:
        timed_out = True

    diagnostics = {
        "http_status": http_status,
        "exception_type": exception_type,
        "tokens_predicted": tokens_predicted,
        "stop_type": stop_type,
        "prompt_len_chars": len(prompt_text),
    }

    return content, latency_ms, timed_out, diagnostics


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

    timeout_sec = ACTION_TIMEOUTS[args.action]

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

        content, latency_ms, timed_out, diag = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=timeout_sec,
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
            f"  timeout_sec:           {timeout_sec}",
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
            f"  --- E7 Diagnostics ---",
            f"  http_status:           {diag['http_status']}",
            f"  exception_type:        {diag['exception_type']}",
            f"  tokens_predicted:      {diag['tokens_predicted']}",
            f"  stop_type:             {diag['stop_type']}",
            f"  prompt_len_chars:      {diag['prompt_len_chars']}",
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

    timeout_sec = ACTION_TIMEOUTS[args.action]
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

        content, latency_ms, timed_out, diag = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=timeout_sec,
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
# DEBUG QUICK MODE
# ============================================================
def run_debug_quick(args, config, prompts):
    """Debug run: 2 prompts per category, 1 repeat, with full diagnostics."""
    # Pick first 2 prompts per category
    cat_counts = {}
    selected = []
    for p in prompts:
        cat = p['category']
        cat_counts.setdefault(cat, 0)
        if cat_counts[cat] < 2:
            selected.append(p)
            cat_counts[cat] += 1

    timeout_sec = ACTION_TIMEOUTS[args.action]
    n_prompts = len(selected)
    cats = sorted(set(p['category'] for p in selected))
    print(f"DEBUG QUICK: {n_prompts} prompts (2/category) x 1 repeat")
    print(f"  Categories: {', '.join(cats)}")
    print(f"  Timeout: {timeout_sec}s (action {args.action})")
    print()

    # Health check + warmup
    if not check_server_health(config['server_url']):
        print("FATAL: Server health check failed.")
        sys.exit(1)
    run_warmup(config['server_url'], timeout_sec)
    print()

    trials = []
    for prompt_row in selected:
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

        content, latency_ms, timed_out, diag = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=timeout_sec,
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

        # Full diagnostic output
        print(f"{'=' * 60}")
        print(f"  {prompt_id} | {category} | exp={ground_truth}")
        print(f"  prompt_len={diag['prompt_len_chars']} chars | n_pred={n_pred}")
        print(f"  http={diag['http_status']} | tokens={diag['tokens_predicted']} "
              f"| stop={diag['stop_type']} | lat={latency_ms:.0f}ms")
        print(f"  raw={repr(content[:120])}")
        print(f"  parsed={repr(answer_parsed)} | correct={correct} | err={error_code}")
        if timed_out:
            print(f"  >>> TIMEOUT! exception={diag['exception_type']}")
        print()

        trials.append({
            "prompt_id": prompt_id, "category": category,
            "error_code": error_code, "correct": int(correct),
            "timeout_flag": int(timed_out), "latency_ms": f"{latency_ms:.0f}",
            "answer_parsed": answer_parsed, "ground_truth": ground_truth,
            "http_status": diag['http_status'],
            "tokens_predicted": diag['tokens_predicted'],
            "stop_type": diag['stop_type'],
            "prompt_len_chars": diag['prompt_len_chars'],
            "answer_raw": content.replace("\n", "\\n")[:200],
        })

    # Per-category summary
    print(f"{'=' * 60}")
    print("DEBUG QUICK SUMMARY:")
    for cat in cats:
        cat_trials = [t for t in trials if t['category'] == cat]
        n = len(cat_trials)
        ok = sum(t['correct'] for t in cat_trials)
        e7 = sum(t['timeout_flag'] for t in cat_trials)
        print(f"  {cat}: {ok}/{n} correct, {e7}/{n} E7 timeouts")

    total_ok = sum(t['correct'] for t in trials)
    total_e7 = sum(t['timeout_flag'] for t in trials)
    print(f"  TOTAL: {total_ok}/{n_prompts} correct, {total_e7}/{n_prompts} E7 timeouts")

    if total_e7 > 0:
        print(f"\n  WARNING: {total_e7} timeouts detected. Check server and prompt lengths.")
    else:
        print(f"\n  All prompts completed without timeout. Safe to run full baseline.")

    # Save debug CSV
    debug_csv = args.out_trials.replace('.csv', '_debug_quick.csv')
    os.makedirs(os.path.dirname(debug_csv), exist_ok=True)
    if trials:
        with open(debug_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=trials[0].keys())
            writer.writeheader()
            writer.writerows(trials)
        print(f"\nDebug CSV saved to: {debug_csv}")


# ============================================================
# PROBE MODE
# ============================================================
PROBE_COUNTS = {"AR": 3, "ALG": 3, "WP": 2, "LOG": 2}  # 10 total
LOG_PROBE_MAX_CHARS = 500  # Skip LOG prompts longer than this in probe mode


def run_probe(args, config, prompts):
    """Probe mode: 3 AR, 3 ALG, 2 WP, 2 LOG × 1 repeat.
    Prints full prompt_text, answer_raw, timeout flag, parsed answer.
    Goal: verify prompt format + timeout are fixed before full baselines."""

    # Select prompts per category (prefer shorter LOG prompts)
    cat_counts = {}
    selected = []
    skipped_log = []
    for p in prompts:
        cat = p['category']
        cat_counts.setdefault(cat, 0)
        if cat_counts[cat] >= PROBE_COUNTS.get(cat, 0):
            continue
        # LOG length guard: skip overly long LOG prompts in probe
        if cat == "LOG" and len(p['prompt_text']) > LOG_PROBE_MAX_CHARS:
            skipped_log.append(f"{p['prompt_id']} ({len(p['prompt_text'])} chars)")
            continue
        selected.append(p)
        cat_counts[cat] += 1

    if skipped_log:
        print(f"  [LOG guard] Skipped {len(skipped_log)} long LOG prompts: {', '.join(skipped_log)}")
        print(f"  [LOG guard] Threshold: {LOG_PROBE_MAX_CHARS} chars (diagnostic only, not scoring)")
        print()

    timeout_sec = ACTION_TIMEOUTS[args.action]
    n_prompts = len(selected)
    cats = sorted(set(p['category'] for p in selected))

    print("=" * 70)
    print(f"PROBE MODE: {n_prompts} prompts × 1 repeat")
    print(f"  Distribution: {', '.join(f'{c}={PROBE_COUNTS[c]}' for c in ['AR', 'ALG', 'WP', 'LOG'])}")
    print(f"  Action: {args.action} (max_tokens={ACTION_BUDGETS[args.action]})")
    print(f"  Timeout: {timeout_sec}s")
    print(f"  Grammar: {'enabled' if args.grammar else 'disabled'}")
    print(f"  Prompt template: {PROMPT_TEMPLATE_VERSION}")
    print("=" * 70)
    print()

    # Health check + warmup
    if not check_server_health(config['server_url']):
        print("FATAL: Server health check failed.")
        sys.exit(1)
    run_warmup(config['server_url'], timeout_sec)
    print()

    trials = []
    for i, prompt_row in enumerate(selected, 1):
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

        content, latency_ms, timed_out, diag = run_inference(
            prompt_text=prompt_text,
            server_url=config['server_url'],
            timeout_sec=timeout_sec,
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

        # Full diagnostic output
        print(f"{'=' * 70}")
        print(f"[{i}/{n_prompts}] {prompt_id} | {category} | expected={ground_truth}")
        print(f"  n_pred={n_pred} | timeout={timeout_sec}s | grammar={'yes' if args.grammar else 'no'}")
        print()
        print(f"  FULL PROMPT SENT:")
        print(f"  {'─' * 60}")
        for line in prompt_text.split('\n'):
            print(f"  │ {line}")
        print(f"  {'─' * 60}")
        print()
        print(f"  FULL RESPONSE:")
        print(f"  {'─' * 60}")
        print(f"  │ {repr(content)}")
        print(f"  {'─' * 60}")
        print()
        print(f"  http={diag['http_status']} | tokens_predicted={diag['tokens_predicted']} "
              f"| stop_type={diag['stop_type']}")
        print(f"  latency={latency_ms:.0f}ms | timed_out={timed_out} | prompt_chars={diag['prompt_len_chars']}")
        print(f"  parsed={repr(answer_parsed)} | correct={correct} | error={error_code}")
        if timed_out:
            print(f"  >>> TIMEOUT after {latency_ms:.0f}ms (exception={diag['exception_type']})")
        print()

        trials.append({
            "prompt_id": prompt_id, "category": category,
            "error_code": error_code, "correct": int(correct),
            "timeout_flag": int(timed_out), "latency_ms": f"{latency_ms:.0f}",
            "answer_parsed": answer_parsed, "ground_truth": ground_truth,
            "http_status": diag['http_status'],
            "tokens_predicted": diag['tokens_predicted'],
            "stop_type": diag['stop_type'],
            "prompt_len_chars": diag['prompt_len_chars'],
            "answer_raw": content.replace("\n", "\\n")[:300],
        })

    # Summary
    print(f"{'=' * 70}")
    print("PROBE SUMMARY:")
    print(f"{'─' * 70}")
    for cat in ['AR', 'ALG', 'WP', 'LOG']:
        cat_trials = [t for t in trials if t['category'] == cat]
        if not cat_trials:
            continue
        n = len(cat_trials)
        ok = sum(t['correct'] for t in cat_trials)
        e7 = sum(t['timeout_flag'] for t in cat_trials)
        e8 = sum(1 for t in cat_trials if t['error_code'] == 'E8')
        print(f"  {cat:4s}: {ok}/{n} correct | {e7} E7 (timeout) | {e8} E8 (parse fail)")

    total_ok = sum(t['correct'] for t in trials)
    total_e7 = sum(t['timeout_flag'] for t in trials)
    total_e8 = sum(1 for t in trials if t['error_code'] == 'E8')
    print(f"{'─' * 70}")
    print(f"  TOTAL: {total_ok}/{n_prompts} correct | {total_e7} E7 | {total_e8} E8")

    if total_e7 == 0 and total_e8 == 0:
        print(f"\n  ALL PROMPTS RESPONDED. Prompt format + timeout look good.")
        print(f"  Safe to proceed with full baseline matrix.")
    elif total_e7 > 0:
        print(f"\n  TIMEOUTS DETECTED. Do NOT run full baseline yet.")
        print(f"  Check server load and consider increasing timeout.")
    else:
        print(f"\n  No timeouts but some parse failures. Check model output format.")

    # Save probe CSV
    probe_csv = args.out_trials.replace('.csv', '_probe.csv')
    os.makedirs(os.path.dirname(probe_csv), exist_ok=True)
    if trials:
        with open(probe_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=trials[0].keys())
            writer.writeheader()
            writer.writerows(trials)
        print(f"\nProbe CSV saved to: {probe_csv}")


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
    ap.add_argument("--debug_quick", action="store_true",
                    help="Debug run: 2 prompts per category, 1 repeat, full logging")
    ap.add_argument("--probe", action="store_true",
                    help="Probe mode: 3 AR, 3 ALG, 2 WP, 2 LOG, 1 repeat, full diagnostics")
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

    # Debug quick mode: 2 prompts per category, 1 repeat, full logging
    if args.debug_quick:
        run_debug_quick(args, config, prompts)
        return

    # Probe mode: 3 AR, 3 ALG, 2 WP, 2 LOG, 1 repeat
    if args.probe:
        run_probe(args, config, prompts)
        return

    # ========================================================
    # FULL RUN
    # ========================================================
    n_pred_base = ACTION_BUDGETS[args.action]
    timeout_sec = ACTION_TIMEOUTS[args.action]
    system_name = f"v1_{args.action.lower()}_{'grammar' if args.grammar else 'nogrammar'}"

    print(f"Running {system_name}")
    print(f"Action: {args.action} (max_tokens={n_pred_base})")
    print(f"Timeout: {timeout_sec}s")
    print(f"Grammar: {'enabled' if args.grammar else 'disabled'}")
    print(f"Parser: {PARSER_VERSION}")
    print(f"Prompt template: {PROMPT_TEMPLATE_VERSION}")
    print(f"Prompts: {len(prompts)}")
    print(f"Repeats: {config['repeats']}")
    print()

    # Health check
    if not check_server_health(config['server_url']):
        print("FATAL: Server health check failed. Is llama-server running?")
        sys.exit(1)

    # Warmup inference
    run_warmup(config['server_url'], timeout_sec)
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
            content, latency_ms, timed_out, diag = run_inference(
                prompt_text=prompt_text,
                server_url=config['server_url'],
                timeout_sec=timeout_sec,
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

                "e7_http_status": diag.get('http_status', ''),
                "e7_exception_type": diag.get('exception_type', ''),
                "e7_tokens_predicted": diag.get('tokens_predicted', ''),
                "e7_stop_type": diag.get('stop_type', ''),
                "e7_prompt_len_chars": diag.get('prompt_len_chars', ''),
                "e7_raw_preview": content.replace("\n", "\\n")[:100] if timed_out else "",

                "model_name": config['model_name'],
                "quantization": config['quantization'],
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 42,
                "timeout_sec": timeout_sec,
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
        "e7_http_status", "e7_exception_type", "e7_tokens_predicted",
        "e7_stop_type", "e7_prompt_len_chars", "e7_raw_preview",
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

    # Per-category breakdown
    categories = sorted(set(t['category'] for t in trials))
    print(f"\n  Per-category breakdown:")
    for cat in categories:
        cat_trials = [t for t in trials if t['category'] == cat]
        n = len(cat_trials)
        ok = sum(t['correct'] for t in cat_trials)
        e7 = sum(t['timeout_flag'] for t in cat_trials)
        e8 = sum(1 for t in cat_trials if t['error_code'] == 'E8')
        print(f"    {cat:4s}: {ok:2d}/{n} correct | {e7} E7 (timeout) | {e8} E8 (parse fail)")

    print(f"\nSaved to: {args.out_trials}")


if __name__ == "__main__":
    main()
