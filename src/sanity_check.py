#!/usr/bin/env python3
"""
Sanity Check - Minimal diagnostic for llama.cpp + Phi-4-Mini inference.

Tests both /completion (raw prompt) and /v1/chat/completions (OpenAI-compat)
to determine which path works correctly with Phi-4-Mini-Instruct.

Logs exact JSON request and response for debugging.

Usage:
    python3 src/sanity_check.py
    python3 src/sanity_check.py --server http://127.0.0.1:8080
    python3 src/sanity_check.py --timeout 60
"""

import argparse
import json
import time
import sys
import requests

DEFAULT_SERVER = "http://127.0.0.1:8080"
DEFAULT_TIMEOUT = 60

# ============================================================
# 8-prompt sanity suite
# ============================================================
SANITY_PROMPTS = [
    # AR: 2 prompts
    {"id": "S_AR1", "cat": "AR", "question": "What is 2 + 2?", "expected": "4",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},
    {"id": "S_AR2", "cat": "AR", "question": "What is 4 - (5 - (3 + 2))?", "expected": "4",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},

    # ALG: 1 prompt
    {"id": "S_ALG1", "cat": "ALG", "question": "Solve for x: 2x + 3 = 7", "expected": "2",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},

    # WP: 1 short prompt
    {"id": "S_WP1", "cat": "WP",
     "question": "Tim has 5 apples. He buys 3 more and gives away 2. How many apples does Tim have?",
     "expected": "6",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},

    # LOG: 1 short + 1 medium
    {"id": "S_LOG1", "cat": "LOG",
     "question": "All cats are animals. Whiskers is a cat. Is Whiskers an animal?",
     "expected": "Yes",
     "system": "You are a logic assistant. Return only Yes or No, nothing else.",
     "n_predict": 6},
    {"id": "S_LOG2", "cat": "LOG",
     "question": "Fiona is round. Fiona is red. If something is round and red then it is nice. Is Fiona nice?",
     "expected": "Yes",
     "system": "You are a logic assistant. Return only Yes or No, nothing else.",
     "n_predict": 6},

    # AR: 1 more (multi-step)
    {"id": "S_AR3", "cat": "AR", "question": "What is 15 * 7?", "expected": "105",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},

    # ALG: 1 more
    {"id": "S_ALG2", "cat": "ALG", "question": "Solve: 5x - 10 = 0", "expected": "2",
     "system": "You are a math assistant. Return only the final numeric answer, nothing else.",
     "n_predict": 12},
]


def check_health(base_url):
    """Check server health."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        data = r.json()
        status = data.get("status", "unknown")
        print(f"  Health: {status} (HTTP {r.status_code})")
        return status == "ok"
    except Exception as e:
        print(f"  Health check FAILED: {e}")
        return False


def test_completion_endpoint(base_url, prompt_text, n_predict, timeout_sec):
    """Test /completion endpoint with raw prompt string."""
    url = f"{base_url}/completion"
    payload = {
        "prompt": prompt_text,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "stop": ["\n", "<|end|>", "<|endoftext|>"],
    }

    return _send_request(url, payload, timeout_sec, "completion")


def test_chat_endpoint(base_url, system_msg, user_msg, n_predict, timeout_sec):
    """Test /v1/chat/completions endpoint with messages array."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "stop": ["\n"],
    }

    return _send_request(url, payload, timeout_sec, "chat")


def _send_request(url, payload, timeout_sec, mode):
    """Send request and return full diagnostics."""
    t0 = time.time()
    result = {
        "mode": mode,
        "url": url,
        "request": payload,
        "http_status": None,
        "content": "",
        "tokens_predicted": None,
        "stop_type": None,
        "timed_out": False,
        "exception": None,
        "latency_ms": 0,
        "raw_response": None,
    }

    try:
        r = requests.post(url, json=payload, timeout=(10.0, float(timeout_sec)))
        result["http_status"] = r.status_code
        j = r.json()
        result["raw_response"] = j

        if mode == "completion":
            result["content"] = j.get("content", "")
            result["tokens_predicted"] = j.get("tokens_predicted")
            result["stop_type"] = j.get("stop_type")
        elif mode == "chat":
            # OpenAI format
            choices = j.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                result["content"] = msg.get("content", "")
                result["stop_type"] = choices[0].get("finish_reason")
            usage = j.get("usage", {})
            result["tokens_predicted"] = usage.get("completion_tokens")

    except requests.exceptions.Timeout:
        result["timed_out"] = True
        result["exception"] = "Timeout"
    except Exception as e:
        result["exception"] = str(e)

    result["latency_ms"] = (time.time() - t0) * 1000
    if result["latency_ms"] >= timeout_sec * 1000:
        result["timed_out"] = True

    return result


def print_result(label, r, expected):
    """Print a single result with full diagnostics."""
    content = r["content"].strip() if r["content"] else ""
    match = content == expected or expected in content
    status = "PASS" if match else ("TIMEOUT" if r["timed_out"] else "WRONG")

    print(f"  [{status:7s}] {label}")
    print(f"    endpoint:  {r['mode']} ({r['url']})")
    print(f"    http:      {r['http_status']}")
    print(f"    latency:   {r['latency_ms']:.0f}ms")
    print(f"    tokens:    {r['tokens_predicted']}")
    print(f"    stop:      {r['stop_type']}")
    print(f"    raw:       {repr(content[:120])}")
    print(f"    expected:  {repr(expected)}")
    if r["timed_out"]:
        print(f"    TIMEOUT:   {r['exception']}")
    if r["exception"] and not r["timed_out"]:
        print(f"    ERROR:     {r['exception']}")
    print()
    return status


def main():
    ap = argparse.ArgumentParser(description="Sanity check for llama.cpp + Phi-4-Mini")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="Server base URL (no /completion)")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout per request (seconds)")
    ap.add_argument("--dump_json", action="store_true", help="Dump full request/response JSON")
    args = ap.parse_args()

    print("=" * 70)
    print("SANITY CHECK: llama.cpp + Phi-4-Mini Inference Pipeline")
    print("=" * 70)
    print(f"  Server:  {args.server}")
    print(f"  Timeout: {args.timeout}s")
    print()

    # Health
    if not check_health(args.server):
        print("\nServer not healthy. Aborting.")
        sys.exit(1)
    print()

    # ========================================
    # Phase 1: Test both endpoints with trivial prompt
    # ========================================
    print("=" * 70)
    print("PHASE 1: Endpoint comparison (trivial prompt: 'What is 2 + 2?')")
    print("=" * 70)
    print()

    # 1a: /completion with raw Phi chat tokens
    raw_prompt = "<|system|>You are a math assistant. Return only the final numeric answer, nothing else.<|end|><|user|>What is 2 + 2?<|end|><|assistant|>"
    r_completion = test_completion_endpoint(args.server, raw_prompt, 12, args.timeout)
    s1 = print_result("/completion (raw Phi tokens)", r_completion, "4")

    if args.dump_json:
        print("    REQUEST JSON:")
        print(f"    {json.dumps(r_completion['request'], indent=2)}")
        print("    RESPONSE JSON:")
        print(f"    {json.dumps(r_completion['raw_response'], indent=2)[:500]}")
        print()

    # 1b: /v1/chat/completions with messages
    r_chat = test_chat_endpoint(
        args.server,
        "You are a math assistant. Return only the final numeric answer, nothing else.",
        "What is 2 + 2?",
        12, args.timeout
    )
    s2 = print_result("/v1/chat/completions (messages)", r_chat, "4")

    if args.dump_json:
        print("    REQUEST JSON:")
        print(f"    {json.dumps(r_chat['request'], indent=2)}")
        print("    RESPONSE JSON:")
        resp_str = json.dumps(r_chat['raw_response'], indent=2) if r_chat['raw_response'] else "null"
        print(f"    {resp_str[:500]}")
        print()

    # 1c: /completion with plain prompt (no special tokens)
    plain_prompt = "What is 2 + 2?\nReturn only the final numeric answer.\nAnswer:"
    r_plain = test_completion_endpoint(args.server, plain_prompt, 12, args.timeout)
    s3 = print_result("/completion (plain, no template)", r_plain, "4")

    # Determine best endpoint
    print("-" * 70)
    results_phase1 = {"completion_phi": s1, "chat_api": s2, "completion_plain": s3}
    working = [k for k, v in results_phase1.items() if v == "PASS"]
    if working:
        best = working[0]
        print(f"  BEST ENDPOINT: {best}")
    else:
        print("  WARNING: No endpoint returned correct answer for '2+2'")
        print("  Check that the model is loaded and the server is responding.")
    print()

    # ========================================
    # Phase 2: Full 8-prompt sanity suite
    # ========================================
    # Pick the endpoint that worked, default to chat API
    if "chat_api" in working:
        use_chat = True
        endpoint_label = "/v1/chat/completions"
    elif "completion_phi" in working:
        use_chat = False
        endpoint_label = "/completion (Phi tokens)"
    else:
        use_chat = True  # Try chat anyway
        endpoint_label = "/v1/chat/completions (fallback)"

    print("=" * 70)
    print(f"PHASE 2: 8-prompt sanity suite via {endpoint_label}")
    print("=" * 70)
    print()

    results = []
    for sp in SANITY_PROMPTS:
        if use_chat:
            r = test_chat_endpoint(
                args.server, sp["system"], sp["question"],
                sp["n_predict"], args.timeout
            )
        else:
            raw = f"<|system|>{sp['system']}<|end|><|user|>{sp['question']}<|end|><|assistant|>"
            r = test_completion_endpoint(args.server, raw, sp["n_predict"], args.timeout)

        status = print_result(f"{sp['id']} ({sp['cat']}): {sp['question'][:50]}", r, sp["expected"])

        if args.dump_json:
            print("    REQUEST:")
            print(f"    {json.dumps(r['request'], indent=2)[:300]}")
            print("    RESPONSE:")
            resp_str = json.dumps(r['raw_response'], indent=2) if r['raw_response'] else "null"
            print(f"    {resp_str[:300]}")
            print()

        results.append({
            "id": sp["id"], "cat": sp["cat"], "status": status,
            "content": r["content"], "expected": sp["expected"],
            "latency_ms": r["latency_ms"], "timed_out": r["timed_out"],
        })

    # ========================================
    # Summary
    # ========================================
    print("=" * 70)
    print("SANITY SUMMARY")
    print("=" * 70)

    for cat in ["AR", "ALG", "WP", "LOG"]:
        cat_results = [r for r in results if r["cat"] == cat]
        n = len(cat_results)
        ok = sum(1 for r in cat_results if r["status"] == "PASS")
        e7 = sum(1 for r in cat_results if r["status"] == "TIMEOUT")
        wrong = sum(1 for r in cat_results if r["status"] == "WRONG")
        print(f"  {cat:4s}: {ok}/{n} PASS | {wrong} WRONG | {e7} TIMEOUT")

    total = len(results)
    total_ok = sum(1 for r in results if r["status"] == "PASS")
    total_e7 = sum(1 for r in results if r["status"] == "TIMEOUT")
    print(f"  {'TOTAL':4s}: {total_ok}/{total} PASS | {total - total_ok - total_e7} WRONG | {total_e7} TIMEOUT")
    print()

    if total_ok == total:
        print("  ALL PASS. Server + prompt format working correctly.")
        print(f"  Recommended endpoint: {endpoint_label}")
        print("  Safe to run probe mode, then full baseline matrix.")
    elif total_e7 > 0:
        print("  TIMEOUTS present. Server may be overloaded or timeout too short.")
        print(f"  Try: --timeout 90, or restart llama-server with fewer threads.")
    else:
        print("  WRONG ANSWERS but no timeouts. Model is responding but not correctly.")
        print("  Check prompt format and model capabilities.")

    print()
    print("Next steps:")
    print("  1. If /v1/chat/completions works: update runner to use chat API")
    print("  2. If /completion works: keep current approach")
    print("  3. If neither works: check model file and server config")


if __name__ == "__main__":
    main()
