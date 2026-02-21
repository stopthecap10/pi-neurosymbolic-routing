#!/usr/bin/env python3
"""
Debug LLM Request - Standalone smoke test for llama.cpp server.

Uses Phi-4 chat template format, matching the baseline runner.
Useful for diagnosing E7 timeouts and server issues.

Usage:
    python3 src/debug_llm_request.py
    python3 src/debug_llm_request.py --server http://127.0.0.1:8080/completion
    python3 src/debug_llm_request.py --timeout 45
"""

import argparse
import time
import requests

DEFAULT_SERVER = "http://127.0.0.1:8080/completion"
DEFAULT_TIMEOUT = 45

SYS_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYS_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."


def phi_prompt(system_msg, question):
    """Build Phi-4 chat template prompt."""
    return f"<|system|>{system_msg}<|end|><|user|>{question}<|end|><|assistant|>"


# Test prompts using Phi chat format
TEST_PROMPTS = [
    {
        "label": "trivial_math",
        "prompt": phi_prompt(SYS_NUMERIC, "What is 2 + 2?"),
        "n_predict": 12,
        "expected": "4",
    },
    {
        "label": "AR_short",
        "prompt": phi_prompt(SYS_NUMERIC, "What is 15 * 7?"),
        "n_predict": 12,
        "expected": "105",
    },
    {
        "label": "ALG_short",
        "prompt": phi_prompt(SYS_NUMERIC, "Solve for x: 3x + 7 = 22"),
        "n_predict": 12,
        "expected": "5",
    },
    {
        "label": "LOG_yesno",
        "prompt": phi_prompt(SYS_YESNO, "If all cats are animals, and Whiskers is a cat, is Whiskers an animal?"),
        "n_predict": 6,
        "expected": "Yes",
    },
    {
        "label": "WP_medium",
        "prompt": phi_prompt(SYS_NUMERIC, "Tim has 5 apples. He buys 3 more and gives away 2. How many apples does Tim have?"),
        "n_predict": 12,
        "expected": "6",
    },
]


def check_health(base_url):
    """Check server health endpoint."""
    health_url = base_url.rsplit('/', 1)[0] + "/health"
    print(f"[1/3] Checking server health: {health_url}")
    try:
        r = requests.get(health_url, timeout=10)
        data = r.json()
        status = data.get("status", "unknown")
        print(f"      Status: {status} (HTTP {r.status_code})")
        return status == "ok"
    except Exception as e:
        print(f"      FAILED: {e}")
        return False


def send_request(server_url, prompt, n_predict, timeout_sec):
    """Send a single completion request and return full diagnostics."""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stop": ["\n", "<|end|>", "<|endoftext|>"],
    }

    t0 = time.time()
    result = {
        "http_status": None,
        "content": "",
        "tokens_predicted": None,
        "stop_type": None,
        "timed_out": False,
        "exception": None,
        "latency_ms": 0,
        "full_response": None,
    }

    try:
        r = requests.post(server_url, json=payload, timeout=(10.0, float(timeout_sec)))
        result["http_status"] = r.status_code
        j = r.json()
        result["content"] = j.get("content", "")
        result["tokens_predicted"] = j.get("tokens_predicted")
        result["stop_type"] = j.get("stop_type")
        result["full_response"] = {k: v for k, v in j.items()
                                    if k not in ("content", "model", "prompt")}
    except requests.exceptions.Timeout:
        result["timed_out"] = True
        result["exception"] = "Timeout"
    except Exception as e:
        result["exception"] = str(e)

    result["latency_ms"] = (time.time() - t0) * 1000

    if result["latency_ms"] >= timeout_sec * 1000:
        result["timed_out"] = True

    return result


def main():
    ap = argparse.ArgumentParser(description="Debug LLM server requests (Phi chat format)")
    ap.add_argument("--server", default=DEFAULT_SERVER, help="Server URL")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    args = ap.parse_args()

    print("=" * 60)
    print("DEBUG LLM REQUEST SMOKE TEST (Phi Chat Format)")
    print("=" * 60)
    print(f"Server:  {args.server}")
    print(f"Timeout: {args.timeout}s")
    print(f"Format:  <|system|>...<|end|><|user|>...<|end|><|assistant|>")
    print()

    # Step 1: Health check
    healthy = check_health(args.server)
    if not healthy:
        print("\nServer is not healthy. Start it with:")
        print("  cd ~/llama.cpp/build/bin && ./llama-server \\")
        print('    -m "/path/to/model.gguf" \\')
        print("    --host 127.0.0.1 --port 8080 -c 4096 -t 4 -ngl 0")
        return

    # Step 2: Warmup
    print(f"\n[2/3] Warmup request...")
    warmup_prompt = phi_prompt(SYS_NUMERIC, "What is 1 + 1?")
    warmup = send_request(args.server, warmup_prompt, 4, args.timeout)
    print(f"      Warmup: {repr(warmup['content'][:40])} in {warmup['latency_ms']:.0f}ms")

    # Step 3: Test prompts
    print(f"\n[3/3] Running {len(TEST_PROMPTS)} test prompts...\n")

    results = []
    for i, tp in enumerate(TEST_PROMPTS, 1):
        r = send_request(args.server, tp["prompt"], tp["n_predict"], args.timeout)

        content = r["content"].strip()
        match = content == tp["expected"] or tp["expected"] in content
        status = "PASS" if match else ("TIMEOUT" if r["timed_out"] else "FAIL")

        print(f"  [{status:7s}] {tp['label']}")
        print(f"           prompt_len={len(tp['prompt'])} chars | n_predict={tp['n_predict']}")
        print(f"           http={r['http_status']} | tokens={r['tokens_predicted']} "
              f"| stop={r['stop_type']} | lat={r['latency_ms']:.0f}ms")
        print(f"           raw={repr(content[:80])}")
        print(f"           expected={repr(tp['expected'])}")
        if r["timed_out"]:
            print(f"           >>> TIMEOUT after {r['latency_ms']:.0f}ms (exception={r['exception']})")
        print()

        results.append({"label": tp["label"], "status": status, "latency_ms": r["latency_ms"]})

    # Summary
    print("=" * 60)
    passes = sum(1 for r in results if r["status"] == "PASS")
    timeouts = sum(1 for r in results if r["status"] == "TIMEOUT")
    fails = sum(1 for r in results if r["status"] == "FAIL")
    print(f"RESULTS: {passes} PASS, {fails} FAIL, {timeouts} TIMEOUT out of {len(results)}")

    if timeouts > 0:
        print(f"\nTIMEOUT DETECTED — the server may be too slow or overloaded.")
        print(f"Try: increase --timeout, reduce -c context, or restart the server.")
    elif fails > 0:
        print(f"\nSome prompts returned wrong answers but did not timeout.")
        print(f"This is expected — the model may simply get some wrong.")
    else:
        print(f"\nAll prompts passed. Server is responding correctly.")
        print(f"Safe to proceed with probe mode, then full baseline.")


if __name__ == "__main__":
    main()
