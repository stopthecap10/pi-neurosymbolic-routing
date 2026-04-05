#!/usr/bin/env python3
"""
Program-of-Thought (PoT) Agent Baseline

The model generates a Python program to solve the problem, then the program
is executed to get the answer. Separates reasoning from computation.

Reference: Chen et al. "Program of Thoughts Prompting" (TMLR 2023)
"""

import re
import time
import requests
import ast
import operator as op
from typing import Dict, Any, Optional
from collections import defaultdict
import io
import contextlib


POT_SYSTEM_PROMPT = """Solve math and logic problems by writing a short Python program.
Output ONLY the Python code, no explanation. End with a print() statement for the answer.

Examples:

Problem: Calculate 4-(5-(3+2)).
```python
result = 4-(5-(3+2))
print(result)
```

Problem: Solve 5*m-18=0 for m.
```python
from sympy import symbols, Eq, solve
m = symbols('m')
sol = solve(Eq(5*m-18, 0), m)
print(sol[0])
```

Problem: A store sells apples for $3 each. Tom buys 7 apples. How much does he spend?
```python
price = 3
quantity = 7
total = price * quantity
print(total)
```

Problem: Felix is a cat. All cats are animals. Is Felix an animal?
```python
facts = {"felix": {"cat"}}
rules = [("cat", "animal")]
for subj, prop in list(facts.items()):
    for trigger, result in rules:
        if trigger in prop:
            facts[subj].add(result)
print("Yes" if "animal" in facts.get("felix", set()) else "No")
```"""


def extract_code(text: str) -> str:
    """Extract Python code from model output."""
    # Try fenced code block first
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: treat whole output as code
    return text.strip()


def safe_execute_code(code: str, timeout_secs: int = 10) -> Optional[str]:
    """Execute generated Python code safely and capture output."""
    # Basic safety checks - block dangerous operations
    forbidden = ['import os', 'import sys', 'import subprocess', '__import__',
                 'open(', 'exec(', 'eval(', 'compile(', 'globals()', 'locals()']
    code_lower = code.lower()
    for f in forbidden:
        if f in code_lower:
            return None

    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, {"__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "list": list,
                "dict": dict,
                "set": set,
                "zip": zip,
                "enumerate": enumerate,
                "__import__": __import__,  # needed for sympy
            }})
        output = stdout_capture.getvalue().strip()
        return output if output else None
    except Exception:
        return None


class PoTAgent:
    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 max_tokens: int = 300, timeout: int = 120):
        self.server_url = server_url.rstrip('/')
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _call_slm(self, messages: list) -> str:
        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
            },
            timeout=(10.0, self.timeout),
        )
        return response.json()["choices"][0]["message"]["content"]

    def solve(self, prompt_text: str, ground_truth: str = "") -> Dict[str, Any]:
        start = time.time()
        slm_calls = 0

        messages = [
            {"role": "system", "content": POT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {prompt_text}"},
        ]

        final_answer = ""
        raw_output = ""
        code = ""

        try:
            raw_output = self._call_slm(messages)
            slm_calls += 1

            code = extract_code(raw_output)
            execution_result = safe_execute_code(code)

            if execution_result is not None:
                final_answer = execution_result.strip()
            else:
                # Retry once with a simpler prompt if code failed
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": "The code failed. Write a simpler version."})
                raw_output2 = self._call_slm(messages)
                slm_calls += 1
                code2 = extract_code(raw_output2)
                result2 = safe_execute_code(code2)
                if result2 is not None:
                    final_answer = result2.strip()

        except Exception as e:
            return {
                "answer": "",
                "tool_called": "code_exec",
                "slm_calls": slm_calls,
                "code": code,
                "raw_output": raw_output,
                "latency_ms": (time.time() - start) * 1000,
                "correct": False,
                "error": str(e),
            }

        latency_ms = (time.time() - start) * 1000

        # Normalize answer
        answer_str = str(final_answer).strip()
        try:
            if answer_str and float(answer_str) == int(float(answer_str)):
                answer_str = str(int(float(answer_str)))
        except (ValueError, OverflowError):
            pass

        return {
            "answer": answer_str,
            "tool_called": "code_exec",
            "slm_calls": slm_calls,
            "code": code,
            "latency_ms": latency_ms,
            "correct": answer_str == ground_truth,
        }


def run_benchmark(csv_path: str, server_url: str = "http://127.0.0.1:8080",
                  max_tokens: int = 300, repeats: int = 3) -> Dict[str, Any]:
    import csv as csv_mod

    agent = PoTAgent(server_url=server_url, max_tokens=max_tokens)
    results = []
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})

    prompts = []
    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            prompts.append(row)

    total_trials = len(prompts) * repeats
    trial_num = 0

    for row in prompts:
        for repeat in range(1, repeats + 1):
            trial_num += 1
            result = agent.solve(row['prompt_text'], row['ground_truth'])
            result['prompt_id'] = row['prompt_id']
            result['category'] = row['category']
            result['ground_truth'] = row['ground_truth']
            result['repeat'] = repeat
            results.append(result)

            cat = row['category']
            by_cat[cat]['total'] += 1
            if result['correct']:
                by_cat[cat]['correct'] += 1

            print(f"[{trial_num}/{total_trials}] {row['prompt_id']} r{repeat}: "
                  f"correct={result['correct']}  "
                  f"calls={result.get('slm_calls', 1)}  "
                  f"latency={result['latency_ms']:.0f}ms")

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    avg_calls = sum(r.get('slm_calls', 1) for r in results) / total if total else 0

    return {
        'total_accuracy': correct / total if total else 0,
        'correct': correct,
        'total': total,
        'repeats': repeats,
        'per_category': dict(by_cat),
        'avg_slm_calls': avg_calls,
        'results': results,
    }
