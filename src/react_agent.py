#!/usr/bin/env python3
"""
ReAct Agent Baseline

Implements the ReAct (Reasoning + Acting) framework for comparison.
The model interleaves Thought/Action/Observation steps to solve problems.
Uses the same symbolic tools as the neurosymbolic router.

Reference: Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)
"""

import re
import time
import requests
import ast
import operator as op
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict


REACT_SYSTEM_PROMPT = """Solve problems using Thought/Action/Observation steps.

Actions available:
  arithmetic_eval(expr) - evaluate arithmetic like 4+5*2
  sympy_solve(eq, var) - solve equation like 3*x+1=10, x
  logic_engine(Yes or No) - answer logic inference question
  direct_answer(number) - give final answer directly
  finish(answer) - return the final answer

Format each step exactly as:
Thought: <reasoning>
Action: <tool_call>
Observation: <result>
...
Action: finish(<answer>)

Examples:
Problem: Calculate 4-(5-(3+2)).
Thought: I need to evaluate this arithmetic expression.
Action: arithmetic_eval(4-(5-(3+2)))
Observation: 4
Action: finish(4)

Problem: Solve 5*m-18=0 for m.
Thought: This is an algebra problem. I will use sympy_solve.
Action: sympy_solve(5*m-18=0, m)
Observation: 18/5
Action: finish(18/5)

Problem: Felix is a cat. All cats are animals. Is Felix an animal?
Thought: I need to apply forward chaining logic.
Action: logic_engine(Yes)
Observation: Yes
Action: finish(Yes)"""


SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg,
}

ACTION_RE = re.compile(
    r'Action:\s*(arithmetic_eval|sympy_solve|logic_engine|direct_answer|finish)\((.+?)\)\s*$',
    re.MULTILINE | re.DOTALL
)


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    try:
        expr = expr.strip().rstrip('.')
        if len(expr) > 200:
            return None
        tree = ast.parse(expr, mode='eval')

        def _eval(node):
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float)):
                    raise ValueError
                return node.value
            elif isinstance(node, ast.BinOp):
                left, right = _eval(node.left), _eval(node.right)
                if isinstance(node.op, ast.Pow) and abs(right) > 10000:
                    raise ValueError
                return SAFE_OPS[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                return SAFE_OPS[type(node.op)](_eval(node.operand))
            raise ValueError

        result = _eval(tree.body)
        if result == int(result):
            return int(result)
        return result
    except Exception:
        return None


def sympy_solve_equation(args_str: str) -> Optional[str]:
    try:
        from sympy import symbols, Eq, solve
        from sympy.parsing.sympy_parser import parse_expr

        parts = args_str.rsplit(',', 1)
        if len(parts) != 2:
            return None
        eq_str = parts[0].strip()
        var_name = parts[1].strip()
        var = symbols(var_name)

        if '=' in eq_str:
            sides = eq_str.split('=', 1)
            lhs = parse_expr(sides[0].strip())
            rhs = parse_expr(sides[1].strip())
            equation = Eq(lhs, rhs)
        else:
            equation = Eq(parse_expr(eq_str), 0)

        solutions = solve(equation, var)
        if solutions:
            sol = solutions[0]
            if sol == int(sol):
                return str(int(sol))
            return str(sol)
        return None
    except Exception:
        return None


def execute_tool(tool_name: str, args: str) -> str:
    if tool_name == "arithmetic_eval":
        result = safe_eval_arithmetic(args)
        return str(result) if result is not None else "Error: could not evaluate"

    elif tool_name == "sympy_solve":
        result = sympy_solve_equation(args)
        return str(result) if result is not None else "Error: could not solve"

    elif tool_name == "logic_engine":
        args_clean = args.strip().strip('"').strip("'")
        if args_clean.lower() in ("yes", "no"):
            return args_clean.capitalize()
        return args_clean

    elif tool_name == "direct_answer":
        nums = re.findall(r'[-+]?\d+(?:\.\d+)?', args)
        return nums[-1] if nums else args.strip()

    elif tool_name == "finish":
        return args.strip()

    return "Unknown tool"


class ReActAgent:
    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 max_tokens: int = 300, timeout: int = 120, max_steps: int = 6):
        self.server_url = server_url.rstrip('/')
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_steps = max_steps

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
                "stop": ["Observation:"],
            },
            timeout=(10.0, self.timeout),
        )
        return response.json()["choices"][0]["message"]["content"]

    def solve(self, prompt_text: str, ground_truth: str = "") -> Dict[str, Any]:
        start = time.time()
        slm_calls = 0
        steps = []

        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem: {prompt_text}"},
        ]

        final_answer = ""
        tool_called = "none"

        try:
            for step in range(self.max_steps):
                output = self._call_slm(messages)
                slm_calls += 1
                steps.append(output)

                # Look for Action in output
                match = ACTION_RE.search(output)
                if not match:
                    # No action found, try to extract any answer
                    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', output)
                    if nums:
                        final_answer = nums[-1]
                    break

                tool_name = match.group(1)
                tool_args = match.group(2).strip()
                tool_called = tool_name

                if tool_name == "finish":
                    final_answer = tool_args.strip()
                    # Normalize
                    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', final_answer)
                    if nums:
                        final_answer = nums[-1]
                    break

                # Execute tool and add observation to context
                observation = execute_tool(tool_name, tool_args)
                messages.append({"role": "assistant", "content": output})
                messages.append({"role": "user", "content": f"Observation: {observation}\n"})

                # If it's a terminal tool (logic_engine, direct_answer), we're done
                if tool_name in ("logic_engine", "direct_answer"):
                    final_answer = observation
                    break

        except Exception as e:
            return {
                "answer": "",
                "tool_called": "error",
                "slm_calls": slm_calls,
                "steps": steps,
                "raw_output": "",
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
            "tool_called": tool_called,
            "slm_calls": slm_calls,
            "steps": len(steps),
            "latency_ms": latency_ms,
            "correct": answer_str == ground_truth,
        }


def run_benchmark(csv_path: str, server_url: str = "http://127.0.0.1:8080",
                  max_tokens: int = 300, repeats: int = 3) -> Dict[str, Any]:
    import csv as csv_mod

    agent = ReActAgent(server_url=server_url, max_tokens=max_tokens)
    results = []
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    tool_usage = defaultdict(int)

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
            tool_usage[result['tool_called']] += 1

            print(f"[{trial_num}/{total_trials}] {row['prompt_id']} r{repeat}: "
                  f"tool={result['tool_called']:<20s} "
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
        'tool_usage': dict(tool_usage),
        'avg_slm_calls': avg_calls,
        'results': results,
    }
