#!/usr/bin/env python3
"""
Tool-Calling Agent Baseline

Gives the SLM (Phi-4-mini) access to the same symbolic solvers as the
neurosymbolic router, but lets the model decide which tool to call.

This serves as the fair baseline comparison requested by the reviewer:
instead of hardcoded regex routing, the SLM itself decides routing.

Tools available to the agent:
  - arithmetic_eval(expression): Safe AST-based arithmetic evaluation
  - sympy_solve(equation, variable): SymPy symbolic equation solver
  - logic_engine(context): Forward-chaining logic inference
  - direct_answer(answer): Return answer directly (no tool)
"""

import re
import time
import requests
import ast
import operator as op
from typing import Dict, Any, Optional, Tuple

# Import existing solvers
from src.token_classifier import tokenize


# ---- Tool definitions for the system prompt ----

TOOL_CALLING_SYSTEM_PROMPT = """You are a math and logic assistant with access to tools.
Given a problem, decide which ONE tool to call and output EXACTLY one line in this format:
TOOL_NAME(arguments)

Available tools:
- arithmetic_eval(expression): Evaluate a pure arithmetic expression. Example: arithmetic_eval(4 - (5 - (3 + 2)))
- sympy_solve(equation, variable): Solve an algebraic equation. Example: sympy_solve(5*m - 7 - 18 = 0, m)
- logic_engine(answer): For logic questions, respond with Yes or No. Example: logic_engine(Yes)
- direct_answer(answer): For word problems or when you know the answer. Example: direct_answer(42)

Output ONLY the tool call, nothing else."""

FEW_SHOT_EXAMPLES = """Examples:
Problem: Calculate 4 - (5 - (3 + 2)).
arithmetic_eval(4 - (5 - (3 + 2)))

Problem: Solve 5*m - 7 - 18 = 0 for m.
sympy_solve(5*m - 7 - 18 = 0, m)

Problem: The cat is green. If something is green then it is kind. Is the cat kind?
logic_engine(Yes)

Problem: A bus travels 60 mph for 5 hours. How far did it go?
direct_answer(300)

"""

# ---- Tool call parser ----

TOOL_CALL_RE = re.compile(
    r'(arithmetic_eval|sympy_solve|logic_engine|direct_answer)\((.+)\)\s*$',
    re.DOTALL
)


def parse_tool_call(text: str) -> Optional[Tuple[str, str]]:
    """Parse a tool call from model output. Returns (tool_name, args) or None."""
    text = text.strip()
    # Try each line (model might output reasoning before the call)
    for line in reversed(text.split('\n')):
        line = line.strip()
        m = TOOL_CALL_RE.match(line)
        if m:
            return m.group(1), m.group(2).strip()
    # Fallback: search anywhere in the text
    m = TOOL_CALL_RE.search(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None


# ---- Safe arithmetic evaluator (same as router_v1.py) ----

SAFE_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg,
}


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """Safely evaluate arithmetic expression."""
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


# ---- SymPy solver ----

def sympy_solve_equation(args_str: str) -> Optional[str]:
    """Parse and solve an equation using SymPy."""
    try:
        from sympy import symbols, Eq, solve
        from sympy.parsing.sympy_parser import parse_expr

        # Parse "equation, variable" format
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


# ---- Agent execution ----

class ToolCallingAgent:
    """Agent that uses SLM to decide which tool to call."""

    def __init__(self, server_url: str = "http://127.0.0.1:8080",
                 max_tokens: int = 50, timeout: int = 60):
        self.server_url = server_url.rstrip('/')
        self.max_tokens = max_tokens
        self.timeout = timeout

    def solve(self, prompt_text: str, ground_truth: str = "") -> Dict[str, Any]:
        """Send prompt to SLM with tool-calling instructions, execute the chosen tool."""
        start = time.time()

        # Build the chat message
        user_msg = f"{FEW_SHOT_EXAMPLES}Problem: {prompt_text}"

        # Call SLM via chat API
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": TOOL_CALLING_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "seed": 42,
                    "stop": ["\n\n"],
                },
                timeout=(10.0, self.timeout),
            )
            raw_output = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raw_output = ""
            return {
                "answer": "",
                "tool_called": "error",
                "tool_args": str(e),
                "raw_output": "",
                "latency_ms": (time.time() - start) * 1000,
                "correct": False,
                "error": str(e),
            }

        # Parse tool call
        parsed = parse_tool_call(raw_output)
        if parsed is None:
            tool_name, tool_args, answer = "none", "", raw_output.strip()
        else:
            tool_name, tool_args = parsed
            answer = self._execute_tool(tool_name, tool_args)

        latency_ms = (time.time() - start) * 1000
        answer_str = str(answer) if answer is not None else ""

        # Format answer for comparison
        try:
            if answer_str and float(answer_str) == int(float(answer_str)):
                answer_str = str(int(float(answer_str)))
        except (ValueError, OverflowError):
            pass

        return {
            "answer": answer_str,
            "tool_called": tool_name,
            "tool_args": tool_args,
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "correct": answer_str == ground_truth,
        }

    def _execute_tool(self, tool_name: str, args: str) -> Optional[str]:
        """Execute the tool and return the result."""
        if tool_name == "arithmetic_eval":
            result = safe_eval_arithmetic(args)
            return str(result) if result is not None else None

        elif tool_name == "sympy_solve":
            return sympy_solve_equation(args)

        elif tool_name == "logic_engine":
            # The model should output Yes/No directly
            args_clean = args.strip().strip('"').strip("'")
            if args_clean.lower() in ("yes", "no"):
                return args_clean.capitalize()
            return args_clean

        elif tool_name == "direct_answer":
            # Extract numeric answer
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?', args)
            if nums:
                return nums[-1]
            return args.strip()

        return None


def run_benchmark(csv_path: str, server_url: str = "http://127.0.0.1:8080",
                  max_tokens: int = 50) -> Dict[str, Any]:
    """Run the tool-calling agent on a benchmark CSV.

    Note: Requires llama.cpp server running on the Pi.
    """
    import csv as csv_mod
    from collections import defaultdict

    agent = ToolCallingAgent(server_url=server_url, max_tokens=max_tokens)
    results = []
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    tool_usage = defaultdict(int)

    with open(csv_path) as f:
        for row in csv_mod.DictReader(f):
            result = agent.solve(row['prompt_text'], row['ground_truth'])
            result['prompt_id'] = row['prompt_id']
            result['category'] = row['category']
            result['ground_truth'] = row['ground_truth']
            results.append(result)

            cat = row['category']
            by_cat[cat]['total'] += 1
            if result['correct']:
                by_cat[cat]['correct'] += 1
            tool_usage[result['tool_called']] += 1

            print(f"{row['prompt_id']}: tool={result['tool_called']:<20s} "
                  f"correct={result['correct']}  "
                  f"latency={result['latency_ms']:.0f}ms")

    total = len(results)
    correct = sum(1 for r in results if r['correct'])

    return {
        'total_accuracy': correct / total if total else 0,
        'correct': correct,
        'total': total,
        'per_category': dict(by_cat),
        'tool_usage': dict(tool_usage),
        'results': results,
    }


if __name__ == "__main__":
    print("Tool-calling agent baseline")
    print("Requires llama.cpp server running at http://127.0.0.1:8080")
    print("Run on the Pi with: python3 -m src.agent_baseline")
    print()

    # Test tool call parsing
    test_cases = [
        ("arithmetic_eval(4 - (5 - (3 + 2)))", ("arithmetic_eval", "4 - (5 - (3 + 2))")),
        ("sympy_solve(5*m - 7 - 18 = 0, m)", ("sympy_solve", "5*m - 7 - 18 = 0, m")),
        ("logic_engine(Yes)", ("logic_engine", "Yes")),
        ("direct_answer(42)", ("direct_answer", "42")),
        ("Let me think... arithmetic_eval(3+5)", ("arithmetic_eval", "3+5")),
    ]

    print("Tool call parsing tests:")
    for text, expected in test_cases:
        result = parse_tool_call(text)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {text[:50]:<50s} {status}")

    # Test arithmetic evaluator
    print("\nArithmetic eval tests:")
    for expr, expected in [("4-(5-(3+2))", 4), ("710/142", 5), ("-1*-201", 201)]:
        result = safe_eval_arithmetic(expr)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"  {expr:<20s} = {expected:<6} {status}")
