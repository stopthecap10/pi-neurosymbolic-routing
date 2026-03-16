#!/usr/bin/env python3
"""
Live Demo — Interactive neurosymbolic router for video recording.
Run on the Pi: python3 src/live_demo.py

Requires: sympy, requests (for WP only), a6_logic_engine.py + a6_logic_parser.py
Start llama.cpp server first for WP: ./llama-server -m <model> --port 8080
"""

import ast
import operator
import os
import re
import sys
import time
from typing import Optional

# ── Safe arithmetic eval (A5) ──────────────────────────────────────

SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def safe_eval_expr(node):
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Non-numeric")
        return node.value
    elif isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10000:
            raise ValueError("Exponent too large")
        return SAFE_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        return SAFE_OPS[type(node.op)](safe_eval_expr(node.operand))
    else:
        raise ValueError(f"Unsupported: {type(node)}")

def safe_eval_arithmetic(expr: str) -> Optional[float]:
    try:
        tree = ast.parse(expr, mode='eval')
        return float(safe_eval_expr(tree.body))
    except Exception:
        return None


# ── Extract arithmetic expression from question text ───────────────

PREFIXES = [
    r'^what\s+is\s+the\s+value\s+of\s+',
    r'^what\s+is\s+',
    r'^calculate\s+',
    r'^compute\s+',
    r'^evaluate\s+',
    r'^find\s+',
    r'^simplify\s+',
]
SKIP_PAT = re.compile(r'^(answer|answer with|return only)\b', re.IGNORECASE)
FLOAT_RE = re.compile(r"[-+]?\d+\.?\d*")

def extract_arithmetic(text: str) -> Optional[str]:
    for raw_line in text.splitlines():
        line = raw_line.strip().rstrip('?.').strip()
        if not line or SKIP_PAT.match(line):
            continue
        if safe_eval_arithmetic(line) is not None:
            return line
        for pat in PREFIXES:
            stripped = re.sub(pat, '', line, flags=re.IGNORECASE).strip().rstrip('?.').strip()
            if stripped and stripped != line and safe_eval_arithmetic(stripped) is not None:
                return stripped
    return None


# ── A4: SymPy algebra solver ──────────────────────────────────────

def solve_algebra(question: str):
    """Extract equation, solve with SymPy. Returns (answer_str, steps_str) or None."""
    try:
        import sympy
    except ImportError:
        return None

    # Find target variable
    target_var = None
    m = re.search(r'\bfor\s+([a-zA-Z])\b', question)
    if m:
        target_var = m.group(1)
    else:
        m = re.search(r'\bwhat\s+is\s+([a-zA-Z])\b', question, re.IGNORECASE)
        if m:
            target_var = m.group(1)
        else:
            m = re.search(r'^Determine\s+([a-zA-Z])\b', question, re.IGNORECASE)
            if m:
                target_var = m.group(1)

    if not target_var:
        # Auto-detect: find single letters next to digits (e.g. 4x, 9n)
        vars_found = set(re.findall(r'\d\s*([a-zA-Z])\b', question))
        if vars_found:
            target_var = sorted(vars_found)[0].lower()
        else:
            return None

    # Clean equation text
    eq_text = question
    eq_text = re.sub(r'^(Solve|Let)\s+', '', eq_text, flags=re.IGNORECASE)
    eq_text = re.sub(r'^Determine\s+[a-zA-Z]\s*,?\s*given\s+that\s+', '', eq_text, flags=re.IGNORECASE)
    eq_text = re.sub(r'\s+for\s+[a-zA-Z]\s*[.?]?\s*$', '', eq_text)
    eq_text = re.sub(r'\s*What\s+is\s+[a-zA-Z]\s*[.?]?\s*$', '', eq_text, flags=re.IGNORECASE)
    eq_text = eq_text.strip().rstrip('.')

    # Insert * between digit and variable: 9x -> 9*x
    eq_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq_text)

    if '=' not in eq_text:
        return None

    eq_strings = [e.strip() for e in eq_text.split(',') if '=' in e]
    if not eq_strings:
        return None

    try:
        var = sympy.Symbol(target_var)
        equations = []
        for eq_str in eq_strings:
            parts = eq_str.split('=')
            if len(parts) != 2:
                continue
            lhs = sympy.sympify(parts[0].strip())
            rhs = sympy.sympify(parts[1].strip())
            equations.append(sympy.Eq(lhs, rhs))

        if not equations:
            return None

        solution = sympy.solve(equations, var)

        if isinstance(solution, dict):
            ans = solution.get(var)
        elif isinstance(solution, list) and len(solution) > 0:
            ans = solution[0]
        else:
            return None

        if ans is None:
            return None

        ans_str = str(ans)
        steps = f"Equation: {', '.join(eq_strings)}\nVariable: {target_var}\nSymPy solve -> {target_var} = {ans_str}"
        return (ans_str, steps)

    except Exception as e:
        return None


# ── A6: Logic engine ──────────────────────────────────────────────

def solve_logic_problem(question: str):
    """Run forward-chaining logic. Returns (answer, trace) or None."""
    try:
        # Add src to path so we can import
        src_dir = os.path.dirname(os.path.abspath(__file__))
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        from a6_logic_engine import solve_logic
    except ImportError:
        return None

    result = solve_logic(question)
    if result["parse_success"]:
        return (result["answer"], result["trace"], result["latency_ms"])
    return None


# ── A2: LLM word problem solver ──────────────────────────────────

SERVER_URL = "http://127.0.0.1:8080/completion"

def solve_word_problem(question: str, token_budget: int = 512):
    """Call llama.cpp server for WP. Returns (answer, raw_output, latency_ms) or None."""
    try:
        import requests
    except ImportError:
        return None

    prompt = (
        f"<|system|>You are a math assistant. Show your work step by step, "
        f"then give the final numeric answer.<|end|>"
        f"<|user|>{question}<|end|>"
        f"<|assistant|>"
    )

    try:
        t0 = time.time()
        resp = requests.post(SERVER_URL, json={
            "prompt": prompt,
            "n_predict": token_budget,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "stop": ["<|end|>", "<|endoftext|>"],
        }, timeout=300)
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 200:
            raw = resp.json().get("content", "").strip()
            # Extract final number
            nums = re.findall(r'[-+]?\d+(?:\.\d+)?', raw)
            answer = nums[-1] if nums else raw
            return (answer, raw, latency_ms)
    except Exception as e:
        return None
    return None


# ── Classifier ────────────────────────────────────────────────────

def classify(question: str) -> str:
    q = question.lower()
    # Logic: RuleTaker-style patterns
    if re.search(r'\b(if someone|if .+ then|all .+ are|every .+ is|question:)\b', q, re.IGNORECASE):
        return "LOG"
    # Algebra: has variables and equations
    if re.search(r'\bsolve\b.*\bfor\s+[a-z]\b', q, re.IGNORECASE):
        return "ALG"
    if re.search(r'\bsolve\b', q, re.IGNORECASE) and re.search(r'\d+\s*[a-z]\s*[+\-]', q):
        return "ALG"
    if re.search(r'\b[a-z]\s*[+\-*/]\s*\d+\s*=', q) or re.search(r'=\s*\d.*\bfor\s+[a-z]', q, re.IGNORECASE):
        return "ALG"
    if re.search(r'\bdetermine\s+[a-z]\b', q, re.IGNORECASE):
        return "ALG"
    # Arithmetic: pure numeric expressions
    expr = extract_arithmetic(question)
    if expr:
        return "AR"
    # Default: word problem
    return "WP"


# ── ANSI colors ───────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
WHITE = "\033[97m"
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_MAGENTA = "\033[45m"

CAT_COLORS = {
    "AR":  (BLUE,    BG_BLUE),
    "ALG": (GREEN,   BG_GREEN),
    "LOG": (YELLOW,  BG_YELLOW),
    "WP":  (MAGENTA, BG_MAGENTA),
}

SOLVER_NAMES = {
    "AR":  "A5 Symbolic Arithmetic Parser",
    "ALG": "A4 SymPy Algebra Solver",
    "LOG": "A6 Forward-Chaining Logic Engine",
    "WP":  "A2 LLM Chain-of-Thought (Phi-4-mini Q6_K)",
}


# ── Display helpers ───────────────────────────────────────────────

def print_banner():
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}   Neurosymbolic Adaptive Router - Live Demo{RESET}")
    print(f"{DIM}   Raspberry Pi 4B (8GB) | Phi-4-mini (3.8B, Q6_K) | V5 Hybrid{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")

def print_divider():
    print(f"{DIM}{'─'*65}{RESET}")

def print_step(step_num, text, color=WHITE):
    print(f"  {DIM}[{step_num}]{RESET} {color}{text}{RESET}")

def print_result_box(category, solver, answer, latency_ms, llm_used, steps=""):
    fg, bg = CAT_COLORS[category]
    print()
    print_divider()
    print(f"  {BOLD}Category:{RESET}  {bg}{BOLD} {category} {RESET}  {DIM}({SOLVER_NAMES[category]}){RESET}")
    print(f"  {BOLD}LLM Used:{RESET}  {'Yes' if llm_used else f'{GREEN}No - Zero LLM tokens{RESET}'}")
    print(f"  {BOLD}Latency:{RESET}   {GREEN}{latency_ms:.1f} ms{RESET}")
    print()
    if steps:
        print(f"  {DIM}Reasoning:{RESET}")
        for line in steps.split('\n'):
            print(f"    {DIM}{line}{RESET}")
        print()
    print(f"  {BOLD}Answer:  {fg}{BOLD}>>> {answer} <<<{RESET}")
    print_divider()
    print()


# ── Main loop ─────────────────────────────────────────────────────

PRESETS = {
    "1": ("What is 347 * 28 + 15?", "AR"),
    "2": ("Calculate 1024 / 16 - 12", "AR"),
    "3": ("Solve 3*x + 7 = 22 for x", "ALG"),
    "4": ("Solve 5*y - 3 = 2*y + 12 for y", "ALG"),
    "5": (
        "Anne is kind. Anne is quiet. Bob is blue. Dave is white. "
        "Gary is blue. Gary is kind. Gary is young. "
        "If someone is blue and not kind then they are quiet. "
        "If someone is quiet and not kind then they are smart.\n\n"
        "Question: Bob is smart.\nAnswer with only Yes or No.\nAnswer:",
        "LOG"
    ),
    "6": (
        "Harry is cold. Harry is quiet. Harry is young. "
        "Charlie is furry. Charlie is green. Charlie is kind. "
        "If someone is quiet then they are white. "
        "If someone is cold and white then they are furry.\n\n"
        "Question: Harry is furry.\nAnswer with only Yes or No.\nAnswer:",
        "LOG"
    ),
    "7": (
        "A bus travels at 60 miles per hour for 5 hours, "
        "then at 40 miles per hour for 3 hours. "
        "What is the total distance traveled?",
        "WP"
    ),
    "8": (
        "Lisa has 3 times as many books as Mark. Mark has 12 books. "
        "How many books do they have in total?",
        "WP"
    ),
}


def run_prompt(question: str, hint_category: str = None):
    # Step 1: Classify
    print_step(1, "Classifying prompt...", CYAN)
    t_class = time.time()
    category = hint_category or classify(question)
    class_ms = (time.time() - t_class) * 1000
    fg, bg = CAT_COLORS[category]
    print_step("", f"Classified as {bg}{BOLD} {category} {RESET} in {class_ms:.1f}ms")

    # Step 2: Route
    print_step(2, f"Routing to {BOLD}{SOLVER_NAMES[category]}{RESET}...", CYAN)

    # Step 3: Solve
    print_step(3, "Solving...", CYAN)

    if category == "AR":
        t0 = time.time()
        expr = extract_arithmetic(question)
        if expr:
            result = safe_eval_arithmetic(expr)
            latency = (time.time() - t0) * 1000
            if result is not None:
                if abs(result - round(result)) < 1e-9:
                    answer = str(int(round(result)))
                else:
                    answer = str(result)
                steps = f"Expression: {expr}\nComputed: {expr} = {answer}"
                print_result_box("AR", "A5", answer, latency, False, steps)
                return
        print(f"  {RED}Could not parse arithmetic expression{RESET}")

    elif category == "ALG":
        t0 = time.time()
        result = solve_algebra(question)
        latency = (time.time() - t0) * 1000
        if result:
            answer, steps = result
            print_result_box("ALG", "A4", answer, latency, False, steps)
            return
        print(f"  {RED}Could not solve algebra problem{RESET}")

    elif category == "LOG":
        t0 = time.time()
        result = solve_logic_problem(question)
        if result:
            answer, trace, latency = result
            print_result_box("LOG", "A6", answer, latency, False, trace)
            return
        print(f"  {RED}Could not parse logic problem{RESET}")

    elif category == "WP":
        print(f"  {DIM}Sending to Phi-4-mini (512 tokens)...{RESET}")
        print(f"  {DIM}(This takes ~2 minutes on Pi 4B){RESET}")
        result = solve_word_problem(question, token_budget=512)
        if result:
            answer, raw, latency = result
            print_result_box("WP", "A2", answer, latency, True, raw)
            return
        print(f"  {RED}LLM server not available. Start with:{RESET}")
        print(f"  {DIM}./llama-server -m <model.gguf> --port 8080{RESET}")


def main():
    print_banner()
    print(f"  {BOLD}Preset prompts:{RESET}")
    print(f"    {BLUE}[1]{RESET} AR: What is 347 * 28 + 15?")
    print(f"    {BLUE}[2]{RESET} AR: Calculate 1024 / 16 - 12")
    print(f"    {GREEN}[3]{RESET} ALG: Solve 3x + 7 = 22 for x")
    print(f"    {GREEN}[4]{RESET} ALG: Solve 5y - 3 = 2y + 12 for y")
    print(f"    {YELLOW}[5]{RESET} LOG: Bob is smart? (forward chaining)")
    print(f"    {YELLOW}[6]{RESET} LOG: Harry is furry? (forward chaining)")
    print(f"    {MAGENTA}[7]{RESET} WP: Bus distance problem (needs LLM)")
    print(f"    {MAGENTA}[8]{RESET} WP: Lisa's books problem (needs LLM)")
    print(f"    {DIM}Or type any math problem. Type 'q' to quit.{RESET}")
    print()

    while True:
        try:
            user_input = input(f"{BOLD}{CYAN}Enter prompt > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('q', 'quit', 'exit'):
            print(f"{DIM}Goodbye!{RESET}")
            break

        if user_input in PRESETS:
            question, hint = PRESETS[user_input]
            print(f"\n  {DIM}Prompt: {question[:80]}{'...' if len(question) > 80 else ''}{RESET}")
            run_prompt(question, hint)
        else:
            run_prompt(user_input)


if __name__ == "__main__":
    main()
