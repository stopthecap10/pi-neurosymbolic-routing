#!/usr/bin/env python3
"""
Hybrid V1 Router - Deterministic Rule-Based Routing
Routes prompts to appropriate actions based on category
"""

import re
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import ast
import operator as op

# Safe arithmetic evaluator for A5 (no eval!)
SAFE_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

def safe_eval_expr(node):
    """Safely evaluate arithmetic expression AST node"""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric constant: {type(node.value)}")
        return node.value
    elif isinstance(node, ast.Num):  # Python <3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
        # Cap exponents to prevent DoS (e.g. 9**9**9**9)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 10000:
            raise ValueError(f"Exponent too large: {right}")
        return SAFE_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_expr(node.operand)
        return SAFE_OPS[type(node.op)](operand)
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """
    Safely evaluate arithmetic expression without using eval()
    Returns None if expression is invalid or unsafe
    """
    try:
        # Parse expression into AST
        tree = ast.parse(expr, mode='eval')
        # Evaluate using safe operations only
        result = safe_eval_expr(tree.body)
        return float(result)
    except Exception:
        return None

# System messages for chat API
SYSTEM_MSG_NUMERIC = "You are a math assistant. Return only the final numeric answer, nothing else."
SYSTEM_MSG_YESNO = "You are a logic assistant. Return only Yes or No, nothing else."

# Regex patterns
INT_RE = re.compile(r"[-+]?\d+")
FLOAT_RE = re.compile(r"[-+]?\d+\.?\d*")

# P3_numeric_context parser (matches run_v1_baseline_matrix.py)
NUM_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?")
CUE_RE = re.compile(
    r'(?:answer\s*(?:is|:)|final\s+(?:numeric\s+)?answer\s*(?:is|:)?'
    r'|answer\b.{0,20}\bis'
    r'|(?:^|\s)[a-z]\s*=)\s*'
    r'([-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)',
    re.IGNORECASE
)
TRAILING_EQ_RE = re.compile(r'^(.+?)\s*=\s*$')
CLEAN_EXPR_RE = re.compile(r'^[\d\s+\-*/().]+$')
HAS_LETTERS_RE = re.compile(r'[a-zA-Z]')


def _safe_eval_expr(expr: str) -> float:
    """Safely evaluate a simple math expression (no eval())."""
    import ast
    import operator
    if len(expr) > 50:
        raise ValueError("Expression too long")
    if not CLEAN_EXPR_RE.match(expr):
        raise ValueError("Non-math characters")
    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError:
        raise ValueError("Invalid syntax")
    _ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        elif isinstance(node, ast.BinOp):
            op_func = _ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported op: {type(node.op).__name__}")
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero")
            return op_func(left, right)
        elif isinstance(node, ast.UnaryOp):
            op_func = _ops.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary op")
            return op_func(_eval_node(node.operand))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        else:
            raise ValueError(f"Unsupported node: {type(node).__name__}")
    return _eval_node(tree)


def _normalize_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return str(value)


def _parse_single_token(raw_token: str) -> str:
    raw_token = raw_token.lstrip('+')
    try:
        if '/' in raw_token:
            parts = raw_token.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                if den == 0:
                    return ""
                return _normalize_value(num / den)
            return ""
        return _normalize_value(float(raw_token))
    except (ValueError, ZeroDivisionError):
        return ""


def parse_numeric_robust(text: str) -> str:
    """P3_numeric_context: Context-aware numeric extraction.
    Priority: direct -> cue phrase -> expression rescue -> single number -> E8."""
    if not text:
        return ""
    cleaned = text.strip().replace(",", "")
    tokens = NUM_TOKEN_RE.findall(cleaned)
    if not tokens:
        return ""
    # Rule 1: Direct numeric output
    direct = re.sub(r'[.\s]+$', '', cleaned)
    if NUM_TOKEN_RE.fullmatch(direct):
        return _parse_single_token(direct)
    # Rule 2: Cue phrase extraction
    cue_matches = CUE_RE.findall(cleaned)
    if cue_matches:
        result = _parse_single_token(cue_matches[-1])
        if result:
            return result
    var_eq_matches = re.findall(
        r'(?:^|\s)([a-zA-Z])\s*=\s*([-+]?\d+(?:\.\d+)?(?:/[-+]?\d+(?:\.\d+)?)?)',
        cleaned
    )
    if var_eq_matches:
        all_eq_parts = re.split(r'\s*=\s*', cleaned)
        last_part = all_eq_parts[-1].strip()
        last_token_match = NUM_TOKEN_RE.search(last_part)
        if last_token_match:
            result = _parse_single_token(last_token_match.group())
            if result:
                return result
        result = _parse_single_token(var_eq_matches[-1][1])
        if result:
            return result
    # Rule 2b: Result after equals (non-variable)
    if '=' in cleaned and not var_eq_matches:
        all_eq_parts = re.split(r'\s*=\s*', cleaned)
        last_part = all_eq_parts[-1].strip()
        if last_part:
            last_token_match = NUM_TOKEN_RE.search(last_part)
            if last_token_match and not HAS_LETTERS_RE.search(last_part):
                result = _parse_single_token(last_token_match.group())
                if result:
                    return result
    # Rule 3: Expression rescue
    eq_match = TRAILING_EQ_RE.match(cleaned)
    if eq_match:
        expr = eq_match.group(1).strip()
        if not HAS_LETTERS_RE.search(expr) and CLEAN_EXPR_RE.match(expr):
            try:
                value = _safe_eval_expr(expr)
                return _normalize_value(value)
            except (ValueError, ZeroDivisionError):
                pass
        return ""
    # Rule 4: Single standalone number (only if minimal text)
    if len(tokens) == 1:
        has_text = HAS_LETTERS_RE.search(cleaned)
        if not has_text:
            return _parse_single_token(tokens[0])
        word_count = len(cleaned.split())
        if word_count <= 5:
            return _parse_single_token(tokens[0])
        return ""
    # Rule 5: Ambiguity -> E8
    has_text = HAS_LETTERS_RE.search(cleaned)
    if has_text:
        return ""
    has_operators = bool(re.search(r'\d\s*[+\-*/]\s*\d', cleaned))
    if len(tokens) > 2 and has_operators:
        return ""
    return _parse_single_token(tokens[-1])

def _derive_base_url(server_url):
    """Derive base URL from server_url (strips /completion path)."""
    if '/completion' in server_url:
        return server_url.rsplit('/completion', 1)[0]
    return server_url.rstrip('/')


def _extract_chat_parts(prompt_text, category):
    """Extract system_msg and user_question from Phi template for chat API."""
    if "<|user|>" in prompt_text:
        user_start = prompt_text.find("<|user|>") + len("<|user|>")
        user_end = prompt_text.find("<|end|>", user_start)
        user_question = prompt_text[user_start:user_end] if user_end > user_start else prompt_text
    else:
        user_question = prompt_text
    system_msg = SYSTEM_MSG_YESNO if category == "LOG" else SYSTEM_MSG_NUMERIC
    return system_msg, user_question


class RouterV1:
    """Deterministic rule-based router for Hybrid V1"""

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any]):
        """
        Initialize router

        Args:
            config: Runtime configuration (server URL, timeouts, etc.)
            routing_decisions: Routing map from v1_decisions_for_hybrid.md
        """
        self.config = config
        self.decisions = routing_decisions
        self.max_escalations = routing_decisions.get('max_escalations', 1)
        self.api_mode = config.get('api_mode', 'chat')

    def route(self, prompt_id: str, category: str, prompt_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Route a single prompt through the appropriate action sequence

        Returns:
            Dictionary with:
                - answer_final: Final parsed answer
                - route_sequence: List of actions attempted
                - final_source: Which action produced the answer
                - correct: Whether answer matches ground_truth
                - error_code: E0-E8 error code
                - total_latency_ms: Total inference time
                - escalations_count: Number of fallback attempts
                - decision_reason: Why this route was chosen
        """
        route_log = []
        escalations = 0
        total_latency = 0

        # Get primary action based on category
        primary_action = self.decisions['category_routes'][category]['action']
        grammar_enabled = self.decisions['category_routes'][category]['grammar_enabled']

        route_log.append({
            'action': primary_action,
            'reason': f'Primary route for {category}',
            'escalation': 0
        })

        # Try primary action
        result = self._execute_action(
            action=primary_action,
            category=category,
            prompt_text=prompt_text,
            grammar_enabled=grammar_enabled,
            ground_truth=ground_truth
        )

        total_latency += result['latency_ms']
        route_log[-1]['result'] = result

        # Check if we need fallback
        needs_fallback = False
        fallback_reason = None

        if result['timeout']:
            needs_fallback = True
            fallback_reason = 'timeout'
        elif not result['parse_success']:
            needs_fallback = True
            fallback_reason = 'parse_failure'
        elif result.get('symbolic_failed', False):
            needs_fallback = True
            fallback_reason = 'symbolic_solver_failure'

        # Attempt fallback if needed and allowed
        if needs_fallback and escalations < self.max_escalations:
            escalations += 1
            fallback_action = self._get_fallback_action(
                primary_action, category, fallback_reason
            )

            if fallback_action:
                route_log.append({
                    'action': fallback_action,
                    'reason': f'Fallback from {primary_action} due to {fallback_reason}',
                    'escalation': escalations
                })

                fallback_result = self._execute_action(
                    action=fallback_action,
                    category=category,
                    prompt_text=prompt_text,
                    grammar_enabled=False,  # Try without grammar on fallback
                    ground_truth=ground_truth
                )

                total_latency += fallback_result['latency_ms']
                route_log[-1]['result'] = fallback_result

                # Use fallback result if it's better
                if fallback_result['parse_success'] and not fallback_result['timeout']:
                    result = fallback_result

        # Build final response
        answer_final = result.get('answer_parsed', '')
        correct = (answer_final == ground_truth)
        error_code = self._determine_error_code(
            category, answer_final, ground_truth,
            result['timeout'], result['parse_success']
        )

        return {
            'answer_final': answer_final,
            'answer_raw': result.get('answer_raw', ''),
            'route_sequence': [r['action'] for r in route_log],
            'route_attempt_sequence': ' → '.join([
                f"{r['action']}({r['reason']})" for r in route_log
            ]),
            'final_source': route_log[-1]['action'],
            'correct': correct,
            'error_code': error_code,
            'total_latency_ms': total_latency,
            'escalations_count': escalations,
            'decision_reason': route_log[0]['reason'],
            'timeout_flag': result['timeout'],
            'parse_success': result['parse_success'],
            'route_log': route_log
        }

    def _execute_action(self, action: str, category: str, prompt_text: str,
                       grammar_enabled: bool, ground_truth: str) -> Dict[str, Any]:
        """Execute a single action and return results"""

        if action == 'A1':
            return self._execute_llm_action(
                prompt_text, max_tokens=12, category=category,
                grammar_enabled=grammar_enabled
            )
        elif action == 'A2':
            return self._execute_llm_action(
                prompt_text, max_tokens=30, category=category,
                grammar_enabled=grammar_enabled
            )
        elif action == 'A4':
            return self._execute_symbolic_extract_solve(prompt_text, category)
        elif action == 'A5':
            return self._execute_symbolic_direct(prompt_text, category)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _execute_llm_action(self, prompt_text: str, max_tokens: int,
                           category: str, grammar_enabled: bool) -> Dict[str, Any]:
        """Execute LLM inference (A1 or A2). Supports chat and completion API modes."""

        is_log = (category == 'LOG')

        # Adjust token budget for LOG
        if is_log:
            n_pred = 6
        else:
            n_pred = max_tokens

        # Get grammar file
        grammar_file = None
        if grammar_enabled:
            if is_log:
                grammar_file = self.config.get('grammar_yesno')
            else:
                grammar_file = self.config.get('grammar_num')

        # Determine effective API mode (grammar forces completion mode)
        effective_mode = self.api_mode
        if grammar_enabled and grammar_file and self.api_mode == "chat":
            effective_mode = "completion"

        # Execute
        t0 = time.time()
        content = ""
        timed_out = False

        if effective_mode == "chat":
            # Chat API mode: /v1/chat/completions
            base_url = _derive_base_url(self.config['server_url'])
            chat_url = f"{base_url}/v1/chat/completions"
            system_msg, user_question = _extract_chat_parts(prompt_text, category)

            request = {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_question},
                ],
                "max_tokens": n_pred,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
                "stop": ["\n"],
            }

            try:
                r = requests.post(
                    chat_url, json=request,
                    timeout=(10.0, float(self.config['timeout_sec'])),
                )
                j = r.json()
                choices = j.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "") or ""
            except requests.exceptions.Timeout:
                timed_out = True
            except Exception as e:
                print(f"ERROR in LLM call: {e}")
                content = ""
        else:
            # Completion mode: /completion
            request = {
                "prompt": prompt_text,
                "n_predict": n_pred,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "seed": 42,
                "stop": ["\n", "<|end|>", "<|endoftext|>"],
            }

            if grammar_file:
                try:
                    with open(grammar_file, 'r') as f:
                        request["grammar"] = f.read()
                except Exception as e:
                    print(f"WARNING: Grammar file {grammar_file} failed to load: {e}")

            try:
                r = requests.post(
                    self.config['server_url'],
                    json=request,
                    timeout=(10.0, float(self.config['timeout_sec'])),
                )
                j = r.json()
                content = j.get("content", "") or ""
            except requests.exceptions.Timeout:
                timed_out = True
            except Exception as e:
                print(f"ERROR in LLM call: {e}")
                content = ""

        latency_ms = (time.time() - t0) * 1000

        if latency_ms >= self.config['timeout_sec'] * 1000:
            timed_out = True

        # Parse answer (P2_numeric_robust for numeric, word-boundary for yesno)
        if is_log:
            answer_parsed = self._extract_yesno(content)
        else:
            answer_parsed = parse_numeric_robust(content)

        parse_success = (answer_parsed != "")

        return {
            'answer_raw': content,
            'answer_parsed': answer_parsed,
            'parse_success': parse_success,
            'timeout': timed_out,
            'latency_ms': latency_ms,
            'symbolic_failed': False
        }

    def _execute_symbolic_direct(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """
        A5: Direct symbolic computation for simple arithmetic
        Extract expression and compute directly (safe, no eval)
        """
        t0 = time.time()

        # Extract arithmetic expression from prompt
        # Look for patterns like "594 divided by 3", "847 + 253", etc.
        expr = self._extract_arithmetic_expression(prompt_text)

        answer_parsed = ""
        symbolic_failed = False

        if expr:
            # Evaluate safely
            result = safe_eval_arithmetic(expr)
            if result is not None:
                # Round to integer if close
                if abs(result - round(result)) < 1e-9:
                    answer_parsed = str(int(round(result)))
                else:
                    answer_parsed = str(result)
            else:
                symbolic_failed = True
        else:
            symbolic_failed = True

        latency_ms = (time.time() - t0) * 1000

        return {
            'answer_raw': f"[A5 symbolic: {expr} = {answer_parsed}]" if expr else "[A5 failed]",
            'answer_parsed': answer_parsed,
            'parse_success': (answer_parsed != ""),
            'timeout': False,
            'latency_ms': latency_ms,
            'symbolic_failed': symbolic_failed
        }

    def _execute_symbolic_extract_solve(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """
        A4: LLM extracts equation, SymPy solves
        For algebra problems
        """
        t0 = time.time()

        # First, use LLM to extract the equation
        # This is a placeholder - would need proper implementation
        # For now, mark as not implemented
        symbolic_failed = True
        answer_parsed = ""

        latency_ms = (time.time() - t0) * 1000

        return {
            'answer_raw': "[A4 not yet implemented]",
            'answer_parsed': answer_parsed,
            'parse_success': False,
            'timeout': False,
            'latency_ms': latency_ms,
            'symbolic_failed': symbolic_failed
        }

    def _extract_arithmetic_expression(self, prompt_text: str) -> Optional[str]:
        """Extract arithmetic expression from prompt text.

        Strategy:
        1. For each non-instruction line, try stripping known word prefixes and
           evaluating the remaining expression directly.  Handles complex multi-step
           expressions like ((-32)/6 - -2)/(104/156).
        2. Fall back to simple 2-number pattern matching for word-form prompts.
        """
        # Common instruction prefixes to strip before the expression
        PREFIXES = [
            r'^what\s+is\s+the\s+value\s+of\s+',
            r'^what\s+is\s+',
            r'^calculate\s+',
            r'^compute\s+',
            r'^evaluate\s+',
            r'^find\s+',
            r'^simplify\s+',
        ]
        # Lines that are part of the instruction wrapper, not the expression
        SKIP_PATTERNS = re.compile(
            r'^(answer|answer with|answer with only|answer:|return only)\b', re.IGNORECASE
        )

        # Step 1: Try each line of the prompt
        for raw_line in prompt_text.splitlines():
            line = raw_line.strip().rstrip('?.').strip()
            if not line or SKIP_PATTERNS.match(line):
                continue

            # Try evaluating the line as-is (bare expression)
            if safe_eval_arithmetic(line) is not None:
                return line

            # Try stripping known word prefixes
            for prefix_pat in PREFIXES:
                stripped = re.sub(prefix_pat, '', line, flags=re.IGNORECASE).strip().rstrip('?.').strip()
                if stripped and stripped != line:
                    if safe_eval_arithmetic(stripped) is not None:
                        return stripped

        # Step 2: Fallback — word-form / 2-number patterns
        text = prompt_text.lower()
        nums = FLOAT_RE.findall(prompt_text)

        if ('divided by' in text) or ('divide' in text):
            if len(nums) >= 2:
                return f"{nums[0]} / {nums[1]}"

        if 'plus' in text:
            if len(nums) >= 2:
                return f"{nums[0]} + {nums[1]}"

        if ('times' in text) or ('multiplied by' in text):
            if len(nums) >= 2:
                return f"{nums[0]} * {nums[1]}"

        if 'minus' in text:
            if len(nums) >= 2:
                return f"{nums[0]} - {nums[1]}"

        return None

    def _get_fallback_action(self, primary_action: str, category: str, reason: str) -> Optional[str]:
        """Determine fallback action based on failure reason"""

        # Timeout fallback: escalate to A2
        if reason == 'timeout':
            if primary_action == 'A1':
                return 'A2'
            else:
                return None  # A2 already timed out, no further fallback

        # Parse failure: try without grammar (escalate to A2)
        if reason == 'parse_failure':
            if primary_action in ['A1', 'A5', 'A4']:
                return 'A2'
            else:
                return None

        # Symbolic solver failure: fallback to neural
        if reason == 'symbolic_solver_failure':
            return 'A1'  # Try fast neural as backup

        return None

    def _extract_last_int(self, text: str) -> str:
        """Extract last integer from text, normalized (no leading zeros)"""
        if not text:
            return ""
        cleaned = text.strip().replace(",", "")
        nums = INT_RE.findall(cleaned)
        if not nums:
            return ""
        result = nums[-1].lstrip('+')
        # Normalize through int() to strip leading zeros: "007" -> "7"
        try:
            return str(int(result))
        except ValueError:
            return result

    def _extract_yesno(self, text: str) -> str:
        """Extract Yes/No from text using word-boundary matching"""
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

    def _determine_error_code(self, category: str, pred: str, expected: str,
                             timed_out: bool, parse_success: bool) -> str:
        """Determine error code"""
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
