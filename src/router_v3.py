#!/usr/bin/env python3
"""
Hybrid V3 Router - WP/LOG Repair Layer
AR→A5, ALG→A4 (frozen from V2)
WP→A2 with A3 repair fallback
LOG→A1 with A3 strict retry fallback
"""

import re
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import ast
import operator as op

# Import SymPy for A4
try:
    from sympy import symbols, Eq, solve
    from sympy.parsing.sympy_parser import parse_expr
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("WARNING: SymPy not available. A4 will be disabled.")

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
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 10000:
            raise ValueError(f"Exponent too large: {right}")
        return SAFE_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = safe_eval_expr(node.operand)
        return SAFE_OPS[type(node.op)](operand)
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")

def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """Safely evaluate arithmetic expression without using eval()"""
    try:
        tree = ast.parse(expr, mode='eval')
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


class RouterV3:
    """V3 Router: WP/LOG repair layer on top of V2"""

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any]):
        self.config = config
        self.decisions = routing_decisions
        self.max_escalations = routing_decisions.get('max_escalations', 2)
        self.api_mode = config.get('api_mode', 'chat')

    def route(self, prompt_id: str, category: str, prompt_text: str, ground_truth: str) -> Dict[str, Any]:
        """Route a single prompt through the appropriate action sequence."""
        route_log = []
        escalations = 0
        total_latency = 0
        repair_attempted = False
        repair_success = False
        repair_trigger_reason = ""
        previous_raw_len = 0
        previous_action_id = ""

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

        # Determine if we need fallback
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

        # Track previous raw output for A3 repair
        previous_raw = result.get('answer_raw', '')

        # Attempt fallback chain (up to max_escalations)
        while needs_fallback and escalations < self.max_escalations:
            escalations += 1
            fallback_action = self._get_fallback_action(
                current_action=route_log[-1]['action'],
                category=category,
                reason=fallback_reason,
                escalation_level=escalations
            )

            if not fallback_action:
                break

            route_log.append({
                'action': fallback_action,
                'reason': f'Fallback from {route_log[-1]["action"]} due to {fallback_reason}',
                'escalation': escalations
            })

            # A3 gets previous raw output for repair prompt
            fallback_result = self._execute_action(
                action=fallback_action,
                category=category,
                prompt_text=prompt_text,
                grammar_enabled=False,
                ground_truth=ground_truth,
                previous_raw=previous_raw
            )

            total_latency += fallback_result['latency_ms']
            route_log[-1]['result'] = fallback_result

            if fallback_action == 'A3':
                repair_attempted = True
                repair_trigger_reason = fallback_reason
                previous_raw_len = len(previous_raw)
                previous_action_id = route_log[-2]['action'] if len(route_log) >= 2 else ""

            if fallback_result['parse_success'] and not fallback_result['timeout']:
                result = fallback_result
                needs_fallback = False
                if fallback_action == 'A3':
                    repair_success = True
            else:
                previous_raw = fallback_result.get('answer_raw', '')
                if fallback_result['timeout']:
                    fallback_reason = 'timeout'
                elif not fallback_result['parse_success']:
                    fallback_reason = 'parse_failure'
                else:
                    needs_fallback = False

        # Build final response
        answer_final = result.get('answer_parsed', '')
        correct = (answer_final == ground_truth)
        error_code = self._determine_error_code(
            category, answer_final, ground_truth,
            result['timeout'], result['parse_success']
        )

        reasoning_mode = self._determine_reasoning_mode(route_log, result)

        return {
            'answer_final': answer_final,
            'answer_raw': result.get('answer_raw', ''),
            'route_sequence': [r['action'] for r in route_log],
            'route_attempt_sequence': ' -> '.join([
                f"{r['action']}({r['reason']})" for r in route_log
            ]),
            'final_source': result.get('final_source', route_log[-1]['action']),
            'correct': correct,
            'error_code': error_code,
            'total_latency_ms': total_latency,
            'escalations_count': escalations,
            'decision_reason': route_log[0]['reason'],
            'timeout_flag': result['timeout'],
            'parse_success': result['parse_success'],
            'symbolic_parse_success': result.get('symbolic_parse_success', False),
            'sympy_solve_success': result.get('sympy_solve_success', False),
            'reasoning_mode': reasoning_mode,
            'repair_attempted': repair_attempted,
            'repair_success': repair_success,
            'repair_trigger_reason': repair_trigger_reason,
            'previous_raw_len': previous_raw_len,
            'previous_action_id': previous_action_id,
            'step_latencies': [(r['action'], r['result']['latency_ms']) for r in route_log if 'result' in r],
            'timeout_policy_version': self.TIMEOUT_POLICY_VERSION,
            'action_timeout_sec_used': result.get('action_timeout_sec_used', ''),
            'timeout_reason': result.get('timeout_reason', ''),
            'route_log': route_log
        }

    def _execute_action(self, action: str, category: str, prompt_text: str,
                       grammar_enabled: bool, ground_truth: str,
                       previous_raw: str = "") -> Dict[str, Any]:
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
        elif action == 'A3':
            return self._execute_repair_action(
                prompt_text, category, previous_raw
            )
        elif action == 'A4':
            return self._execute_symbolic_extract_solve(prompt_text, category)
        elif action == 'A5':
            return self._execute_symbolic_direct(prompt_text, category)
        else:
            raise ValueError(f"Unknown action: {action}")

    # Baseline-compatible default timeouts (unchanged from V1/V2)
    ACTION_TIMEOUTS = {"A1": 45, "A2": 60}

    # V3 hybrid policy: per-action cost caps (frozen before evaluation)
    # These are declared as part of the hybrid controller design, not ad-hoc changes.
    # Baselines use ACTION_TIMEOUTS only. V3 uses CATEGORY_TIMEOUTS where defined.
    TIMEOUT_POLICY_VERSION = "v3_cost_cap_1"
    CATEGORY_TIMEOUTS = {
        ("WP", "A2"): 20,   # WP primary: 30 tok should finish in ~15s on Pi
        ("WP", "A3"): 20,   # WP repair: 12 tok, just final answer
        ("LOG", "A1"): 15,  # LOG primary: 6 tok, one word
        ("LOG", "A3"): 15,  # LOG retry: 6 tok, one word
    }

    def _execute_llm_action(self, prompt_text: str, max_tokens: int,
                           category: str, grammar_enabled: bool) -> Dict[str, Any]:
        """Execute LLM inference (A1 or A2). Supports chat and completion API modes."""

        is_log = (category == 'LOG')

        if is_log:
            n_pred = 6
        else:
            n_pred = max_tokens

        action_name = "A1" if max_tokens <= 12 else "A2"
        # Use category-specific timeout if available, else default
        timeout_sec = self.CATEGORY_TIMEOUTS.get(
            (category, action_name),
            self.ACTION_TIMEOUTS.get(action_name, self.config['timeout_sec'])
        )

        grammar_file = None
        if grammar_enabled:
            if is_log:
                grammar_file = self.config.get('grammar_yesno')
            else:
                grammar_file = self.config.get('grammar_num')

        effective_mode = self.api_mode
        if grammar_enabled and grammar_file and self.api_mode == "chat":
            effective_mode = "completion"

        t0 = time.time()
        content = ""
        timed_out = False

        if effective_mode == "chat":
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
                    timeout=(10.0, float(timeout_sec)),
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
                    timeout=(10.0, float(timeout_sec)),
                )
                j = r.json()
                content = j.get("content", "") or ""
            except requests.exceptions.Timeout:
                timed_out = True
            except Exception as e:
                print(f"ERROR in LLM call: {e}")
                content = ""

        latency_ms = (time.time() - t0) * 1000

        if latency_ms >= timeout_sec * 1000:
            timed_out = True

        if is_log:
            answer_parsed = self._extract_yesno(content)
        else:
            answer_parsed = parse_numeric_robust(content)

        parse_success = (answer_parsed != "")

        # Determine timeout source for logging
        cat_key = (category, action_name)
        timeout_reason = "action_cap" if cat_key in self.CATEGORY_TIMEOUTS else "global"

        return {
            'answer_raw': content,
            'answer_parsed': answer_parsed,
            'parse_success': parse_success,
            'timeout': timed_out,
            'latency_ms': latency_ms,
            'symbolic_failed': False,
            'symbolic_parse_success': False,
            'sympy_solve_success': False,
            'final_source': 'llm',
            'action_timeout_sec_used': timeout_sec,
            'timeout_reason': timeout_reason,
        }

    def _execute_repair_action(self, prompt_text: str, category: str,
                               previous_raw: str) -> Dict[str, Any]:
        """
        A3: Repair/continuation action.
        For WP: sends continuation prompt with previous partial output as context.
        For LOG: sends strict retry prompt.
        """
        raw_question = self._extract_user_question(prompt_text)
        is_log = (category == 'LOG')

        if category == "WP":
            # Truncate previous output to avoid exceeding context
            prev_truncated = previous_raw[:200].strip()
            if prev_truncated:
                repair_question = (
                    f"{raw_question}\n\n"
                    f"Your work so far: {prev_truncated}\n\n"
                    f"What is the final numeric answer?"
                )
            else:
                repair_question = (
                    f"{raw_question}\n\n"
                    f"What is the final numeric answer?"
                )
            system_msg = SYSTEM_MSG_NUMERIC
            n_pred = 12
        elif category == "LOG":
            repair_question = (
                f"{raw_question}\n\n"
                f"Answer with exactly one word: Yes or No."
            )
            system_msg = SYSTEM_MSG_YESNO
            n_pred = 6
        else:
            # A3 not designed for AR/ALG (they have symbolic paths)
            return {
                'answer_raw': '[A3 not applicable]',
                'answer_parsed': '',
                'parse_success': False,
                'timeout': False,
                'latency_ms': 0,
                'symbolic_failed': False,
                'symbolic_parse_success': False,
                'sympy_solve_success': False,
                'final_source': 'none'
            }

        # Use category-specific timeout for A3 (cost-capped)
        timeout_sec = self.CATEGORY_TIMEOUTS.get(
            (category, "A3"),
            self.ACTION_TIMEOUTS.get("A1", 45)
        )

        t0 = time.time()
        content = ""
        timed_out = False

        # Always use chat API for repair prompts
        base_url = _derive_base_url(self.config['server_url'])
        chat_url = f"{base_url}/v1/chat/completions"

        request = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": repair_question},
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
                timeout=(10.0, float(timeout_sec)),
            )
            j = r.json()
            choices = j.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
        except requests.exceptions.Timeout:
            timed_out = True
        except Exception as e:
            print(f"ERROR in A3 repair call: {e}")
            content = ""

        latency_ms = (time.time() - t0) * 1000

        if latency_ms >= timeout_sec * 1000:
            timed_out = True

        # Parse
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
            'symbolic_failed': False,
            'symbolic_parse_success': False,
            'sympy_solve_success': False,
            'final_source': 'repair',
            'action_timeout_sec_used': timeout_sec,
            'timeout_reason': 'action_cap',
        }

    def _extract_user_question(self, prompt_text: str) -> str:
        """Extract the raw user question from Phi chat template."""
        if "<|user|>" in prompt_text and "<|end|>" in prompt_text:
            start = prompt_text.find("<|user|>") + len("<|user|>")
            end = prompt_text.find("<|end|>", start)
            if end > start:
                return prompt_text[start:end].strip()
        return prompt_text

    def _execute_symbolic_direct(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """A5: Direct symbolic computation for simple arithmetic"""
        t0 = time.time()

        raw_question = self._extract_user_question(prompt_text)
        expr = self._extract_arithmetic_expression(raw_question)

        answer_parsed = ""
        symbolic_failed = False
        symbolic_parse_success = False

        if expr:
            symbolic_parse_success = True
            result = safe_eval_arithmetic(expr)
            if result is not None:
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
            'symbolic_failed': symbolic_failed,
            'symbolic_parse_success': symbolic_parse_success,
            'sympy_solve_success': False,
            'final_source': 'symbolic'
        }

    def _execute_symbolic_extract_solve(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """A4: Deterministic equation extraction + SymPy solve for ALG."""
        t0 = time.time()

        if not SYMPY_AVAILABLE:
            return {
                'answer_raw': "[A4 disabled: SymPy not installed]",
                'answer_parsed': "",
                'parse_success': False,
                'timeout': False,
                'latency_ms': (time.time() - t0) * 1000,
                'symbolic_failed': True,
                'symbolic_parse_success': False,
                'sympy_solve_success': False,
                'final_source': 'none'
            }

        raw_question = self._extract_user_question(prompt_text)

        # Extract target variable
        target_var = None
        target_match = re.search(r'\bfor\s+([a-zA-Z])\b\s*[.?]?\s*$', raw_question)
        if target_match:
            target_var = target_match.group(1)
        else:
            what_match = re.search(r'\bwhat\s+is\s+([a-zA-Z])\b\s*[.?]?\s*$', raw_question, re.IGNORECASE)
            if what_match:
                target_var = what_match.group(1)

        if not target_var:
            return self._a4_fail(t0, "target_var_missing", raw_question)

        # Extract equations
        eq_text = raw_question
        eq_text = re.sub(r'^(Solve|Let)\s+', '', eq_text, flags=re.IGNORECASE)
        eq_text = re.sub(r'\s+for\s+[a-zA-Z]\s*[.?]?\s*$', '', eq_text)
        eq_text = re.sub(r'\s*What\s+is\s+[a-zA-Z]\s*[.?]?\s*$', '', eq_text, flags=re.IGNORECASE)
        eq_text = eq_text.strip().rstrip('.')

        if not eq_text or '=' not in eq_text:
            return self._a4_fail(t0, "eq_extract_fail", raw_question)

        eq_strings = [e.strip() for e in eq_text.split(',') if '=' in e]
        if not eq_strings:
            return self._a4_fail(t0, "eq_extract_fail", raw_question)

        try:
            answer_parsed, sympy_success, parse_success = self._solve_with_sympy(
                eq_strings, target_var
            )
            latency_ms = (time.time() - t0) * 1000

            return {
                'answer_raw': f"[A4: {eq_strings} solve {target_var} -> {answer_parsed}]",
                'answer_parsed': answer_parsed,
                'parse_success': parse_success,
                'timeout': False,
                'latency_ms': latency_ms,
                'symbolic_failed': not sympy_success,
                'symbolic_parse_success': parse_success,
                'sympy_solve_success': sympy_success,
                'final_source': 'sympy' if sympy_success else 'none'
            }

        except Exception as e:
            return self._a4_fail(t0, f"sympy_error: {str(e)[:40]}", raw_question)

    def _a4_fail(self, t0, reason, raw_question):
        """Helper for A4 failure returns."""
        return {
            'answer_raw': f"[A4 failed: {reason}] {raw_question[:60]}",
            'answer_parsed': "",
            'parse_success': False,
            'timeout': False,
            'latency_ms': (time.time() - t0) * 1000,
            'symbolic_failed': True,
            'symbolic_parse_success': False,
            'sympy_solve_success': False,
            'final_source': 'none'
        }

    def _solve_with_sympy(self, eq_strings: list, target_var: str) -> Tuple[str, bool, bool]:
        """Solve system of equations with SymPy for a given target variable."""
        try:
            all_vars_set = set()
            for eq_str in eq_strings:
                found = re.findall(r'\b([a-zA-Z])\b', eq_str)
                all_vars_set.update(found)

            sym_dict = {v: symbols(v) for v in all_vars_set}
            target_sym = sym_dict.get(target_var)
            if target_sym is None:
                return ("", False, False)

            equations = []
            for eq_str in eq_strings:
                if '=' not in eq_str:
                    continue
                parts = eq_str.split('=')
                if len(parts) != 2:
                    continue
                lhs = parts[0].strip()
                rhs = parts[1].strip()
                lhs_expr = parse_expr(lhs, local_dict=sym_dict)
                rhs_expr = parse_expr(rhs, local_dict=sym_dict)
                equations.append(Eq(lhs_expr, rhs_expr))

            if not equations:
                return ("", False, False)

            solutions = solve(equations, list(sym_dict.values()))

            answer_val = None
            if isinstance(solutions, dict):
                answer_val = solutions.get(target_sym)
            elif isinstance(solutions, list) and len(solutions) > 0:
                sol = solutions[0]
                if isinstance(sol, dict):
                    answer_val = sol.get(target_sym)
                elif isinstance(sol, tuple):
                    var_list = list(sym_dict.values())
                    if target_sym in var_list:
                        idx = var_list.index(target_sym)
                        if idx < len(sol):
                            answer_val = sol[idx]
                else:
                    answer_val = sol

            if answer_val is None:
                return ("", False, False)

            try:
                sol_float = float(answer_val)
                if abs(sol_float - round(sol_float)) < 1e-9:
                    answer = str(int(round(sol_float)))
                else:
                    answer = str(sol_float)
                return (answer, True, True)
            except (ValueError, TypeError, OverflowError):
                return (str(answer_val), True, True)

        except Exception as e:
            return ("", False, False)

    def _extract_arithmetic_expression(self, prompt_text: str) -> Optional[str]:
        """Extract arithmetic expression from prompt text."""
        PREFIXES = [
            r'^what\s+is\s+the\s+value\s+of\s+',
            r'^what\s+is\s+',
            r'^calculate\s+',
            r'^compute\s+',
            r'^evaluate\s+',
            r'^find\s+',
            r'^simplify\s+',
        ]
        SKIP_PATTERNS = re.compile(
            r'^(answer|answer with|answer with only|answer:|return only)\b', re.IGNORECASE
        )

        for raw_line in prompt_text.splitlines():
            line = raw_line.strip().rstrip('?.').strip()
            if not line or SKIP_PATTERNS.match(line):
                continue

            if safe_eval_arithmetic(line) is not None:
                return line

            for prefix_pat in PREFIXES:
                stripped = re.sub(prefix_pat, '', line, flags=re.IGNORECASE).strip().rstrip('?.').strip()
                if stripped and stripped != line:
                    if safe_eval_arithmetic(stripped) is not None:
                        return stripped

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

    def _get_fallback_action(self, current_action: str, category: str, reason: str, escalation_level: int) -> Optional[str]:
        """
        V3 fallback chains:
        - AR: A5 -> A1 -> A2
        - ALG: A4 -> A1 -> A2
        - WP: A2 -> A3 (repair)
        - LOG: A1 -> A3 (repair)
        """

        if escalation_level == 1:
            if current_action == 'A5':
                return 'A1'
            elif current_action == 'A4':
                return 'A1'
            elif current_action == 'A2' and category == 'WP':
                return 'A3'  # WP: A2 failed -> repair
            elif current_action == 'A1' and category == 'LOG':
                return 'A3'  # LOG: A1 failed -> strict retry
            elif current_action == 'A1':
                return 'A2'
            else:
                return None

        elif escalation_level == 2:
            if current_action == 'A1':
                return 'A2'
            else:
                return None

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

    def _determine_reasoning_mode(self, route_log: List[Dict], final_result: Dict) -> str:
        """Determine reasoning mode based on final action"""
        final_action = route_log[-1]['action']

        if final_action in ['A5', 'A4']:
            return 'symbolic'
        elif final_action == 'A1':
            return 'short'
        elif final_action == 'A2':
            return 'extended'
        elif final_action == 'A3':
            return 'repair'
        else:
            return 'unknown'
