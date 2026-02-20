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
        return node.value
    elif isinstance(node, ast.Num):  # Python <3.8
        return node.n
    elif isinstance(node, ast.BinOp):
        left = safe_eval_expr(node.left)
        right = safe_eval_expr(node.right)
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
    except:
        return None

# Regex patterns
INT_RE = re.compile(r"[-+]?\d+")
FLOAT_RE = re.compile(r"[-+]?\d+\.?\d*")

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
        """Execute LLM inference (A1 or A2)"""

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

        # Build request
        request = {
            "prompt": prompt_text,
            "n_predict": n_pred,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }

        if grammar_file:
            try:
                with open(grammar_file, 'r') as f:
                    request["grammar"] = f.read()
            except:
                pass  # Grammar file not found, continue without

        # Execute
        t0 = time.time()
        content = ""
        timed_out = False

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

        # Parse answer
        if is_log:
            answer_parsed = self._extract_yesno(content)
        else:
            answer_parsed = self._extract_last_int(content)

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
        """Extract arithmetic expression from prompt text"""
        text = prompt_text.lower()

        # Extract all numbers first
        nums = FLOAT_RE.findall(prompt_text)

        # Pattern: "X / Y" or "X divided by Y"
        if ('/' in text) or ('divided by' in text) or ('divide' in text):
            if len(nums) >= 2:
                return f"{nums[0]} / {nums[1]}"

        # Pattern: "X + Y" or "X plus Y"
        if ('+' in text) or ('plus' in text):
            if len(nums) >= 2:
                return f"{nums[0]} + {nums[1]}"

        # Pattern: "X * Y" or "X times Y" or "X multiplied by Y"
        if ('*' in text) or ('times' in text) or ('multiplied by' in text):
            if len(nums) >= 2:
                return f"{nums[0]} * {nums[1]}"

        # Pattern: "X - Y" or "X minus Y"
        # Look for subtraction operator (not a negative sign at start of number)
        if (' - ' in prompt_text) or ('minus' in text):
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
        """Extract last integer from text"""
        if not text:
            return ""
        cleaned = text.strip().replace(",", "")
        nums = INT_RE.findall(cleaned)
        if not nums:
            return ""
        result = nums[-1].lstrip('+')
        return result

    def _extract_yesno(self, text: str) -> str:
        """Extract Yes/No from text"""
        if not text:
            return ""
        t = text.lower().strip()
        yes_pos = t.rfind("yes")
        no_pos = t.rfind("no")
        if yes_pos == -1 and no_pos == -1:
            return ""
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
