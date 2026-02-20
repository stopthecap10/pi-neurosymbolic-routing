#!/usr/bin/env python3
"""
Hybrid V2 Router - Enhanced Routing with A4 Symbolic Solver
Adds A4 (LLM extract + SymPy solve) for ALG category
Enhanced fallback chains with max 2 escalations
"""

import re
import time
import requests
from typing import Dict, List, Any, Optional, Tuple
import ast
import operator as op

# Import SymPy for A4
try:
    from sympy import symbols, Eq, solve, sympify
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

class RouterV2:
    """Enhanced router for Hybrid V2 with A4 support"""

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any]):
        """
        Initialize router

        Args:
            config: Runtime configuration (server URL, timeouts, etc.)
            routing_decisions: Routing map
        """
        self.config = config
        self.decisions = routing_decisions
        self.max_escalations = routing_decisions.get('max_escalations', 2)

    def route(self, prompt_id: str, category: str, prompt_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Route a single prompt through the appropriate action sequence

        Returns:
            Dictionary with:
                - answer_final: Final parsed answer
                - route_sequence: List of actions attempted
                - final_source: Which action produced the answer (llm/sympy)
                - correct: Whether answer matches ground_truth
                - error_code: E0-E8 error code
                - total_latency_ms: Total inference time
                - escalations_count: Number of fallback attempts
                - decision_reason: Why this route was chosen
                - symbolic_parse_success: Whether symbolic extraction worked
                - sympy_solve_success: Whether SymPy solved the equation
                - reasoning_mode: short/extended/symbolic
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
                break  # No more fallbacks available

            route_log.append({
                'action': fallback_action,
                'reason': f'Fallback from {route_log[-1]["action"]} due to {fallback_reason}',
                'escalation': escalations
            })

            fallback_result = self._execute_action(
                action=fallback_action,
                category=category,
                prompt_text=prompt_text,
                grammar_enabled=False,  # Disable grammar on fallback
                ground_truth=ground_truth
            )

            total_latency += fallback_result['latency_ms']
            route_log[-1]['result'] = fallback_result

            # Use fallback result if it's better
            if fallback_result['parse_success'] and not fallback_result['timeout']:
                result = fallback_result
                needs_fallback = False  # Success, stop escalating
            else:
                # Check if we need another fallback
                if fallback_result['timeout']:
                    fallback_reason = 'timeout'
                elif not fallback_result['parse_success']:
                    fallback_reason = 'parse_failure'
                else:
                    needs_fallback = False  # Parsed something, stop

        # Build final response
        answer_final = result.get('answer_parsed', '')
        correct = (answer_final == ground_truth)
        error_code = self._determine_error_code(
            category, answer_final, ground_truth,
            result['timeout'], result['parse_success']
        )

        # Determine reasoning mode
        reasoning_mode = self._determine_reasoning_mode(route_log, result)

        return {
            'answer_final': answer_final,
            'answer_raw': result.get('answer_raw', ''),
            'route_sequence': [r['action'] for r in route_log],
            'route_attempt_sequence': ' → '.join([
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
            'symbolic_failed': False,
            'symbolic_parse_success': False,
            'sympy_solve_success': False,
            'final_source': 'llm'
        }

    def _execute_symbolic_direct(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """
        A5: Direct symbolic computation for simple arithmetic
        Extract expression and compute directly (safe, no eval)
        """
        t0 = time.time()

        # Extract arithmetic expression from prompt
        expr = self._extract_arithmetic_expression(prompt_text)

        answer_parsed = ""
        symbolic_failed = False
        symbolic_parse_success = False

        if expr:
            symbolic_parse_success = True
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
            'symbolic_failed': symbolic_failed,
            'symbolic_parse_success': symbolic_parse_success,
            'sympy_solve_success': False,
            'final_source': 'symbolic'
        }

    def _execute_symbolic_extract_solve(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """
        A4: LLM extracts equation, SymPy solves
        For algebra problems

        ROBUST VERSION: Fail fast with strict extraction
        """
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

        # Step 1: Extract BASE prompt (remove instruction suffix to avoid template collision)
        # prompt_text format: "{base_prompt}\n{instruction}\nAnswer:"
        # We need just the base prompt for clean extraction
        base_prompt = prompt_text.split('\nAnswer with only')[0].strip()

        # Build CLEAN extraction prompt (no collision with answer template)
        # STRICT: Only ask for equation, short generation
        extraction_prompt = f"{base_prompt}\nWrite only the equation:"

        # Call LLM with STRICT limits (8 tokens max for equation like "2*x-6=x")
        extraction_request = {
            "prompt": extraction_prompt,
            "n_predict": 12,  # Strict: just enough for equation
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        }

        extraction_t0 = time.time()
        extraction_content = ""
        extraction_timeout = False

        try:
            r = requests.post(
                self.config['server_url'],
                json=extraction_request,
                timeout=(10.0, 20.0),  # Shorter timeout for extraction only
            )
            j = r.json()
            extraction_content = j.get("content", "") or ""
        except requests.exceptions.Timeout:
            extraction_timeout = True
        except Exception as e:
            print(f"ERROR in A4 extraction: {e}")
            extraction_content = ""

        extraction_latency = (time.time() - extraction_t0) * 1000

        # Fail fast if extraction timed out or is empty
        if extraction_timeout or not extraction_content.strip():
            return {
                'answer_raw': f"[A4 extraction failed: timeout={extraction_timeout}]",
                'answer_parsed': "",
                'parse_success': False,
                'timeout': extraction_timeout,
                'latency_ms': extraction_latency,
                'symbolic_failed': True,
                'symbolic_parse_success': False,
                'sympy_solve_success': False,
                'final_source': 'none'
            }

        # Step 2: Parse equation and solve with SymPy
        equation_text = extraction_content.strip()

        # Quick validation: must contain '=' and variable 'x'
        if '=' not in equation_text or 'x' not in equation_text.lower():
            return {
                'answer_raw': f"[A4 parse failed: {equation_text}]",
                'answer_parsed': "",
                'parse_success': False,
                'timeout': False,
                'latency_ms': extraction_latency,
                'symbolic_failed': True,
                'symbolic_parse_success': False,
                'sympy_solve_success': False,
                'final_source': 'none'
            }

        try:
            # Try to solve with SymPy
            sympy_t0 = time.time()
            answer_parsed, sympy_success, parse_success = self._solve_with_sympy(equation_text)
            sympy_latency = (time.time() - sympy_t0) * 1000

            total_latency = extraction_latency + sympy_latency

            return {
                'answer_raw': f"[A4: {equation_text[:50]} → {answer_parsed}]",
                'answer_parsed': answer_parsed,
                'parse_success': parse_success,
                'timeout': False,
                'latency_ms': total_latency,
                'symbolic_failed': not sympy_success,
                'symbolic_parse_success': parse_success,
                'sympy_solve_success': sympy_success,
                'final_source': 'sympy' if sympy_success else 'none'
            }

        except Exception as e:
            total_latency = extraction_latency + ((time.time() - t0) * 1000 - extraction_latency)
            return {
                'answer_raw': f"[A4 SymPy error: {str(e)[:50]}]",
                'answer_parsed': "",
                'parse_success': False,
                'timeout': False,
                'latency_ms': total_latency,
                'symbolic_failed': True,
                'symbolic_parse_success': False,
                'sympy_solve_success': False,
                'final_source': 'none'
            }

    def _solve_with_sympy(self, equation_text: str) -> Tuple[str, bool, bool]:
        """
        Solve equation with SymPy
        Returns: (answer_str, sympy_solve_success, parse_success)
        """
        try:
            # Define common variables
            x = symbols('x')

            # Try different equation formats
            # Format 1: "x = expression" → solve for x
            if '=' in equation_text:
                parts = equation_text.split('=')
                if len(parts) == 2:
                    lhs = parts[0].strip()
                    rhs = parts[1].strip()

                    # Parse both sides
                    lhs_expr = parse_expr(lhs, local_dict={'x': x})
                    rhs_expr = parse_expr(rhs, local_dict={'x': x})

                    # Solve equation
                    equation = Eq(lhs_expr, rhs_expr)
                    solutions = solve(equation, x)

                    if solutions and len(solutions) > 0:
                        # Take first solution
                        sol = solutions[0]
                        # Convert to float if possible
                        try:
                            sol_float = float(sol)
                            # Round to int if close
                            if abs(sol_float - round(sol_float)) < 1e-9:
                                answer = str(int(round(sol_float)))
                            else:
                                answer = str(sol_float)
                            return (answer, True, True)
                        except:
                            return (str(sol), True, True)

            # If we get here, parsing failed
            return ("", False, False)

        except Exception as e:
            return ("", False, False)

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
            r'^(answer|answer with|answer with only|answer:)\b', re.IGNORECASE
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

    def _get_fallback_action(self, current_action: str, category: str, reason: str, escalation_level: int) -> Optional[str]:
        """
        Determine fallback action based on failure reason and escalation level

        V2 fallback chains:
        - AR: A5 → A1 → A2
        - ALG: A4 → A1 → A2
        - WP: A1 → A2
        - LOG: A1 (strict, no fallback unless timeout/parse)
        """

        # Escalation level 1
        if escalation_level == 1:
            if current_action == 'A5':
                return 'A1'  # Symbolic failed, try fast neural
            elif current_action == 'A4':
                return 'A1'  # SymPy failed, try fast neural
            elif current_action == 'A1':
                return 'A2'  # A1 failed, try extended neural
            else:
                return None  # A2 already failed, no more fallbacks

        # Escalation level 2
        elif escalation_level == 2:
            if current_action == 'A1':
                return 'A2'  # Final fallback to extended neural
            else:
                return None  # No more fallbacks

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

    def _determine_reasoning_mode(self, route_log: List[Dict], final_result: Dict) -> str:
        """Determine reasoning mode based on final action"""
        final_action = route_log[-1]['action']

        if final_action in ['A5', 'A4']:
            return 'symbolic'
        elif final_action == 'A1':
            return 'short'
        elif final_action == 'A2':
            return 'extended'
        else:
            return 'unknown'
