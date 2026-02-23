#!/usr/bin/env python3
"""
Hybrid V5 Router — A6 Symbolic Logic + CoT WP + V3.1 base

Changes from V3.1:
  - LOG: A6 (symbolic forward-chaining solver) primary, A1 fallback
  - WP: Chain-of-thought mode with configurable token budget
  - AR/ALG: unchanged from V3.1

A6 solves RuleTaker-style logic problems via:
  1. Pattern-based NL parsing to structured facts/rules
  2. Forward-chaining inference to fixed point
  3. Closed-world query evaluation

WP CoT mode:
  - System prompt encourages step-by-step reasoning
  - Higher token budget (150) allows model to finish reasoning
  - Parser extracts "Answer: X" or last number from CoT output
"""

import re
import time
from typing import Dict, Any

from router_v3 import RouterV3, parse_numeric_robust, _derive_base_url
from a6_logic_engine import solve_logic

# CoT system prompt for WP
SYSTEM_MSG_WP_COT = (
    "Solve this math problem. Put your final answer in \\boxed{}. Answer with the fewest amount of words."
)


class RouterV5(RouterV3):
    """V5 Router: V3.1 base + A6 symbolic logic for LOG."""

    ROUTER_VERSION = "v5.0"
    TIMEOUT_POLICY_VERSION = "v5.0_a6_logic"

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any],
                 wp_token_budget: int = 30, wp_cot: bool = False):
        super().__init__(config, routing_decisions)
        self.wp_token_budget = wp_token_budget
        self.wp_cot = wp_cot
        # Scale WP timeout proportionally with token budget
        # Default: 20s for 30 tokens. Scale linearly.
        if wp_token_budget != 30:
            wp_timeout = int(20 * wp_token_budget / 30)
            self.CATEGORY_TIMEOUTS = dict(self.CATEGORY_TIMEOUTS)
            self.CATEGORY_TIMEOUTS[("WP", "A2")] = wp_timeout
            self.TIMEOUT_POLICY_VERSION = f"v5.0_a6_logic_wp{wp_token_budget}"
            if wp_cot:
                self.TIMEOUT_POLICY_VERSION += "_cot"
            print(f"[V5] WP A2 timeout scaled: {wp_timeout}s for {wp_token_budget} tokens")
        if wp_cot:
            print(f"[V5] WP chain-of-thought mode ENABLED (budget={wp_token_budget})")

    def _execute_action(self, action: str, category: str, prompt_text: str,
                        grammar_enabled: bool, ground_truth: str,
                        previous_raw: str = "") -> Dict[str, Any]:
        """Execute action — adds A6, WP CoT, and WP token budget override."""
        if action == 'A6':
            return self._execute_a6_logic(prompt_text, category)
        # WP with chain-of-thought
        if action == 'A2' and category == 'WP' and self.wp_cot:
            return self._execute_wp_cot(prompt_text)
        # Override A2 token budget for WP if configured differently
        if action == 'A2' and category == 'WP' and self.wp_token_budget != 30:
            return self._execute_llm_action(
                prompt_text, max_tokens=self.wp_token_budget,
                category=category, grammar_enabled=grammar_enabled
            )
        return super()._execute_action(
            action, category, prompt_text, grammar_enabled, ground_truth, previous_raw
        )

    def _execute_wp_cot(self, prompt_text: str) -> Dict[str, Any]:
        """
        WP chain-of-thought: let the model reason step-by-step,
        then extract the final answer from its output.
        """
        import requests

        raw_question = self._extract_user_question(prompt_text)

        timeout_sec = self.CATEGORY_TIMEOUTS.get(
            ("WP", "A2"),
            self.ACTION_TIMEOUTS.get("A2", 60)
        )

        t0 = time.time()
        content = ""
        timed_out = False

        base_url = _derive_base_url(self.config['server_url'])
        chat_url = f"{base_url}/v1/chat/completions"

        request = {
            "messages": [
                {"role": "system", "content": SYSTEM_MSG_WP_COT},
                {"role": "user", "content": raw_question},
            ],
            "max_tokens": self.wp_token_budget,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
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
            print(f"ERROR in WP CoT call: {e}")
            content = ""

        latency_ms = (time.time() - t0) * 1000

        if latency_ms >= timeout_sec * 1000:
            timed_out = True

        # Parse answer from CoT output
        answer_parsed = self._extract_cot_answer(content)
        parse_success = (answer_parsed != "")

        cat_key = ("WP", "A2")
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
            'final_source': 'llm_cot',
            'action_timeout_sec_used': timeout_sec,
            'timeout_reason': timeout_reason,
        }

    def _extract_cot_answer(self, text: str) -> str:
        """
        Extract final numeric answer from chain-of-thought output.

        Priority:
        0. \\boxed{X} (Qwen-Math native format)
        1. "Answer: X" or "answer is X" pattern
        2. Last number after "=" in the text
        3. Fall back to parse_numeric_robust
        """
        if not text:
            return ""

        # 0. Look for \boxed{X} (Qwen-Math native output)
        boxed_match = re.search(r'\\boxed\{([-+]?\d+(?:\.\d+)?)\}', text)
        if boxed_match:
            val = boxed_match.group(1)
            try:
                f = float(val)
                if abs(f - round(f)) < 1e-9:
                    return str(int(round(f)))
                return str(f)
            except ValueError:
                pass

        # 1. Look for explicit "Answer: X" pattern
        answer_match = re.search(
            r'(?:answer|final answer)\s*(?:is|:)\s*([-+]?\d+(?:\.\d+)?)',
            text, re.IGNORECASE
        )
        if answer_match:
            val = answer_match.group(1)
            try:
                f = float(val)
                if abs(f - round(f)) < 1e-9:
                    return str(int(round(f)))
                return str(f)
            except ValueError:
                pass

        # 2. Look for last "= X" pattern (common in step-by-step math)
        eq_matches = re.findall(
            r'=\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)',
            text
        )
        if eq_matches:
            val = eq_matches[-1].replace(",", "")
            try:
                f = float(val)
                if abs(f - round(f)) < 1e-9:
                    return str(int(round(f)))
                return str(f)
            except ValueError:
                pass

        # 3. Fallback to existing parser
        return parse_numeric_robust(text)

    def _execute_a6_logic(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """A6: Symbolic forward-chaining logic solver."""
        # Extract the raw question from Phi chat template
        raw_question = self._extract_user_question(prompt_text)

        result = solve_logic(raw_question)

        answer_parsed = ""
        parse_success = False

        if result["parse_success"] and result["answer"] in ("Yes", "No"):
            answer_parsed = result["answer"]
            parse_success = True

        return {
            'answer_raw': f"[A6: {result['answer']}] {result['trace'][:200]}",
            'answer_parsed': answer_parsed,
            'parse_success': parse_success,
            'timeout': False,
            'latency_ms': result['latency_ms'],
            'symbolic_failed': not parse_success,
            'symbolic_parse_success': result['parse_success'],
            'sympy_solve_success': False,
            'final_source': 'logic_symbolic' if parse_success else 'none',
            'action_timeout_sec_used': 0,
            'timeout_reason': 'none',
            # A6-specific fields
            'a6_parse_success': result['parse_success'],
            'a6_n_facts': result['n_facts'],
            'a6_n_rules': result['n_rules'],
            'a6_n_derived': result['n_derived'],
            'a6_rule_fired': result['rule_fired'],
            'a6_pattern': result['pattern'],
        }

    def _determine_reasoning_mode(self, route_log, final_result):
        """Override to recognize A6 logic symbolic mode."""
        final_action = route_log[-1]['action']
        if final_action == 'A6':
            return 'logic_symbolic'
        return super()._determine_reasoning_mode(route_log, final_result)

    def _get_fallback_action(self, current_action: str, category: str,
                             reason: str, escalation_level: int):
        """
        V5 fallback chains:
        - AR: A5 -> A1 -> A2
        - ALG: A4 -> A1 -> A2
        - WP: A2 -> A3 (repair)
        - LOG: A6 -> A1 (symbolic logic first, LLM fallback)
        """
        if escalation_level == 1:
            if current_action == 'A6' and category == 'LOG':
                return 'A1'  # A6 failed -> try LLM
            # Everything else same as V3.1
            return super()._get_fallback_action(
                current_action, category, reason, escalation_level
            )

        return super()._get_fallback_action(
            current_action, category, reason, escalation_level
        )
