#!/usr/bin/env python3
"""
Hybrid V5 Router — A6 Symbolic Logic + V3.1 base

Changes from V3.1:
  - LOG: A6 (symbolic forward-chaining solver) primary, A1 fallback
  - AR/ALG/WP: unchanged from V3.1

A6 solves RuleTaker-style logic problems via:
  1. Pattern-based NL parsing to structured facts/rules
  2. Forward-chaining inference to fixed point
  3. Closed-world query evaluation

If A6 can't parse the prompt or returns Unknown, falls back to A1.
"""

import time
from typing import Dict, Any

from router_v3 import RouterV3
from a6_logic_engine import solve_logic


class RouterV5(RouterV3):
    """V5 Router: V3.1 base + A6 symbolic logic for LOG."""

    ROUTER_VERSION = "v5.0"
    TIMEOUT_POLICY_VERSION = "v5.0_a6_logic"

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any],
                 wp_token_budget: int = 30):
        super().__init__(config, routing_decisions)
        self.wp_token_budget = wp_token_budget
        # Scale WP timeout proportionally with token budget
        # Default: 20s for 30 tokens. Scale linearly.
        if wp_token_budget != 30:
            wp_timeout = int(20 * wp_token_budget / 30)
            self.CATEGORY_TIMEOUTS = dict(self.CATEGORY_TIMEOUTS)
            self.CATEGORY_TIMEOUTS[("WP", "A2")] = wp_timeout
            self.TIMEOUT_POLICY_VERSION = f"v5.0_a6_logic_wp{wp_token_budget}"
            print(f"[V5] WP A2 timeout scaled: {wp_timeout}s for {wp_token_budget} tokens")

    def _execute_action(self, action: str, category: str, prompt_text: str,
                        grammar_enabled: bool, ground_truth: str,
                        previous_raw: str = "") -> Dict[str, Any]:
        """Execute action — adds A6 and WP token budget override."""
        if action == 'A6':
            return self._execute_a6_logic(prompt_text, category)
        # Override A2 token budget for WP if configured differently
        if action == 'A2' and category == 'WP' and self.wp_token_budget != 30:
            return self._execute_llm_action(
                prompt_text, max_tokens=self.wp_token_budget,
                category=category, grammar_enabled=grammar_enabled
            )
        return super()._execute_action(
            action, category, prompt_text, grammar_enabled, ground_truth, previous_raw
        )

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
