#!/usr/bin/env python3
"""
Hybrid V6 Router — A7 Symbolic WP Solver + LLM Verification + V5 base

Architecture (neurosymbolic):
  - WP: A7 (symbolic solver) computes answer, LLM verifies (logged only)
  - LOG: A6 (forward-chain logic)
  - AR/ALG: unchanged (A5/A4)

The LLM verification is LOGGED ONLY — it never overrides A7's answer.
This gives us analysis data ("LLM agreed with symbolic on X%") without
risking a weak verifier vetoing correct symbolic answers.

Fallback chains (if symbolic fails to parse):
  AR  -> A5 -> A1 -> A2
  ALG -> A4 -> A1 -> A2
  LOG -> A6 -> A1
  WP  -> A7 -> A2 -> A3
"""

import re
import time
import requests
from typing import Dict, Any

from router_v5 import RouterV5
from a7_wp_solver import solve_wp


def _derive_base_url(server_url: str) -> str:
    """Derive base URL from llama.cpp server URL."""
    # server_url is like http://localhost:8080/completion
    # We need http://localhost:8080
    if '/completion' in server_url:
        return server_url.replace('/completion', '')
    return server_url.rstrip('/')


class RouterV6(RouterV5):
    """V6 Router: V5 + A7 symbolic WP solver + logged-only LLM verification."""

    ROUTER_VERSION = "v6.0"
    TIMEOUT_POLICY_VERSION = "v6.0_a7_wp_verified"

    def _execute_action(self, action: str, category: str, prompt_text: str,
                        grammar_enabled: bool, ground_truth: str,
                        previous_raw: str = "") -> Dict[str, Any]:
        """Execute action — adds A7 for WP."""
        if action == 'A7':
            return self._execute_a7_wp(prompt_text, category)
        return super()._execute_action(
            action, category, prompt_text, grammar_enabled, ground_truth, previous_raw
        )

    def _execute_a7_wp(self, prompt_text: str, category: str) -> Dict[str, Any]:
        """A7: Symbolic word problem solver + logged-only LLM verification."""
        # Extract the raw question from Phi chat template
        raw_question = self._extract_user_question(prompt_text)

        result = solve_wp(raw_question)

        answer_parsed = ""
        parse_success = False

        if result['parse_success'] and result['answer']:
            answer_parsed = result['answer']
            parse_success = True

        # Run logged-only LLM verification (does NOT change the answer)
        llm_verify = self._llm_verify_wp(raw_question, answer_parsed) if parse_success else {
            'llm_agrees': None,
            'llm_verify_raw': '',
            'llm_verify_latency_ms': 0,
            'llm_verify_error': 'skipped_no_symbolic_answer',
        }

        return {
            'answer_raw': f"[A7: {result['answer']}] strategy={result['strategy']} steps={result['n_steps']}",
            'answer_parsed': answer_parsed,
            'parse_success': parse_success,
            'timeout': False,
            'latency_ms': result['latency_ms'] + llm_verify['llm_verify_latency_ms'],
            'symbolic_failed': not parse_success,
            'symbolic_parse_success': result['parse_success'],
            'sympy_solve_success': False,
            'final_source': 'wp_symbolic' if parse_success else 'none',
            'action_timeout_sec_used': 0,
            'timeout_reason': 'none',
            # A7-specific fields
            'a7_parse_success': result['parse_success'],
            'a7_strategy': result['strategy'],
            'a7_n_steps': result['n_steps'],
            'a7_steps': '|'.join(result['steps'][:5]),  # First 5 steps for CSV
            # LLM verification fields (logged only, does not affect answer)
            'llm_agrees': llm_verify['llm_agrees'],
            'llm_verify_raw': llm_verify['llm_verify_raw'],
            'llm_verify_latency_ms': llm_verify['llm_verify_latency_ms'],
            'llm_verify_error': llm_verify.get('llm_verify_error', ''),
        }

    def _llm_verify_wp(self, question: str, symbolic_answer: str) -> Dict[str, Any]:
        """
        Ask the LLM: "The answer to this problem is [X]. Is that correct? Yes/No"

        This is LOGGED ONLY — it never changes the final answer.
        Gives us analysis data on LLM-symbolic agreement.
        """
        verify_prompt = (
            f"{question}\n\n"
            f"The answer is {symbolic_answer}. Is that correct? "
            f"Reply with only Yes or No."
        )
        system_msg = "You are a math verification assistant. Reply with only Yes or No."

        # Use chat API with 6-token budget (just need Yes/No)
        base_url = _derive_base_url(self.config['server_url'])
        chat_url = f"{base_url}/v1/chat/completions"

        request = {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": verify_prompt},
            ],
            "max_tokens": 6,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "stop": ["\n"],
        }

        t0 = time.time()
        content = ""
        error = ""

        try:
            r = requests.post(
                chat_url, json=request,
                timeout=(10.0, 45.0),  # 45s timeout (same as A1)
            )
            j = r.json()
            choices = j.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
        except requests.exceptions.Timeout:
            error = "timeout"
        except Exception as e:
            error = str(e)[:100]

        latency_ms = (time.time() - t0) * 1000

        # Parse Yes/No from response
        llm_agrees = None
        content_lower = content.strip().lower()
        if content_lower.startswith('yes'):
            llm_agrees = True
        elif content_lower.startswith('no'):
            llm_agrees = False

        return {
            'llm_agrees': llm_agrees,
            'llm_verify_raw': content.strip()[:100],
            'llm_verify_latency_ms': latency_ms,
            'llm_verify_error': error,
        }

    def _determine_reasoning_mode(self, route_log, final_result):
        """Override to recognize A7 WP symbolic mode."""
        final_action = route_log[-1]['action']
        if final_action == 'A7':
            return 'wp_symbolic'
        return super()._determine_reasoning_mode(route_log, final_result)

    def _get_fallback_action(self, current_action: str, category: str,
                             reason: str, escalation_level: int):
        """
        V6 fallback chains:
        - AR: A5 -> A1 -> A2
        - ALG: A4 -> A1 -> A2
        - WP: A7 -> A2 -> A3 (symbolic WP first, LLM fallback)
        - LOG: A6 -> A1
        """
        if escalation_level == 1:
            if current_action == 'A7' and category == 'WP':
                return 'A2'  # A7 failed -> try LLM
            return super()._get_fallback_action(
                current_action, category, reason, escalation_level
            )

        if escalation_level == 2:
            if category == 'WP':
                return 'A3'  # A2 failed -> repair
            return super()._get_fallback_action(
                current_action, category, reason, escalation_level
            )

        return super()._get_fallback_action(
            current_action, category, reason, escalation_level
        )
