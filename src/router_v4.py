#!/usr/bin/env python3
"""
Hybrid V4 Router - Calibrated Selective Compute

Wraps V3.1 routing logic with a calibrated P(correct) controller.
- AR→A5, ALG→A4: deterministic, no calibration needed
- WP: A2 → check p_correct → escalate to A3R if p < tau
- LOG: A1 → compute p_correct → log score (no fallback in V4-min)

The calibrator is a logistic regression loaded from JSON.
Inference is pure Python (sigmoid dot-product), no sklearn needed.
"""

import json
import time
from typing import Dict, Any, Optional

from router_v3 import RouterV3
from router_features import (
    extract_router_features,
    feature_vector_from_dict,
    predict_p_correct,
    FEATURE_NAMES_V1,
    FEATURE_VERSION,
)


class RouterV4(RouterV3):
    """V4 Router: V3.1 base + calibrated selective-compute controller."""

    ROUTER_VERSION = "v4.0"

    def __init__(self, config: Dict[str, Any], routing_decisions: Dict[str, Any],
                 calibrator_path: Optional[str] = None, tau: float = 0.70):
        super().__init__(config, routing_decisions)
        self.tau = tau
        self.calibrator = None
        self.calibrator_version = "none"
        self.calibrator_loaded = False

        if calibrator_path:
            self._load_calibrator(calibrator_path)

    def _load_calibrator(self, path: str):
        """Load calibrator JSON (coefficients + intercept)."""
        try:
            with open(path, 'r') as f:
                cal = json.load(f)

            self.calibrator = {
                "coefficients": cal["coefficients"],
                "intercept": cal["intercept"],
                "feature_names": cal["feature_names"],
            }
            self.calibrator_version = cal.get("version", "unknown")
            self.calibrator_loaded = True
            print(f"[V4] Calibrator loaded: {path} (version={self.calibrator_version})")
        except Exception as e:
            print(f"[V4] WARNING: Failed to load calibrator from {path}: {e}")
            self.calibrator = None
            self.calibrator_loaded = False

    def _compute_p_correct(self, category: str, attempt_idx: int, action: str,
                           result: Dict[str, Any],
                           prev_parse_fail: bool = False,
                           prev_timeout: bool = False) -> tuple:
        """
        Compute P(correct) for a routing attempt.
        Returns (p_correct, feature_extraction_ms, calibrator_ms)
        """
        if not self.calibrator_loaded:
            return -1.0, 0.0, 0.0

        t0 = time.time()
        features = extract_router_features(
            category=category,
            attempt_idx=attempt_idx,
            action=action,
            parse_success=result.get('parse_success', False),
            timeout_flag=result.get('timeout', False),
            answer_raw=result.get('answer_raw', ''),
            symbolic_parse_success=result.get('symbolic_parse_success', False),
            prev_attempt_failed_parse=prev_parse_fail,
            prev_attempt_timeout=prev_timeout,
        )
        vec = feature_vector_from_dict(features)
        feat_ms = (time.time() - t0) * 1000

        t1 = time.time()
        p = predict_p_correct(
            vec,
            self.calibrator["coefficients"],
            self.calibrator["intercept"],
        )
        cal_ms = (time.time() - t1) * 1000

        return p, feat_ms, cal_ms

    def route(self, prompt_id: str, category: str, prompt_text: str,
              ground_truth: str) -> Dict[str, Any]:
        """
        Route with calibrated selective compute.

        AR/ALG: deterministic (bypass calibrator)
        WP: A2 → p_correct check → A3R if p < tau
        LOG: A1 → p_correct logged (no fallback in V4-min)
        """
        t_route_start = time.time()

        # AR and ALG: deterministic, bypass calibrator entirely
        if category in ("AR", "ALG"):
            result = super().route(prompt_id, category, prompt_text, ground_truth)
            routing_logic_ms = (time.time() - t_route_start) * 1000
            result.update({
                'tau': self.tau,
                'p_correct': -1.0,  # N/A for deterministic
                'p_correct_pre_escalation': -1.0,
                'p_correct_post_escalation': -1.0,
                'calibrator_version': self.calibrator_version,
                'calibrator_loaded': self.calibrator_loaded,
                'decision_reason': f'deterministic_{category}_{result["route_sequence"][0]}',
                'feature_extraction_ms': 0.0,
                'calibrator_ms': 0.0,
                'routing_logic_ms': routing_logic_ms,
            })
            return result

        # WP and LOG: calibrated routing
        return self._route_calibrated(
            prompt_id, category, prompt_text, ground_truth, t_route_start
        )

    def _route_calibrated(self, prompt_id: str, category: str, prompt_text: str,
                          ground_truth: str, t_route_start: float) -> Dict[str, Any]:
        """Calibrated routing for WP and LOG."""

        route_log = []
        total_latency = 0
        total_feat_ms = 0.0
        total_cal_ms = 0.0
        p_correct_pre = -1.0
        p_correct_post = -1.0
        decision_reason = ""

        # Get primary action
        primary_action = self.decisions['category_routes'][category]['action']
        grammar_enabled = self.decisions['category_routes'][category]['grammar_enabled']

        # Execute primary action
        result = self._execute_action(
            action=primary_action,
            category=category,
            prompt_text=prompt_text,
            grammar_enabled=grammar_enabled,
            ground_truth=ground_truth,
        )
        total_latency += result['latency_ms']

        route_log.append({
            'action': primary_action,
            'reason': f'Primary route for {category}',
            'escalation': 0,
            'result': result,
        })

        # Compute p_correct for primary
        p_correct, feat_ms, cal_ms = self._compute_p_correct(
            category=category,
            attempt_idx=1,
            action=primary_action,
            result=result,
        )
        total_feat_ms += feat_ms
        total_cal_ms += cal_ms
        p_correct_pre = p_correct

        # Decision: accept or escalate?
        escalated = False
        previous_raw = result.get('answer_raw', '')

        if category == "WP":
            # WP: escalate to A3R if p_correct < tau and result is salvageable
            if result['timeout'] or not result['parse_success']:
                if self.calibrator_loaded and p_correct >= 0 and p_correct >= self.tau:
                    decision_reason = f"accept_p>=tau (p={p_correct:.3f}>=tau={self.tau})"
                elif result['timeout'] and not previous_raw.strip():
                    decision_reason = f"stop_empty_timeout (p={p_correct:.3f})"
                else:
                    # Escalate to A3R
                    escalated = True
                    decision_reason = f"escalate_p<tau (p={p_correct:.3f}<tau={self.tau})"

                    fallback_result = self._execute_action(
                        action='A3',
                        category=category,
                        prompt_text=prompt_text,
                        grammar_enabled=False,
                        ground_truth=ground_truth,
                        previous_raw=previous_raw,
                    )
                    total_latency += fallback_result['latency_ms']

                    route_log.append({
                        'action': 'A3',
                        'reason': f'Escalate from {primary_action} (p={p_correct:.3f}<tau={self.tau})',
                        'escalation': 1,
                        'result': fallback_result,
                    })

                    # Compute p_correct for A3R
                    p_post, feat_ms2, cal_ms2 = self._compute_p_correct(
                        category=category,
                        attempt_idx=2,
                        action='A3',
                        result=fallback_result,
                        prev_parse_fail=not result['parse_success'],
                        prev_timeout=result['timeout'],
                    )
                    total_feat_ms += feat_ms2
                    total_cal_ms += cal_ms2
                    p_correct_post = p_post

                    # Use A3R result if it parsed
                    if fallback_result['parse_success'] and not fallback_result['timeout']:
                        result = fallback_result
            else:
                decision_reason = f"accept_parse_ok (p={p_correct:.3f})"

        elif category == "LOG":
            # LOG: A1 only, log p_correct but no fallback
            if result['parse_success'] and not result['timeout']:
                decision_reason = f"accept_LOG_A1 (p={p_correct:.3f})"
            elif result['timeout']:
                decision_reason = f"stop_LOG_timeout (p={p_correct:.3f})"
            else:
                decision_reason = f"stop_LOG_parse_fail (p={p_correct:.3f})"

        # Build final response
        answer_final = result.get('answer_parsed', '')
        correct = (answer_final == ground_truth)
        error_code = self._determine_error_code(
            category, answer_final, ground_truth,
            result['timeout'], result['parse_success']
        )
        reasoning_mode = self._determine_reasoning_mode(route_log, result)
        routing_logic_ms = (time.time() - t_route_start) * 1000

        return {
            'answer_final': answer_final,
            'answer_raw': result.get('answer_raw', ''),
            'route_sequence': [r['action'] for r in route_log],
            'route_attempt_sequence': ' -> '.join(
                f"{r['action']}({r['reason']})" for r in route_log
            ),
            'final_source': result.get('final_source', route_log[-1]['action']),
            'correct': correct,
            'error_code': error_code,
            'total_latency_ms': total_latency,
            'escalations_count': 1 if escalated else 0,
            'decision_reason': decision_reason,
            'timeout_flag': result['timeout'],
            'parse_success': result['parse_success'],
            'symbolic_parse_success': result.get('symbolic_parse_success', False),
            'sympy_solve_success': result.get('sympy_solve_success', False),
            'reasoning_mode': reasoning_mode,
            'repair_attempted': escalated,
            'repair_success': escalated and result.get('parse_success', False) and not result.get('timeout', False),
            'repair_trigger_reason': 'calibrated_escalation' if escalated else '',
            'previous_raw_len': len(previous_raw) if escalated else 0,
            'previous_action_id': primary_action if escalated else '',
            'step_latencies': [(r['action'], r['result']['latency_ms']) for r in route_log],
            'timeout_policy_version': self.TIMEOUT_POLICY_VERSION,
            'action_timeout_sec_used': result.get('action_timeout_sec_used', ''),
            'timeout_reason': result.get('timeout_reason', ''),
            'route_log': route_log,
            # V4-specific fields
            'tau': self.tau,
            'p_correct': p_correct_post if escalated and p_correct_post >= 0 else p_correct_pre,
            'p_correct_pre_escalation': p_correct_pre,
            'p_correct_post_escalation': p_correct_post,
            'calibrator_version': self.calibrator_version,
            'calibrator_loaded': self.calibrator_loaded,
            'feature_extraction_ms': total_feat_ms,
            'calibrator_ms': total_cal_ms,
            'routing_logic_ms': routing_logic_ms,
        }
