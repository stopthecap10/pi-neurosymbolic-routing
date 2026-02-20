# V2 Deterministic Router - CORRECTED SPECIFICATION

**Status:** APPROVED - Ready for implementation
**Date:** 2026-02-11
**Critical Fixes:** All 8 issues from review addressed

---

## Route Definitions (Predefined Actions)

```python
# Action definitions (frozen before T3)
ACTIONS = {
    "A1_short": {
        "method": "llm",
        "tokens": 12,
        "description": "Standard LLM decode"
    },
    "A2_extended": {
        "method": "llm",
        "tokens": 30,
        "description": "Extended reasoning decode (CoT-style)"
    },
    "A3_verify": {
        "method": "llm_then_sympy_verify",
        "tokens": 12,
        "description": "LLM answer + SymPy verification"
    },
    "A4_extract_solve": {
        "method": "llm_extract_then_sympy",
        "tokens": 20,
        "description": "LLM extract equation + SymPy solve"
    },
    "A5_symbolic": {
        "method": "direct_compute",
        "tokens": 0,
        "description": "Pure symbolic computation (no LLM)"
    }
}
```

## Routing Policy (To Be Determined by Testing)

**AR:** A5_symbolic (confirmed - direct compute is optimal)
**LOG:** A1_short (confirmed - already 100%)
**ALG:** A1 vs A2 vs A4 - **NEEDS TESTING**
**WP:** A1 vs A2 - **NEEDS TESTING**

## Safe Arithmetic Parser (Fix #1)

```python
import sympy as sp

def safe_arithmetic_eval(expr_str):
    """Safe arithmetic evaluation - no raw eval()"""
    try:
        result = sp.sympify(expr_str, evaluate=True)
        return float(result)
    except:
        return None
```

## Extended Reasoning (Fix #2)

A2_extended is a FORMAL ACTION with 30-token budget.
Not a hidden boost - it's declared and logged.

## V2 Logging Schema

```python
trial = {
    # V1 fields (all kept)
    "system": "v2_router",

    # V2 routing fields
    "route_chosen": "A4_extract_solve",
    "route_attempt_sequence": "A4,A2",
    "escalations_count": 1,
    "final_answer_source": "sympy",
    "decision_reason": "category=ALG,initial=A4",

    # Action configuration
    "action_budget_profile": "A4:20tok,A2:30tok",
    "reasoning_mode": "extended",
    "prompt_template_version": "v1.0",
    "max_tokens_used": 20,

    # Symbolic execution
    "symbolic_parse_success": 1,
    "llm_response": "2x + 5 = 17",
    "sympy_equation": "Eq(2*x + 5, 17)",
    "sympy_solution": "6",

    # Grammar tracking
    "grammar_version": "G1-relaxed" if grammar else "none",
}
```

## Fallback Policy

```python
FALLBACK_POLICY = {
    "A5_symbolic": {
        "parse_fail": "A1_short",
        "max_retries": 1
    },
    "A4_extract_solve": {
        "extraction_fail": "A1_short",
        "sympy_fail": "A2_extended",
        "timeout": "A2_extended",
        "max_retries": 2
    },
    "A2_extended": {
        "parse_fail": None,
        "timeout": None,
        "max_retries": 0
    },
    "A1_short": {
        "parse_fail": "A2_extended",
        "timeout": "A2_extended",
        "max_retries": 1
    }
}
```

## Success Criteria (Neutral, Measurable)

1. V2 runs complete with full logging coverage
2. V2 shows measurable improvement on at least one weak category (ALG or WP)
3. AR route demonstrates cost reduction while maintaining accuracy
4. Route traces are reproducible
5. No regressions on strong categories (AR, LOG)

## Implementation Checklist

- [ ] Build safe arithmetic parser (no eval)
- [ ] Define ACTIONS dict with token budgets
- [ ] **TEST A1 vs A2 on ALG prompts**
- [ ] **TEST A1 vs A2 on WP prompts**
- [ ] Implement A5_symbolic for AR
- [ ] Implement A4_extract_solve for ALG
- [ ] Implement chosen action for WP (A1 or A2 based on testing)
- [ ] Implement A1_short for LOG with strict parsing
- [ ] Add all V2 logging fields
- [ ] Implement fallback policy
- [ ] Test on smoke dataset
- [ ] Test on tier1_mini
- [ ] Generate V1+V2 comparison summary

## Time Budget

**Realistic:** 4-6 hours total
- A1 vs A2 testing: 1 hour
- V2 implementation: 2 hours
- Testing & debugging: 1-2 hours
- Summary generation: 30 min

---

**NEXT STEP:** Run A1 vs A2 comparison on ALG and WP categories to make data-driven routing decisions.
