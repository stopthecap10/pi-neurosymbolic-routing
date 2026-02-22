#!/usr/bin/env python3
"""Quick test: A4 deterministic solver on all 10 ALG prompts."""
from src.router_v2 import RouterV2

config = {'server_url': 'http://localhost:8080/completion', 'timeout_sec': 20, 'api_mode': 'chat'}
decisions = {'category_routes': {'ALG': {'action': 'A4', 'grammar_enabled': False}}, 'max_escalations': 2}
router = RouterV2(config, decisions)

tests = [
    ('Solve 27 = -4*d - 5*z - 9, 4*z + 12 = d for d.', '-4'),
    ('Solve c + c - 5*p = -25, 5 = 2*c + p for c.', '0'),
    ('Solve 0 = -37*o + 41*o - 8 for o.', '2'),
    ('Solve 15 = o + 2*t + 4, -18 = -3*o - t for o.', '5'),
    ('Let 4*t**3 - 648*t**2 + 34992*t - 629856 = 0. What is t?', '54'),
    ('Solve 4*b - 4 = 4*p, 0 = -2*b - 2*p - 2*p + 14 for b.', '3'),
    ('Solve 5*m - 7 - 18 = 0 for m.', '5'),
    ('Solve -554*z = -562*z - 24 for z.', '-3'),
    ('Solve -2*m - 4*n = -4*m - 20, -4*n + 20 = -m for m.', '0'),
    ('Solve -9*f + 13*f = -28 for f.', '-7'),
]

passed = 0
for i, (q, expected) in enumerate(tests, 1):
    prompt = f'<|system|>You are a math assistant. Return only the final numeric answer, nothing else.<|end|><|user|>{q}<|end|><|assistant|>'
    result = router._execute_symbolic_extract_solve(prompt, 'ALG')
    answer = result['answer_parsed']
    ok = answer == expected
    if ok:
        passed += 1
    status = 'OK' if ok else 'FAIL'
    print(f"  ALG_{i:02d}: {status}  got={answer!r}  expected={expected!r}  sympy={result['sympy_solve_success']}  lat={result['latency_ms']:.1f}ms")
    if not ok:
        print(f"         raw={result['answer_raw']}")

print(f"\nA4 Result: {passed}/10 correct")
