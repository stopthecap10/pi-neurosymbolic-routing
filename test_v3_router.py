#!/usr/bin/env python3
"""Quick test: V3 router A4/A5 frozen, A3 repair method."""
from src.router_v3 import RouterV3

config = {'server_url': 'http://localhost:8080/completion', 'timeout_sec': 20, 'api_mode': 'chat'}
decisions = {
    'category_routes': {
        'AR': {'action': 'A5', 'grammar_enabled': False},
        'ALG': {'action': 'A4', 'grammar_enabled': False},
        'WP': {'action': 'A2', 'grammar_enabled': False},
        'LOG': {'action': 'A1', 'grammar_enabled': False},
    },
    'max_escalations': 2,
}
router = RouterV3(config, decisions)

# Test A5 (AR)
prompt = '<|system|>You are a math assistant.<|end|><|user|>Calculate 4-(5-(3+2)).<|end|><|assistant|>'
r = router._execute_symbolic_direct(prompt, 'AR')
ans = r['answer_parsed']
assert ans == '4', f"A5 failed: got {ans}"
print(f"A5 AR: {ans} == 4  OK")

# Test all 10 ALG via A4
alg_tests = [
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

for i, (q, exp) in enumerate(alg_tests, 1):
    p = f'<|system|>You are a math assistant.<|end|><|user|>{q}<|end|><|assistant|>'
    r = router._execute_symbolic_extract_solve(p, 'ALG')
    ans = r['answer_parsed']
    assert ans == exp, f"A4 ALG_{i} failed: got {ans}, expected {exp}"
    print(f"A4 ALG_{i:02d}: {ans} == {exp}  OK")

# Test A3 repair builds prompt (no LLM, just verify method runs)
r = router._execute_repair_action(
    '<|system|>Math.<|end|><|user|>How many chocolate bars?<|end|><|assistant|>',
    'WP',
    'The bus traveled 60 miles'
)
assert r['final_source'] == 'repair', f"A3 source wrong: {r['final_source']}"
print(f"A3 WP repair: method OK (would need LLM for actual test)")

# Test fallback chains
fb = router._get_fallback_action('A2', 'WP', 'parse_failure', 1)
assert fb == 'A3', f"WP fallback wrong: {fb}"
print(f"WP fallback: A2 -> {fb}  OK")

fb = router._get_fallback_action('A1', 'LOG', 'parse_failure', 1)
assert fb == 'A3', f"LOG fallback wrong: {fb}"
print(f"LOG fallback: A1 -> {fb}  OK")

fb = router._get_fallback_action('A5', 'AR', 'symbolic_solver_failure', 1)
assert fb == 'A1', f"AR fallback wrong: {fb}"
print(f"AR fallback: A5 -> {fb}  OK")

fb = router._get_fallback_action('A4', 'ALG', 'symbolic_solver_failure', 1)
assert fb == 'A1', f"ALG fallback wrong: {fb}"
print(f"ALG fallback: A4 -> {fb}  OK")

print("\nAll V3 router tests PASSED.")
