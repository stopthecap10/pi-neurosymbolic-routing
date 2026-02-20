# A1 vs A2 Controlled Comparison Test - FINAL

**Status:** Ready to run
**All 7 pre-run checks addressed**

---

## Test Controls (FROZEN)

```python
FROZEN_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "timeout_sec": 20,
    "prompt_template": "{prompt}\nAnswer with only the final number.\nAnswer:",
    "prompt_template_version": "v1.0",
    "model": "Phi-4-Mini Q6_K",
    "parser": "extract_last_int()",
    "grammar": "none",  # Fixed: no grammar for both A1 and A2
    "repeats": 3,
}

VARIABLE_PARAM = {
    "A1_short": {"tokens": 12},
    "A2_extended": {"tokens": 30},
}
```

## Decision Rule (With Quality Checks)

```python
For each category (ALG/WP):
    Compute:
        Δacc = acc_A2 - acc_A1
        Δenergy = median_energy_A2 - median_energy_A1
        Δlatency = median_latency_A2 - median_latency_A1
        Δtimeout = timeout_rate_A2 - timeout_rate_A1
        Δparse_fail = parse_fail_A2 - parse_fail_A1

    Decision:
        IF Δacc >= 3pp AND Δtimeout < 10pp AND Δparse_fail < 10pp:
            → Use A2_extended for that category
        ELIF both A1 and A2 have accuracy < 50% (for ALG only):
            → Use A4_extract_solve (SymPy) instead
        ELSE:
            → Use A1_short (cheaper), A2 as escalation
```

---

## Commands to Run on Pi

```bash
# 1. Sync updated test scripts
./scripts/sync_to_pi.sh

# 2. SSH into Pi
ssh stopthecap10@192.168.1.179
cd ~/pi-neurosymbolic-routing

# 3. Run all 4 tests
```

### Test 1: ALG with A1 (12 tokens)
```bash
python3 src/test_action_comparison.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --category ALG \
  --action A1 \
  --out_trials outputs/alg_a1.csv
```

### Test 2: ALG with A2 (30 tokens)
```bash
python3 src/test_action_comparison.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --category ALG \
  --action A2 \
  --out_trials outputs/alg_a2.csv
```

### Test 3: WP with A1 (12 tokens)
```bash
python3 src/test_action_comparison.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --category WP \
  --action A1 \
  --out_trials outputs/wp_a1.csv
```

### Test 4: WP with A2 (30 tokens)
```bash
python3 src/test_action_comparison.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --category WP \
  --action A2 \
  --out_trials outputs/wp_a2.csv
```

### Generate Comparison Summary
```bash
python3 src/compare_a1_a2.py
```

---

## Outputs

**Individual CSVs:**
- outputs/alg_a1.csv
- outputs/alg_a2.csv
- outputs/wp_a1.csv
- outputs/wp_a2.csv

**Summary files:**
- outputs/a1_a2_comparison_summary.md (routing decisions)
- outputs/a1_a2_comparison_by_category.csv (metrics table)

**CSV Fields (all logged):**
- run_id, timestamp, prompt_id, category, action_type
- token_budget, grammar_enabled, grammar_version, prompt_template_version
- answer_parsed, correct, parse_success
- latency_ms_total, timeout_flag
- **energy_start_mwh, energy_end_mwh, energy_delta_mwh, energy_per_prompt_mwh**
- error_code
- model_name, quantization, temperature, top_p, seed, timeout_sec

---

## Time Estimate

- Run 4 tests: 15 minutes (with energy recording)
- Generate comparison: 1 minute
- **Total: ~20 minutes**

---

## Success Criteria

After comparison, we'll have:
1. ✅ Data-driven routing rules for ALG and WP
2. ✅ Energy/latency tradeoffs quantified
3. ✅ Quality checks (timeout/parse) validated
4. ✅ Clear decision for V2 implementation

---

**Ready to run!**
