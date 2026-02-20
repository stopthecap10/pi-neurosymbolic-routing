# V1 Baseline Matrix - Complete 2×2 Factorial

**Status:** Ready to run
**Purpose:** Establish clean baselines before building Hybrid V1

---

## Matrix Design

**Factor 1: Action (Token Budget)**
- A1: 12 tokens
- A2: 30 tokens

**Factor 2: Grammar**
- Grammar: enabled
- No-grammar: disabled

**4 Configs (2×2):**
1. `v1_a1_grammar` - 12 tokens + grammar
2. `v1_a1_nogrammar` - 12 tokens + no grammar
3. `v1_a2_grammar` - 30 tokens + grammar
4. `v1_a2_nogrammar` - 30 tokens + no grammar

---

## Frozen Parameters (ALL configs)

```python
FROZEN = {
    "model": "Phi-4-Mini Q6_K",
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 42,
    "timeout_sec": 20,
    "prompt_template": "{prompt}\n{instruction}\nAnswer:",
    "prompt_template_version": "v1.0",
    "parser": "extract_last_int() / extract_yesno()",
    "energy_method": "usb_power_meter_delta_mwh",
    "repeats": 3,
}

VARIABLE = {
    "action_id": ["A1", "A2"],
    "grammar_enabled": [0, 1],
}
```

---

## Commands to Run on Pi

```bash
# Sync code
cd ~/pi-neurosymbolic-routing  # On Mac
./scripts/sync_to_pi.sh

# SSH into Pi
ssh stopthecap10@192.168.1.179
cd ~/pi-neurosymbolic-routing
```

### Config 1: v1_a1_grammar
```bash
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --action A1 \
  --grammar \
  --out_trials outputs/v1_a1_grammar.csv
```

### Config 2: v1_a1_nogrammar
```bash
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --action A1 \
  --out_trials outputs/v1_a1_nogrammar.csv
```

### Config 3: v1_a2_grammar
```bash
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --action A2 \
  --grammar \
  --out_trials outputs/v1_a2_grammar.csv
```

### Config 4: v1_a2_nogrammar
```bash
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/tier1_mini.csv \
  --action A2 \
  --out_trials outputs/v1_a2_nogrammar.csv
```

---

## Time Estimate

- Per config: ~3 minutes (20 prompts × 3 repeats = 60 inferences)
- **Total: ~15 minutes for all 4 configs**

---

## Output Files

- `outputs/v1_a1_grammar.csv` - 60 trials
- `outputs/v1_a1_nogrammar.csv` - 60 trials
- `outputs/v1_a2_grammar.csv` - 60 trials
- `outputs/v1_a2_nogrammar.csv` - 60 trials

**Total: 240 trials** (complete factorial baseline)

---

## Next Step: Generate Summary

After all 4 configs complete, run:

```bash
python3 src/summarize_v1_matrix.py
```

This creates:
- `outputs/t1_baseline_summary.md` - Main summary
- `outputs/t1_baseline_by_category.csv` - Metrics table
- `outputs/v1_decisions_for_hybrid_v1.md` - Routing decisions

---

## What We'll Learn

1. **Main effects:**
   - Does A2 (30 tok) improve accuracy vs A1 (12 tok)?
   - Does grammar help or hurt?

2. **Interactions:**
   - Does grammar benefit depend on token budget?
   - Which categories benefit from each factor?

3. **Category-specific routing rules:**
   - AR: best config?
   - ALG: best config?
   - WP: best config?
   - LOG: best config?

Then we build Hybrid V1 using data-driven rules!
