# Complete Execution Workflow - ISEF Project

**Created**: 2026-02-12
**Purpose**: Step-by-step guide to execute the complete research pipeline

---

## Overview

This workflow implements your ISEF research plan:
1. **V1 Baselines** - Establish LLM-only performance ceiling
2. **Hybrid V1** - Deterministic rule-based routing
3. **Hybrid V2** - Refined routing with better fallbacks
4. **V3/V4** - Verification and calibration (future)

---

## Prerequisites

### On Raspberry Pi
- [ ] llama.cpp server running on port 8080
- [ ] Phi-4-Mini Q6_K model loaded
- [ ] USB power meter connected and readable
- [ ] All code synced from Mac to Pi

### Sync Code to Pi
```bash
# On Mac
bash scripts/sync_to_pi.sh
```

### Verify Server
```bash
# On Pi
curl http://127.0.0.1:8080/health
```

---

## Phase 1: V1 Baseline Matrix

### Step 1.1: Smoke Test (Optional but Recommended)
```bash
# On Pi
cd ~/edge-ai/pi-neurosymbolic-routing
bash scripts/run_smoke_test.sh
```

**Expected output**: 16 trials (4 prompts × 4 configs), ~2-3 minutes

**Pass criteria**:
- All 4 configs complete without crashes
- Energy readings recorded successfully
- No unexpected E8 parse failures

### Step 1.2: Full Baseline Matrix

```bash
# On Pi
bash scripts/run_baseline_matrix.sh
```

**What this does**:
- Runs all 4 configs: (A1/A2) × (grammar/no-grammar)
- 20 prompts × 3 repeats = 60 trials per config
- Total: 240 trials
- Duration: ~60-120 minutes

**Manual steps during execution**:
1. Config 1/4 starts → Enter starting mWh
2. Config 1/4 ends → Enter ending mWh
3. Repeat for configs 2-4

**Output files**:
```
outputs/v1_a1_grammar.csv
outputs/v1_a1_nogrammar.csv
outputs/v1_a2_grammar.csv
outputs/v1_a2_nogrammar.csv
```

### Step 1.3: Generate Summary

```bash
# On Pi (or can sync back to Mac and run there)
python3 src/summarize_v1_matrix.py
```

**Output files**:
```
outputs/t1_baseline_summary.md         # Overall results and key findings
outputs/t1_baseline_by_category.csv    # Per-category breakdown
```

**Review**:
1. Open `outputs/t1_baseline_summary.md`
2. Check overall accuracy, latency, energy
3. Identify which categories are weak (expect ALG, WP ~40%)
4. Note which config is best overall

---

## Phase 2: Make Routing Decisions

### Step 2.1: Fill Decisions File

Open `outputs/v1_decisions_for_hybrid.md` and fill in:

1. **Canonical baseline** - Which config to compare against?
2. **AR routing** - Likely A5 (symbolic direct) since already 100%
3. **ALG routing** - A1 or A4 (A4 not fully implemented yet)
4. **WP routing** - A1 with fallback to A2?
5. **LOG routing** - A1 (fast yes/no) since already 100%
6. **Grammar policy** - Enable or disable per category?
7. **Fallback rules** - How to handle timeouts/parse fails?

**Example decisions**:
```
Canonical baseline: v1_a1_nogrammar (cheapest with good accuracy)
AR  → A5 (symbolic direct, cost savings)
ALG → A1 (A4 needs implementation)
WP  → A1 with fallback to A2
LOG → A1 (already perfect)
Grammar: disabled (no benefit observed)
Max escalations: 1
```

### Step 2.2: Update Router (If Needed)

If you want to change the routing map from the hardcoded default:

Edit `src/run_hybrid_v1.py` function `load_routing_decisions()` to match your filled decisions file.

---

## Phase 3: Hybrid V1

### Step 3.1: Smoke Test

```bash
# On Pi
bash scripts/run_hybrid_v1_smoke.sh
```

**Expected**:
- 4 trials (1 per category)
- See routing in action: AR→A5, ALG→A1, etc.
- Duration: ~1 minute

### Step 3.2: Full Hybrid V1 Run

```bash
# On Pi
bash scripts/run_hybrid_v1.sh
```

**What this does**:
- Runs 20 prompts × 3 repeats = 60 trials
- Each trial routes through deterministic router
- Logs routing decisions and fallback attempts
- Duration: ~10-15 minutes

**Manual steps**:
1. Enter starting mWh
2. Let it run (watch routing decisions in output)
3. Enter ending mWh

**Output file**:
```
outputs/hybrid_v1.csv
```

### Step 3.3: Compare to Baseline

Create `src/compare_hybrid_v1_to_baseline.py`:

```python
#!/usr/bin/env python3
"""Compare Hybrid V1 to canonical baseline"""

import csv
from pathlib import Path

# Load canonical baseline (e.g., v1_a1_nogrammar)
baseline_path = Path("outputs/v1_a1_nogrammar.csv")
hybrid_path = Path("outputs/hybrid_v1.csv")

baseline_trials = list(csv.DictReader(open(baseline_path)))
hybrid_trials = list(csv.DictReader(open(hybrid_path)))

# Calculate metrics
def calc_metrics(trials):
    total = len(trials)
    correct = sum(int(t['correct']) for t in trials)

    latencies = [float(t['total_latency_ms']) for t in trials if not int(t.get('timeout_flag', 0))]
    median_lat = sorted(latencies)[len(latencies)//2] if latencies else 0

    energy_vals = []
    for t in trials:
        e = t.get('energy_per_prompt_mwh', 'NA')
        if e != 'NA':
            try:
                energy_vals.append(float(e))
            except:
                pass
    median_energy = sorted(energy_vals)[len(energy_vals)//2] if energy_vals else None

    return {
        'accuracy': correct / total if total > 0 else 0,
        'median_latency_ms': median_lat,
        'median_energy_mwh': median_energy
    }

baseline_metrics = calc_metrics(baseline_trials)
hybrid_metrics = calc_metrics(hybrid_trials)

print("="*60)
print("HYBRID V1 vs BASELINE COMPARISON")
print("="*60)
print()
print(f"Baseline (v1_a1_nogrammar):")
print(f"  Accuracy: {baseline_metrics['accuracy']*100:.1f}%")
print(f"  Latency:  {baseline_metrics['median_latency_ms']:.0f} ms")
if baseline_metrics['median_energy_mwh']:
    print(f"  Energy:   {baseline_metrics['median_energy_mwh']:.2f} mWh")
print()
print(f"Hybrid V1:")
print(f"  Accuracy: {hybrid_metrics['accuracy']*100:.1f}%")
print(f"  Latency:  {hybrid_metrics['median_latency_ms']:.0f} ms")
if hybrid_metrics['median_energy_mwh']:
    print(f"  Energy:   {hybrid_metrics['median_energy_mwh']:.2f} mWh")
print()
print("Improvement:")
acc_delta = (hybrid_metrics['accuracy'] - baseline_metrics['accuracy']) * 100
print(f"  Accuracy: {acc_delta:+.1f} pp")

if baseline_metrics['median_latency_ms'] > 0:
    lat_delta_pct = ((hybrid_metrics['median_latency_ms'] / baseline_metrics['median_latency_ms']) - 1) * 100
    print(f"  Latency:  {lat_delta_pct:+.1f}%")

if hybrid_metrics['median_energy_mwh'] and baseline_metrics['median_energy_mwh']:
    energy_delta_pct = ((hybrid_metrics['median_energy_mwh'] / baseline_metrics['median_energy_mwh']) - 1) * 100
    print(f"  Energy:   {energy_delta_pct:+.1f}%")
```

Run:
```bash
python3 src/compare_hybrid_v1_to_baseline.py
```

### Step 3.4: Decision Gate

**If Hybrid V1 shows improvement** (accuracy OR cost):
- ✅ Proceed to Hybrid V2

**If Hybrid V1 shows NO improvement**:
- Review routing decisions
- Check if A5 (symbolic) is working correctly for AR
- Consider implementing A4 for ALG
- Debug fallback logic

---

## Phase 4: Hybrid V2 (After V1 Success)

### Implementation Notes

Hybrid V2 = Hybrid V1 + refinements:
- Better fallback chains
- Stronger symbolic verification
- More detailed logging

**Key changes from V1 to V2**:
1. Implement A4 (LLM extract + SymPy solve) for ALG
2. Refine fallback chains (e.g., AR: A5→A1→A2, ALG: A4→A1→A2)
3. Add symbolic verification for WP numeric answers
4. Improve routing decision logging

**Files to create**:
- `src/router_v2.py` - Enhanced router
- `src/run_hybrid_v2.py` - V2 runner
- `scripts/run_hybrid_v2.sh` - Run script

**Testing process** (same as V1):
1. Smoke test
2. Full T1 run
3. Compare to V1 and baseline
4. Document improvements

---

## Phase 5: Scale to Tier 2 and Tier 3 (Future)

After Hybrid V2 proves successful on T1:

### Tier 2 (Calibration)
- 200 prompts
- Use for calibrating V4 controller
- Same test procedure

### Tier 3 (Final Evaluation)
- 300 prompts
- Final results for ISEF paper
- Full statistical analysis

---

## Troubleshooting

### Server not responding
```bash
# Check server status
curl http://127.0.0.1:8080/health

# Restart server (adjust path to your llama.cpp)
cd ~/edge-ai/llama.cpp
./llama-server -m ~/edge-ai/models/microsoft_Phi-4-mini-instruct-Q6_K.gguf -c 2048 --port 8080
```

### Parse failures (E8)
- Check prompt formatting in tier1_mini.csv
- Verify grammar files exist and are valid
- Review extraction regex in router

### Energy readings inconsistent
- Reset USB meter before each run
- Take readings at exact same time (before first trial, after last trial)
- Document any interruptions

### Router not routing correctly
- Check routing decisions in load_routing_decisions()
- Verify category detection logic
- Add debug prints to router.route() method

---

## Quick Reference Commands

```bash
# === ON MAC ===
# Sync code to Pi
bash scripts/sync_to_pi.sh

# === ON PI ===
# Baseline smoke test
bash scripts/run_smoke_test.sh

# Full baseline matrix
bash scripts/run_baseline_matrix.sh

# Generate summary
python3 src/summarize_v1_matrix.py

# Hybrid V1 smoke
bash scripts/run_hybrid_v1_smoke.sh

# Hybrid V1 full
bash scripts/run_hybrid_v1.sh

# Compare
python3 src/compare_hybrid_v1_to_baseline.py
```

---

## Expected Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1.1 | Baseline smoke test | 3 min |
| 1.2 | Full baseline matrix | 90 min |
| 1.3 | Generate summary | 1 min |
| 2 | Make routing decisions | 20 min |
| 3.1 | Hybrid V1 smoke | 1 min |
| 3.2 | Hybrid V1 full | 15 min |
| 3.3 | Compare results | 2 min |
| **Total** | | **~2.5 hours** |

---

## Next Steps After This Workflow

1. Document V1 and Hybrid V1 results in ISEF paper
2. Implement Hybrid V2 with A4 and better fallbacks
3. Run Hybrid V2 and compare
4. Scale to Tier 2 for calibration
5. Implement V4 (calibrated controller)
6. Run Tier 3 for final evaluation
7. Write complete results section

---

**Good luck! You've got this. 🚀**
