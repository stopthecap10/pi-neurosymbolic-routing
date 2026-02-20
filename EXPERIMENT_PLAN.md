# ISEF Experiment Plan: Neurosymbolic Routing on Edge Devices

## Overview

This experiment evaluates the effectiveness of neurosymbolic routing (LLM + symbolic solver) for mathematical and logical reasoning on resource-constrained edge devices (Raspberry Pi 4B, 8GB RAM).

## Models

### Baseline: Qwen2.5-3B-Instruct (Q8)
- **Purpose**: General-purpose LLM baseline
- **Size**: ~2GB
- **Expected GSM8K**: ~50-60%
- **Use Case**: What developers would typically deploy on edge devices

### Specialist: Qwen2.5-Math-7B-Instruct (Q4)
- **Purpose**: Math-specialized LLM
- **Size**: ~4.3GB
- **Expected GSM8K**: ~91.6%
- **Use Case**: Domain-specific optimization

### Symbolic Solver: SymPy
- **Purpose**: Handles algebraic and arithmetic problems
- **Expected Accuracy**: 80-95% on ALG/AR categories

## Experimental Conditions

### 1. Grammar Constraints
Tests whether grammar-constrained decoding reduces extraction failures (E8 errors).

**Hypothesis**: Grammar constraints enable reliable structured output regardless of model accuracy.

### 2. Model Comparison
Tests whether specialized models outperform general-purpose models.

**Hypothesis**: Domain-specific models (Math-7B) outperform general models (3B) on math tasks.

### 3. Hybrid Routing
Tests whether routing difficult problems to symbolic solvers improves overall accuracy.

**Hypothesis**: Hybrid systems outperform pure LLM approaches by leveraging symbolic reasoning.

## Test Matrix

| System | Model | Grammar | SymPy | Script |
|--------|-------|---------|-------|--------|
| Baseline-3B-NG | Qwen-3B | No | No | `run_t1_nogrammar_qwen.sh` |
| Baseline-3B-G | Qwen-3B | Yes | No | `run_t1_grammar_qwen.sh` |
| Specialist-NG | Math-7B | No | No | `run_t1_nogrammar_qwen_math.sh` |
| Specialist-G | Math-7B | Yes | No | `run_t1_grammar_qwen_math.sh` |
| Hybrid-V1 | Math-7B | Yes | Yes | `run_hybrid_v1_qwen_math.sh` |
| Hybrid-V2 | Math-7B | Yes | Yes | `run_hybrid_v2_qwen_math.sh` |

## Dataset

**Tier-1 Benchmark** (40 prompts × 3 repeats = 120 trials)
- WP (Word Problems): 10 from GSM8K
- AR (Arithmetic): 10 from SVAMP
- ALG (Algebra): 10 from AQuA-RAT
- LOG (Logic): 10 from BoolQ

## Routing Strategy (Hybrid V1)

```
if category in (ALG, AR):
    route to SymPy
elif category in (WP, LOG):
    route to LLM
```

## Expected Results

| System | WP | AR | ALG | LOG | Overall |
|--------|----|----|-----|-----|---------|
| Baseline-3B-G | 50-60% | 30-40% | 0-10% | 75-80% | ~40% |
| Specialist-G | 85-90% | 40-50% | 0-10% | 75-80% | ~50% |
| Hybrid-V1 | 85-90% | 80-95% | 80-95% | 75-80% | ~85% |

## Key Metrics

1. **Extraction Failure Rate (E8)**
   - Grammar vs No-Grammar comparison
   - Should show: Grammar → 0% E8, No-Grammar → 40-60% E8

2. **Accuracy by Category**
   - Baseline vs Specialist vs Hybrid
   - Shows value of specialization and symbolic routing

3. **Energy Efficiency**
   - Joules per query
   - Shows edge deployment is practical

4. **Inference Time**
   - Seconds per query
   - Demonstrates real-time feasibility

## Research Questions

1. **RQ1**: Do grammar constraints reduce extraction failures on edge LLMs?
   - **Metric**: E8 rate (Grammar vs No-Grammar)

2. **RQ2**: Do specialized models outperform general-purpose models?
   - **Metric**: Accuracy (Math-7B vs 3B)

3. **RQ3**: Does neurosymbolic routing improve overall performance?
   - **Metric**: Overall accuracy (Hybrid vs LLM-only)

4. **RQ4**: Is edge deployment practical for real-time applications?
   - **Metric**: Inference time and energy consumption

## Significance

This research demonstrates that:
1. Grammar-constrained decoding enables reliable structured output from edge LLMs
2. Model specialization + symbolic routing achieves near-optimal performance
3. Edge devices can deploy sophisticated reasoning systems with proper architecture
4. Neurosymbolic approaches unlock capabilities beyond pure neural methods

## Scripts Created

- ✅ `scripts/run_t1_grammar_qwen.sh` (Baseline with grammar)
- ✅ `scripts/run_t1_nogrammar_qwen.sh` (Baseline without grammar)
- ✅ `scripts/run_t1_grammar_qwen_math.sh` (Specialist with grammar)
- ✅ `scripts/run_t1_nogrammar_qwen_math.sh` (Specialist without grammar)
- ⏳ `scripts/run_hybrid_v1_qwen_math.sh` (To be updated)
- ⏳ `scripts/run_hybrid_v2_qwen_math.sh` (To be updated)
