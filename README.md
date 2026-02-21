# Neuro-Symbolic Routing for Energy-Efficient Edge AI Inference

**Research Project** | Raspberry Pi 4B | Phi-4-Mini (Q6_K) + SymPy

## Abstract

This project investigates whether **neuro-symbolic routing** — selectively bypassing neural inference with symbolic solvers — can improve both accuracy and energy efficiency on edge hardware. We implement a deterministic routing system on a Raspberry Pi 4B that classifies math problems by category and routes them to the most efficient solver: direct symbolic evaluation for arithmetic, SymPy-based equation solving for algebra, and constrained neural inference for word problems and logic.

## Research Question

Can a hybrid neuro-symbolic router on resource-constrained edge hardware (Raspberry Pi 4B, 8GB RAM) achieve higher accuracy with lower energy consumption compared to neural-only inference?

## Method

### System Architecture

```
Input Prompt --> Category Classifier --> Router
                                          |-- A5: Symbolic (AST eval)     --> AR prompts
                                          |-- A4: LLM + SymPy solve      --> ALG prompts
                                          |-- A1/A2: Neural (12/30 tok)  --> WP prompts
                                          |-- A1: Neural (6 tok)         --> LOG prompts
```

### Experimental Protocol

1. **Baseline Matrix**: 4 configs (A1/A2 x grammar/no-grammar) x 40 prompts x 3 repeats = 480 trials
2. **Hybrid V1**: Deterministic routing with A5 symbolic solver for arithmetic
3. **Hybrid V2**: Enhanced with A4 (LLM extraction + SymPy equation solver) for algebra
4. **Energy**: Measured via USB power meter (delta mWh per run)

### Dataset

- **40 prompts** (Tier-1), balanced: 10 AR / 10 ALG / 10 WP / 10 LOG
- Sources: DeepMind Mathematics v1.0, GSM8K, RuleTaker
- All sourced from published datasets (no hand-written prompts)
- Full provenance tracked per prompt (dataset, split, record ID, field mapping version)

### Hardware

- Raspberry Pi 4 Model B (8GB RAM)
- Phi-4-Mini quantized to Q6_K (3.0 GB)
- llama.cpp inference server (CPU-only, 4 threads)
- USB power meter for energy measurement

## Repository Structure

```
pi-neurosymbolic-routing/
├── src/
│   ├── router_v1.py                 # V1 router (A5 symbolic for AR)
│   ├── router_v2.py                 # V2 router (+ A4 SymPy for ALG)
│   ├── run_v1_baseline_matrix.py    # Baseline runner (A1/A2 x grammar)
│   ├── run_hybrid_v1.py             # Hybrid V1 runner
│   ├── run_hybrid_v2.py             # Hybrid V2 runner
│   ├── build_official_splits.py     # Dataset builder (from raw sources)
│   ├── rebuild_t1_balanced.py       # T1 rebalancing script
│   ├── summarize_v1_matrix.py       # Baseline summary generator
│   └── compare_hybrid_v1_to_baseline.py  # Comparison report
├── data/
│   └── splits/                      # Official benchmark splits
│       ├── industry_tier1_40_v2.csv # 40-prompt balanced T1 (production)
│       ├── industry_tier2_200.csv   # 200-prompt T2
│       └── split_manifest.json      # Provenance manifest
├── configs/
│   └── run_tier1.yaml               # Frozen inference config
├── grammars/                        # GBNF grammars for constrained decoding
├── scripts/                         # Shell scripts for Pi execution
└── tests/                           # Test suite
```

## Reproducing Results

### Prerequisites

- Raspberry Pi 4B (8GB) with Debian/Ubuntu
- Python 3.9+
- llama.cpp compiled for ARM64
- Phi-4-Mini Q6_K model file (~3 GB)
- USB power meter (optional, for energy measurement)

### Setup

```bash
git clone https://github.com/stopthecap10/pi-neurosymbolic-routing.git
cd pi-neurosymbolic-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Start Inference Server

```bash
cd ~/llama.cpp/build/bin
./llama-server \
  -m /path/to/microsoft_Phi-4-mini-instruct-Q6_K.gguf \
  --host 0.0.0.0 --port 8080 -c 4096 -ngl 0 &
```

### Step 2: Run Baseline Matrix (4 configs)

```bash
cd ~/pi-neurosymbolic-routing

# A1 (12 tokens) without grammar
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/industry_tier1_40_v2.csv \
  --action A1 --out_trials outputs/official/v1_a1_nogrammar.csv --split_role official

# A1 with grammar
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/industry_tier1_40_v2.csv \
  --action A1 --grammar --out_trials outputs/official/v1_a1_grammar.csv --split_role official

# A2 (30 tokens) without grammar
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/industry_tier1_40_v2.csv \
  --action A2 --out_trials outputs/official/v1_a2_nogrammar.csv --split_role official

# A2 with grammar
python3 src/run_v1_baseline_matrix.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/industry_tier1_40_v2.csv \
  --action A2 --grammar --out_trials outputs/official/v1_a2_grammar.csv --split_role official
```

### Step 3: Generate Summary + Run Hybrid

```bash
python3 src/summarize_v1_matrix.py --out_dir outputs/official

python3 src/run_hybrid_v1.py \
  --config configs/run_tier1.yaml \
  --csv data/splits/industry_tier1_40_v2.csv \
  --out_trials outputs/official/hybrid_v1.csv --split_role official

python3 src/compare_hybrid_v1_to_baseline.py --out_dir outputs/official
```

### Automated Full Run

```bash
bash scripts/run_baseline_matrix.sh
```

## Frozen Parameters

All inference parameters are frozen for reproducibility:

| Parameter | Value |
|-----------|-------|
| Model | Phi-4-Mini Q6_K |
| Temperature | 0.0 |
| Seed | 42 |
| A1 token budget | 12 |
| A2 token budget | 30 |
| Timeout | 20 sec |
| Repeats | 3 per prompt |
| Parser | P2_numeric_robust |
| Prompt template | PT2_frozen |

## Key Design Decisions

- **Deterministic routing**: Category determines solver; no learned classifier needed
- **Symbolic-first for arithmetic**: AST-based evaluation is faster and more accurate than LLM inference
- **Escalation fallback**: A5 -> A1 -> A2 for AR; A4 -> A1 -> A2 for ALG
- **Energy measurement**: USB power meter delta (mWh) across entire run, divided by prompt count
- **No hand-written prompts**: All 40 T1 prompts sourced from published academic datasets

## Dataset Sources

| Category | Source | Citation |
|----------|--------|----------|
| AR (Arithmetic) | DeepMind Mathematics v1.0 | Saxton et al., 2019 (arXiv:1904.01557) |
| ALG (Algebra) | DeepMind Mathematics v1.0 | Saxton et al., 2019 |
| WP (Word Problems) | GSM8K | Cobbe et al., 2021 |
| LOG (Logic) | RuleTaker | Clark et al., 2020 |

## License

MIT
