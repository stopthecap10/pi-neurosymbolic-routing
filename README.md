# Pi Neuro-Symbolic Routing (ISEF) — Phi-2 + SymPy (Raspberry Pi 4B)

This repo benchmarks **LLM-only** vs **neuro-symbolic routing** on a **Raspberry Pi 4B**.
Main idea: route math-like problems to **SymPy** (or use SymPy to verify/fix), so you can reduce compute cost on edge devices.

## What's included
- `src/pi_neuro_routing/`: Refactored Python package with core logic, runners, benchmarks, and analysis tools
- `data/benchmarks/`: Benchmark datasets (Tier1-40, Tier2-400, Tier3-1000)
- `grammars/`: GBNF grammars for constrained decoding (organized into final/ and experimental/)
- `tests/`: Test suite
- `docs/`: Documentation including migration guide
- `archive/`: Archived backup files and legacy data (not deleted, just organized)
- `scripts/`: Shell scripts for running experiments
- Unified CLI: `src/pi_neuro_routing/cli/run_experiment.py` - single entry point for all experiments

## Directory Structure

```
pi-neurosymbolic-routing/
├── src/pi_neuro_routing/      # Main package
│   ├── core/                  # Core runner logic
│   ├── runners/               # Specific implementations
│   ├── benchmarks/            # Benchmark generation
│   ├── analysis/              # Analysis tools
│   ├── utils/                 # Utilities
│   └── cli/                   # CLI interface
├── data/benchmarks/           # Organized datasets
├── grammars/final/            # Production grammars
├── tests/                     # Test suite
└── docs/                      # Documentation
```

See [docs/MIGRATION.md](docs/MIGRATION.md) for details on the refactored structure.

## Quickstart (Pi)

### 0) Setup (once)
```bash
cd ~/pi-neurosymbolic-routing
python3 -m venv .venv
source .venv/bin/activate
pip install -e .  # Install package in development mode
```

### 1) Running Experiments

**Unified CLI** (recommended):
```bash
# Baseline with grammar constraints
python -m pi_neuro_routing.cli.run_experiment --mode baseline --grammar \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_grammar.csv \
    --trials_out results/baseline_grammar_trials.csv \
    --verbose

# Baseline without grammar constraints
python -m pi_neuro_routing.cli.run_experiment --mode baseline --no-grammar \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_nogrammar.csv \
    --verbose

# Both variants in one run
python -m pi_neuro_routing.cli.run_experiment --mode baseline --both \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_both.csv \
    --trials_out results/baseline_both_trials.csv
```

**Legacy scripts** (original files in `src/`, still work):
```bash
# These still work if you prefer the original interface
python src/run_baseline_phi2_server.py --csv data/benchmarks/industry_tier2_400.csv --out results/out.csv
python src/run_hybrid_v1.py
python src/run_hybrid_v2.py
```

For more details on the CLI and migration, see [docs/MIGRATION.md](docs/MIGRATION.md).

## Tier-1 Phi-2 (Pi) quick commands

From repo root on the Pi:

- Reset energy log (archives to `outputs/_archive/`):  
  `bash scripts/reset_energy_log.sh`
- Start llama.cpp server in tmux (Phi-2 Q8):  
  `tmux new -s phi2 -d 'cd ~/llama.cpp && ./build/bin/llama-server -m /home/stopthecap10/edge-ai/models/phi-2.Q8_0.gguf -c 2048 -t 4 --host 127.0.0.1 --port 8080'`
- Tier-1 grammar baseline + energy logging (defaults: repeats=3, timeout_s=20, warmup_per_prompt=1, n_pred_num=12, n_pred_log=6):  
  `bash scripts/run_t1_grammar.sh`
- Tier-1 no-grammar baseline + energy logging (same defaults, no grammar sent):  
  `bash scripts/run_t1_nogrammar.sh`
- Verify Tier-1 CSV layout and preview prompts (2 per category):  
  `bash scripts/check_t1_prompts.py --csv data/benchmarks/industry_tier1_40.csv`

Outputs land in `outputs/` with run-specific timestamps; each run appends one row to `outputs/energy_log.csv`. Override defaults via env vars (e.g., `MODEL_FILE`, `SERVER_URL`, `CSV_PATH`, `REPEATS`, `TIMEOUT_S`, `WARMUP_PER_PROMPT`).

## Results (Tier-2 / 400 prompts on Raspberry Pi 4B)

| System | Accuracy (ALL) | Median latency (s) |
|---|---:|---:|
| Phi-2 LLM-only (grammar) | 66.5% | 1.7535 |
| Hybrid v1 (router: SymPy for AR/ALG) | 87.5% | 0.7681 |
| Hybrid v2 (LLM + SymPy verify/fallback) | 89.5% | 8.8793 |

Notes:
- AR/ALG/LOG/WP are Arithmetic / Algebra / Logic / Word Problems.
- Latencies are medians across repeated trials (see output CSVs for details).
- Hybrid v2 improves accuracy via verification/fallback, but adds overhead.
