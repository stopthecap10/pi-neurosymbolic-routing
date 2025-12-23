# Pi Neuro-Symbolic Routing (ISEF) — Phi-2 + SymPy (Raspberry Pi 4B)

This repo benchmarks **LLM-only** vs **neuro-symbolic routing** on a **Raspberry Pi 4B**.
Main idea: route math-like problems to **SymPy** (or use SymPy to verify/fix), so you can reduce compute cost on edge devices.

## What’s included
- `src/`: baseline + hybrid runners
- `data/`: prompt sets (Tier2-400, Tier3-1000)
- `grammars/`: GBNF grammars for constrained decoding
- `artifacts/` (optional): archived run outputs
- `sheet_summary*.csv`: summarized results used for plots/poster

## Quickstart (Pi)

### 0) Setup (once)
```bash
cd ~/isef_repo_public
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

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
