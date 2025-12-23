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
