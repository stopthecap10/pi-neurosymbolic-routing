# Pi Neuro-Symbolic Routing (ISEF)

This repo benchmarks LLM-only vs neuro-symbolic routing on a Raspberry Pi.

## What’s included
- `src/`: baseline + hybrid runners
- `data/`: prompt sets (Tier2-400, Tier3-1000)
- `grammars/`: GBNF grammars for constrained decoding
- `outputs/`: summary CSVs + case-study CSVs used in the poster

## Quick start (Pi)
1) Start llama.cpp server (Phi-2 GGUF), example:
```bash
~/llama.cpp/build/bin/llama-server -m ~/edge-ai/models/phi-2-Q4_K_M.gguf -t 4 -c 512 --host 127.0.0.1 --port 8080
