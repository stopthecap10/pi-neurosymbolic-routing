#!/usr/bin/env bash
set -euo pipefail

# Tier-1 Qwen2.5-Math-7B Q4 baseline without grammar (64 tokens for reasoning)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Missing .venv. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

source .venv/bin/activate
mkdir -p outputs

CSV_PATH="${CSV_PATH:-data/benchmarks/industry_tier1_40.csv}"
SERVER_URL="${SERVER_URL:-http://192.168.1.179:8080/completion}"
MODEL_FILE="${MODEL_FILE:-/home/stopthecap10/edge-ai/models/qwen2.5-math-7b-instruct-q4_k_m.gguf}"
REPEATS="${REPEATS:-3}"
TIMEOUT_S="${TIMEOUT_S:-30}"
WARMUP_PER_PROMPT="${WARMUP_PER_PROMPT:-1}"
N_PRED_NUM="${N_PRED_NUM:-64}"
N_PRED_LOG="${N_PRED_LOG:-32}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OUT_CSV="outputs/T1_qwen2.5_math_7b_q4_nogrammar_64tok_r${REPEATS}_${RUN_ID}.csv"
TRIALS_CSV="outputs/T1_qwen2.5_math_7b_q4_nogrammar_64tok_r${REPEATS}_${RUN_ID}_trials.csv"

python3 src/run_phi2_server_runner_clean_nogrammar.py \
  --csv "$CSV_PATH" \
  --out "$OUT_CSV" \
  --trials_out "$TRIALS_CSV" \
  --server_url "$SERVER_URL" \
  --timeout_s "$TIMEOUT_S" \
  --repeats "$REPEATS" \
  --warmup_per_prompt "$WARMUP_PER_PROMPT" \
  --n_pred_num "$N_PRED_NUM" \
  --n_pred_log "$N_PRED_LOG"

echo ""
echo "Results saved to:"
echo "  Summary: $OUT_CSV"
echo "  Trials:  $TRIALS_CSV"
