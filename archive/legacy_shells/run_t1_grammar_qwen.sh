#!/usr/bin/env bash
set -euo pipefail

# Tier-1 Qwen2.5-3B Q8 baseline with grammar constraints + energy logging

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
  echo "Missing .venv. Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

source .venv/bin/activate
mkdir -p outputs

CSV_PATH="${CSV_PATH:-data/benchmarks/industry_tier1_40.csv}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080/completion}"
MODEL_FILE="${MODEL_FILE:-/home/stopthecap10/edge-ai/models/qwen2.5-3b-instruct-q8_0.gguf}"
REPEATS="${REPEATS:-3}"
TIMEOUT_S="${TIMEOUT_S:-25}"
WARMUP_PER_PROMPT="${WARMUP_PER_PROMPT:-1}"
N_PRED_NUM="${N_PRED_NUM:-7}"
N_PRED_LOG="${N_PRED_LOG:-3}"
NUM_GRAMMAR_FILE="${NUM_GRAMMAR_FILE:-grammars/grammar_phi2_answer_int_strict_final.gbnf}"
YESNO_GRAMMAR_FILE="${YESNO_GRAMMAR_FILE:-grammars/grammar_phi2_answer_yesno_strict_final.gbnf}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OUT_CSV="outputs/T1_qwen2.5_3b_q8_grammar_r${REPEATS}_${RUN_ID}.csv"
TRIALS_CSV="outputs/T1_qwen2.5_3b_q8_grammar_r${REPEATS}_${RUN_ID}_trials.csv"

python3 src/log_run_energy.py \
  --system qwen2.5_3b_q8_grammar \
  --tier T1 \
  --csv_used "$CSV_PATH" \
  --model_file "$MODEL_FILE" \
  --repeats "$REPEATS" \
  --out_csv "$OUT_CSV" \
  --trials_csv "$TRIALS_CSV" \
  --log_csv outputs/energy_log.csv \
  -- \
  python3 src/run_phi2_server_runner_safe.py \
    --csv "$CSV_PATH" \
    --out "$OUT_CSV" \
    --trials_out "$TRIALS_CSV" \
    --server_url "$SERVER_URL" \
    --timeout_s "$TIMEOUT_S" \
    --repeats "$REPEATS" \
    --warmup_per_prompt "$WARMUP_PER_PROMPT" \
    --n_pred_num "$N_PRED_NUM" \
    --n_pred_log "$N_PRED_LOG" \
    --num_grammar_file "$NUM_GRAMMAR_FILE" \
    --yesno_grammar_file "$YESNO_GRAMMAR_FILE"
