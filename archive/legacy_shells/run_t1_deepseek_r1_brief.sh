#!/usr/bin/env bash
set -euo pipefail

# Tier-1 DeepSeek-R1-Distill-Qwen-7B Q4 with brief Answer: mode (no grammar, stop sequence)

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
MODEL_FILE="${MODEL_FILE:-/home/stopthecap10/edge-ai/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf}"
REPEATS="${REPEATS:-3}"
TIMEOUT_S="${TIMEOUT_S:-15}"
WARMUP_PER_PROMPT="${WARMUP_PER_PROMPT:-1}"
N_PRED="${N_PRED:-20}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OUT_CSV="outputs/T1_deepseek_r1_q4_brief_r${REPEATS}_${RUN_ID}.csv"
TRIALS_CSV="outputs/T1_deepseek_r1_q4_brief_r${REPEATS}_${RUN_ID}_trials.csv"

echo "========================================="
echo "DeepSeek-R1 Brief Answer Mode Test"
echo "========================================="
echo "Using prompt format: '<problem>\\nAnswer:'"
echo "Stop sequence: \\n"
echo "Max tokens: $N_PRED"
echo "Timeout: ${TIMEOUT_S}s"
echo "Repeats per prompt: $REPEATS"
echo "========================================="
echo ""

python3 src/run_deepseek_r1_brief.py \
  --csv "$CSV_PATH" \
  --out "$OUT_CSV" \
  --trials_out "$TRIALS_CSV" \
  --server_url "$SERVER_URL" \
  --timeout_s "$TIMEOUT_S" \
  --repeats "$REPEATS" \
  --warmup_per_prompt "$WARMUP_PER_PROMPT" \
  --n_pred "$N_PRED"

echo ""
echo "========================================="
echo "Test completed!"
echo "Results saved to:"
echo "  Summary: $OUT_CSV"
echo "  Trials:  $TRIALS_CSV"
echo "========================================="
echo ""

# Calculate and display accuracy
python3 << 'EOFPYTHON'
import csv
import sys

summary_file = sys.argv[1]

with open(summary_file, 'r') as f:
    rows = list(csv.DictReader(f))

# Calculate per-category accuracy
categories = {}
for row in rows:
    cat = row['category']
    acc = float(row['acc'])
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(acc)

print("Category Accuracy:")
print("-" * 40)
total_correct = 0
total_problems = 0
for cat in sorted(categories.keys()):
    accs = categories[cat]
    avg_acc = sum(accs) / len(accs)
    correct = sum(accs)
    total = len(accs)
    total_correct += correct
    total_problems += total
    print(f"  {cat:4s}: {correct:.0f}/{total} = {avg_acc*100:5.1f}%")

overall_acc = total_correct / total_problems if total_problems > 0 else 0
print("-" * 40)
print(f"  TOTAL: {total_correct:.0f}/{total_problems} = {overall_acc*100:5.1f}%")
print()

EOFPYTHON "$OUT_CSV"
