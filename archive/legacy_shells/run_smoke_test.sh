#!/usr/bin/env bash
#
# Smoke Test Runner for V1 Baseline Matrix
# Runs 1 prompt per category across all 4 configs (16 trials total)
#

set -e

echo "======================================================================"
echo "V1 BASELINE MATRIX - SMOKE TEST"
echo "======================================================================"
echo ""
echo "This will run 4 prompts (1 per category) × 4 configs = 16 trials"
echo "Expected duration: ~2-3 minutes"
echo ""
echo "Pass criteria:"
echo "  - No crashes"
echo "  - All rows written correctly"
echo "  - Energy delta present and non-negative"
echo "  - Parser works (no E8 unless legitimate parse failure)"
echo "  - Timeout and error codes set correctly"
echo ""
read -p "Press ENTER to continue or Ctrl+C to abort..."
echo ""

# Config paths
CONFIG="configs/run_tier1_smoke.yaml"
CSV="data/splits/tier1_smoke.csv"
OUT_DIR="outputs/smoke"

# Create output directory
mkdir -p "$OUT_DIR"

# Timestamp for this smoke test run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "CONFIG 1/4: baseline_a1_grammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --grammar \
  --out_trials "$OUT_DIR/smoke_a1_grammar_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "CONFIG 2/4: baseline_a1_nogrammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --out_trials "$OUT_DIR/smoke_a1_nogrammar_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "CONFIG 3/4: baseline_a2_grammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --grammar \
  --out_trials "$OUT_DIR/smoke_a2_grammar_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "CONFIG 4/4: baseline_a2_nogrammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --out_trials "$OUT_DIR/smoke_a2_nogrammar_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "SMOKE TEST COMPLETE"
echo "======================================================================"
echo ""
echo "Validating results..."
echo ""

# Simple validation
total_files=0
total_rows=0
for f in "$OUT_DIR"/smoke_*_${TIMESTAMP}.csv; do
  if [ -f "$f" ]; then
    total_files=$((total_files + 1))
    rows=$(tail -n +2 "$f" | wc -l | tr -d ' ')
    total_rows=$((total_rows + rows))
    echo "✓ $(basename "$f"): $rows trials"
  fi
done

echo ""
if [ "$total_files" -eq 4 ] && [ "$total_rows" -eq 16 ]; then
  echo "✅ SMOKE TEST PASSED"
  echo "   - All 4 configs completed"
  echo "   - All 16 trials recorded (4 prompts × 4 configs × 1 repeat)"
  echo ""
  echo "Next step: Run full T1 baseline matrix with:"
  echo "   bash scripts/run_baseline_matrix.sh"
else
  echo "❌ SMOKE TEST FAILED"
  echo "   - Expected: 4 files, 16 rows"
  echo "   - Got: $total_files files, $total_rows rows"
  exit 1
fi
