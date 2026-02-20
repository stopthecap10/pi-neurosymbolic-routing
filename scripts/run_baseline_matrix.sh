#!/usr/bin/env bash
#
# Full T1 Baseline Matrix Runner
# Runs all 20 prompts × 3 repeats × 4 configs = 240 trials
#

set -e

echo "======================================================================"
echo "V1 BASELINE MATRIX - FULL T1 RUN"
echo "======================================================================"
echo ""
echo "This will run:"
echo "  - 38 prompts (official T1)"
echo "  - 3 repeats per prompt"
echo "  - 4 configs (A1/A2 × grammar/no-grammar)"
echo "  - Total: 456 trials"
echo ""
echo "Expected duration: ~60-120 minutes"
echo ""
echo "IMPORTANT:"
echo "  - Make sure llama.cpp server is running on port 8080"
echo "  - Have USB power meter ready for energy readings"
echo "  - Run on Pi, not Mac"
echo ""
read -p "Press ENTER to continue or Ctrl+C to abort..."
echo ""

# Config paths
CONFIG="configs/run_tier1.yaml"
CSV="data/splits/industry_tier1_40.csv"
OUT_DIR="outputs/official"

# Create output directory
mkdir -p "$OUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "======================================================================"
echo "CONFIG 1/4: baseline_a1_grammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --grammar \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a1_grammar_${TIMESTAMP}.csv"

echo ""
echo "Config 1/4 complete. Take a short break if needed."
echo ""
read -p "Press ENTER to continue to config 2/4..."
echo ""

echo "======================================================================"
echo "CONFIG 2/4: baseline_a1_nogrammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a1_nogrammar_${TIMESTAMP}.csv"

echo ""
echo "Config 2/4 complete. Take a short break if needed."
echo ""
read -p "Press ENTER to continue to config 3/4..."
echo ""

echo "======================================================================"
echo "CONFIG 3/4: baseline_a2_grammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --grammar \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a2_grammar_${TIMESTAMP}.csv"

echo ""
echo "Config 3/4 complete. Take a short break if needed."
echo ""
read -p "Press ENTER to continue to config 4/4 (final)..."
echo ""

echo "======================================================================"
echo "CONFIG 4/4: baseline_a2_nogrammar"
echo "======================================================================"
python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a2_nogrammar_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "FULL T1 BASELINE MATRIX COMPLETE"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - $OUT_DIR/v1_a1_grammar_${TIMESTAMP}.csv"
echo "  - $OUT_DIR/v1_a1_nogrammar_${TIMESTAMP}.csv"
echo "  - $OUT_DIR/v1_a2_grammar_${TIMESTAMP}.csv"
echo "  - $OUT_DIR/v1_a2_nogrammar_${TIMESTAMP}.csv"
echo ""
echo "Next step: Generate summary with:"
echo "   python3 src/summarize_v1_matrix.py --out_dir $OUT_DIR"
