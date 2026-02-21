#!/usr/bin/env bash
#
# Full T1 Baseline Matrix Runner
# Runs 40 prompts × 3 repeats × 4 configs = 480 trials
# Then generates summary, runs Hybrid V1, and comparison
#

set -e

echo "======================================================================"
echo "V1 BASELINE MATRIX - FULL T1 RUN"
echo "======================================================================"
echo ""
echo "This will run:"
echo "  - 40 prompts (official T1 v2, balanced)"
echo "  - 3 repeats per prompt"
echo "  - 4 configs (A1/A2 × grammar/no-grammar)"
echo "  - Total: 480 baseline trials + 120 hybrid V1 trials"
echo ""

# Config paths
CONFIG="configs/run_tier1.yaml"
CSV="data/splits/industry_tier1_40_v2.csv"
OUT_DIR="outputs/official"

# Create output directory
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "CONFIG 1/4: baseline_a1_nogrammar"
echo "======================================================================"
echo "skip
skip" | python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a1_nogrammar.csv"

echo ""
echo "Config 1/4 complete."
echo ""

echo "======================================================================"
echo "CONFIG 2/4: baseline_a1_grammar"
echo "======================================================================"
echo "skip
skip" | python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A1 \
  --grammar \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a1_grammar.csv"

echo ""
echo "Config 2/4 complete."
echo ""

echo "======================================================================"
echo "CONFIG 3/4: baseline_a2_nogrammar"
echo "======================================================================"
echo "skip
skip" | python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a2_nogrammar.csv"

echo ""
echo "Config 3/4 complete."
echo ""

echo "======================================================================"
echo "CONFIG 4/4: baseline_a2_grammar"
echo "======================================================================"
echo "skip
skip" | python3 src/run_v1_baseline_matrix.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --action A2 \
  --grammar \
  --split_role official \
  --out_trials "$OUT_DIR/v1_a2_grammar.csv"

echo ""
echo "======================================================================"
echo "BASELINE MATRIX COMPLETE - GENERATING SUMMARY"
echo "======================================================================"
echo ""

python3 src/summarize_v1_matrix.py --out_dir "$OUT_DIR"

echo ""
echo "======================================================================"
echo "RUNNING HYBRID V1"
echo "======================================================================"
echo ""

echo "skip
skip" | python3 src/run_hybrid_v1.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --out_trials "$OUT_DIR/hybrid_v1.csv" \
  --split_role official

echo ""
echo "======================================================================"
echo "GENERATING COMPARISON"
echo "======================================================================"
echo ""

python3 src/compare_hybrid_v1_to_baseline.py --out_dir "$OUT_DIR"

echo ""
echo "======================================================================"
echo "ALL DONE"
echo "======================================================================"
echo ""
echo "Output files:"
echo "  - $OUT_DIR/v1_a1_nogrammar.csv"
echo "  - $OUT_DIR/v1_a1_grammar.csv"
echo "  - $OUT_DIR/v1_a2_nogrammar.csv"
echo "  - $OUT_DIR/v1_a2_grammar.csv"
echo "  - $OUT_DIR/t1_baseline_summary.md"
echo "  - $OUT_DIR/v1_decisions_for_hybrid.md"
echo "  - $OUT_DIR/hybrid_v1.csv"
echo "  - $OUT_DIR/v1_vs_baseline.md"
