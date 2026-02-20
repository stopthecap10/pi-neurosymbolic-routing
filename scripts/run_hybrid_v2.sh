#!/usr/bin/env bash
#
# Hybrid V2 Full Run
# Enhanced router with A4 (LLM extract + SymPy solve) for ALG
#

set -e

echo "======================================================================"
echo "HYBRID V2 - FULL T1 RUN"
echo "======================================================================"
echo ""
echo "This will run:"
echo "  - 38 prompts (official T1)"
echo "  - 3 repeats per prompt"
echo "  - Enhanced routing with A4 symbolic solver"
echo "  - Total: 114 trials"
echo ""
echo "Routing map (V2):"
echo "  AR  → A5 → A1 → A2 (symbolic direct with 2-level fallback)"
echo "  ALG → A4 → A1 → A2 (LLM extract + SymPy with 2-level fallback)"
echo "  WP  → A1 → A2 (neural with fallback)"
echo "  LOG → A1 (strict yes/no)"
echo ""
echo "Max escalations: 2"
echo ""
echo "Expected duration: ~15-20 minutes"
echo ""
echo "IMPORTANT:"
echo "  - Make sure llama.cpp server is running on port 8080"
echo "  - Make sure SymPy is installed: pip3 install sympy"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python3 src/run_hybrid_v2.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --split_role official \
  --out_trials "$OUT_DIR/hybrid_v2_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "HYBRID V2 COMPLETE"
echo "======================================================================"
echo ""
echo "Output file: $OUT_DIR/hybrid_v2_${TIMESTAMP}.csv"
echo ""
echo "Next steps:"
echo "  1. Compare V2 to V1:"
echo "     python3 src/compare_v1_vs_v2.py --out_dir $OUT_DIR"
echo "  2. Compare V2 to baseline:"
echo "     python3 src/compare_baseline_vs_v2.py"
echo "  3. Review ALG improvements (expected: 40% → 80%+)"
