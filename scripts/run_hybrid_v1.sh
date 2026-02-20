#!/usr/bin/env bash
#
# Hybrid V1 Full Run
# Runs deterministic rule-based router on official T1 (40 prompts)
#

set -e

echo "======================================================================"
echo "HYBRID V1 - FULL T1 RUN"
echo "======================================================================"
echo ""
echo "This will run:"
echo "  - 40 prompts (official T1)"
echo "  - 3 repeats per prompt"
echo "  - Deterministic rule-based routing"
echo "  - Total: 120 trials"
echo ""
echo "Routing map:"
echo "  AR  → A5 (symbolic direct computation)"
echo "  ALG → A1 (neural, will upgrade to A4 later)"
echo "  WP  → A1 → (fallback) A2"
echo "  LOG → A1 (fast yes/no)"
echo ""
echo "Expected duration: ~10-15 minutes"
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python3 src/run_hybrid_v1.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --split_role official \
  --out_trials "$OUT_DIR/hybrid_v1_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "HYBRID V1 COMPLETE"
echo "======================================================================"
echo ""
echo "Output file: $OUT_DIR/hybrid_v1_${TIMESTAMP}.csv"
echo ""
echo "Next steps:"
echo "  1. Compare Hybrid V1 to baselines:"
echo "     python3 src/compare_hybrid_v1_to_baseline.py --out_dir $OUT_DIR"
echo "  2. If results are good, proceed to Hybrid V2 refinement"
