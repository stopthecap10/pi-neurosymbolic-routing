#!/usr/bin/env bash
#
# Hybrid V1 Smoke Test
# Tests routing logic with 1 prompt per category
#

set -e

echo "======================================================================"
echo "HYBRID V1 - SMOKE TEST"
echo "======================================================================"
echo ""
echo "This will run Hybrid V1 router with 4 prompts (1 per category)"
echo "Expected duration: ~1 minute"
echo ""
echo "Routing map:"
echo "  AR  → A5 (symbolic direct)"
echo "  ALG → A1 (neural, A4 not implemented yet)"
echo "  WP  → A1 (neural with A2 fallback)"
echo "  LOG → A1 (fast yes/no)"
echo ""
read -p "Press ENTER to continue or Ctrl+C to abort..."
echo ""

# Config paths
CONFIG="configs/run_tier1_smoke.yaml"
CSV="data/splits/tier1_smoke.csv"
OUT_DIR="outputs/smoke"

# Create output directory
mkdir -p "$OUT_DIR"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

python3 src/run_hybrid_v1.py \
  --config "$CONFIG" \
  --csv "$CSV" \
  --out_trials "$OUT_DIR/smoke_hybrid_v1_${TIMESTAMP}.csv"

echo ""
echo "======================================================================"
echo "HYBRID V1 SMOKE TEST COMPLETE"
echo "======================================================================"
echo ""
echo "Next step: Run full Hybrid V1 with:"
echo "   bash scripts/run_hybrid_v1.sh"
