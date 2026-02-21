#!/usr/bin/env bash
#
# Hybrid V2 Smoke Test
# Quick test to verify A4 routing works correctly
#

set -e

echo "======================================================================"
echo "HYBRID V2 - SMOKE TEST"
echo "======================================================================"
echo ""
echo "This will run:"
echo "  - 4 prompts (1 per category)"
echo "  - 1 repeat per prompt"
echo "  - Total: 4 trials"
echo ""
echo "Routing map (V2):"
echo "  AR  → A5 → A1 → A2 (symbolic direct with fallback)"
echo "  ALG → A4 → A1 → A2 (LLM extract + SymPy with fallback)"
echo "  WP  → A1 → A2 (neural with fallback)"
echo "  LOG → A1 (strict yes/no)"
echo ""
echo "Max escalations: 2"
echo ""
echo "Expected duration: ~30 seconds"
echo ""
echo "IMPORTANT:"
echo "  - Make sure llama.cpp server is running on port 8080"
echo "  - Make sure SymPy is installed: pip3 install sympy"
echo "  - Run on Pi, not Mac"
echo ""
read -p "Press ENTER to continue or Ctrl+C to abort..."
echo ""

# Create smoke test prompts (1 per category)
SMOKE_CSV="data/splits/t1_smoke.csv"

# Check if smoke file exists, if not create it
if [ ! -f "$SMOKE_CSV" ]; then
    echo "Creating smoke test file from industry_tier1_40.csv..."
    head -n 1 data/splits/industry_tier1_40.csv > "$SMOKE_CSV"
    # Get first prompt from each category
    grep "^AR000001," data/splits/industry_tier1_40.csv >> "$SMOKE_CSV"
    grep "^ALG000001," data/splits/industry_tier1_40.csv >> "$SMOKE_CSV"
    grep "^WP000001," data/splits/industry_tier1_40.csv >> "$SMOKE_CSV"
    grep "^LOG000001," data/splits/industry_tier1_40.csv >> "$SMOKE_CSV"
    echo "Created $SMOKE_CSV"
fi

# Config paths
CONFIG="configs/run_tier1.yaml"
OUT_DIR="outputs"

# Temporarily set repeats to 1 for smoke test
python3 -c "
import yaml
with open('$CONFIG', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['repeats'] = 1
with open('$CONFIG.smoke', 'w') as f:
    yaml.dump(cfg, f)
"

python3 src/run_hybrid_v2.py \
  --config "$CONFIG.smoke" \
  --csv "$SMOKE_CSV" \
  --out_trials "$OUT_DIR/hybrid_v2_smoke.csv"

# Cleanup temp config
rm -f "$CONFIG.smoke"

echo ""
echo "======================================================================"
echo "SMOKE TEST COMPLETE"
echo "======================================================================"
echo ""
echo "Output file: outputs/hybrid_v2_smoke.csv"
echo ""
echo "Next steps:"
echo "  1. Review the output to verify routing worked correctly"
echo "  2. Check that ALG trials show A4 routing (symbolic_parse_success, sympy_solve_success)"
echo "  3. If smoke test looks good, run full V2:"
echo "     bash scripts/run_hybrid_v2.sh"
