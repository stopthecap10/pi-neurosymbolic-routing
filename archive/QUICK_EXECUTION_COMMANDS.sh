#!/usr/bin/env bash
#
# OFFICIAL SPLIT BUILD & RERUN - Command Sequence
# Copy-paste these blocks in order
#
# Status: dev → official protocol transition
# Date: 2026-02-12
#

set -e

echo "=========================================================================="
echo "OFFICIAL SPLIT BUILD & RERUN - EXECUTION SEQUENCE"
echo "=========================================================================="
echo ""
echo "This script contains all commands needed to transition from"
echo "tier1_mini.csv (dev) → industry_tier1_40.csv (official protocol)"
echo ""
echo "Execute blocks in order, checking output after each block."
echo ""
echo "=========================================================================="

# ============================================================================
# BLOCK 1: SETUP (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 1: SETUP & VALIDATION (ON MAC)"
echo "========================================================================"
echo ""

cd ~/Documents/pi-neurosymbolic-routing

# Install datasets library
pip3 install datasets

# Validate dataset access
python3 src/validate_dataset_builder.py

# Expected: ✅ ALL CHECKS PASSED

# ============================================================================
# BLOCK 2: BUILD OFFICIAL SPLITS (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 2: BUILD OFFICIAL SPLITS (ON MAC)"
echo "========================================================================"
echo ""

# Build all three tiers
python3 src/build_official_splits.py --out_dir data/splits --seed 42

# Validate output
wc -l data/splits/industry_tier1_40.csv    # Should be 41 (40 + header)
wc -l data/splits/industry_tier2_200.csv   # Should be 201
wc -l data/splits/industry_tier3_300.csv   # Should be 301

# Check provenance
head -2 data/splits/industry_tier1_40.csv

# Verify source_type
cut -d',' -f8 data/splits/industry_tier1_40.csv | sort -u
# Should show: "source_type" and "dataset_raw" only

# Check category distribution
tail -n +2 data/splits/industry_tier1_40.csv | cut -d',' -f2 | sort | uniq -c
# Should show: 10 ALG, 10 AR, 10 LOG, 10 WP

echo ""
echo "✓ If all checks pass, proceed to BLOCK 3"
echo ""

# ============================================================================
# BLOCK 3: ARCHIVE DEV ARTIFACTS (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 3: ARCHIVE DEV ARTIFACTS (ON MAC)"
echo "========================================================================"
echo ""

# Create archive structure
mkdir -p archive/tier1_mini_dev/{splits,outputs,notes}

# Copy dev files (COPY first, not move)
cp data/splits/tier1_mini.csv archive/tier1_mini_dev/splits/
cp outputs/hybrid_v1*.csv archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/hybrid_v1*.md archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/v1_*.csv archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/v1_*.md archive/tier1_mini_dev/notes/ 2>/dev/null || true

# Create archive README
cat > archive/tier1_mini_dev/README.md << 'EOF'
# Dev Dataset Results (tier1_mini.csv)

**Status**: Development/debugging dataset (NOT for final ISEF claims)

tier1_mini.csv: 20 manually created prompts (5 per category)
- Useful for fast iteration during development
- Violates Section 3 protocol (no HuggingFace provenance)

Official splits for final claims:
- industry_tier1_40.csv (40 prompts, full provenance)
- industry_tier2_200.csv (200 prompts)
- industry_tier3_300.csv (300 prompts)
EOF

# Verify archive
ls -R archive/tier1_mini_dev/

echo ""
echo "✓ Dev artifacts archived. Proceed to BLOCK 4"
echo ""

# ============================================================================
# BLOCK 4: SYNC TO PI (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 4: SYNC OFFICIAL SPLITS TO PI (ON MAC)"
echo "========================================================================"
echo ""

# Sync all official files
scp data/splits/industry_tier1_40.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/industry_tier2_200.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/industry_tier3_300.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/split_manifest.json stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/split_build_report.md stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/

echo ""
echo "✓ Files synced to Pi. Now SSH to Pi for BLOCK 5"
echo ""

# ============================================================================
# BLOCK 5: VERIFY ON PI (ON PI)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 5: VERIFY ON PI (ON PI)"
echo "========================================================================"
echo ""
echo "SSH to Pi and run:"
echo ""
echo "ssh stopthecap10@edgeai"
echo "cd ~/pi-neurosymbolic-routing"
echo ""
echo "# Verify files"
echo "ls -lh data/splits/industry_tier*.csv"
echo ""
echo "# Check counts"
echo "wc -l data/splits/industry_tier1_40.csv"
echo "# Should be: 41"
echo ""
echo "# Check first row"
echo "head -2 data/splits/industry_tier1_40.csv"
echo ""
echo "# Verify server running"
echo "curl http://127.0.0.1:8080/health"
echo ""
echo "# If server not running:"
echo "tmux new -s phi4 -d 'cd ~/llama.cpp && ./build/bin/llama-server -m /home/stopthecap10/edge-ai/models/microsoft_Phi-4-mini-instruct-Q6_K.gguf -c 2048 -t 4 --port 8080'"
echo ""

# ============================================================================
# BLOCK 6: UPDATE SCRIPTS (ON PI)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 6: UPDATE SCRIPTS TO USE OFFICIAL T1 (ON PI)"
echo "========================================================================"
echo ""
echo "On Pi, edit these scripts to change CSV path:"
echo ""
echo "scripts/run_baseline_matrix.sh"
echo "scripts/run_hybrid_v1.sh"
echo "scripts/run_hybrid_v2.sh"
echo ""
echo "Change:"
echo "  FROM: CSV=\"data/splits/tier1_mini.csv\""
echo "  TO:   CSV=\"data/splits/industry_tier1_40.csv\""
echo ""
echo "Or run this sed command:"
echo ""
echo "sed -i.bak 's|tier1_mini.csv|industry_tier1_40.csv|g' scripts/run_baseline_matrix.sh"
echo "sed -i.bak 's|tier1_mini.csv|industry_tier1_40.csv|g' scripts/run_hybrid_v1.sh"
echo "sed -i.bak 's|tier1_mini.csv|industry_tier1_40.csv|g' scripts/run_hybrid_v2.sh"
echo ""
echo "Verify:"
echo "grep CSV= scripts/run_baseline_matrix.sh"
echo ""

# ============================================================================
# BLOCK 7: RUN OFFICIAL BASELINE (ON PI)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 7: RUN OFFICIAL BASELINE MATRIX (ON PI)"
echo "========================================================================"
echo ""
echo "This will take ~2-3 hours"
echo ""
echo "cd ~/pi-neurosymbolic-routing"
echo "mkdir -p outputs/official"
echo ""
echo "# Run baseline matrix (4 configs × 40 prompts × 3 repeats = 480 trials)"
echo "bash scripts/run_baseline_matrix.sh"
echo ""
echo "Expected outputs:"
echo "  - outputs/v1_a1_grammar.csv"
echo "  - outputs/v1_a1_nogrammar.csv"
echo "  - outputs/v1_a2_grammar.csv"
echo "  - outputs/v1_a2_nogrammar.csv"
echo ""
echo "Move to outputs/official/:"
echo "mv outputs/v1_*.csv outputs/official/"
echo ""

# ============================================================================
# BLOCK 8: GENERATE BASELINE SUMMARY (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 8: GENERATE BASELINE SUMMARY (ON MAC)"
echo "========================================================================"
echo ""
echo "Sync results from Pi:"
echo ""
echo "scp stopthecap10@edgeai:~/pi-neurosymbolic-routing/outputs/official/v1_*.csv outputs/official/"
echo ""
echo "# Generate summary (you may need to create/adapt this script)"
echo "python3 src/summarize_v1_matrix.py --input_dir outputs/official/"
echo ""

# ============================================================================
# BLOCK 9: RUN HYBRID V1 (ON PI)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 9: RUN HYBRID V1 ON OFFICIAL T1 (ON PI)"
echo "========================================================================"
echo ""
echo "cd ~/pi-neurosymbolic-routing"
echo ""
echo "# Smoke test first"
echo "bash scripts/run_hybrid_v1_smoke.sh"
echo ""
echo "# Full run (40 prompts × 3 repeats)"
echo "bash scripts/run_hybrid_v1.sh"
echo ""
echo "# Move output to official/"
echo "mv outputs/hybrid_v1.csv outputs/official/"
echo ""

# ============================================================================
# BLOCK 10: RUN HYBRID V2 (ON PI)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 10: RUN HYBRID V2 ON OFFICIAL T1 (ON PI)"
echo "========================================================================"
echo ""
echo "cd ~/pi-neurosymbolic-routing"
echo ""
echo "# Smoke test first (verify A4 latency < 10s)"
echo "bash scripts/run_hybrid_v2_smoke.sh"
echo ""
echo "# Full run"
echo "bash scripts/run_hybrid_v2.sh"
echo ""
echo "# Move output to official/"
echo "mv outputs/hybrid_v2.csv outputs/official/"
echo ""

# ============================================================================
# BLOCK 11: GENERATE COMPARISONS (ON MAC)
# ============================================================================

echo ""
echo "========================================================================"
echo "BLOCK 11: GENERATE COMPARISONS (ON MAC)"
echo "========================================================================"
echo ""
echo "# Sync all results from Pi"
echo "scp stopthecap10@edgeai:~/pi-neurosymbolic-routing/outputs/official/*.csv outputs/official/"
echo ""
echo "# Generate V1 vs V2 comparison"
echo "python3 src/compare_v1_vs_v2.py"
echo ""
echo "# Generate baseline vs V1"
echo "python3 src/compare_baseline_vs_v1.py"
echo ""

# ============================================================================
# COMPLETION
# ============================================================================

echo ""
echo "========================================================================"
echo "COMPLETION CHECKLIST"
echo "========================================================================"
echo ""
echo "Official splits built and validated:"
echo "  [ ] industry_tier1_40.csv (40 prompts)"
echo "  [ ] industry_tier2_200.csv (200 prompts)"
echo "  [ ] industry_tier3_300.csv (300 prompts)"
echo "  [ ] split_manifest.json"
echo "  [ ] split_build_report.md"
echo ""
echo "Dev artifacts archived:"
echo "  [ ] archive/tier1_mini_dev/ created"
echo "  [ ] tier1_mini.csv archived"
echo "  [ ] Dev results archived"
echo ""
echo "Official runs completed:"
echo "  [ ] Baseline matrix on official T1"
echo "  [ ] Hybrid V1 on official T1"
echo "  [ ] Hybrid V2 on official T1"
echo ""
echo "Documentation updated:"
echo "  [ ] README.md explains split transition"
echo "  [ ] Master Plan Section 3 marked complete"
echo ""
echo "Next steps:"
echo "  1. Freeze official T1 results"
echo "  2. Move to T2 (200 prompts) for V3/V4"
echo "  3. Reserve T3 (300 prompts) for final eval"
echo ""
echo "========================================================================"
