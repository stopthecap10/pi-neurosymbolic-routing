#!/bin/bash
# Smoke test for E8 extraction fix
# Tests 2 AR prompts with debug output to verify prompt construction and model responses

set -e

# Find first 2 AR prompts from the CSV
CSV_FILE="data/industry_tier2_400.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

# Extract first 2 AR rows (using Python to handle multi-line CSV fields)
echo "Creating test CSV with 2 AR prompts..."
python3 create_smoke_test_csv.py

echo ""
echo "Test CSV contents:"
cat /tmp/smoke_test_ar.csv
echo ""

# Run the safe runner with debug mode enabled
echo "========================================="
echo "Running safe runner with --debug mode..."
echo "========================================="
echo ""

python3 src/run_phi2_server_runner_safe.py \
    --csv /tmp/smoke_test_ar.csv \
    --out /tmp/smoke_test_ar_summary.csv \
    --trials_out /tmp/smoke_test_ar_trials.csv \
    --repeats 1 \
    --debug

echo ""
echo "========================================="
echo "Results:"
echo "========================================="
echo ""
echo "Summary:"
cat /tmp/smoke_test_ar_summary.csv
echo ""
echo "Trials (with raw content):"
cat /tmp/smoke_test_ar_trials.csv

echo ""
echo "Smoke test complete!"
