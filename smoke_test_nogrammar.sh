#!/bin/bash
# Smoke test for no-grammar runner
# Tests 1 AR and 1 LOG prompt, shows repr(content) and stop_type

set -e

CSV_FILE="data/industry_tier2_400.csv"

if [ ! -f "$CSV_FILE" ]; then
    echo "ERROR: CSV file not found: $CSV_FILE"
    exit 1
fi

# Create test CSV with 1 AR and 1 LOG prompt
echo "Creating test CSV with 1 AR and 1 LOG prompt..."
python3 - <<'PYEOF'
import csv

input_csv = "data/industry_tier2_400.csv"
output_csv = "/tmp/smoke_test_nogrammar.csv"

with open(input_csv, 'r', encoding='utf-8') as fin:
    reader = csv.DictReader(fin)

    ar_rows = []
    log_rows = []

    for row in reader:
        if row['category'] == 'AR' and len(ar_rows) < 1:
            ar_rows.append(row)
        elif row['category'] == 'LOG' and len(log_rows) < 1:
            log_rows.append(row)

        if len(ar_rows) >= 1 and len(log_rows) >= 1:
            break

    with open(output_csv, 'w', encoding='utf-8', newline='') as fout:
        if ar_rows or log_rows:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(ar_rows + log_rows)
            print(f"Created {output_csv} with {len(ar_rows)} AR and {len(log_rows)} LOG prompts")
        else:
            print("No prompts found!")
PYEOF

echo ""
echo "Test CSV contents:"
cat /tmp/smoke_test_nogrammar.csv
echo ""

# Run the no-grammar runner with debug mode enabled
echo "========================================="
echo "Running no-grammar runner with --debug mode..."
echo "========================================="
echo ""

python3 src/run_phi2_server_runner_clean_nogrammar.py \
    --csv /tmp/smoke_test_nogrammar.csv \
    --out /tmp/smoke_test_nogrammar_summary.csv \
    --trials_out /tmp/smoke_test_nogrammar_trials.csv \
    --repeats 1 \
    --debug 2>&1

echo ""
echo "========================================="
echo "Results:"
echo "========================================="
echo ""
echo "Summary:"
cat /tmp/smoke_test_nogrammar_summary.csv
echo ""
echo "Trials (with raw content):"
cat /tmp/smoke_test_nogrammar_trials.csv

echo ""
echo "Smoke test complete!"
