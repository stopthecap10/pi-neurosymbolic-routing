#!/usr/bin/env python3
"""Extract first 2 AR prompts for smoke testing."""
import csv

input_csv = "data/industry_tier2_400.csv"
output_csv = "/tmp/smoke_test_ar.csv"

with open(input_csv, 'r', encoding='utf-8') as fin:
    reader = csv.DictReader(fin)

    ar_rows = []
    for row in reader:
        if row['category'] == 'AR':
            ar_rows.append(row)
            if len(ar_rows) >= 2:
                break

    with open(output_csv, 'w', encoding='utf-8', newline='') as fout:
        if ar_rows:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(ar_rows)
            print(f"Created {output_csv} with {len(ar_rows)} AR prompts")
        else:
            print("No AR prompts found!")
