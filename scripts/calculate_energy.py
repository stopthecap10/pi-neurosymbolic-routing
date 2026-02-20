#!/usr/bin/env python3
"""
Calculate energy metrics from manual USB power meter readings
Fill in start_mwh and end_mwh in v1_energy_log.csv, then run this script
"""

import csv
from pathlib import Path

def calculate_energy():
    energy_log = Path("outputs/v1_energy_log.csv")

    # Read the log
    rows = []
    with open(energy_log, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Calculate derived fields if start/end are filled
            if row['start_mwh'] and row['end_mwh']:
                start = float(row['start_mwh'])
                end = float(row['end_mwh'])
                delta = end - start
                num_prompts = int(row['num_prompts']) if row['num_prompts'] else 20
                num_inferences = int(row['num_inferences']) if row['num_inferences'] else 60

                row['delta_mwh'] = f"{delta:.2f}"
                row['energy_per_prompt_mwh'] = f"{delta/num_prompts:.2f}"
                row['energy_per_inference_mwh'] = f"{delta/num_inferences:.2f}"

            rows.append(row)

    # Write back
    with open(energy_log, 'w', newline='') as f:
        fieldnames = ['run_date', 'system', 'start_mwh', 'end_mwh', 'delta_mwh',
                     'num_prompts', 'num_inferences', 'energy_per_prompt_mwh',
                     'energy_per_inference_mwh', 'meter_model', 'notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Energy calculations complete!")
    print(f"📊 Results in: {energy_log}")
    print()

    # Print summary
    for row in rows:
        if row.get('delta_mwh'):
            print(f"{row['system']:20s}: {row['delta_mwh']:>8s} mWh total, "
                  f"{row['energy_per_prompt_mwh']:>6s} mWh/prompt")

if __name__ == "__main__":
    calculate_energy()
