#!/usr/bin/env bash
set -euo pipefail

# Archive existing outputs/energy_log.csv (if any) and create a fresh file with header

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ARCHIVE_DIR="outputs/_archive"
LOG_PATH="outputs/energy_log.csv"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$ARCHIVE_DIR"

if [ -f "$LOG_PATH" ]; then
  mv "$LOG_PATH" "${ARCHIVE_DIR}/energy_log.csv.${TIMESTAMP}"
  echo "Archived existing log to ${ARCHIVE_DIR}/energy_log.csv.${TIMESTAMP}"
else
  echo "No existing energy_log.csv found; creating a fresh one."
fi

python3 - <<'PY'
from pathlib import Path
import csv

log_path = Path("outputs/energy_log.csv")
log_path.parent.mkdir(parents=True, exist_ok=True)
with log_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "run_id","system","tier","csv_used","model_file","repeats",
        "start_mWh","end_mWh","elapsed_s","exit_code","out_csv","trials_csv"
    ])
print(f"Wrote fresh header to {log_path}")
PY
