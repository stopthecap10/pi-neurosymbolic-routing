#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8080/completion}"

echo "[0/4] Checking llama-server is reachable at $SERVER_URL ..."
if ! curl -sS -m 2 "$SERVER_URL" >/dev/null 2>&1; then
  echo "means the server isn't running."
  echo "Start it in another terminal first, e.g.:"
  echo "  ~/llama.cpp/build/bin/llama-server -m ~/edge-ai/models/phi-2-Q4_K_M.gguf -t 4 -c 512 --host 127.0.0.1 --port 8080"
  exit 1
fi
echo "server reachable"

echo "[1/4] Running baseline Phi-2 server (grammar + no-grammar)..."
python3 src/run_baseline_phi2_server.py

echo "[2/4] Running hybrid v1..."
python3 src/run_hybrid_v1.py

echo "[3/4] Running hybrid v2..."
python3 src/run_hybrid_v2.py

echo "[4/4] Building sheet summary CSV..."
python3 scripts/make_sheet_table.py

echo " Done. Upload sheet_summary.csv to Google Sheets."
