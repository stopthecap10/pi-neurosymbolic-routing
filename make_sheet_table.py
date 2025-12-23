from pathlib import Path
import pandas as pd

def latest_run(dir_name: str, file_name: str) -> Path:
    d = Path(dir_name)
    for r in sorted(d.glob("*"), reverse=True):
        if (r / file_name).exists():
            return r
    raise FileNotFoundError(f"No completed run in {dir_name} containing {file_name}")

rows = []

# --- Baseline (runs_server) ---
base_run = latest_run("runs_server", "baseline_results.csv")
base = pd.read_csv(base_run / "baseline_results.csv")
for (variant, cat), sub in base.groupby(["variant", "category"]):
    rows.append({
        "system": "baseline_phi2_server",
        "run_tag": base_run.name,
        "variant": variant,
        "category": cat,
        "n": int(len(sub)),
        "accuracy_pct": float(sub["correct"].mean() * 100),
        "median_latency_s": float(pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()),
    })
# overall per variant
for variant, sub in base.groupby("variant"):
    rows.append({
        "system": "baseline_phi2_server",
        "run_tag": base_run.name,
        "variant": variant,
        "category": "ALL",
        "n": int(len(sub)),
        "accuracy_pct": float(sub["correct"].mean() * 100),
        "median_latency_s": float(pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()),
    })

# --- Hybrid v1 ---
h1_run = latest_run("runs_hybrid_v1", "hybrid_results.csv")
h1 = pd.read_csv(h1_run / "hybrid_results.csv")
for cat, sub in h1.groupby("category"):
    rows.append({
        "system": "hybrid_v1",
        "run_tag": h1_run.name,
        "variant": "hybrid_v1",
        "category": cat,
        "n": int(len(sub)),
        "accuracy_pct": float(sub["correct"].mean() * 100),
        "median_latency_s": float(pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()),
    })
rows.append({
    "system": "hybrid_v1",
    "run_tag": h1_run.name,
    "variant": "hybrid_v1",
    "category": "ALL",
    "n": int(len(h1)),
    "accuracy_pct": float(h1["correct"].mean() * 100),
    "median_latency_s": float(pd.to_numeric(h1["latency_wall_median_s"], errors="coerce").median()),
})

# --- Hybrid v2 ---
h2_run = latest_run("runs_hybrid_v2", "hybrid_v2_results.csv")
h2 = pd.read_csv(h2_run / "hybrid_v2_results.csv")
for cat, sub in h2.groupby("category"):
    rows.append({
        "system": "hybrid_v2",
        "run_tag": h2_run.name,
        "variant": "hybrid_v2",
        "category": cat,
        "n": int(len(sub)),
        "accuracy_pct": float(sub["correct"].mean() * 100),
        "median_latency_s": float(pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()),
    })
rows.append({
    "system": "hybrid_v2",
    "run_tag": h2_run.name,
    "variant": "hybrid_v2",
    "category": "ALL",
    "n": int(len(h2)),
    "accuracy_pct": float(h2["correct"].mean() * 100),
    "median_latency_s": float(pd.to_numeric(h2["latency_wall_median_s"], errors="coerce").median()),
})

out = pd.DataFrame(rows).sort_values(["system","variant","category"])
out_path = Path("sheet_summary.csv")
out.to_csv(out_path, index=False)

print("✅ wrote", out_path)
print("\n--- preview ---")
print(out.to_string(index=False))
