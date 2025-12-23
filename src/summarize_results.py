import pandas as pd
from pathlib import Path

# pick latest run that actually has baseline_results.csv
runs = sorted(Path("runs_server").glob("*"), reverse=True)
latest = None
for r in runs:
    if (r / "baseline_results.csv").exists():
        latest = r
        break
if latest is None:
    raise FileNotFoundError("No completed run found with baseline_results.csv")

csv_path = latest / "baseline_results.csv"
df = pd.read_csv(csv_path)

print("Loaded:", csv_path)
print()

for v in sorted(df["variant"].unique()):
    sub = df[df["variant"] == v]
    acc = sub["correct"].mean() * 100
    lat = pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()
    print(f"{v:22s}  acc={acc:5.1f}%   median_wall_s={lat:.2f}")

print("\nPer-category:")
for (v, c), sub in df.groupby(["variant", "category"]):
    acc = sub["correct"].mean() * 100
    lat = pd.to_numeric(sub["latency_wall_median_s"], errors="coerce").median()
    print(f"{v:22s} {c:4s}  acc={acc:5.1f}%   median_wall_s={lat:.2f}")
