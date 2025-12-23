#!/usr/bin/env python3
import csv, glob
from pathlib import Path
from statistics import median

def latest_completed_run(root="runs_server"):
    runs = sorted(Path(root).glob("*"), reverse=True)
    for r in runs:
        if (r/"baseline_results_trials.csv").exists():
            return r
    raise SystemExit("No completed run found in runs_server/")

def ffloat(x):
    try: return float(x)
    except: return None

def iint(x):
    try: return int(x)
    except: return 0

def main():
    run = latest_completed_run()
    trials_path = run/"baseline_results_trials.csv"

    rows = list(csv.DictReader(open(trials_path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit("No rows in trials CSV.")

    # group by (variant, category)
    groups = {}
    for r in rows:
        v = r.get("variant","")
        c = r.get("category","")
        groups.setdefault((v,c), []).append(r)
        groups.setdefault((v,"ALL"), []).append(r)

    out = []
    for (v,c), rr in sorted(groups.items()):
        acc = sum(iint(x.get("correct","0")) for x in rr) / len(rr) * 100.0
        lats = [ffloat(x.get("latency_wall_s","")) for x in rr]
        lats = [x for x in lats if x is not None]
        med = median(lats) if lats else ""
        out.append({
            "system": "baseline_phi2_server",
            "run_tag": run.name,
            "variant": v,
            "category": c,
            "n": len(rr),
            "accuracy_pct": acc,
            "median_latency_s": med,
        })

    out_path = Path("sheet_summary.csv")
    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(out)

    print("✅ appended to:", out_path)
    print("run_tag:", run.name)
    print("source:", trials_path)

if __name__ == "__main__":
    main()
