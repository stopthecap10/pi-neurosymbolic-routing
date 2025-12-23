import csv, json
from pathlib import Path

def latest_completed_run() -> Path:
    for r in sorted(Path("runs_server").glob("*"), reverse=True):
        if (r / "baseline_results_trials.csv").exists():
            return r
    raise FileNotFoundError("No completed run found with baseline_results_trials.csv")

latest = latest_completed_run()
trials = latest / "baseline_results_trials.csv"
rows = list(csv.DictReader(open(trials, newline="", encoding="utf-8")))

bad = [r for r in rows if r.get("status") == "ok" and int(r["correct"]) == 0]

print("LATEST (completed):", latest)
print("total trials:", len(rows))
print("incorrect trials:", len(bad))

seen = set()
dedup = []
for r in bad:
    k = (r["id"], r["variant"])
    if k in seen:
        continue
    seen.add(k)
    dedup.append(r)

print("\nIncorrect (dedup by id+variant):", len(dedup))
for r in dedup[:50]:
    print(f'{r["id"]:6s} {r["category"]:4s} {r["variant"]:22s} expected={r["expected_answer"]!r} got={r["final_output"]!r}')

print("\nSample raw JSON for first 3 failures:")
for r in dedup[:3]:
    p = Path(r["log_file"])
    j = json.load(open(p, encoding="utf-8"))
    gs = j.get("generation_settings", {}) or {}
    g = gs.get("grammar", "")
    print(f"\n--- {p.name} ---")
    print("prompt_tail:", j.get("prompt", "")[-160:].replace("\n", "\\n"))
    print("content:", repr(j.get("content", "")))
    print("grammar_on:", bool(g))
    print("grammar_head:", repr(g[:80]))
