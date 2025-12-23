#!/usr/bin/env python3
import argparse, csv, os, sys
from collections import defaultdict, Counter
from statistics import median

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pick_existing(row, keys):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return ""

def median_of(rows, key_candidates):
    vals = []
    for r in rows:
        v = as_float(pick_existing(r, key_candidates))
        if v is not None:
            vals.append(v)
    return median(vals) if vals else None

def normalize_bool(x):
    if x is None: return 0
    s = str(x).strip().lower()
    if s in ("1","true","t","yes","y"): return 1
    return 0

def majority_output(rows):
    outs = [r.get("final_output","") for r in rows if r.get("final_output","") != ""]
    if not outs:
        return ""
    c = Counter(outs)
    # deterministic tie-break
    best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best

def majority_correct(rows):
    # if "correct" exists, compute majority of outputs then compare to expected when available
    maj = majority_output(rows)
    if maj == "":
        return 0
    expected = rows[0].get("expected_answer","")
    return 1 if str(maj).strip() == str(expected).strip() else 0

def index_by_id(rows):
    by = defaultdict(list)
    for r in rows:
        _id = r.get("id","").strip()
        if _id:
            by[_id].append(r)
    return by

def load_prompts(prompts_csv):
    prompts = {}
    for r in read_csv(prompts_csv):
        _id = r.get("id","").strip()
        if not _id:
            continue
        prompts[_id] = {
            "category": r.get("category","").strip(),
            "prompt": r.get("prompt","").strip(),
            "expected_answer": r.get("expected_answer","").strip(),
        }
    return prompts

def resolve_trials_csv(arg):
    # accept either a direct CSV path or a run directory containing *_trials.csv
    if os.path.isfile(arg):
        return arg
    if os.path.isdir(arg):
        # try common names
        candidates = [
            os.path.join(arg, "baseline_results_trials.csv"),
            os.path.join(arg, "hybrid_results_trials.csv"),
            os.path.join(arg, "hybrid_v2_results_trials.csv"),
            os.path.join(arg, "hybrid_v1_results_trials.csv"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        # fallback: any *trials*.csv
        for fn in os.listdir(arg):
            if "trials" in fn and fn.endswith(".csv"):
                return os.path.join(arg, fn)
    raise FileNotFoundError(f"Could not find trials CSV from: {arg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompts CSV (e.g. baseline_prompts_tier2_400.csv)")
    ap.add_argument("--baseline", required=True, help="baseline trials CSV or run dir (runs_server/<TAG>)")
    ap.add_argument("--hybrid_v1", required=True, help="hybrid v1 trials CSV or run dir (runs_hybrid_v1/<TAG>)")
    ap.add_argument("--hybrid_v2", required=True, help="hybrid v2 trials CSV or run dir (runs_hybrid_v2/<TAG>)")
    ap.add_argument("--out", default="case_studies_candidates.csv")
    ap.add_argument("--top", type=int, default=20, help="top K per case type")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)

    base_path = resolve_trials_csv(args.baseline)
    v1_path   = resolve_trials_csv(args.hybrid_v1)
    v2_path   = resolve_trials_csv(args.hybrid_v2)

    base_rows = read_csv(base_path)
    v1_rows   = read_csv(v1_path)
    v2_rows   = read_csv(v2_path)

    base_by = index_by_id(base_rows)
    v1_by   = index_by_id(v1_rows)
    v2_by   = index_by_id(v2_rows)

    all_ids = sorted(set(base_by) | set(v1_by) | set(v2_by))

    # Build per-id summary
    summary = {}
    for _id in all_ids:
        p = prompts.get(_id, {"category":"", "prompt":"", "expected_answer":""})
        base = base_by.get(_id, [])
        v1   = v1_by.get(_id, [])
        v2   = v2_by.get(_id, [])

        base_out = majority_output(base)
        base_ok  = majority_correct(base) if base else 0
        base_lat = median_of(base, ["latency_wall_median_s","latency_wall_s","latency_wall","wall_s","latency_s"])

        v1_out = majority_output(v1)
        v1_ok  = majority_correct(v1) if v1 else 0
        v1_lat = median_of(v1, ["latency_wall_median_s","latency_wall_s","latency_wall","wall_s","latency_s"])

        v2_out = majority_output(v2)
        v2_ok  = majority_correct(v2) if v2 else 0
        v2_lat = median_of(v2, ["latency_wall_median_s","latency_wall_s","latency_wall","wall_s","latency_s"])

        # extra fields if present (grab first row)
        v2_route = v2[0].get("route","") if v2 else ""
        v2_sym_expr = pick_existing(v2[0], ["sym_expr","sympy_expr","verifier_expr","expr"]) if v2 else ""
        v2_llm_answer = pick_existing(v2[0], ["llm_answer","llm_output","raw_llm_answer"]) if v2 else ""

        summary[_id] = {
            "id": _id,
            "category": p.get("category",""),
            "prompt": p.get("prompt",""),
            "expected_answer": p.get("expected_answer",""),
            "baseline_output": base_out,
            "baseline_correct": base_ok,
            "baseline_latency_s": "" if base_lat is None else f"{base_lat:.5f}",
            "hybrid_v1_output": v1_out,
            "hybrid_v1_correct": v1_ok,
            "hybrid_v1_latency_s": "" if v1_lat is None else f"{v1_lat:.5f}",
            "hybrid_v2_output": v2_out,
            "hybrid_v2_correct": v2_ok,
            "hybrid_v2_latency_s": "" if v2_lat is None else f"{v2_lat:.5f}",
            "hybrid_v2_route": v2_route,
            "hybrid_v2_sym_expr": v2_sym_expr,
            "hybrid_v2_llm_answer": v2_llm_answer,
        }

    # Case type builders
    def score_delta_lat(row):
        a = as_float(row["hybrid_v2_latency_s"])
        b = as_float(row["hybrid_v1_latency_s"])
        if a is None or b is None:
            return None
        return a - b

    baseline_wrong_v2_right = []
    v1_wrong_v2_right = []
    all_fail = []
    v2_fail = []
    latency_tradeoff = []

    for _id, r in summary.items():
        if r["baseline_correct"] == 0 and r["hybrid_v2_correct"] == 1:
            baseline_wrong_v2_right.append(r)
        if r["hybrid_v1_correct"] == 0 and r["hybrid_v2_correct"] == 1:
            v1_wrong_v2_right.append(r)
        if r["baseline_correct"] == 0 and r["hybrid_v1_correct"] == 0 and r["hybrid_v2_correct"] == 0:
            all_fail.append(r)
        if r["hybrid_v2_correct"] == 0:
            v2_fail.append(r)
        if r["hybrid_v1_correct"] == 1 and r["hybrid_v2_correct"] == 1:
            d = score_delta_lat(r)
            if d is not None:
                rr = dict(r)
                rr["delta_latency_s"] = f"{d:.5f}"
                latency_tradeoff.append(rr)

    # Ranking heuristics
    # - prioritize LOG for failures
    def cat_boost(r):
        return 0 if r.get("category","") == "LOG" else 1

    baseline_wrong_v2_right = sorted(baseline_wrong_v2_right, key=lambda r: (cat_boost(r), r["id"]))[:args.top]
    v1_wrong_v2_right       = sorted(v1_wrong_v2_right, key=lambda r: (cat_boost(r), r["id"]))[:args.top]
    all_fail                = sorted(all_fail, key=lambda r: (cat_boost(r), r["id"]))[:args.top]
    v2_fail                 = sorted(v2_fail, key=lambda r: (cat_boost(r), r["id"]))[:args.top]
    latency_tradeoff        = sorted(latency_tradeoff, key=lambda r: -float(r["delta_latency_s"]))[:args.top]

    # Write one combined CSV
    out_rows = []
    def add_block(case_type, rows, note_template):
        for r in rows:
            rr = dict(r)
            rr["case_type"] = case_type
            rr["suggested_note"] = note_template
            out_rows.append(rr)

    add_block("BASELINE_WRONG__V2_CORRECT", baseline_wrong_v2_right,
              "Baseline wrong, Hybrid v2 correct (good poster example).")
    add_block("V1_WRONG__V2_CORRECT", v1_wrong_v2_right,
              "Hybrid v1 wrong, Hybrid v2 correct (upgrade from v1).")
    add_block("ALL_FAIL_HARD_CASE", all_fail,
              "All systems fail (use as known limitation; often LOG).")
    add_block("V2_FAIL", v2_fail,
              "Hybrid v2 fails (use as remaining weakness example; check LOG).")
    add_block("LATENCY_TRADEOFF_BOTH_CORRECT", latency_tradeoff,
              "Both correct; shows accuracy/latency tradeoff (v2 slower/faster).")

    # ensure columns exist
    cols = [
        "case_type","id","category","prompt","expected_answer",
        "baseline_output","baseline_correct","baseline_latency_s",
        "hybrid_v1_output","hybrid_v1_correct","hybrid_v1_latency_s",
        "hybrid_v2_output","hybrid_v2_correct","hybrid_v2_latency_s",
        "delta_latency_s",
        "hybrid_v2_route","hybrid_v2_sym_expr","hybrid_v2_llm_answer",
        "suggested_note"
    ]
    for r in out_rows:
        for c in cols:
            if c not in r:
                r[c] = ""

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print("✅ wrote", args.out)
    print("Inputs:")
    print("  baseline:", base_path)
    print("  hybrid_v1:", v1_path)
    print("  hybrid_v2:", v2_path)
    print()
    # quick counts
    print("Counts (all IDs scanned):", len(all_ids))
    print("  BASELINE_WRONG__V2_CORRECT:", len(baseline_wrong_v2_right))
    print("  V1_WRONG__V2_CORRECT:", len(v1_wrong_v2_right))
    print("  ALL_FAIL_HARD_CASE:", len(all_fail))
    print("  V2_FAIL:", len(v2_fail))
    print("  LATENCY_TRADEOFF_BOTH_CORRECT:", len(latency_tradeoff))

if __name__ == "__main__":
    main()
