import argparse, csv, time, re, sys
from pathlib import Path
import requests
from statistics import median

INT_RE = re.compile(r"[-+]?\d+")

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def extract_last_int(text: str) -> str:
    if not text:
        return ""
    # Remove commas just in case
    nums = INT_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else ""

def extract_yesno(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    if "yes" in t:
        return "Yes"
    if "no" in t:
        return "No"
    return ""

def err_code(category: str, pred: str, expected: str, timed_out: bool) -> str:
    if timed_out:
        return "E7"
    if pred == "":
        return "E8"
    if pred == expected:
        return "E0"
    c = category.upper()
    if c == "AR":
        return "E1"
    if c == "ALG":
        return "E3"
    if c == "LOG":
        return "E2"
    if c == "WP":
        return "E5"
    return "E8"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials_out", required=True)
    ap.add_argument("--server_url", default="http://127.0.0.1:8080/completion")
    ap.add_argument("--timeout_s", type=float, default=20.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup_per_prompt", type=int, default=0)
    ap.add_argument("--n_pred_num", type=int, default=32)
    ap.add_argument("--n_pred_log", type=int, default=8)
    ap.add_argument("--num_grammar_file", default="grammars/grammar_int_only_strict.gbnf")
    ap.add_argument("--yesno_grammar_file", default="grammars/grammar_yesno_only.gbnf")
    args = ap.parse_args()

    # Validate grammar files exist
    num_path = Path(args.num_grammar_file)
    yesno_path = Path(args.yesno_grammar_file)
    if not num_path.exists():
        print(f"ERROR: num_grammar_file not found: {args.num_grammar_file}", file=sys.stderr)
        sys.exit(1)
    if not yesno_path.exists():
        print(f"ERROR: yesno_grammar_file not found: {args.yesno_grammar_file}", file=sys.stderr)
        sys.exit(1)

    num_grammar = load_text(args.num_grammar_file)
    yesno_grammar = load_text(args.yesno_grammar_file)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    trials = []
    summaries = []

    for i, row in enumerate(rows, start=1):
        pid = (row.get("id") or "").strip()
        cat = (row.get("category") or "").strip()
        prompt_raw = row.get("prompt") or ""
        expected_raw = row.get("expected_answer") or ""

        if not pid or not cat or not expected_raw:
            print(f"ERROR: Row {i} missing required fields (id={pid!r}, cat={cat!r}, expected={expected_raw!r})", file=sys.stderr)
            continue

        prompt = prompt_raw.rstrip() + "\nAnswer: "
        expected = expected_raw.strip()

        # HTTP timeout should be longer than experiment timeout to avoid false cutoffs
        http_timeout = (10.0, args.timeout_s + 15.0)

        for w in range(args.warmup_per_prompt):
            try:
                requests.post(
                    args.server_url,
                    json={"prompt": prompt, "n_predict": 8, "temperature": 0.0, "grammar": ""},
                    timeout=http_timeout,
                )
            except Exception:
                pass

        trial_times = []
        trial_preds = []
        trial_errs = []

        for t in range(1, args.repeats + 1):
            n_pred = args.n_pred_log if cat.upper() == "LOG" else args.n_pred_num
            grammar = yesno_grammar if cat.upper() == "LOG" else num_grammar

            raw = ""
            tokens_predicted = None
            tokens_evaluated = None
            t0 = time.monotonic()
            try:
                r = requests.post(
                    args.server_url,
                    json={"prompt": prompt, "n_predict": int(n_pred), "temperature": 0.0, "grammar": grammar},
                    timeout=http_timeout,
                )
                j = r.json()
                raw = j.get("content", "") or ""
                tokens_predicted = j.get("tokens_predicted")
                tokens_evaluated = j.get("tokens_evaluated")
            except requests.exceptions.ReadTimeout:
                # HTTP timed out - but still check wall time below
                pass
            except Exception:
                # Treat other failures as parse failures
                pass

            dt = time.monotonic() - t0

            # E7 is determined by wall time, not HTTP timeout
            timed_out = dt >= args.timeout_s

            if cat.upper() == "LOG":
                pred = extract_yesno(raw)
            else:
                pred = extract_last_int(raw)

            e = err_code(cat, pred, expected, timed_out)

            # Debug output for E7 (timeout) or E8 (extraction failure)
            if e in ("E7", "E8"):
                debug_parts = [pid, cat, f"{dt:.3f}"]
                debug_parts.append(str(tokens_predicted) if tokens_predicted is not None else "None")
                debug_parts.append(str(tokens_evaluated) if tokens_evaluated is not None else "None")
                debug_parts.append(repr(raw[:160]) if raw else "''")
                print(" ".join(debug_parts), file=sys.stderr, flush=True)

            # Required single-line print (no extra jargon)
            print(f"{pid} {cat} #{t}/{args.repeats} time_s={dt:.3f} ans={pred} exp={expected} err={e}", flush=True)

            trials.append({
                "id": pid, "category": cat, "trial": t,
                "time_s": f"{dt:.6f}",
                "pred": pred, "expected": expected, "err": e,
                "raw": raw.replace("\n", "\\n"),
            })

            trial_times.append(dt)
            trial_preds.append(pred)
            trial_errs.append(e)

        # Summary: median latency, majority pred (ties -> first)
        med_t = median(trial_times) if trial_times else 0.0
        counts = {}
        for p in trial_preds:
            counts[p] = counts.get(p, 0) + 1
        majority = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if counts else ""

        ok = 1 if majority == expected else 0
        e_sum = err_code(cat, majority, expected, timed_out=False)

        summaries.append({
            "id": pid,
            "category": cat,
            "median_time_s": f"{med_t:.6f}",
            "pred": majority,
            "expected": expected,
            "ok": ok,
            "err": e_sum,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.trials_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","category","trial","time_s","pred","expected","err","raw"])
        w.writeheader()
        w.writerows(trials)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","category","median_time_s","pred","expected","ok","err"])
        w.writeheader()
        w.writerows(summaries)

if __name__ == "__main__":
    main()
