import argparse, csv, time, re, sys
from pathlib import Path
import requests
from statistics import median

INT_RE = re.compile(r"[-+]?\d+")

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def is_degenerate_pattern(s: str) -> bool:
    """Detect if string has repetitive patterns (e.g., 050505, 000000, 170005000)."""
    if len(s) < 6:
        return False

    # Check for repeating single char (000000)
    for i in range(len(s) - 5):
        if len(set(s[i:i+6])) == 1:  # 6 identical chars in a row
            return True

    # Check for repeating 2-char patterns anywhere in string (050505)
    if len(s) >= 6:
        for i in range(len(s) - 5):
            pattern = s[i:i+2]
            # Check if this 2-char pattern repeats at least 3 times consecutively
            if s[i:i+6] == pattern * 3:
                return True

    # Check for repeating 3-char patterns (000500050005)
    if len(s) >= 9:
        for i in range(len(s) - 8):
            pattern = s[i:i+3]
            if s[i:i+9] == pattern * 3:  # Same 3-char pattern repeated 3 times
                return True

    # Check for repeating 4-char patterns (00050005)
    if len(s) >= 8:
        for i in range(len(s) - 7):
            pattern = s[i:i+4]
            if s[i:i+8] == pattern * 2:  # Same 4-char pattern repeated 2 times
                return True

    return False

def extract_last_int(text: str) -> str:
    """Extract last integer from text, stripping whitespace and removing commas."""
    if not text:
        return ""
    # Strip whitespace and remove commas
    cleaned = text.strip().replace(",", "")
    nums = INT_RE.findall(cleaned)
    return nums[-1] if nums else ""

def extract_yesno(text: str) -> str:
    """Find last occurrence of Yes or No (case-insensitive), return canonical Yes/No."""
    if not text:
        return ""
    # Strip problematic tokens that model might generate (including partial tokens)
    cleaned = text.replace("<|question_end|>", "").replace("<|question_end|", "").replace("<|endoftext|>", "").replace("<|", "").strip()

    # If after cleaning we have nothing, return empty
    if not cleaned:
        return ""

    t = cleaned.lower()
    # Find all occurrences of yes/no
    yes_pos = t.rfind("yes")
    no_pos = t.rfind("no")

    if yes_pos == -1 and no_pos == -1:
        return ""
    # Return the one that appears last
    if yes_pos > no_pos:
        return "Yes"
    else:
        return "No"

def err_code(category: str, pred: str, expected: str, timed_out: bool, degenerate: bool = False) -> str:
    if timed_out or degenerate:
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
    ap.add_argument("--timeout_s", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup_per_prompt", type=int, default=0)
    ap.add_argument("--n_pred_num", type=int, default=12)
    ap.add_argument("--n_pred_log", type=int, default=6)
    ap.add_argument("--num_grammar_file", default="grammars/grammar_phi2_answer_int_strict_final.gbnf")
    ap.add_argument("--yesno_grammar_file", default="grammars/grammar_phi2_answer_yesno_strict_final.gbnf")
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
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
        pid = row["id"].strip()
        cat = row["category"].strip()
        # Strip problematic tokens that might be in CSV, then add "\nAnswer: "
        prompt = row["prompt"].replace("<|question_end|>", "").replace("<|endoftext|>", "").rstrip() + "\nAnswer: "
        expected = row["expected_answer"].strip()

        # HTTP timeout should be longer than experiment timeout to avoid false cutoffs
        http_timeout = (10.0, float(args.timeout_s) + 15.0)

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
            error_field = None
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
                error_field = j.get("error")
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
                # Debug: detect "Answer:" + whitespace only (no digits)
                if args.debug and not pred and raw.strip().startswith("Answer:"):
                    # Check if raw has "Answer:" but no digits
                    after_answer = raw.split("Answer:", 1)[-1] if "Answer:" in raw else ""
                    if after_answer and not any(c.isdigit() for c in after_answer):
                        print(f"DEBUG: Answer+whitespace only for {pid} {cat}, content={repr(raw)}, len={len(raw)}", file=sys.stderr, flush=True)

            # Degeneracy guard: detect runaway loops
            degenerate = False
            if cat.upper() != "LOG":
                # Check 1: Answer too long (>18 digits)
                if len(pred) > 18:
                    degenerate = True
                # Check 2: Repetitive pattern detected (e.g., 050505, 00050000)
                elif is_degenerate_pattern(pred):
                    degenerate = True
                # Check 3: Suspiciously long (>10 digits) - most real answers are shorter
                elif len(pred) > 10:
                    degenerate = True

            e = err_code(cat, pred, expected, timed_out, degenerate)

            # Debug output for E7s
            if args.debug and e == "E7":
                debug_parts = [
                    f"E7: {pid} {cat}",
                    f"wall_time={dt:.3f}s",
                ]
                if degenerate:
                    if len(pred) > 18:
                        debug_parts.append(f"DEGENERACY: {len(pred)} digits (>18)")
                    elif is_degenerate_pattern(pred):
                        debug_parts.append(f"DEGENERACY: pattern '{pred}'")
                    elif len(pred) > 10:
                        debug_parts.append(f"DEGENERACY: {len(pred)} digits (>10)")
                if tokens_predicted is not None:
                    debug_parts.append(f"tokens_pred={tokens_predicted}")
                if tokens_evaluated is not None:
                    debug_parts.append(f"tokens_eval={tokens_evaluated}")
                if raw:
                    debug_parts.append(f"content={repr(raw[:160])}")
                print(" ".join(debug_parts), file=sys.stderr, flush=True)

            # Debug output for E8 extraction failures
            if args.debug and e == "E8":
                print(f"E8: {pid} {cat} content={repr(raw[:120])} error={repr(error_field)}", file=sys.stderr, flush=True)

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
        # Check degeneracy for summary
        degenerate_sum = False
        if cat.upper() != "LOG":
            if len(majority) > 18 or is_degenerate_pattern(majority) or len(majority) > 10:
                degenerate_sum = True
        e_sum = err_code(cat, majority, expected, timed_out=False, degenerate=degenerate_sum)

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
