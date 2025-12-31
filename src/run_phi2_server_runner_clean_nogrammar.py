import argparse, csv, time, re, sys
from pathlib import Path
import requests
from statistics import median

INT_RE = re.compile(r"[-+]?\d+")
NUM_RE = re.compile(r"[-+]?\d+\.?\d*")  # Matches integers and decimals

def extract_last_int(text: str) -> str:
    if not text:
        return ""
    # Use NUM_RE to handle decimals like "17.0", then convert to int
    # Take FIRST number to avoid picking up numbers from model's rambling (e.g., "192\n\nExercise 3:")
    nums = NUM_RE.findall(text.replace(",", ""))
    if not nums:
        return ""
    try:
        return str(int(float(nums[0])))
    except (ValueError, OverflowError):
        return ""

def extract_yesno(text: str) -> str:
    if not text:
        return ""
    # Check for yes/no or true/false (case insensitive)
    m = re.search(r"(yes|no|true|false)", text, flags=re.I)
    if not m:
        return ""
    w = m.group(1).lower()
    # Map true→Yes, false→No for consistency with expected answers
    if w in ("yes", "true"):
        return "Yes"
    elif w in ("no", "false"):
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
    ap.add_argument("--timeout_s", type=float, default=60.0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n_pred_num", type=int, default=8)
    ap.add_argument("--n_pred_log", type=int, default=6)
    ap.add_argument("--warmup_per_prompt", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    trials = []
    summary_rows = []

    for row in rows:
        pid = row["id"].strip()
        cat = row["category"].strip()
        expected = row["expected_answer"].strip()
        base_prompt = row["prompt"].rstrip()
        # Always use consistent suffix for all categories
        prompt = base_prompt + "\nAnswer: "

        is_log = (cat.upper() == "LOG")
        n_pred = args.n_pred_log if is_log else args.n_pred_num

        # Warmup: run dummy inferences to populate KV cache
        http_timeout = (10.0, float(args.timeout_s) + 15.0)
        for _ in range(args.warmup_per_prompt):
            try:
                requests.post(
                    args.server_url,
                    json={"prompt": prompt, "n_predict": 8, "temperature": 0.0},
                    timeout=http_timeout,
                )
            except Exception:
                pass

        times = []
        oks = []

        for t in range(1, args.repeats + 1):
            t0 = time.time()
            timed_out = False
            content = ""
            try:
                r = requests.post(
                    args.server_url,
                    json={"prompt": prompt, "n_predict": int(n_pred), "temperature": 0.0},
                    timeout=(10.0, float(args.timeout_s)),
                )
                content = r.json().get("content", "")
            except requests.exceptions.Timeout:
                timed_out = True
            except Exception:
                content = ""

            dt = time.time() - t0
            pred = extract_yesno(content) if is_log else extract_last_int(content)
            if args.debug and pred == "" and not timed_out:
                print(f"DEBUG {pid} {cat} content_prefix={repr(content[:120])}", file=sys.stderr)
            err = err_code(cat, pred, expected, timed_out)
            ok = 1 if err == "E0" else 0

            print(f"{pid} {cat} #{t}/{args.repeats} time_s={dt:.3f} ans={pred} exp={expected} err={err}", flush=True)

            trials.append({"id": pid,"category": cat,"trial": t,"time_s": f"{dt:.6f}","pred": pred,"expected": expected,"err": err,"ok": ok})
            times.append(dt); oks.append(ok)

            # Small delay to let server settle between requests
            if t < args.repeats:
                time.sleep(0.1)

        summary_rows.append({"id": pid,"category": cat,"expected": expected,"acc": f"{sum(oks)/len(oks):.3f}","median_time_s": f"{median(times):.6f}"})

    with open(args.trials_out, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w=_csv.DictWriter(f, fieldnames=["id","category","trial","time_s","pred","expected","err","ok"])
        w.writeheader(); w.writerows(trials)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w=_csv.DictWriter(f, fieldnames=["id","category","expected","acc","median_time_s"])
        w.writeheader(); w.writerows(summary_rows)

if __name__ == "__main__":
    main()
