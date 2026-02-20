import argparse, csv, time, re, sys
from pathlib import Path
import requests
from statistics import median

INT_RE = re.compile(r"[-+]?\d+")
NUM_RE = re.compile(r"[-+]?\d+\.?\d*")  # Matches integers and decimals

def extract_last_int(text: str) -> str:
    """Extract last integer from text using regex [-+]?\d+, ignoring continuation markers"""
    if not text:
        return ""

    # Split at common continuation markers to avoid extracting from generated exercises
    for marker in ["\n\nExercise", "\n\n\nExercise", "\nExercise", "\n\n#", "\n\nProblem", "\n\nQuestion"]:
        if marker in text:
            text = text.split(marker)[0]
            break

    # Strip whitespace and remove commas
    cleaned = text.strip().replace(",", "")

    # Remove common answer prefixes that might confuse extraction
    cleaned = cleaned.replace("x = ", "").replace("x=", "")
    cleaned = cleaned.replace("Answer: ", "").replace("Answer:", "")
    cleaned = cleaned.strip()

    # Use INT_RE to find all integers in the text
    nums = INT_RE.findall(cleaned)
    if not nums:
        return ""
    # Return the LAST integer found (before continuation markers)
    return nums[-1]

def extract_yesno(text: str) -> str:
    """Find last occurrence of Yes or No (case-insensitive), ignoring continuation markers"""
    if not text:
        return ""

    # Split at common continuation markers to avoid extracting from generated exercises
    for marker in ["\n\nExercise", "\n\n\nExercise", "\nExercise", "\n\n#", "\n\nProblem", "\n\nQuestion"]:
        if marker in text:
            text = text.split(marker)[0]
            break

    # Clean up text
    cleaned = text.strip()
    if not cleaned:
        return ""

    # Find LAST occurrence of yes/no using rfind
    t = cleaned.lower()
    yes_pos = t.rfind("yes")
    no_pos = t.rfind("no")

    if yes_pos == -1 and no_pos == -1:
        return ""

    # Return the one that appears last
    if yes_pos > no_pos:
        return "Yes"
    else:
        return "No"

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
    ap.add_argument("--n_pred_num", type=int, default=12)
    ap.add_argument("--n_pred_log", type=int, default=6)
    ap.add_argument("--warmup_per_prompt", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Enable debug output")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    trials = []
    summary_rows = []

    for i, row in enumerate(rows, start=1):
        pid = row["id"].strip()
        cat = row["category"].strip()
        expected = row["expected_answer"].strip()

        # Build safe prompt: strip tokens, ensure clean construction
        # NEVER include <|question_end|> or <|endoftext|> - they cause empty outputs
        base_question = row["prompt"].replace("<|question_end|>", "").replace("<|endoftext|>", "").strip()

        # Remove any trailing "Answer:" or "Exercise:" patterns to avoid duplication
        for pattern in ["\nAnswer:", "Answer:", "\nExercise:", "Exercise:"]:
            if base_question.endswith(pattern):
                base_question = base_question[:-len(pattern)].rstrip()

        # Check if dataset prompt already contains instruction text
        if cat.upper() == "LOG":
            has_instruction = "yes or no" in base_question.lower() or "answer with only yes" in base_question.lower()
        else:  # AR, ALG, WP - all numeric
            has_instruction = "final number" in base_question.lower() or "answer with only the" in base_question.lower()

        # Build prompt: add instruction only if not already present
        if has_instruction:
            # Instruction already in prompt, just add Answer: suffix
            prompt = f"{base_question}\nAnswer: "
        else:
            # Add instruction before Answer: suffix
            if cat.upper() == "LOG":
                instruction = "Answer with only Yes or No."
            else:
                instruction = "Answer with only the final number."
            prompt = f"{base_question}\n{instruction}\nAnswer: "

        # Debug: print first prompt construction once
        if args.debug and i == 1:
            print(f"DEBUG: First prompt (last 200 chars): {repr(prompt[-200:])}", file=sys.stderr, flush=True)

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
            content = ""
            tokens_predicted = None
            timed_out = False
            try:
                r = requests.post(
                    args.server_url,
                    json={
                        "prompt": prompt,
                        "n_predict": int(n_pred),
                        "temperature": 0.0,
                    },
                    timeout=(10.0, float(args.timeout_s)),
                )
                j = r.json()
                content = j.get("content", "") or ""
                tokens_predicted = j.get("tokens_predicted")
            except requests.exceptions.Timeout:
                timed_out = True
            except Exception:
                content = ""

            dt = time.time() - t0
            if dt >= args.timeout_s:
                timed_out = True
            pred = extract_yesno(content) if is_log else extract_last_int(content)

            if pred == "" and not timed_out:
                content_prefix = content.replace("\n", "\\n")[:160]
                print(f"E8 {pid} {cat} content={repr(content_prefix)}", file=sys.stderr, flush=True)

            err = err_code(cat, pred, expected, timed_out)
            ok = 1 if err == "E0" else 0

            print(f"{pid} {cat} #{t}/{args.repeats} time_s={dt:.3f} ans={pred} exp={expected} err={err}", flush=True)

            trials.append({"id": pid,"category": cat,"trial": t,"time_s": f"{dt:.6f}","pred": pred,"expected": expected,"err": err,"ok": ok,"raw": content.replace("\n", "\\n")})
            times.append(dt); oks.append(ok)

            # Small delay to let server settle between requests
            if t < args.repeats:
                time.sleep(0.1)

        summary_rows.append({"id": pid,"category": cat,"expected": expected,"acc": f"{sum(oks)/len(oks):.3f}","median_time_s": f"{median(times):.6f}"})

    with open(args.trials_out, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w=_csv.DictWriter(f, fieldnames=["id","category","trial","time_s","pred","expected","err","ok","raw"])
        w.writeheader(); w.writerows(trials)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        w=_csv.DictWriter(f, fieldnames=["id","category","expected","acc","median_time_s"])
        w.writeheader(); w.writerows(summary_rows)

if __name__ == "__main__":
    main()
