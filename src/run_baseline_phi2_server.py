import csv, json, re, time
from pathlib import Path
from datetime import datetime
from statistics import median
import requests

SERVER_URL = "http://127.0.0.1:8080/completion"
PROMPTS_CSV = "data/baseline_prompts.csv"

TIMEOUT_S = 90
REPEATS = 1
WARMUP_PER_PROMPT = 0

N_PRED_NUM = 32
N_PRED_LOG = 8

# Make prompts consistent (Phi-2 likes this)
PROMPT_SUFFIX = "\n<|question_end|>Answer:"

GRAMMAR_YESNO_ONLY_FILE = "grammars/grammar_yesno_only.gbnf"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs_server") / RUN_TAG
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

YESNO_ANY = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
INT_ANY = re.compile(r"[-+]?\d+")

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def normalize_prompt(p: str) -> str:
    p = p.replace("\r\n", "\n").rstrip()
    # strip any accidental existing suffixes
    for s in ["\n<|question_end|>Answer:", "<|question_end|>Answer:", "\nAnswer:", "Answer:"]:
        if p.endswith(s):
            p = p[:-len(s)].rstrip()
    return p + PROMPT_SUFFIX

def call_completion(prompt: str, n_predict: int, grammar_text: str):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        # IMPORTANT: always send grammar, even if empty, to avoid "sticky" server state
        "grammar": grammar_text,
    }
    t0 = time.time()
    r = requests.post(SERVER_URL, json=payload, timeout=TIMEOUT_S)
    wall_s = time.time() - t0

    status = "ok" if r.ok else f"http_{r.status_code}"
    try:
        j = r.json()
    except Exception:
        j = {"content": r.text}

    content = j.get("content", "")
    timings = j.get("timings", {}) or {}
    # compute time: prompt_ms + predicted_ms (if provided)
    compute_s = ""
    if isinstance(timings, dict):
        pm = timings.get("prompt_ms")
        gm = timings.get("predicted_ms")
        if isinstance(pm, (int, float)) and isinstance(gm, (int, float)):
            compute_s = (pm + gm) / 1000.0

    return status, wall_s, compute_s, content, j

def extract_answer(category: str, content: str) -> str:
    if category.upper() == "LOG":
        m = YESNO_ANY.search(content)
        return (m.group(1).title() if m else "").strip()
    m = INT_ANY.search(content)
    return (m.group(0) if m else "").strip()

def main():
    # load prompts
    prompts = []
    with open(PROMPTS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            prompts.append(r)

    yesno_grammar = read_text(GRAMMAR_YESNO_ONLY_FILE)

    variants = [
        ("phi2_server_nogrammar", False),
        ("phi2_server_grammar", True),
    ]

    trial_rows = []
    summary_rows = []

    def flush_csvs():
        if trial_rows:
            tpath = RUN_DIR / "baseline_results_trials.csv"
            with open(tpath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
                w.writeheader()
                w.writerows(trial_rows)

        if summary_rows:
            spath = RUN_DIR / "baseline_results.csv"
            with open(spath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)

    try:
        for row in prompts:
            pid = row["id"].strip()
            cat = row["category"].strip()
            expected = str(row["expected_answer"]).strip()

            prompt_send = normalize_prompt(row["prompt"])

            for variant, wants_grammar in variants:
                # ONLY apply Yes/No grammar to LOG when variant is "grammar"
                if wants_grammar and cat.upper() == "LOG":
                    grammar_text = yesno_grammar
                else:
                    # IMPORTANT: empty string clears grammar on servers that keep slot state
                    grammar_text = ""

                n_pred = N_PRED_LOG if cat.upper() == "LOG" else N_PRED_NUM

                # warmup (not counted)
                for _ in range(WARMUP_PER_PROMPT):
                    call_completion(prompt_send, n_pred, grammar_text)

                walls = []
                computes = []
                outs = []
                corrects = []

                for t in range(1, REPEATS + 1):
                    status, wall_s, compute_s, content, raw = call_completion(prompt_send, n_pred, grammar_text)
                    got = extract_answer(cat, content)
                    ok = int(status == "ok" and got == expected)

                    log_path = LOG_DIR / f"{pid}__{variant}__t{t}.json"
                    raw_out = dict(raw)
                    raw_out["prompt"] = prompt_send
                    raw_out["expected_answer"] = expected
                    raw_out["final_output"] = got
                    raw_out["status"] = status
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(raw_out, f, ensure_ascii=False, indent=2)

                    trial_rows.append({
                        "id": pid,
                        "category": cat,
                        "variant": variant,
                        "trial": t,
                        "status": status,
                        "latency_wall_s": f"{wall_s:.4f}",
                        "latency_compute_s": (f"{compute_s:.4f}" if compute_s != "" else ""),
                        "expected_answer": expected,
                        "final_output": got,
                        "correct": ok,
                        "log_file": str(log_path),
                    })

                    walls.append(wall_s)
                    if compute_s != "":
                        computes.append(float(compute_s))
                    outs.append(got)
                    corrects.append(ok)

                    eco = "E0" if ok else "E1"
                    print(f"{pid} [{variant}] t{t}/{REPEATS} {status} wall={wall_s:.2f}s -> {got} ({eco})", flush=True)

                # majority vote final output
                final = max(set(outs), key=outs.count) if outs else ""
                correct_majority = int(sum(corrects) >= (REPEATS // 2 + 1))

                summary_rows.append({
                    "id": pid,
                    "category": cat,
                    "variant": variant,
                    "expected_answer": expected,
                    "final_output": final,
                    "correct": correct_majority,
                    "latency_wall_median_s": f"{median(walls):.4f}" if walls else "",
                    "latency_compute_median_s": f"{median(computes):.4f}" if computes else "",
                })

                flush_csvs()

    except KeyboardInterrupt:
        print("\n⚠️ interrupted: writing partial CSVs...", flush=True)
        flush_csvs()
        raise

    flush_csvs()
    print(f"\n✅ wrote {RUN_DIR/'baseline_results.csv'}")
    print(f"✅ wrote {RUN_DIR/'baseline_results_trials.csv'}")
    print(f"📁 raw logs in {LOG_DIR}")

if __name__ == "__main__":
    main()
