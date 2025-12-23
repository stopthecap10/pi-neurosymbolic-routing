import csv, json, re, time
from pathlib import Path
from datetime import datetime
from statistics import median
import requests

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

SERVER_URL = "http://127.0.0.1:8080/completion"
PROMPTS_CSV = "data/baseline_prompts.csv"

TIMEOUT_S = 90
REPEATS = 1   # dev fast; later set 3-5 for real stats
WARMUP_PER_PROMPT = 0

N_PRED_NUM = 32
N_PRED_LOG = 8

PROMPT_SUFFIX = "\n<|question_end|>Answer:"

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs_hybrid_v1") / RUN_TAG
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

YESNO_ANY = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
INT_ANY = re.compile(r"[-+]?\d+")

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# ------- Helpers -------

def normalize_prompt(p: str) -> str:
    p = p.replace("\r\n", "\n").rstrip()
    for s in ["\n<|question_end|>Answer:", "<|question_end|>Answer:", "\nAnswer:", "Answer:"]:
        if p.endswith(s):
            p = p[:-len(s)].rstrip()
    return p + PROMPT_SUFFIX

def call_llm(prompt: str, n_predict: int):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "grammar": "",   # ALWAYS empty for now
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
    return status, wall_s, content, j

def extract_llm_answer(category: str, content: str) -> str:
    if category.upper() == "LOG":
        m = YESNO_ANY.search(content)
        return (m.group(1).title() if m else "").strip()
    m = INT_ANY.search(content)
    return (m.group(0) if m else "").strip()

def sympy_arithmetic(prompt: str) -> tuple[str, str]:
    # Extract after "What is"
    # Example: "Answer with only the final number. What is (45 + 55) / 5?"
    m = re.search(r"What is (.+?)\??$", prompt.strip())
    if not m:
        raise ValueError("Could not find 'What is ...' expression")
    expr_raw = m.group(1).strip()
    expr = parse_expr(expr_raw, transformations=TRANSFORMS)
    val = sp.simplify(expr)
    return expr_raw, str(int(val))

def sympy_algebra(prompt: str) -> tuple[str, str]:
    # Extract after "Solve for x:"
    # Example: "Solve for x: 4(x - 2) = 20."
    m = re.search(r"Solve for x:\s*(.+)$", prompt.strip())
    if not m:
        raise ValueError("Could not find 'Solve for x:' equation")
    eq_raw = m.group(1).strip().rstrip(".")
    if "=" not in eq_raw:
        raise ValueError("No '=' found in equation")
    left_raw, right_raw = [s.strip() for s in eq_raw.split("=", 1)]
    x = sp.Symbol("x")
    left = parse_expr(left_raw, transformations=TRANSFORMS, local_dict={"x": x})
    right = parse_expr(right_raw, transformations=TRANSFORMS, local_dict={"x": x})
    sol = sp.solve(sp.Eq(left, right), x)
    if not sol:
        raise ValueError("No solution")
    return eq_raw, str(int(sol[0]))

def route(category: str) -> str:
    c = category.strip().upper()
    if c in ("AR", "ALG"):
        return "SYM"
    if c in ("LOG", "WP"):
        return "LLM"
    return "SKIP"

def correctness(expected: str, got: str, category: str) -> int:
    if not expected or not got:
        return 0
    if category.upper() == "LOG":
        return int(expected.strip().lower() == got.strip().lower())
    try:
        return int(int(expected.strip()) == int(got.strip()))
    except Exception:
        return 0

# Error codes (yours + 2 new ones)
# E0 correct
# E1 arithmetic error (wrong answer on AR/WP)
# E2 logical inference error (wrong on LOG)
# E3 algebra manipulation error (wrong on ALG)
# E6 instruction-following failure (blank/malformed)
# E7 symbolic parse/solver failure (NEW)
# E8 timeout/server error (NEW)

def error_code(expected: str, got: str, category: str, route_taken: str, status: str) -> str:
    if status != "ok":
        return "E8"
    if got == "":
        return "E6"
    if correctness(expected, got, category):
        return "E0"
    c = category.upper()
    if route_taken == "SYM":
        return "E7" if got == "" else ("E3" if c == "ALG" else "E1")
    # LLM side
    if c == "LOG":
        return "E2"
    if c == "ALG":
        return "E3"
    return "E1"

def main():
    prompts = list(csv.DictReader(open(PROMPTS_CSV, newline="", encoding="utf-8")))

    trial_rows = []
    summary_rows = []

    def flush():
        if trial_rows:
            p = RUN_DIR / "hybrid_results_trials.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
                w.writeheader()
                w.writerows(trial_rows)
        if summary_rows:
            p = RUN_DIR / "hybrid_results.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)

    for row in prompts:
        pid = row["id"].strip()
        cat = row["category"].strip()
        prompt_base = row["prompt"]
        expected = str(row["expected_answer"]).strip()

        r = route(cat)
        if r == "SKIP":
            continue

        walls = []
        outs = []

        for t in range(1, REPEATS + 1):
            status = "ok"
            wall_s = 0.0
            got = ""
            sym_expr = ""
            sym_out = ""
            llm_content = ""
            raw = {}

            t0 = time.time()
            try:
                if r == "SYM":
                    if cat.upper() == "AR":
                        sym_expr, sym_out = sympy_arithmetic(prompt_base)
                        got = sym_out
                    else:
                        sym_expr, sym_out = sympy_algebra(prompt_base)
                        got = sym_out
                    wall_s = time.time() - t0
                else:
                    prompt_send = normalize_prompt(prompt_base)
                    # warmup
                    for _ in range(WARMUP_PER_PROMPT):
                        call_llm(prompt_send, N_PRED_LOG if cat.upper()=="LOG" else N_PRED_NUM)

                    status, wall_s, llm_content, raw = call_llm(
                        prompt_send,
                        N_PRED_LOG if cat.upper()=="LOG" else N_PRED_NUM,
                    )
                    got = extract_llm_answer(cat, llm_content)
            except Exception as e:
                status = "err"
                wall_s = time.time() - t0
                raw = {"error": str(e)}

            ok = correctness(expected, got, cat)
            eco = error_code(expected, got, cat, r, status)

            # save raw log json
            log_path = LOG_DIR / f"{pid}__{r}__t{t}.json"
            log_obj = {
                "id": pid,
                "category": cat,
                "route": r,
                "prompt": prompt_base,
                "expected_answer": expected,
                "final_output": got,
                "status": status,
                "latency_wall_s": wall_s,
                "error_code": eco,
                "sym_expr": sym_expr,
                "sym_output": sym_out,
                "llm_content": llm_content,
                "llm_raw": raw,
            }
            log_path.write_text(json.dumps(log_obj, indent=2), encoding="utf-8", errors="replace")

            trial_rows.append({
                "id": pid,
                "category": cat,
                "route": r,
                "trial": t,
                "status": status,
                "expected_answer": expected,
                "final_output": got,
                "correct": int(ok),
                "error_code": eco,
                "latency_wall_s": f"{wall_s:.4f}",
                "sym_expr": sym_expr,
                "sym_output": sym_out,
                "log_file": str(log_path),
            })

            walls.append(wall_s)
            outs.append(got)

            print(f"{pid} [{r}] t{t}/{REPEATS} {status} wall={wall_s:.2f}s -> {got} ({eco})", flush=True)

            flush()

        final = max(set(outs), key=outs.count) if outs else ""
        summary_rows.append({
            "id": pid,
            "category": cat,
            "route": r,
            "expected_answer": expected,
            "final_output": final,
            "correct": correctness(expected, final, cat),
            "latency_wall_median_s": f"{median(walls):.4f}" if walls else "",
        })
        flush()

    flush()
    print(f"\n✅ wrote {RUN_DIR/'hybrid_results.csv'}")
    print(f"✅ wrote {RUN_DIR/'hybrid_results_trials.csv'}")
    print(f"📁 raw logs in {LOG_DIR}")

if __name__ == "__main__":
    main()
