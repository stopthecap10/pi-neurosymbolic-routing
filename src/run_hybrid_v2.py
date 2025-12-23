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

# ========= CONFIG =========
SERVER_URL = "http://127.0.0.1:8080/completion"
PROMPTS_CSV = "data/baseline_prompts.csv"

TIMEOUT_S = 120
REPEATS = 1           # later: 3
WARMUP_PER_PROMPT = 0 # keep 0 unless you need warming

N_PRED_NUM = 32
N_PRED_LOG = 8
TEMP = 0

PROMPT_SUFFIX = "\n<|question_end|>Answer:"

GRAMMAR_YESNO_FILE = "grammars/grammar_yesno.gbnf"
USE_YESNO_GRAMMAR = True

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs_hybrid_v2") / RUN_TAG
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ========= Regex / parsing helpers =========
YESNO_ANY = re.compile(r"\b(Yes|No)\b", re.IGNORECASE)
INT_ANY   = re.compile(r"[-+]?\d+")

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def load_yesno_grammar() -> str:
    try:
        return Path(GRAMMAR_YESNO_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""

YESNO_GRAMMAR = load_yesno_grammar() if USE_YESNO_GRAMMAR else ""

def normalize_prompt(p: str) -> str:
    p = p.replace("\r\n", "\n").rstrip()
    # strip any existing answer markers at end
    for s in ["\n<|question_end|>Answer:", "<|question_end|>Answer:", "\nAnswer:", "Answer:"]:
        if p.endswith(s):
            p = p[:-len(s)].rstrip()
    return p + PROMPT_SUFFIX

def call_llm(prompt: str, n_predict: int, grammar: str = ""):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": TEMP,
        "grammar": grammar or "",
    }
    t0 = time.time()
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=TIMEOUT_S)
        wall_s = time.time() - t0
        if not r.ok:
            return f"http_{r.status_code}", wall_s, "", {"text": r.text}
        j = r.json()
        return "ok", wall_s, j.get("content", ""), j
    except Exception as e:
        wall_s = time.time() - t0
        return "err", wall_s, "", {"error": str(e)}

def extract_num(content: str) -> str:
    m = INT_ANY.search(content)
    return (m.group(0) if m else "").strip()

def extract_yesno(content: str) -> str:
    m = YESNO_ANY.search(content)
    return (m.group(1).title() if m else "").strip()

def correctness(expected: str, got: str, category: str) -> int:
    if not expected or not got:
        return 0
    c = category.strip().upper()
    if c == "LOG":
        return int(expected.strip().lower() == got.strip().lower())
    try:
        return int(int(expected.strip()) == int(got.strip()))
    except Exception:
        return 0

# ========= SymPy helpers =========
def sympy_arithmetic_from_prompt(prompt: str) -> tuple[str, str]:
    m = re.search(r"What is (.+?)\??$", prompt.strip())
    if not m:
        raise ValueError("AR: could not find 'What is ...' expression")
    expr_raw = m.group(1).strip()
    expr = parse_expr(expr_raw, transformations=TRANSFORMS)
    val = sp.simplify(expr)
    return expr_raw, str(int(val))

def sympy_algebra_from_prompt(prompt: str) -> tuple[str, str]:
    m = re.search(r"Solve for x:\s*(.+)$", prompt.strip())
    if not m:
        raise ValueError("ALG: could not find 'Solve for x:' equation")
    eq_raw = m.group(1).strip().rstrip(".")
    if "=" not in eq_raw:
        raise ValueError("ALG: no '=' found")
    left_raw, right_raw = [s.strip() for s in eq_raw.split("=", 1)]
    x = sp.Symbol("x")
    left = parse_expr(left_raw, transformations=TRANSFORMS, local_dict={"x": x})
    right = parse_expr(right_raw, transformations=TRANSFORMS, local_dict={"x": x})
    sol = sp.solve(sp.Eq(left, right), x)
    if not sol:
        raise ValueError("ALG: no solution")
    return eq_raw, str(int(sol[0]))

# ========= WP verifier (FIXED + safer policy) =========
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
LITERS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*liters?\b", re.IGNORECASE)

def wp_percent_used_remaining(prompt: str):
    # "A tank holds 50 liters. 30% is used. How many liters remain?"
    mL = LITERS_RE.search(prompt)
    mP = PCT_RE.search(prompt)
    if not mL or not mP:
        return None, None, "missing_total_or_pct"
    total = float(mL.group(1))
    used_pct = float(mP.group(1))
    remain = total * (1.0 - used_pct / 100.0)
    if abs(remain - round(remain)) < 1e-9:
        remain = int(round(remain))
    sym_expr = f"{total}*(1-({used_pct}/100))"
    return str(remain), sym_expr, None

def sympy_word_problem(prompt: str) -> tuple[str, str]:
    p = prompt.strip()

    # percent-used tank (WP08-style) — handled by robust regex
    ans, expr, err = wp_percent_used_remaining(p)
    if err is None and ans is not None:
        return expr, ans

    # simple templates
    m = re.search(r"box has (\d+).+buy (\d+) box", p, re.IGNORECASE)
    if m:
        a, b = map(int, m.groups()); return f"{a}*{b}", str(a*b)

    m = re.search(r"length (\d+).+width (\d+).+area", p, re.IGNORECASE)
    if m:
        L, W = map(int, m.groups()); return f"{L}*{W}", str(L*W)

    m = re.search(r"have (\d+).+give away (\d+).+remain", p, re.IGNORECASE)
    if m:
        a, b = map(int, m.groups()); return f"{a}-{b}", str(a-b)

    m = re.search(r"travels (\d+).+in (\d+) hour.+average speed", p, re.IGNORECASE)
    if m:
        d, t = map(int, m.groups()); return f"{d}/{t}", str(int(d/t))

    m = re.search(r"sells (\d+) .+ for \$?(\d+).+1 .* cost", p, re.IGNORECASE)
    if m:
        n, total = map(int, m.groups()); return f"{total}/{n}", str(int(total/n))

    m = re.search(r"needs (\d+) cup.+per batch.+for (\d+) batch", p, re.IGNORECASE)
    if m:
        per, batches = map(int, m.groups()); return f"{per}*{batches}", str(per*batches)

    m = re.search(r"There are (\d+).+groups of (\d+).+How many groups", p, re.IGNORECASE)
    if m:
        total, size = map(int, m.groups()); return f"{total}/{size}", str(int(total/size))

    m = re.search(r"moves (\d+).+in (\d+) hour.+speed", p, re.IGNORECASE)
    if m:
        d, t = map(int, m.groups()); return f"{d}/{t}", str(int(d/t))

    m = re.search(r"read (\d+) pages per day for (\d+) day", p, re.IGNORECASE)
    if m:
        per, days = map(int, m.groups()); return f"{per}*{days}", str(per*days)

    raise ValueError("WP: could not parse into math expression")

# ========= LOG verifier (self-consistency) =========
def log_verifier(question_prompt: str) -> tuple[str, dict]:
    q = question_prompt.strip()
    base = q if q.lower().startswith("answer with only") else ("Answer with only Yes or No. " + q)

    p1 = normalize_prompt(base)
    p2 = normalize_prompt(base + " Answer only Yes or No.")
    p3 = normalize_prompt("Answer with only Yes or No. Think carefully. " + q)

    g = YESNO_GRAMMAR if YESNO_GRAMMAR else ""

    s1, t1, c1, j1 = call_llm(p1, N_PRED_LOG, grammar=g)
    a1 = extract_yesno(c1)

    s2, t2, c2, j2 = call_llm(p2, N_PRED_LOG, grammar=g)
    a2 = extract_yesno(c2)

    meta = {
        "attempts": [
            {"status": s1, "wall_s": t1, "answer": a1, "content": c1, "prompt": p1},
            {"status": s2, "wall_s": t2, "answer": a2, "content": c2, "prompt": p2},
        ],
        "used_grammar": bool(g),
    }

    if s1 == "ok" and s2 == "ok" and a1 in ("Yes","No") and a1 == a2:
        meta["final_rule"] = "agree_2"
        return a1, meta

    s3, t3, c3, j3 = call_llm(p3, N_PRED_LOG, grammar=g)
    a3 = extract_yesno(c3)
    meta["attempts"].append({"status": s3, "wall_s": t3, "answer": a3, "content": c3, "prompt": p3})

    votes = [a for a in (a1,a2,a3) if a in ("Yes","No")]
    if votes:
        yes = votes.count("Yes"); no = votes.count("No")
        if yes > no:
            meta["final_rule"] = "majority_3"; return "Yes", meta
        if no > yes:
            meta["final_rule"] = "majority_3"; return "No", meta
        meta["final_rule"] = "tie_pick_last_valid"
        return votes[-1], meta

    meta["final_rule"] = "no_valid_answer"
    return "", meta

# ========= Error codes =========
# Your codes:
# E0 correct
# E1 arithmetic computation error (AR/WP numeric wrong)
# E2 logical inference error (LOG wrong)
# E3 algebraic manipulation error (ALG wrong)
# E4 hallucinated step (not used here)
# E5 partial reasoning (not used here)
# E6 instruction following failure (blank/malformed)
# Added:
# E7 verification/solver failure
# E8 timeout/server error

def pick_error(category: str, expected: str, got: str, status: str, sym_ok: bool) -> str:
    if status != "ok":
        return "E8"
    if got == "":
        return "E6"
    if correctness(expected, got, category):
        return "E0"
    c = category.upper()
    if not sym_ok and c in ("AR","ALG","WP"):
        return "E7"
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
            p = RUN_DIR / "hybrid_v2_results_trials.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
                w.writeheader()
                w.writerows(trial_rows)
        if summary_rows:
            p = RUN_DIR / "hybrid_v2_results.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)

    for row in prompts:
        pid = row["id"].strip()
        cat = row["category"].strip().upper()
        prompt_base = row["prompt"]
        expected = str(row.get("expected_answer","")).strip()

        walls = []
        finals = []

        for t in range(1, REPEATS + 1):
            status = "ok"
            wall_s = 0.0

            llm_content = ""
            llm_answer = ""
            final_answer = ""

            verify_used = False
            verify_pass = ""
            fallback_used = False
            sym_expr = ""
            sym_out = ""
            sym_ok = True
            route = "LLM"

            raw_llm = {}
            raw_meta = {}

            try:
                if cat == "LOG":
                    t0 = time.time()
                    final_answer, meta = log_verifier(prompt_base)
                    wall_s = time.time() - t0
                    raw_meta = meta
                    route = "LLM_LOG_SELFCONSIST"

                else:
                    prompt_send = normalize_prompt(prompt_base)

                    for _ in range(WARMUP_PER_PROMPT):
                        call_llm(prompt_send, N_PRED_NUM, grammar="")

                    status, wall_s, llm_content, raw_llm = call_llm(prompt_send, N_PRED_NUM, grammar="")
                    llm_answer = extract_num(llm_content)
                    final_answer = llm_answer

                    # numeric verification for AR/ALG/WP
                    verify_used = True
                    try:
                        if cat == "AR":
                            sym_expr, sym_out = sympy_arithmetic_from_prompt(prompt_base)
                            if llm_answer != "" and int(llm_answer) == int(sym_out):
                                verify_pass = "1"
                                route = "LLM+VERIFY_PASS"
                            else:
                                verify_pass = "0"
                                final_answer = sym_out
                                fallback_used = True
                                route = "LLM+VERIFY_FAIL_FALLBACK_SYM"

                        elif cat == "ALG":
                            sym_expr, sym_out = sympy_algebra_from_prompt(prompt_base)
                            if llm_answer != "" and int(llm_answer) == int(sym_out):
                                verify_pass = "1"
                                route = "LLM+VERIFY_PASS"
                            else:
                                verify_pass = "0"
                                final_answer = sym_out
                                fallback_used = True
                                route = "LLM+VERIFY_FAIL_FALLBACK_SYM"

                        elif cat == "WP":
                            # IMPORTANT POLICY:
                            # WP verifier is best-effort. It should NOT override LLM when it disagrees.
                            sym_expr, sym_out = sympy_word_problem(prompt_base)
                            if llm_answer != "" and sym_out != "" and int(llm_answer) == int(sym_out):
                                verify_pass = "1"
                                route = "LLM+VERIFY_PASS"
                                final_answer = llm_answer
                            else:
                                verify_pass = "0"
                                route = "LLM+VERIFY_MISMATCH_KEEP_LLM"
                                final_answer = llm_answer  # keep LLM

                        else:
                            verify_used = False
                            route = "LLM"

                    except Exception as e:
                        sym_ok = False
                        verify_used = True
                        verify_pass = ""
                        raw_meta = {"verifier_error": str(e)}
                        route = "LLM+VERIFY_ERROR_KEEP_LLM"
                        final_answer = llm_answer

            except Exception as e:
                status = "err"
                raw_meta = {"exception": str(e)}

            eco = pick_error(cat, expected, final_answer, status, sym_ok)
            ok = correctness(expected, final_answer, cat)

            log_path = LOG_DIR / f"{pid}__{cat}__t{t}.json"
            log_obj = {
                "id": pid,
                "category": cat,
                "trial": t,
                "route": route,
                "prompt": prompt_base,
                "expected_answer": expected,
                "llm_content": llm_content,
                "llm_answer": llm_answer,
                "final_output": final_answer,
                "correct": int(ok),
                "error_code": eco,
                "latency_wall_s": wall_s,
                "verify_used": int(bool(verify_used)),
                "verify_pass": verify_pass,
                "fallback_used": int(bool(fallback_used)),
                "sym_expr": sym_expr,
                "sym_output": sym_out,
                "sym_ok": sym_ok,
                "llm_raw": raw_llm,
                "meta": raw_meta,
            }
            log_path.write_text(json.dumps(log_obj, indent=2), encoding="utf-8", errors="replace")

            trial_rows.append({
                "id": pid,
                "category": cat,
                "trial": t,
                "route": route,
                "expected_answer": expected,
                "final_output": final_answer,
                "correct": int(ok),
                "error_code": eco,
                "latency_wall_s": f"{wall_s:.4f}",
                "verify_used": int(bool(verify_used)),
                "verify_pass": verify_pass,
                "fallback_used": int(bool(fallback_used)),
                "sym_expr": sym_expr,
                "sym_output": sym_out,
                "log_file": str(log_path),
            })

            walls.append(wall_s)
            finals.append(final_answer)

            print(f"{pid} [{cat}] t{t}/{REPEATS} {status} wall={wall_s:.2f}s -> {final_answer} ({eco}) route={route}", flush=True)
            flush()

        # summary row
        final = finals[0] if finals else ""
        summary_rows.append({
            "id": pid,
            "category": cat,
            "route": "mixed",
            "expected_answer": expected,
            "final_output": final,
            "correct": correctness(expected, final, cat),
            "latency_wall_median_s": f"{median(walls):.4f}" if walls else "",
        })
        flush()

    print(f"\n✅ wrote {RUN_DIR/'hybrid_v2_results.csv'}")
    print(f"✅ wrote {RUN_DIR/'hybrid_v2_results_trials.csv'}")
    print(f"📁 raw logs in {LOG_DIR}")

if __name__ == "__main__":
    main()
