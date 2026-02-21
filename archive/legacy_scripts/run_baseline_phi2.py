import csv
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime
from statistics import median

# ====== CONFIG ======
BIN   = str(Path.home() / "llama.cpp/build/bin/llama-completion")
MODEL = str(Path.home() / "edge-ai/models/phi-2-Q4_K_M.gguf")

PROMPTS_CSV = "data/baseline_prompts.csv"

TIMEOUT_S = 60
THREADS = 4
CTX = 512

# repeats per (prompt, variant)
REPEATS = 3

# keep outputs short
N_PRED_NUM = 32   # AR/ALG/WP
N_PRED_LOG = 8    # LOG

GRAMMAR_YESNO = "grammars/grammar_yesno.gbnf"
GRAMMAR_INT   = "grammars/grammar_phi2_answer_int.gbnf"

# If llama-completion supports --no-warmup, we use it
AUTO_NO_WARMUP = True

RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("runs") / RUN_TAG
LOG_DIR = RUN_DIR / "logs"
RUN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

YESNO_ANY = re.compile(r"\b(Yes|No)\b")
INT_ANY   = re.compile(r"[+-]?\d+")

# ---------- Error codes ----------
# Your taxonomy:
# E0 = No error (correct)
# E1 = Arithmetic Computation Error
# E2 = Logical Inference Error
# E3 = Algebraic Manipulation Error
# E4 = Hallucinated Step (not auto-detected in baseline runner)
# E5 = Partial Reasoning (not auto-detected)
# E6 = Instruction Following Failure (not auto-detected reliably)
#
# Added for runner robustness:
# E7 = Timeout
# E8 = Parse failure / malformed output


def tool_supports(flag: str) -> bool:
    try:
        p = subprocess.run([BIN, "--help"], capture_output=True, text=True, timeout=5)
        return flag in (p.stdout + p.stderr)
    except Exception:
        return False


def kind_from_category(category: str) -> str | None:
    cat = category.strip().upper()
    if cat == "LOG":
        return "log"
    if cat in ("AR", "ALG", "WP"):
        return "num"
    return None


def n_predict_for(kind: str) -> int:
    return N_PRED_LOG if kind == "log" else N_PRED_NUM


def grammar_for(variant: str, kind: str) -> str | None:
    # Two baseline variants:
    # - phi2_nogrammar: no grammar
    # - phi2_grammar:   LOG uses Yes/No grammar; NUM uses Phi-2 Answer:int grammar
    if variant == "phi2_nogrammar":
        return None
    if kind == "log":
        return GRAMMAR_YESNO
    if kind == "num":
        return GRAMMAR_INT
    return None


def extract_num(stdout: str) -> str:
    # Best rule: find first line containing Answer: and extract first integer
    for line in stdout.splitlines():
        if "Answer:" in line:
            m = INT_ANY.search(line)
            if m:
                return m.group(0)
    # fallback: last integer anywhere in stdout
    hits = INT_ANY.findall(stdout)
    return hits[-1] if hits else ""


def extract_yesno(stdout: str) -> str:
    hits = YESNO_ANY.findall(stdout)
    return hits[-1] if hits else ""


def compute_correct(expected: str, got: str, kind: str) -> int:
    if not expected or not got:
        return 0
    exp = expected.strip()
    got = got.strip()
    if kind == "log":
        return 1 if exp.lower() == got.lower() else 0
    try:
        return 1 if int(exp) == int(got) else 0
    except Exception:
        return 0


def assign_error_code(status: str, got: str, correct: int, category: str) -> str:
    if status == "timeout":
        return "E7"
    if got.strip() == "":
        return "E8"
    if correct == 1:
        return "E0"

    cat = category.strip().upper()
    if cat in ("AR", "WP"):
        return "E1"
    if cat == "ALG":
        return "E3"
    if cat == "LOG":
        return "E2"
    return "E6"  # fallback if category unexpected


def run_llm(prompt: str, grammar_file: str | None, n_predict: int):
    cmd = [
        BIN, "-m", MODEL,
        "-t", str(THREADS),
        "-c", str(CTX),
        "-n", str(n_predict),
        "-no-cnv",
        "--temp", "0",
        "-p", prompt,
    ]

    if AUTO_NO_WARMUP and tool_supports("--no-warmup"):
        cmd.append("--no-warmup")

    if grammar_file:
        cmd += ["--grammar-file", grammar_file]

    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,   # stdout + stderr separate
            text=True,
            timeout=TIMEOUT_S
        )
        lat = time.time() - t0
        return "ok", lat, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        lat = time.time() - t0
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        return "timeout", lat, out, err


def main():
    with open(PROMPTS_CSV, newline="") as f:
        prompts = list(csv.DictReader(f))

    variants = ["phi2_nogrammar", "phi2_grammar"]

    # We write two outputs:
    # 1) trials CSV = every single run (good for auditing)
    # 2) summary CSV = one row per (id, variant) using median latency (good for plots/stats)
    trials_rows = []
    summary_rows = []

    for variant in variants:
        for r in prompts:
            rid = r.get("id", "").strip()
            cat = r.get("category", "").strip()
            prompt = r.get("prompt", "")
            expected = r.get("expected_answer", "").strip()

            kind = kind_from_category(cat)
            if not kind:
                continue

            grammar = grammar_for(variant, kind)
            n_pred = n_predict_for(kind)

            # run repeats
            latencies_ok = []
            got_first_ok = ""
            status_any_timeout = False

            for t in range(1, REPEATS + 1):
                status, lat, stdout, stderr = run_llm(prompt, grammar, n_pred)

                if status == "timeout":
                    status_any_timeout = True
                    got = ""
                    correct = 0
                else:
                    # parse ONLY stdout (stderr contains perf numbers)
                    got = extract_yesno(stdout) if kind == "log" else extract_num(stdout)
                    correct = compute_correct(expected, got, kind)
                    latencies_ok.append(lat)
                    if got_first_ok == "" and got.strip() != "":
                        got_first_ok = got.strip()

                ecode = assign_error_code(status, got, correct, cat)

                # raw log per trial
                log_path = LOG_DIR / f"{rid}__{variant}__t{t}.txt"
                log_path.write_text(
                    "=== PROMPT ===\n" + prompt +
                    "\n\n=== STDOUT ===\n" + (stdout or "") +
                    "\n\n=== STDERR ===\n" + (stderr or ""),
                    encoding="utf-8",
                    errors="replace"
                )

                trials_rows.append({
                    "id": rid,
                    "category": cat,
                    "variant": variant,
                    "trial": str(t),
                    "grammar_used": grammar or "",
                    "expected_answer": expected,
                    "final_output": got.strip(),
                    "correct": str(correct),
                    "error_code": ecode,
                    "latency_s": f"{lat:.4f}",
                })

                print(f"{rid} [{variant}] t{t}/{REPEATS} {status} {lat:.2f}s -> {got.strip()} ({ecode})")

            # summary row per (id, variant)
            if latencies_ok:
                med_lat = median(latencies_ok)
                status_summary = "ok" if not status_any_timeout else "ok_with_timeouts"
            else:
                med_lat = ""
                status_summary = "timeout"

            # correctness in summary: based on first successful parsed output (temp=0 should be stable)
            if status_summary == "timeout":
                final_out = ""
                corr = 0
                ecode = "E7"
            else:
                final_out = got_first_ok
                corr = compute_correct(expected, final_out, kind)
                # if we couldn't parse anything even though it wasn't timeout
                if final_out.strip() == "":
                    ecode = "E8"
                else:
                    ecode = assign_error_code("ok", final_out, corr, cat)

            summary_rows.append({
                "id": rid,
                "category": cat,
                "variant": variant,
                "grammar_used": grammar or "",
                "prompt": prompt,
                "expected_answer": expected,
                "final_output": final_out,
                "correct": str(corr),
                "error_code": ecode,
                "latency_median_s": (f"{med_lat:.4f}" if med_lat != "" else ""),
                "status": status_summary,
            })

    # Write trials CSV
    trials_csv = RUN_DIR / "baseline_results_trials.csv"
    with open(trials_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials_rows[0].keys()))
        w.writeheader()
        w.writerows(trials_rows)

    # Write summary CSV
    summary_csv = RUN_DIR / "baseline_results.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n✅ wrote {summary_csv}")
    print(f"✅ wrote {trials_csv}")
    print(f"📁 raw logs in {LOG_DIR}")


if __name__ == "__main__":
    main()
