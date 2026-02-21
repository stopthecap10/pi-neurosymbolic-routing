import argparse, csv, json, sys, time, statistics
from pathlib import Path
import requests
import re

INT_RE = re.compile(r"[-+]?\d+")

def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def safe_int(s: str):
    try:
        return int(s.strip())
    except Exception:
        return None

def extract_numeric(text: str) -> str:
    t = (text or "").strip()
    nums = INT_RE.findall(t.replace(",", ""))
    return nums[-1] if nums else ""

def extract_yesno(text: str) -> str:
    t = (text or "").strip().lower()
    parts = re.split(r"[^a-z]+", t)
    parts = [p for p in parts if p]
    has_yes = "yes" in parts
    has_no  = "no" in parts
    if has_yes and not has_no:
        return "Yes"
    if has_no and not has_yes:
        return "No"
    return ""

def call_server(url: str, prompt: str, n_predict: int, grammar: str, timeout_s: float):
    payload = {
        "prompt": prompt,
        "n_predict": int(n_predict),
        "temperature": 0.0,
        "grammar": grammar or "",
    }
    t0 = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=(10.0, float(timeout_s)))
        wall = time.perf_counter() - t0
    except requests.exceptions.ReadTimeout:
        return {"ok_http": False, "wall": time.perf_counter()-t0, "err": "E7", "raw": ""}
    except requests.exceptions.ConnectTimeout:
        return {"ok_http": False, "wall": time.perf_counter()-t0, "err": "E7", "raw": ""}
    except requests.exceptions.ConnectionError:
        return {"ok_http": False, "wall": time.perf_counter()-t0, "err": "E7", "raw": ""}
    except Exception:
        return {"ok_http": False, "wall": time.perf_counter()-t0, "err": "E7", "raw": ""}

    if r.status_code != 200:
        return {"ok_http": False, "wall": wall, "err": "E7", "raw": ""}

    try:
        j = r.json()
    except json.JSONDecodeError:
        return {"ok_http": True, "wall": wall, "err": "E8", "raw": ""}

    return {"ok_http": True, "wall": wall, "err": "E0", "raw": j.get("content", "")}

def wrong_err_for_category(cat: str) -> str:
    c = (cat or "").strip().upper()
    if c == "AR":
        return "E1"
    if c == "ALG":
        return "E3"
    if c == "LOG":
        return "E2"
    if c == "WP":
        return "E5"
    return "E5"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials_out", required=True)
    ap.add_argument("--server_url", required=True)
    ap.add_argument("--timeout_s", type=float, default=60)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup_per_prompt", type=int, default=0)
    ap.add_argument("--n_pred_num", type=int, default=32)
    ap.add_argument("--n_pred_log", type=int, default=8)
    ap.add_argument("--num_grammar_file", required=True)
    ap.add_argument("--yesno_grammar_file", required=True)
    ap.add_argument("--max_per_category", type=int, default=0)   # 0 = no limit
    ap.add_argument("--max_total", type=int, default=0)          # 0 = no limit (smoke tests)
    args = ap.parse_args()

    num_grammar = load_text(args.num_grammar_file)
    yn_grammar  = load_text(args.yesno_grammar_file)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.trials_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8") as f_in, \
         open(args.out, "w", newline="", encoding="utf-8") as f_out, \
         open(args.trials_out, "w", newline="", encoding="utf-8") as f_trials:

        rdr = csv.DictReader(f_in)
        outw = csv.writer(f_out)
        triw = csv.writer(f_trials)

        outw.writerow(["id","category","expected_answer","final_answer","correct","err","wall_s","chosen_trial"])
        triw.writerow(["id","category","trial","answer","expected_answer","correct","err","wall_s","raw"])

        cat_counts = {}
        total_kept = 0

        for row in rdr:
            pid = row["id"].strip()
            cat = row["category"].strip()
            prompt = row["prompt"]
            expected = row["expected_answer"].strip()

            # enforce per-category limit (for T1/T2/T3)
            if args.max_per_category > 0:
                if cat_counts.get(cat, 0) >= args.max_per_category:
                    continue

            if args.max_total > 0 and total_kept >= args.max_total:
                break

            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            total_kept += 1

            if cat.upper() == "LOG":
                grammar = yn_grammar
                n_pred = args.n_pred_log
                extractor = extract_yesno
            else:
                grammar = num_grammar
                n_pred = args.n_pred_num
                extractor = extract_numeric

            # warmup (does not count)
            for _ in range(max(0, args.warmup_per_prompt)):
                call_server(args.server_url, prompt, n_pred, grammar, args.timeout_s)

            trials = []
            for t in range(1, args.repeats + 1):
                resp = call_server(args.server_url, prompt, n_pred, grammar, args.timeout_s)
                raw = resp["raw"]
                wall = resp["wall"]

                if resp["err"] != "E0":
                    ans = ""
                    err = resp["err"]
                    ok = 0
                else:
                    ans = extractor(raw).strip()
                    if ans == "":
                        err = "E8"
                        ok = 0
                    else:
                        # correctness
                        a_i = safe_int(ans)
                        e_i = safe_int(expected)
                        if a_i is not None and e_i is not None:
                            ok = int(a_i == e_i)
                        else:
                            ok = int(ans == expected)

                        err = "E0" if ok == 1 else wrong_err_for_category(cat)

                print(f"{pid} {cat} #{t}/{args.repeats} time_s={wall:.3f} ans={ans} exp={expected} err={err}", flush=True)

                triw.writerow([pid, cat, t, ans, expected, ok, err, f"{wall:.6f}", (raw or "").replace("\n","\\n")[:2000]])
                trials.append((wall, t, ans, ok, err))

            # median wall trial for summary
            med_wall = statistics.median([w for (w, *_r) in trials])
            chosen = min(trials, key=lambda x: abs(x[0] - med_wall))
            wall, t_idx, ans, ok, err = chosen
            outw.writerow([pid, cat, expected, ans, ok, err, f"{wall:.6f}", t_idx])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("CTRL-C: stopped.")
        sys.exit(130)
