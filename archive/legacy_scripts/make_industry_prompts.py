import argparse
import csv
import math
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset

SEED_DEFAULT = 42

INT_RE = re.compile(r"[-+]?\d+")

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

def parse_int_from_text(s: Any) -> Optional[int]:
    """Extract an integer from common dataset answer formats."""
    if s is None:
        return None
    s = _safe_str(s).strip().replace(",", "")

    # GSM8K style: rationale then "#### <answer>"
    if "####" in s:
        tail = s.split("####")[-1].strip()
        m = INT_RE.search(tail)
        return int(m.group(0)) if m else None

    # Plain integer in text
    m = INT_RE.search(s)
    return int(m.group(0)) if m else None

def float_to_int_if_close(x: Any, eps: float = 1e-9) -> Optional[int]:
    """Convert numeric result to int iff it's very close to an integer."""
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    r = round(v)
    if abs(v - r) <= eps:
        return int(r)
    return None

def reservoir_sample(
    it: Iterable[Dict[str, Any]],
    k: int,
    rng: random.Random,
    predicate,
) -> List[Dict[str, Any]]:
    """Uniform sample of size k from an iterable, with filtering predicate."""
    out: List[Dict[str, Any]] = []
    seen = 0
    for ex in it:
        if not predicate(ex):
            continue
        seen += 1
        if len(out) < k:
            out.append(ex)
        else:
            j = rng.randrange(seen)
            if j < k:
                out[j] = ex
    return out

@dataclass
class Row:
    id: str
    category: str
    prompt: str
    expected_answer: str
    source: str

def load_split(name: str, config: Optional[str], split: str, streaming: bool) -> Any:
    if config is None:
        return load_dataset(name, split=split, streaming=streaming)
    return load_dataset(name, config, split=split, streaming=streaming)

def build_AR(n: int, rng: random.Random) -> Tuple[List[Row], str]:
    # Calc-mawps provides expression + result
    name = "MU-NLPC/Calc-mawps"
    split = "train"
    ds = load_split(name, None, split=split, streaming=True)

    def ok(ex):
        expr = _safe_str(ex.get("expression")).strip()
        resi = float_to_int_if_close(ex.get("result"))
        return bool(expr) and (resi is not None)

    sample = reservoir_sample(ds, n, rng, ok)

    rows: List[Row] = []
    for i, ex in enumerate(sample, 1):
        expr = _safe_str(ex.get("expression")).strip()
        ans = float_to_int_if_close(ex.get("result"))
        assert ans is not None
        prompt = (
            "Compute the value of the following expression.\n"
            "Answer with only the final number.\n\n"
            f"{expr}\n"
        )
        rows.append(Row(
            id=f"AR{i:04d}",
            category="AR",
            prompt=prompt,
            expected_answer=str(ans),
            source=f"{name}:{split}:expression"
        ))
    note = f"AR: {name} split={split} streaming=True (used: expression->result; integers only)"
    return rows, note

def build_ALG(n: int, rng: random.Random) -> Tuple[List[Row], str]:
    # Calc-mawps provides equation (solve for x) + result
    name = "MU-NLPC/Calc-mawps"
    split = "train"
    ds = load_split(name, None, split=split, streaming=True)

    def ok(ex):
        eq = _safe_str(ex.get("equation")).strip()
        resi = float_to_int_if_close(ex.get("result"))
        # Ensure it looks like an equation with x somewhere
        return bool(eq) and ("x" in eq.lower()) and (resi is not None)

    sample = reservoir_sample(ds, n, rng, ok)

    rows: List[Row] = []
    for i, ex in enumerate(sample, 1):
        eq = _safe_str(ex.get("equation")).strip()
        ans = float_to_int_if_close(ex.get("result"))
        assert ans is not None
        prompt = (
            "Solve for x.\n"
            "Answer with only the final number.\n\n"
            f"{eq}\n"
        )
        rows.append(Row(
            id=f"ALG{i:04d}",
            category="ALG",
            prompt=prompt,
            expected_answer=str(ans),
            source=f"{name}:{split}:equation"
        ))
    note = f"ALG: {name} split={split} streaming=True (used: equation(x)->result; integers only)"
    return rows, note

def build_WP(n: int, rng: random.Random) -> Tuple[List[Row], str]:
    # Mix GSM8K + SVAMP roughly 50/50
    n_gsm = n // 2
    n_sv = n - n_gsm

    rows: List[Row] = []

    # GSM8K
    gsm_name = "openai/gsm8k"
    gsm_split = "test"
    gsm = load_split(gsm_name, "main", split=gsm_split, streaming=False)

    def gsm_ok(ex):
        q = _safe_str(ex.get("question")).strip()
        a = parse_int_from_text(ex.get("answer"))
        return bool(q) and (a is not None)

    gsm_list = [ex for ex in gsm if gsm_ok(ex)]
    rng.shuffle(gsm_list)
    gsm_pick = gsm_list[:n_gsm]

    for i, ex in enumerate(gsm_pick, 1):
        q = _safe_str(ex.get("question")).strip()
        a = parse_int_from_text(ex.get("answer"))
        assert a is not None
        prompt = (
            "Solve the following word problem.\n"
            "Answer with only the final number.\n\n"
            f"{q}\n"
        )
        rows.append(Row(
            id=f"WP_GSM{i:04d}",
            category="WP",
            prompt=prompt,
            expected_answer=str(a),
            source=f"{gsm_name}:{gsm_split}"
        ))

    # SVAMP (ChilleD/SVAMP) - fields vary; handle common ones
    sv_name = "ChilleD/SVAMP"
    sv_split = "test"
    sv = load_split(sv_name, None, split=sv_split, streaming=False)

    def sv_question(ex):
        # Common SVAMP fields: Body, Question; sometimes "question" exists
        body = _safe_str(ex.get("Body")).strip()
        ques = _safe_str(ex.get("Question")).strip()
        if not ques:
            ques = _safe_str(ex.get("question")).strip()
        if body and ques:
            return f"{body} {ques}".strip()
        return (body or ques).strip()

    def sv_answer(ex):
        # Common field: "Answer" or "answer"
        a = ex.get("Answer", ex.get("answer"))
        # Sometimes stored numeric already
        if isinstance(a, (int, float)):
            return float_to_int_if_close(a)
        return parse_int_from_text(a)

    sv_list = []
    for ex in sv:
        q = sv_question(ex)
        a = sv_answer(ex)
        if q and (a is not None):
            sv_list.append((q, a))

    rng.shuffle(sv_list)
    sv_pick = sv_list[:n_sv]

    for i, (q, a) in enumerate(sv_pick, 1):
        prompt = (
            "Solve the following word problem.\n"
            "Answer with only the final number.\n\n"
            f"{q}\n"
        )
        rows.append(Row(
            id=f"WP_SV{i:04d}",
            category="WP",
            prompt=prompt,
            expected_answer=str(a),
            source=f"{sv_name}:{sv_split}"
        ))

    note = (
        f"WP: {gsm_name} (main/{gsm_split}) + {sv_name} ({sv_split}); "
        f"integers only; ratio approx {n_gsm}:{n_sv}"
    )
    return rows, note

def build_LOG(n: int, rng: random.Random) -> Tuple[List[Row], str]:
    # BoolQ provides passage + question + boolean answer
    name = "google/boolq"
    split = "train"
    ds = load_split(name, None, split=split, streaming=False)

    rows: List[Row] = []

    def fmt_yesno(b: Any) -> str:
        return "Yes" if bool(b) else "No"

    # Filter to keep passages from being too huge on the Pi (latency sanity).
    # Keep first 900 chars of passage.
    pool = []
    for ex in ds:
        q = _safe_str(ex.get("question")).strip()
        p = _safe_str(ex.get("passage")).strip()
        a = ex.get("answer")
        if q and p and isinstance(a, (bool, int)):
            pool.append((q, p[:900], bool(a)))

    rng.shuffle(pool)
    pick = pool[:n]

    for i, (q, p, a) in enumerate(pick, 1):
        prompt = (
            "Read the passage and answer the question.\n"
            "Answer with only Yes or No.\n\n"
            "Passage:\n"
            f"{p}\n\n"
            "Question:\n"
            f"{q}\n"
        )
        rows.append(Row(
            id=f"LOG{i:04d}",
            category="LOG",
            prompt=prompt,
            expected_answer=fmt_yesno(a),
            source=f"{name}:{split}"
        ))

    note = f"LOG: {name} split={split} (passage truncated to 900 chars; Yes/No labels)"
    return rows, note

def write_csv(path: str, rows: List[Row]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["id", "category", "prompt", "expected_answer"])
        for r in rows:
            w.writerow([r.id, r.category, r.prompt, r.expected_answer])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--tier1", type=int, default=40)
    ap.add_argument("--tier2", type=int, default=400)
    ap.add_argument("--tier3", type=int, default=1000)
    args = ap.parse_args()

    if args.tier1 % 4 or args.tier2 % 4 or args.tier3 % 4:
        raise SystemExit("tier sizes must be divisible by 4 to keep 25% per category.")

    rng = random.Random(args.seed)

    per3 = args.tier3 // 4

    ar, ar_note = build_AR(per3, rng)
    alg, alg_note = build_ALG(per3, rng)
    wp, wp_note = build_WP(per3, rng)
    log, log_note = build_LOG(per3, rng)

    tier3_rows = ar + alg + log + wp  # order doesn't matter; shuffle for mixing
    rng.shuffle(tier3_rows)

    # Nested tiers: tier1 subset of tier2 subset of tier3
    tier2_rows = tier3_rows[: args.tier2]
    tier1_rows = tier3_rows[: args.tier1]

    out1 = f"{args.outdir}/industry_tier1_{args.tier1}.csv"
    out2 = f"{args.outdir}/industry_tier2_{args.tier2}.csv"
    out3 = f"{args.outdir}/industry_tier3_{args.tier3}.csv"
    notes = f"{args.outdir}/industry_dataset_notes.txt"

    write_csv(out1, tier1_rows)
    write_csv(out2, tier2_rows)
    write_csv(out3, tier3_rows)

    with open(notes, "w", encoding="utf-8") as f:
        f.write("Industry prompt set generation notes\n")
        f.write(f"generated_utc={datetime.utcnow().isoformat()}Z\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"tier1={args.tier1} tier2={args.tier2} tier3={args.tier3}\n")
        f.write("category_mix=AR:25% ALG:25% LOG:25% WP:25%\n\n")
        f.write(ar_note + "\n")
        f.write(alg_note + "\n")
        f.write(log_note + "\n")
        f.write(wp_note + "\n")
        f.write("\nfilters: integers-only for AR/ALG/WP; LOG uses Yes/No\n")
        f.write("prompt_fairness: identical prompts across systems; grammars affect decoding only\n")

    print("Wrote:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
    print(" ", notes)

if __name__ == "__main__":
    main()
