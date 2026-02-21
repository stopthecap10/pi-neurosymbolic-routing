import argparse, csv, random, sys
from collections import defaultdict

CATS = ["AR", "ALG", "LOG", "WP"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_cat", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shuffle_within_cat", action="store_true")
    args = ap.parse_args()

    # Load
    by_cat = defaultdict(list)
    header = None
    with open(args.in_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames
        for r in rdr:
            cat = (r.get("category","") or "").strip().upper()
            if cat in CATS:
                by_cat[cat].append(r)

    # Check enough
    for cat in CATS:
        n = len(by_cat[cat])
        if n < args.per_cat:
            print(f"FAIL: {cat} has {n}, need {args.per_cat}")
            sys.exit(2)

    rng = random.Random(args.seed)

    # Choose per category
    chosen = {}
    for cat in CATS:
        rows = list(by_cat[cat])
        if args.shuffle_within_cat:
            rng.shuffle(rows)
        chosen[cat] = rows[:args.per_cat]

    # Interleave AR,ALG,LOG,WP to keep ordering clean and balanced
    out_rows = []
    for i in range(args.per_cat):
        for cat in CATS:
            out_rows.append(chosen[cat][i])

    # Write
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"OK: wrote {len(out_rows)} rows to {args.out_csv} ({args.per_cat} each)")

if __name__ == "__main__":
    main()
