import csv
import argparse
from collections import defaultdict

CAT_ORDER = ["AR", "ALG", "LOG", "WP"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--map_csv", required=True)
    args = ap.parse_args()

    with open(args.in_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    buckets = defaultdict(list)
    for r in rows:
        cat = (r["category"] or "").strip().upper()
        buckets[cat].append(r)

    out_rows = []
    mapping = []

    for cat in CAT_ORDER:
        items = buckets.get(cat, [])
        for i, r in enumerate(items, 1):
            old_id = r["id"]
            new_id = f"{cat}{i:03d}"
            r2 = dict(r)
            r2["id"] = new_id
            out_rows.append(r2)
            mapping.append({"new_id": new_id, "old_id": old_id, "category": cat})

    # Write outputs
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","category","prompt","expected_answer"])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in ["id","category","prompt","expected_answer"]})

    with open(args.map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["new_id","old_id","category"])
        w.writeheader()
        for m in mapping:
            w.writerow(m)

if __name__ == "__main__":
    main()
