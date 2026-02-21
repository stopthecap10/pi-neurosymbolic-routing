import argparse, csv, sys, re
from collections import Counter

def validate_integer(value: str) -> bool:
    """Check if value is a valid integer string (optional leading -)."""
    return bool(re.match(r'^-?\d+$', value.strip()))

def validate_yesno(value: str) -> bool:
    """Check if value is exactly 'Yes' or 'No'."""
    return value.strip() in ["Yes", "No"]

def main():
    ap = argparse.ArgumentParser(
        description="Validate benchmark CSV format and content"
    )
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--total", type=int, required=True, help="Expected total rows")
    ap.add_argument("--per_cat", type=int, required=True, help="Expected rows per category")
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Check total count
    if len(rows) < args.total:
        print(f"FAIL total rows: got {len(rows)}, need {args.total}")
        sys.exit(1)

    rows = rows[:args.total]
    c = Counter(r["category"].strip() for r in rows)

    ok = True
    errors = []

    # Check category distribution
    print("\n=== Category Distribution ===")
    for cat in ["AR","ALG","LOG","WP"]:
        got = c.get(cat, 0)
        if got != args.per_cat:
            ok = False
            print(f"FAIL {cat}: got {got}, need {args.per_cat}")
        else:
            print(f"OK {cat}: {got}")

    # Check content validity
    print("\n=== Content Validation ===")
    empty_prompts = []
    empty_expected = []
    invalid_log_answers = []
    invalid_numeric_answers = []

    for i, row in enumerate(rows, start=1):
        row_id = row.get("id", f"row_{i}")
        category = row.get("category", "").strip()
        prompt = row.get("prompt", "").strip()
        expected = row.get("expected_answer", "").strip()

        # Check for empty prompts
        if not prompt:
            empty_prompts.append(row_id)

        # Check for empty expected answers
        if not expected:
            empty_expected.append(row_id)

        # Category-specific validation
        if category == "LOG":
            if not validate_yesno(expected):
                invalid_log_answers.append(f"{row_id} (got: '{expected}')")
        elif category in ["AR", "ALG", "WP"]:
            if not validate_integer(expected):
                invalid_numeric_answers.append(f"{row_id} (got: '{expected}')")

    # Report content issues
    if empty_prompts:
        ok = False
        print(f"FAIL: {len(empty_prompts)} empty prompts: {', '.join(empty_prompts[:5])}")
    else:
        print(f"OK: All prompts non-empty")

    if empty_expected:
        ok = False
        print(f"FAIL: {len(empty_expected)} empty expected answers: {', '.join(empty_expected[:5])}")
    else:
        print(f"OK: All expected answers non-empty")

    if invalid_log_answers:
        ok = False
        print(f"FAIL: {len(invalid_log_answers)} LOG answers not Yes/No: {', '.join(invalid_log_answers[:5])}")
    else:
        print(f"OK: All LOG answers are Yes or No")

    if invalid_numeric_answers:
        ok = False
        print(f"FAIL: {len(invalid_numeric_answers)} numeric answers invalid: {', '.join(invalid_numeric_answers[:5])}")
    else:
        print(f"OK: All numeric answers are valid integers")

    # Final result
    print("\n=== Summary ===")
    if ok:
        print("✓ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("✗ VALIDATION FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
