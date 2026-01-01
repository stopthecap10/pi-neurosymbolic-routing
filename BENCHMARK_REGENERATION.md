# Benchmark Regeneration Guide

This document provides the exact commands to regenerate the industry-standard benchmark CSVs on your Raspberry Pi.

## Overview

The benchmark system has been updated to use:
- **WP (Word Problems)**: Real GSM8K dataset from HuggingFace
- **AR (Arithmetic)**: Deterministically generated synthetic problems
- **ALG (Algebra)**: Deterministically generated synthetic problems
- **LOG (Logic)**: Deterministically generated synthetic problems

All generation is **fully deterministic** and **reproducible** using seed 1234.

## Prerequisites

Ensure you have the required Python packages:

```bash
pip3 install datasets --break-system-packages
```

(The `--break-system-packages` flag is needed on Raspberry Pi OS with externally managed Python environments)

## Regenerating All Benchmarks

### Single Command (All Tiers)

```bash
python3 src/build_benchmark.py --seed 1234 --output-dir data
```

This will generate:
- `data/industry_tier1_40.csv` (40 samples: 10 per category)
- `data/industry_tier2_400.csv` (400 samples: 100 per category)
- `data/industry_tier3_1000.csv` (1000 samples: 250 per category)

### With Custom Seed

To use a different random seed:

```bash
python3 src/build_benchmark.py --seed 9999 --output-dir data
```

## Validation

After generation, validate the CSVs:

```bash
# Validate Tier 1
python3 src/validate_prompt_csv.py --csv data/industry_tier1_40.csv --total 40 --per_cat 10

# Validate Tier 2
python3 src/validate_prompt_csv.py --csv data/industry_tier2_400.csv --total 400 --per_cat 100

# Validate Tier 3
python3 src/validate_prompt_csv.py --csv data/industry_tier3_1000.csv --total 1000 --per_cat 250
```

Expected output for each:
```
=== Category Distribution ===
OK AR: [count]
OK ALG: [count]
OK LOG: [count]
OK WP: [count]

=== Content Validation ===
OK: All prompts non-empty
OK: All expected answers non-empty
OK: All LOG answers are Yes or No
OK: All numeric answers are valid integers

=== Summary ===
✓ ALL CHECKS PASSED
```

## Quality Inspection

Inspect random samples from each category:

```bash
python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv
```

This prints 2 random samples per category for manual review.

To see different samples, change the seed:

```bash
python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv --seed 99
```

## File Structure

After regeneration, you will have:

```
data/
├── industry_tier1_40.csv      # 40 samples (10 per category)
├── industry_tier2_400.csv     # 400 samples (100 per category)
├── industry_tier3_1000.csv    # 1000 samples (250 per category)
└── datasets.md                # Dataset documentation
```

## CSV Format

Each CSV has the following structure:

```csv
id,category,prompt,expected_answer
AR001,AR,"Compute the value of the following expression.
Answer with only the final number.

99 + 46
",145
```

### Categories

- **AR**: Arithmetic (addition, subtraction, multiplication)
- **ALG**: Algebra (linear equations: x = a op b, x + a = b, x - a = b)
- **LOG**: Logic (Yes/No deductive reasoning)
- **WP**: Word Problems (GSM8K grade-school math)

### ID Format

IDs follow the pattern `{CATEGORY}{NUMBER:03d}`:
- AR001, AR002, ..., AR250
- ALG001, ALG002, ..., ALG250
- LOG001, LOG002, ..., LOG250
- WP001, WP002, ..., WP250

## Prompt Suffixes

All prompts end with the appropriate instruction to match grammar constraints:

- **Numeric categories** (AR, ALG, WP): `"Answer with only the final number."`
- **Logic category** (LOG): `"Answer with only Yes or No."`

## Expected Answers

- **AR/ALG/WP**: Integer strings (e.g., "145", "-133", "74")
- **LOG**: Exactly "Yes" or "No"

## Determinism Guarantee

Running the generation with the same seed will **always** produce identical CSVs:

```bash
# These two runs will produce byte-for-byte identical files
python3 src/build_benchmark.py --seed 1234 --output-dir data
python3 src/build_benchmark.py --seed 1234 --output-dir data
```

## Notes

- The WP category downloads real problems from GSM8K via HuggingFace (requires internet)
- AR, ALG, LOG categories are generated locally (no internet required)
- Generation typically takes 30-60 seconds depending on network speed
- All samples are guaranteed to have correct answers (no labeling errors)

## Troubleshooting

### datasets library not found

```bash
pip3 install datasets --break-system-packages
```

### Network issues with GSM8K

The script will fail if it cannot download GSM8K. Ensure internet connectivity and try again.

### Validation fails

If validation fails, regenerate the CSVs:

```bash
python3 src/build_benchmark.py --seed 1234 --output-dir data
```

Then validate again.

## Files Modified/Created

This benchmark overhaul created/modified the following files:

**Created:**
- `src/build_benchmark.py` - Main benchmark generation script
- `src/smoke_test_samples.py` - Sample inspection tool
- `data/datasets.md` - Dataset documentation
- `BENCHMARK_REGENERATION.md` - This file

**Modified:**
- `src/validate_prompt_csv.py` - Enhanced with content validation

**Generated:**
- `data/industry_tier1_40.csv`
- `data/industry_tier2_400.csv`
- `data/industry_tier3_1000.csv`

## For More Information

See `data/datasets.md` for detailed information about the datasets and generation methodology.
