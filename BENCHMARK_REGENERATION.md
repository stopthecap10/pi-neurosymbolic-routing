# Benchmark Regeneration Guide

This guide provides exact commands to regenerate industry-standard benchmark CSVs on your Raspberry Pi using real public datasets.

## Quick Start

To regenerate all three benchmark tiers:

```bash
python3 src/build_benchmark.py --seed 1234
```

This single command will:
1. Download GSM8K, DeepMind Mathematics, and RuleTaker datasets
2. Generate all three tier CSVs with deterministic sampling
3. Cache datasets locally for future runs

**Expected Output**:
- `data/industry_tier1_40.csv` (40 samples: 10 per category)
- `data/industry_tier2_400.csv` (400 samples: 100 per category)
- `data/industry_tier3_1000.csv` (1000 samples: 250 per category)

## Prerequisites

### Install Required Packages

```bash
pip3 install datasets --break-system-packages
```

The `--break-system-packages` flag is needed on Raspberry Pi OS with externally managed Python environments.

### Verify Installation

```bash
python3 -c "import datasets; print(datasets.__version__)"
```

### Internet Connection

First-time dataset download requires internet connectivity. Datasets are cached locally at `data/cache/` for subsequent runs.

## Datasets Used

This implementation uses **100% real industry-standard datasets**:

1. **WP (Word Problems)**: GSM8K test split (1,319 samples)
   - Source: https://huggingface.co/datasets/gsm8k
   - License: MIT

2. **AR (Arithmetic)**: SVAMP train split (700 samples)
   - Source: https://huggingface.co/datasets/ChilleD/SVAMP
   - License: MIT

3. **ALG (Algebra)**: AQuA-RAT train split (97,467 samples)
   - Source: https://huggingface.co/datasets/aqua_rat
   - License: Apache 2.0

4. **LOG (Logic)**: BoolQ train split (9,427 samples)
   - Source: https://huggingface.co/datasets/google/boolq
   - License: CC BY-SA 3.0

All datasets are publicly available and well-cited in academic literature.

See `data/datasets.md` for detailed dataset information and citations.

## Command-Line Options

### Generate All Tiers (Default)

```bash
python3 src/build_benchmark.py --seed 1234
```

### Generate Specific Tier

```bash
# Tier 1 only (40 samples)
python3 src/build_benchmark.py --tier T1

# Tier 2 only (400 samples)
python3 src/build_benchmark.py --tier T2

# Tier 3 only (1000 samples)
python3 src/build_benchmark.py --tier T3
```

### Custom Seed

To use a different random seed for sampling:

```bash
python3 src/build_benchmark.py --seed 9999
```

**Note**: Using a different seed will produce different samples but maintain the same format and distribution.

### Custom Directories

```bash
python3 src/build_benchmark.py \
  --seed 1234 \
  --output-dir data \
  --cache-dir data/cache
```

**Parameters**:
- `--output-dir`: Where to write CSV files (default: `data`)
- `--cache-dir`: Where to cache downloaded datasets (default: `data/cache`)

## Validation

After generation, validate the CSVs to ensure correctness:

### Validate All Tiers

```bash
# Validate Tier 1
python3 src/validate_prompt_csv.py --csv data/industry_tier1_40.csv --total 40 --per_cat 10

# Validate Tier 2
python3 src/validate_prompt_csv.py --csv data/industry_tier2_400.csv --total 400 --per_cat 100

# Validate Tier 3
python3 src/validate_prompt_csv.py --csv data/industry_tier3_1000.csv --total 1000 --per_cat 250
```

### Expected Validation Output

```
=== Category Distribution ===
OK AR: 10
OK ALG: 10
OK LOG: 10
OK WP: 10

=== Content Validation ===
OK: All prompts non-empty
OK: All expected answers non-empty
OK: All LOG answers are Yes or No
OK: All numeric answers are valid integers

=== Summary ===
✓ ALL CHECKS PASSED
```

## Quality Inspection

Manually inspect random samples:

```bash
# Default: 2 samples per category with seed 42
python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv

# Custom sample count and seed
python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv --samples-per-cat 5 --seed 99
```

This displays sample prompts and answers for manual quality review.

## CSV Format

Each generated CSV follows this exact schema:

```csv
id,category,prompt,expected_answer
AR001,AR,"Compute the value of the following expression.
Answer with only the final number.

15 + 27
",42
```

### Categories

- **AR**: Arithmetic computations (add, subtract, multiply)
- **ALG**: Algebra equations (solve for x)
- **LOG**: Logical reasoning (Yes/No questions)
- **WP**: Word problems (grade-school math)

### ID Format

IDs are deterministic and follow the pattern `{CATEGORY}{NUMBER:03d}`:
- `AR001`, `AR002`, ..., `AR250`
- `ALG001`, `ALG002`, ..., `ALG250`
- `LOG001`, `LOG002`, ..., `LOG250`
- `WP001`, `WP002`, ..., `WP250`

### Prompt Suffixes

All prompts end with the appropriate instruction:

- **AR/ALG/WP**: `"Answer with only the final number."`
- **LOG**: `"Answer with only Yes or No."`

### Expected Answer Format

- **AR/ALG/WP**: Integer string (e.g., `"145"`, `"-22"`, `"0"`)
- **LOG**: Exactly `"Yes"` or `"No"` (case-sensitive)

## Tier Configurations

| Tier | File | Total | Per Category | Use Case |
|------|------|-------|--------------|----------|
| Tier 1 | `industry_tier1_40.csv` | 40 | 10 | Quick smoke tests |
| Tier 2 | `industry_tier2_400.csv` | 400 | 100 | Standard benchmarks |
| Tier 3 | `industry_tier3_1000.csv` | 1000 | 250 | Comprehensive evaluation |

## Determinism Guarantee

Running the benchmark builder with the same seed produces **byte-for-byte identical** CSV files:

```bash
# These two runs produce identical output
python3 src/build_benchmark.py --seed 1234
python3 src/build_benchmark.py --seed 1234
```

This ensures:
- Reproducible experiments
- Consistent comparisons across runs
- Verifiable results

## Disk Space and Runtime

**First Run** (with dataset download):
- Disk: ~500MB for cached datasets
- Time: 5-15 minutes (depending on internet speed)
- Network: ~200MB download

**Subsequent Runs** (using cache):
- Disk: No additional space (overwrites CSVs)
- Time: 30-60 seconds
- Network: No download needed

## Troubleshooting

### "ModuleNotFoundError: No module named 'datasets'"

```bash
pip3 install datasets --break-system-packages
```

### "Connection error" or "Network timeout"

Ensure internet connectivity and try again. Datasets are downloaded from HuggingFace:
- https://huggingface.co/datasets/gsm8k
- https://huggingface.co/datasets/deepmind/math_dataset
- https://huggingface.co/datasets/metaeval/ruletaker

### "WARNING: Only found X valid samples (needed Y)"

Some datasets may have fewer samples with valid integer answers. The builder will generate as many as possible. If the shortage is severe, try:
1. Using a different seed: `--seed 5678`
2. Checking dataset availability on HuggingFace

### Validation Fails

If validation fails after generation:

```bash
# Regenerate with verbose output
python3 src/build_benchmark.py --seed 1234

# Check the output logs for errors
# Then validate again
python3 src/validate_prompt_csv.py --csv data/industry_tier2_400.csv --total 400 --per_cat 100
```

### Slow Generation on Raspberry Pi

The first run downloads large datasets and may take 10-15 minutes. Subsequent runs use cached data and complete in under 1 minute.

To speed up:
1. Use `--tier T1` to generate only Tier 1 for testing
2. Ensure cache directory has sufficient space
3. Verify network speed if downloads are slow

## File Structure

After successful generation:

```
pi-neurosymbolic-routing/
├── data/
│   ├── industry_tier1_40.csv      # Generated Tier 1
│   ├── industry_tier2_400.csv     # Generated Tier 2
│   ├── industry_tier3_1000.csv    # Generated Tier 3
│   ├── datasets.md                # Dataset documentation
│   └── cache/                     # Cached datasets (auto-created)
│       ├── downloads/
│       └── ...
├── src/
│   ├── build_benchmark.py         # Main builder script
│   ├── validate_prompt_csv.py     # Validation tool
│   └── smoke_test_samples.py      # Manual inspection tool
└── BENCHMARK_REGENERATION.md      # This file
```

## Updating Benchmarks

To regenerate benchmarks with the latest dataset versions:

```bash
# Clear cache to force re-download
rm -rf data/cache/*

# Regenerate all tiers
python3 src/build_benchmark.py --seed 1234
```

**Note**: This may produce different samples if the underlying datasets have been updated on HuggingFace.

## Using Different Seeds for Experiments

To create multiple benchmark variants for cross-validation:

```bash
# Variant A (seed 1234)
python3 src/build_benchmark.py --seed 1234 --output-dir data/variant_a

# Variant B (seed 5678)
python3 src/build_benchmark.py --seed 5678 --output-dir data/variant_b

# Variant C (seed 9999)
python3 src/build_benchmark.py --seed 9999 --output-dir data/variant_c
```

Each variant will have different samples but identical format and distribution.

## Integration with Evaluation Pipeline

After generating benchmarks, use them in your evaluation runs:

```bash
# Run baseline evaluation on Tier 2
python3 src/run_baseline_phi2_server.py --csv data/industry_tier2_400.csv

# Run neurosymbolic routing on Tier 2
python3 src/run_phi2_server_runner_clean.py --csv data/industry_tier2_400.csv
```

## Version Control

The generated CSV files should be checked into version control:

```bash
git add data/industry_tier*.csv
git commit -m "Regenerate industry benchmarks with real datasets (seed 1234)"
```

**Cache directory** (`data/cache/`) should be added to `.gitignore` as it's large and auto-generated.

## Citation

If you use these benchmarks in publications, cite the original dataset papers. See `data/datasets.md` for BibTeX entries for:
- GSM8K (Cobbe et al., 2021)
- DeepMind Mathematics (Saxton et al., 2019)
- RuleTaker (Clark et al., 2020)
- ProofWriter (Tafjord et al., 2020)

## Support

For issues with:
- **Dataset loading**: Check HuggingFace status and internet connection
- **CSV format**: Run validation tool to identify specific issues
- **Sample quality**: Use smoke test tool to manually review samples

For questions about specific datasets, refer to their original papers and repositories linked in `data/datasets.md`.
