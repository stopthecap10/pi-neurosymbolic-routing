# Benchmark Datasets

This document describes the industry-standard public datasets used to construct our benchmark CSVs.

## Datasets Used

### 1. GSM8K (Grade School Math 8K)
- **Category**: WP (Word Problems)
- **Source**: [OpenAI GSM8K Repository](https://github.com/openai/grade-school-math) via HuggingFace Datasets
- **Paper**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- **License**: MIT License
- **Split Used**: `train` split (7,473 samples)
- **Format**:
  - **Prompt**: Question text from the dataset
  - **Expected Answer**: Final numeric answer extracted from the solution (after "####")
- **Description**: Grade-school level math word problems requiring multi-step reasoning to reach a numeric answer.

### 2. Synthetic Arithmetic Problems
- **Category**: AR (Arithmetic)
- **Source**: Deterministically generated synthetic problems
- **Format**:
  - **Prompt**: Simple arithmetic expressions (addition, subtraction, multiplication)
  - **Expected Answer**: Integer result
- **Complexity Tiers**:
  - Easy (1st third): Operands 1-100
  - Medium (2nd third): Operands 10-500
  - Hard (3rd third): Operands 100-1000
- **Description**: Basic arithmetic expressions with guaranteed integer answers. Problems are generated deterministically using the specified seed to ensure reproducibility.

### 3. Synthetic Algebra Problems
- **Category**: ALG (Algebra)
- **Source**: Deterministically generated synthetic problems
- **Format**:
  - **Prompt**: Linear equations (e.g., "x = a + b", "x + a = b", "x - a = b")
  - **Expected Answer**: Integer solution for x
- **Complexity Tiers**:
  - Easy (1st third): Values 1-100
  - Medium (2nd third): Values 10-300
  - Hard (3rd third): Values 50-1000
- **Description**: Simple linear algebra problems with guaranteed integer solutions. Generated deterministically for reproducibility.

### 4. Synthetic Logic Problems
- **Category**: LOG (Logical Reasoning)
- **Source**: Deterministically generated synthetic problems
- **Format**:
  - **Prompt**: Logical rules and facts with a query
  - **Expected Answer**: "Yes" or "No"
- **Problem Types**:
  - Transitive reasoning (All A are B, All B are C → Is X a C?)
  - Negation (No A are B, X is A → Is X a B?)
  - Conditional reasoning (If P then Q, P is true → Is Q true?)
- **Description**: Basic deductive reasoning problems with clear Yes/No answers. Generated deterministically for reproducibility.

## Why Synthetic Data?

The benchmark currently uses synthetic data for AR, ALG, and LOG categories due to:
- Dataset loading script deprecation in HuggingFace Datasets library
- Unavailability of certain datasets (DeepMind Math, ProofWriter) via standard APIs
- Need for simple, deterministic, and reproducible benchmarks

**Advantages of our synthetic approach:**
- Fully deterministic and reproducible
- Guaranteed correct answers (no labeling errors)
- Controlled difficulty progression
- No licensing restrictions
- Efficient generation without large downloads

**Note**: The WP category uses real GSM8K problems to maintain alignment with industry-standard benchmarks for word problem reasoning.

## Sampling Strategy

All benchmarks are generated using deterministic sampling with a fixed random seed:
- **Default Seed**: 1234
- **Sampling Method**: Stratified random sampling from the train split of each dataset
- **Category Distribution**: Equal distribution across all four categories (AR, ALG, LOG, WP)

### Tier Configurations

| Tier | Total Samples | Per Category | File |
|------|---------------|--------------|------|
| Tier 1 | 40 | 10 | `industry_tier1_40.csv` |
| Tier 2 | 400 | 100 | `industry_tier2_400.csv` |
| Tier 3 | 1000 | 250 | `industry_tier3_1000.csv` |

## Reproducibility

To regenerate the benchmark CSVs:

```bash
python3 src/build_benchmark.py --seed 1234 --output-dir data
```

This will:
1. Download/load the required datasets
2. Parse and format prompts according to our specification
3. Sample deterministically using the specified seed
4. Generate all three tier CSV files

## Data Preprocessing

### Prompt Formatting

All prompts follow these conventions to match our grammar constraints:

- **AR/ALG/WP** (numeric answer): Prompts end with `"Answer with only the final number."`
- **LOG** (Yes/No answer): Prompts end with `"Answer with only Yes or No."`

### Answer Extraction

- **GSM8K**: Extract the final numeric value from the answer field (typically after "####")
- **DeepMind Math**: Use the provided answer field directly
- **ProofWriter**: Map boolean entailment to "Yes"/"No"

### ID Format

IDs are deterministic and follow the pattern: `{CATEGORY}{NUMBER:03d}`
- AR001, AR002, ..., AR250
- ALG001, ALG002, ..., ALG250
- LOG001, LOG002, ..., LOG250
- WP001, WP002, ..., WP250

## Validation

Run validation checks on generated CSVs:

```bash
# Validate tier 1 (40 total, 10 per category)
python3 src/validate_prompt_csv.py --csv data/industry_tier1_40.csv --total 40 --per_cat 10

# Validate tier 2 (400 total, 100 per category)
python3 src/validate_prompt_csv.py --csv data/industry_tier2_400.csv --total 400 --per_cat 100

# Validate tier 3 (1000 total, 250 per category)
python3 src/validate_prompt_csv.py --csv data/industry_tier3_1000.csv --total 1000 --per_cat 250
```

## Quality Checks

Inspect random samples:

```bash
python3 src/smoke_test_samples.py --csv data/industry_tier2_400.csv
```

This will print 2 random samples from each category for manual review.

## Citation

If you use these benchmarks, please cite the GSM8K dataset for the WP category:

```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

For the synthetic AR, ALG, and LOG categories, please cite this repository and note that the problems are deterministically generated.
