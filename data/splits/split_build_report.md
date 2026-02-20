# Split Build Report

**Build Date**: 2026-02-12T22:57:21.094483
**Builder**: src/build_official_splits.py v1.1
**Global Seed**: 42
**Category Seeds**: AR=42, ALG=43, WP=42, LOG=42

---

## Dataset Sources

| Category | Dataset | Source | Loader |
|----------|---------|--------|--------|
| AR | DeepMind Math v1.0 | Google Storage (raw tarball) | tarball |
| ALG | DeepMind Math v1.0 | Google Storage (raw tarball) | tarball |
| WP | GSM8K | `openai/gsm8k` (HuggingFace) | datasets lib |
| LOG | RuleTaker | `tasksource/ruletaker` (HuggingFace) | datasets lib |

## Load Summary

| Category | Records Loaded | Modules |
|----------|----------------|---------|
| AR | 1,779,165 | arithmetic__add_or_sub, arithmetic__add_sub_multiple, arithmetic__mul, arithmetic__div, arithmetic__mixed |
| ALG | 1,524,399 | algebra__linear_1d, algebra__linear_2d, algebra__polynomial_roots |
| WP | 1,319 | openai/gsm8k/main/test |
| LOG | 480,152 | tasksource/ruletaker/train |

## Exclusions by Reason

**AR**:
- duplicate_text: 192,448
- non_numeric: 1,554,165

**ALG**:
- duplicate_text: 13,791
- non_numeric: 475,599

**LOG**:
- duplicate_text: 139,420

## Tier Counts

| Tier | AR | ALG | WP | LOG | Total |
|------|-----|-----|-----|-----|-------|
| T1 | 10 | 10 | 10 | 10 | 40 |
| T2 | 50 | 50 | 50 | 50 | 200 |
| T3 | 75 | 75 | 75 | 75 | 300 |

## Validation Results

| Check | Result |
|-------|--------|
| Schema completeness | PASSED |
| Provenance completeness | PASSED |
| source_type = dataset_raw | PASSED |
| Category quotas exact | PASSED |
| Label validity | PASSED |
| No duplicates within tiers | PASSED |
| Cross-tier disjointness | PASSED |
| 10-row sanity check | PASSED |

## Spot Check (T1 Samples)

### AR - AR_000001
- **Source**: `https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz`
- **Record ID**: `arithmetic__add_sub_multiple_easy_94395`
- **Prompt**: Calculate 4 - (5 - (3 + 2))....
- **Answer**: `4`
- **source_type**: `dataset_raw`

### ALG - ALG_000001
- **Source**: `https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz`
- **Record ID**: `algebra__linear_2d_easy_545323`
- **Prompt**: Solve 27 = -4*d - 5*z - 9, 4*z + 12 = d for d....
- **Answer**: `-4`
- **source_type**: `dataset_raw`

### WP - WP_000001
- **Source**: `openai/gsm8k/main`
- **Record ID**: `gsm8k_test_291`
- **Prompt**: 4 adults and 8 children are to share 8 packets of chocolate bars. Each packet contains 5 chocolate bars. If each adult g...
- **Answer**: `2`
- **source_type**: `dataset_raw`

### LOG - LOG_000001
- **Source**: `tasksource/ruletaker`
- **Record ID**: `ruletaker_train_65961`
- **Prompt**: The bald eagle is green. The bald eagle is young. The cat likes the bald eagle. The dog is kind. The squirrel is big. Th...
- **Answer**: `Yes`
- **source_type**: `dataset_raw`

## Overlap Verification

Cross-tier disjointness verified by source key and text hash.
Sampling order: T3 first, T2 from remaining, T1 from remaining.
Zero overlap across all tier pairs.