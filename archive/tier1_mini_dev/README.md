# archive/tier1_mini_dev

Development dataset used before official split protocol was established.

## What this is

`splits/tier1_mini.csv` contained 20 manually written prompts (5 per category: AR, ALG, WP, LOG).
These were hardcoded in `src/build_tier1_mini.py` as Python dict literals.

The `dataset` column labels ("deepmind_math_arithmetic", "ruletaker", etc.) were style indicators
only - NOT actual HuggingFace dataset sources. No provenance fields.

## Why it was replaced

Section 3 of the Master Plan requires:
- All prompts pulled directly from source datasets (no synthetic/manual prompts)
- Full 14-field provenance schema per row
- source_type = dataset_raw
- Hard-fail validation

## Official replacement

`data/splits/industry_tier1_40.csv` - 40 prompts (10/category)
Built by `src/build_official_splits.py` from:
- AR/ALG: DeepMind Mathematics v1.0 (raw tarball, Google Storage)
- WP: openai/gsm8k (HuggingFace, test split)
- LOG: tasksource/ruletaker (HuggingFace, train split)

## Archived date

2026-02-19
