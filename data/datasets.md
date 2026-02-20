# Benchmark Datasets (Official - Section 3 Protocol)

This document describes the **real public datasets** used to construct official evaluation splits for the Pi Neurosymbolic Routing project.

**All splits follow Section 3 protocol**: no synthetic prompts, no paraphrasing, full provenance, `source_type=dataset_raw`.

---

## Overview

| Category | Dataset | Source | Loader |
|----------|---------|--------|--------|
| **AR** (Arithmetic) | DeepMind Mathematics v1.0 | Google Storage (raw tarball) | tarball parser |
| **ALG** (Algebra) | DeepMind Mathematics v1.0 | Google Storage (raw tarball) | tarball parser |
| **WP** (Word Problems) | GSM8K | `openai/gsm8k` (HuggingFace) | datasets lib |
| **LOG** (Logical Entailment) | RuleTaker | `tasksource/ruletaker` (HuggingFace) | datasets lib |

---

## 1. DeepMind Mathematics v1.0 - AR & ALG Categories

**Source**: https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz

**Paper**: [Analysing Mathematical Reasoning Abilities of Neural Models](https://arxiv.org/abs/1904.01557) (Saxton et al., 2019)

**License**: Apache 2.0

**Version**: v1.0 (frozen)

**Loader**: Raw tarball download + text file parsing (alternating question/answer lines)

**Why raw tarball?** The HuggingFace `deepmind/math_dataset` wrapper uses deprecated dataset scripts (`trust_remote_code` removed in datasets>=4.0). The raw tarball from Google Storage is the primary authoritative source.

### AR (Arithmetic) Modules

| Module | Records (train-easy) |
|--------|---------------------|
| `arithmetic__add_or_sub` | ~170K |
| `arithmetic__add_sub_multiple` | ~667K |
| `arithmetic__mul` | ~212K |
| `arithmetic__div` | ~398K |
| `arithmetic__mixed` | ~333K |
| **Total** | **~1.78M** |

### ALG (Algebra) Modules

| Module | Records (train-easy) |
|--------|---------------------|
| `algebra__linear_1d` | ~667K |
| `algebra__linear_2d` | ~667K |
| `algebra__polynomial_roots` | ~191K |
| **Total** | **~1.52M** |

### Field Mapping

- `prompt_text` ← original `question` line (whitespace-normalized only)
- `ground_truth` ← original `answer` line (numeric-only filter applied)
- `source_module` ← module name (e.g., `arithmetic__add_or_sub`)
- Difficulty: `train-easy` (suitable for edge models)

### Filtering

- Non-numeric answers excluded (fractions, expressions, etc.)
- Empty questions/answers excluded
- Duplicates removed by (source, record_id) and text hash

---

## 2. GSM8K - WP Category

**Source**: `openai/gsm8k` on HuggingFace (config: `main`, split: `test`)

**Paper**: [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168) (Cobbe et al., 2021)

**License**: MIT License

**Split Used**: `test` (1,319 samples)

### Field Mapping

- `prompt_text` ← `question` field (whitespace-normalized only)
- `ground_truth` ← final number extracted after `####` separator
- `source_answer_raw` ← full answer with rationale (for audit)

### Filtering

- Must contain `####` separator
- Final answer must be integer (non-numeric excluded)
- Empty questions/answers excluded

### Example

```
Question: "Janet's ducks lay 16 eggs per day..."
Answer: "Janet sells 16 - 3 - 4 = 9 duck eggs... #### 18"
ground_truth: "18"
```

---

## 3. RuleTaker - LOG Category

**Source**: `tasksource/ruletaker` on HuggingFace (split: `train`)

**Paper**: [Transformers as Soft Reasoners over Language](https://arxiv.org/abs/2002.05867) (Clark et al., 2020)

**Split Used**: `train` (480,152 samples before dedup)

### Field Mapping

- `prompt_text` ← fixed template: `{context}\n\nQuestion: {question}`
- `ground_truth` ← deterministic label mapping:
  - `"entailment"` → `"Yes"`
  - `"not entailment"` → `"No"`
- `source_answer_raw` ← original label string
- `source_module` ← depth config (e.g., `depth-1`, `depth-2`)

### Filtering

- Empty context/question excluded
- Invalid labels excluded
- Duplicates removed by text hash

### Example

```
Context: "Anne is quiet. Bob is kind..."
Question: "Bob is kind."
Label: "entailment"
ground_truth: "Yes"
```

---

## Tier Structure (Section 3.3)

| Tier | Per Category | Total | Role |
|------|-------------|-------|------|
| T1 | 10 | 40 | Smoke + design decisions (official but small) |
| T2 | 50 | 200 | Development + calibrator training |
| T3 | 75 | 300 | Final held-out claims only |

### Sampling Order (Section 3.9)

1. T3 sampled first (largest held-out pool)
2. T2 from remaining
3. T1 from remaining

This guarantees zero cross-tier overlap by construction.

### Seeds

- Global seed: 42
- AR: 42, ALG: 43, WP: 42, LOG: 42
- Tier offsets: stable MD5 hash of tier name

---

## Output Files

- `data/splits/industry_tier1_40.csv` - 40 rows
- `data/splits/industry_tier2_200.csv` - 200 rows
- `data/splits/industry_tier3_300.csv` - 300 rows
- `data/splits/split_manifest.json` - full build metadata
- `data/splits/split_build_report.md` - human-readable report

---

## Reproducibility

```bash
# Regenerate identical splits
python3 src/build_official_splits.py --out_dir data/splits --seed 42 --cache_dir data/cache

# Validate datasets are accessible
python3 src/validate_dataset_builder.py
```

Running with same seed produces byte-for-byte identical CSVs (given same dataset versions).

---

## Schema (Section 3.4)

Every row in all tier CSVs contains:

| Field | Description |
|-------|-------------|
| `prompt_id` | Unique ID (e.g., `AR_000001`) |
| `category` | One of: AR, ALG, WP, LOG |
| `dataset_name` | Short name (deepmind_math, gsm8k, ruletaker) |
| `dataset_source` | Full URL or HF path |
| `dataset_version` | Version string |
| `source_split` | Original split (train-easy, test, train) |
| `source_record_id` | Stable source identifier |
| `source_type` | Always `dataset_raw` |
| `field_map_version` | `fm_v1.0` |
| `prompt_text` | Original question (whitespace-normalized) |
| `ground_truth` | Target answer (numeric or Yes/No) |
| `source_answer_raw` | Original answer from source |
| `source_module` | Module/config info |
| `source_meta_json` | Additional metadata as JSON |

---

## Citation

**DeepMind Mathematics**:
```bibtex
@article{saxton2019analysing,
  title={Analysing Mathematical Reasoning Abilities of Neural Models},
  author={Saxton, David and Grefenstette, Edward and Hill, Felix and Kohli, Pushmeet},
  journal={arXiv preprint arXiv:1904.01557},
  year={2019}
}
```

**GSM8K**:
```bibtex
@article{cobbe2021training,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and others},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

**RuleTaker**:
```bibtex
@article{clark2020transformers,
  title={Transformers as Soft Reasoners over Language},
  author={Clark, Peter and Tafjord, Oyvind and Richardson, Kyle},
  journal={arXiv preprint arXiv:2002.05867},
  year={2020}
}
```

---

## License Notes

- **DeepMind Mathematics**: Apache 2.0
- **GSM8K**: MIT License
- **RuleTaker**: CC BY 4.0

All generated CSV files are derived works and should be attributed to original dataset authors.

---

## Dev Dataset (Archived)

The original `data/splits/tier1_mini.csv` (20 manually written prompts) was used for development/debugging only. It has been archived to `archive/tier1_mini_dev/` and is NOT used for official claims. See `archive/tier1_mini_dev/README.md` for details.
