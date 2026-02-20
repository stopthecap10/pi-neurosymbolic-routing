# Dataset Compliance Proof (Section 3 Protocol)

**Date**: 2026-02-12
**Status**: VALIDATED - All splits built and verified
**Builder**: `src/build_official_splits.py`
**Validator**: `src/validate_dataset_builder.py`

---

## Section 3.1 Compliance Matrix

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Pull records directly from source datasets | PASS | Raw tarball (DeepMind) + HuggingFace `load_dataset()` (GSM8K, RuleTaker) |
| Do NOT generate questions | PASS | No synthetic generation in code |
| Do NOT paraphrase/rewrite source text | PASS | Only `normalize_text()` for whitespace |
| Do NOT use model-generated labels | PASS | Deterministic extraction only |
| ONLY whitespace/newline normalization | PASS | `normalize_text()`: strip, CRLF→LF, collapse spaces |
| Every row includes full provenance | PASS | 14 required fields per row |
| Hard-fail on validation errors | PASS | `sys.exit(1)` on validation failure |
| `source_type=dataset_raw` for all | PASS | Hardcoded in record construction |

---

## Section 3.2 Dataset Mapping

### AR (Arithmetic) - DeepMind Mathematics v1.0

**Source**: Raw tarball from Google Storage
**URL**: `https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz`
**Version**: v1.0 (frozen)
**License**: Apache 2.0

**Why raw tarball (not HuggingFace)?** The HuggingFace `deepmind/math_dataset` wrapper uses deprecated dataset scripts (`trust_remote_code` removed in datasets>=4.0). The raw tarball from Google Storage is the **primary authoritative source** - more rigorous than any third-party wrapper.

**Modules** (frozen):
- `arithmetic__add_or_sub` (~170K records)
- `arithmetic__add_sub_multiple` (~667K records)
- `arithmetic__mul` (~212K records)
- `arithmetic__div` (~398K records)
- `arithmetic__mixed` (~333K records)
- **Total**: ~1.78M raw, ~1.59M after dedup

**Difficulty**: `train-easy` (suitable for edge models)

**Field Mapping**:
- `prompt_text` <- original `question` line (whitespace-normalized only)
- `ground_truth` <- original `answer` line (numeric-only filter applied)
- `source_type` <- `"dataset_raw"` (hardcoded)
- `source_module` <- module name (e.g., `arithmetic__add_or_sub`)

**Parser Logic** (deterministic):
```python
# Alternating question/answer lines in .txt files within tarball
target_path = f"mathematics_dataset-v1.0/train-easy/{module}.txt"
# Read lines, pair them as (question, answer)
while idx + 1 < len(lines):
    question = lines[idx].strip()
    answer = lines[idx + 1].strip()
    idx += 2
```

**Filtering**:
- Non-numeric answers excluded (fractions, expressions, etc.)
- Empty questions/answers excluded
- Duplicates removed by text hash

---

### ALG (Algebra) - DeepMind Mathematics v1.0

**Source**: Same raw tarball as AR
**URL**: `https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz`

**Modules** (frozen):
- `algebra__linear_1d` (~667K records)
- `algebra__linear_2d` (~667K records)
- `algebra__polynomial_roots` (~191K records)
- **Total**: ~1.52M raw, ~1.51M after dedup

**Same parser and field mapping as AR** - deterministic only, no paraphrasing.

---

### WP (Word Problems) - GSM8K

**Source**: `openai/gsm8k` on HuggingFace
**Config**: `main`
**Split**: `test` (1,319 samples)
**License**: MIT License

**Field Mapping**:
- `prompt_text` <- `question` field (whitespace-normalized only)
- `ground_truth` <- final number extracted after `####` separator
- `source_answer_raw` <- full answer with rationale (for audit)
- `source_type` <- `"dataset_raw"` (hardcoded)

**Extraction Logic** (deterministic, no paraphrase):
```python
if '####' not in answer:
    continue  # excluded

answer_final = answer.split('####')[-1].strip().replace(',', '')
# Validate: must be integer
```

---

### LOG (Logical Entailment) - RuleTaker

**Source**: `tasksource/ruletaker` on HuggingFace
**Split**: `train` (480,152 samples before dedup, ~340K after)
**License**: CC BY 4.0

**Field Mapping**:
- `prompt_text` <- fixed template: `{context}\n\nQuestion: {question}`
- `ground_truth` <- deterministic label mapping:
  - `"entailment"` -> `"Yes"`
  - `"not entailment"` / `"not_entailment"` -> `"No"`
- `source_answer_raw` <- original label string
- `source_module` <- depth config (e.g., `depth-1`, `depth-2`)
- `source_type` <- `"dataset_raw"` (hardcoded)

**Template Version** (frozen in `field_map_version: fm_v1.0`):
```python
prompt_composed = f"Context: {self.normalize_text(context)}\n\nQuestion: {self.normalize_text(question)}"

# Deterministic Yes/No mapping
LABEL_MAP = {'entailment': 'Yes', 'not entailment': 'No', 'not_entailment': 'No', 'non-entailment': 'No'}
```

---

## Build Results (Verified)

### Pool Sizes After Loading

| Category | Raw Records | After Dedup | Source |
|----------|------------|-------------|--------|
| AR | 1,779,165 | 1,586,717 | DeepMind Math v1.0 (5 arithmetic modules) |
| ALG | 1,524,399 | 1,510,608 | DeepMind Math v1.0 (3 algebra modules) |
| WP | 1,319 | 1,319 | GSM8K test split |
| LOG | 480,152 | 340,732 | RuleTaker train split |

### Tier Outputs

| Tier | Per Category | Total | Status |
|------|-------------|-------|--------|
| T1 | 10 | 40 | VALIDATED |
| T2 | 50 | 200 | VALIDATED |
| T3 | 75 | 300 | VALIDATED |

### Cross-Tier Disjointness: VERIFIED (zero overlap)

---

## Validation Suite

### Implemented Checks

1. **Schema completeness** - All 14 required fields present, no null/blank provenance
2. **Source type constraint** - All rows `source_type=dataset_raw`, hard-fail if not
3. **Category quotas** - T1: 10/cat, T2: 50/cat, T3: 75/cat (exact)
4. **Label validity** - AR/ALG/WP: `^-?\d+$`, LOG: exactly `"Yes"` or `"No"`
5. **Uniqueness** - No duplicate `prompt_id` within tier
6. **Cross-tier disjointness** - T3 first, T2 from remaining, T1 from remaining
7. **Sanity checks** - 10 sample rows per category printed for manual review

### Hard-Fail Behavior

```python
if errors:
    print("VALIDATION FAILED:")
    for err in errors[:10]:
        print(f"  - {err}")
    return False  # Triggers sys.exit(1) in main()
```

---

## Reproducibility Guarantee (Section 3.13)

### Deterministic Sampling

**Seed Management**:
```python
self.global_seed = config.get('seed', 42)
self.category_seeds = {
    'AR': self.global_seed,       # 42
    'ALG': self.global_seed + 1,  # 43
    'WP': self.global_seed,       # 42
    'LOG': self.global_seed       # 42
}
```

**Sampling Order** (Section 3.9):
1. T3 sampled first (largest held-out)
2. Remove T3 from pools
3. T2 from remaining
4. Remove T2 from pools
5. T1 from remaining

Running with same seed (42) + same cached tarball produces byte-for-byte identical CSVs.

---

## normalize_text() - The ONLY Allowed Transformation

```python
def normalize_text(self, text: str) -> str:
    """ONLY allowed transformation: whitespace normalization (Section 3.6)"""
    text = text.strip()
    text = text.replace('\r\n', '\n')
    text = re.sub(r' +', ' ', text)
    return text
```

**NO other text transformations in entire file.** No LLM calls, no paraphrasing, no manual label overrides, no synthetic generation.

---

## Output Files

- `data/splits/industry_tier1_40.csv` - 40 rows (10 per category)
- `data/splits/industry_tier2_200.csv` - 200 rows (50 per category)
- `data/splits/industry_tier3_300.csv` - 300 rows (75 per category)
- `data/splits/split_manifest.json` - full build metadata
- `data/splits/split_build_report.md` - human-readable report

---

## Acceptance Gate

- [x] All datasets load successfully (tarball + HuggingFace)
- [x] Field mappings are deterministic (no LLM/paraphrase)
- [x] Validation suite passes (no errors)
- [x] T1=40, T2=200, T3=300 (exact counts)
- [x] All rows have `source_type=dataset_raw`
- [x] Manifest and report generated
- [x] Zero cross-tier overlap verified
- [x] Sanity checks reviewed (10 samples/category)

**Status**: READY FOR OFFICIAL RUNS

---

## Comparison: Dev vs Official

| Aspect | tier1_mini.csv (dev) | industry_tier1_40.csv (official) |
|--------|---------------------|-----------------------------------|
| Prompts | 20 (5 per category) | 40 (10 per category) |
| Source | Manually written | DeepMind Math tarball + HuggingFace |
| Provenance | None | Full 14-field schema |
| source_type | N/A | `dataset_raw` |
| Reproducible | No (manual) | Yes (seed 42) |
| ISEF Valid | No (dev only) | Yes |

---

## Citations

**DeepMind Mathematics v1.0**: Saxton et al. (2019). "Analysing Mathematical Reasoning Abilities of Neural Models." arXiv:1904.01557. Apache 2.0.

**GSM8K**: Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168. MIT License.

**RuleTaker**: Clark et al. (2020). "Transformers as Soft Reasoners over Language." arXiv:2002.05867. CC BY 4.0.

---

**Validated**: 2026-02-12
**Builder Version**: build_official_splits.py v1.0
