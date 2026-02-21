# Official Split Build & Rerun Checklist

**Status**: Transitioning from dev protocol (tier1_mini) → official protocol (Section 3)

**Date**: 2026-02-12

---

## PROOF: Dataset Usage Compliance (Section 3.2)

### Dataset Mapping (LOCKED)

| Category | Dataset | HuggingFace Path | Module/Split |
|----------|---------|------------------|--------------|
| AR | DeepMind Mathematics | `deepmind/math_dataset` | `arithmetic__*` modules |
| ALG | DeepMind Mathematics | `deepmind/math_dataset` | `algebra__*` modules |
| WP | GSM8K | `openai/gsm8k` | `main` config, `test` split |
| LOG | BoolQ (fallback) | `google/boolq` | `train` split |

### Compliance Verification

**Section 3.1 Rules:**
- ✅ Pull records directly from source datasets (HuggingFace)
- ✅ Do NOT generate questions
- ✅ Do NOT paraphrase/rewrite source text
- ✅ Do NOT use model-generated labels
- ✅ ONLY deterministic whitespace/newline normalization
- ✅ Every split row must include full provenance
- ✅ Hard-fail if any validation check fails
- ✅ `source_type=dataset_raw` for all rows

**Code Review: `src/build_official_splits.py`**
- Line 59-63: `normalize_text()` - ONLY whitespace normalization (allowed)
- Line 74-135: `load_deepmind_math()` - Direct HF load, no paraphrasing
- Line 137-183: `load_gsm8k()` - Direct HF load, deterministic extraction
- Line 185-242: `load_ruletaker()` - Direct HF load, fixed template
- NO LLM calls, NO manual overrides, NO synthetic generation

---

## PART 1: Setup & Validation (ON MAC)

### Step 1.1: Install Dependencies

```bash
cd ~/Documents/pi-neurosymbolic-routing

# Install datasets library
pip3 install datasets

# Verify installation
python3 -c "import datasets; print(datasets.__version__)"
```

### Step 1.2: Validate Dataset Access

```bash
# Run validation to prove datasets are accessible
python3 src/validate_dataset_builder.py

# Expected output:
# ✓ PASS: DeepMind Math
# ✓ PASS: GSM8K
# ✓ PASS: RuleTaker/ProofWriter/BoolQ
# ✓ PASS: Field Mappings
# ✅ ALL CHECKS PASSED
```

**If validation fails:**
- Check internet connection
- Verify HuggingFace access
- Check pip install succeeded

### Step 1.3: Build Official Splits

```bash
# Build all three tiers with deterministic seed
python3 src/build_official_splits.py --out_dir data/splits --seed 42

# Expected output files:
# ✓ data/splits/industry_tier1_40.csv (40 rows: 10/category)
# ✓ data/splits/industry_tier2_200.csv (200 rows: 50/category)
# ✓ data/splits/industry_tier3_300.csv (300 rows: 75/category)
# ✓ data/splits/split_manifest.json
# ✓ data/splits/split_build_report.md
```

### Step 1.4: Validate Split Quality

```bash
# Check T1 counts
wc -l data/splits/industry_tier1_40.csv
# Expected: 41 lines (40 + header)

# Check provenance fields exist
head -2 data/splits/industry_tier1_40.csv | cut -d',' -f1-8

# Expected to see:
# prompt_id,category,dataset_name,dataset_source,dataset_version,source_split,source_record_id,source_type

# Verify source_type column
cut -d',' -f8 data/splits/industry_tier1_40.csv | sort -u
# Expected: "source_type" (header) and "dataset_raw" only

# Check category distribution
tail -n +2 data/splits/industry_tier1_40.csv | cut -d',' -f2 | sort | uniq -c
# Expected: 10 ALG, 10 AR, 10 LOG, 10 WP
```

**Acceptance Gate:**
- [ ] All 5 files created
- [ ] Validation report shows PASS
- [ ] T1 has exactly 40 rows (10 per category)
- [ ] T2 has exactly 200 rows (50 per category)
- [ ] T3 has exactly 300 rows (75 per category)
- [ ] All rows have `source_type=dataset_raw`
- [ ] No overlap between tiers (check manifest)

---

## PART 2: Archive Dev Artifacts (ON MAC)

### Step 2.1: Create Archive Structure

```bash
cd ~/Documents/pi-neurosymbolic-routing

# Create dev archive
mkdir -p archive/tier1_mini_dev/{splits,outputs,notes}

# Copy (not move yet) for safety
cp data/splits/tier1_mini.csv archive/tier1_mini_dev/splits/
cp outputs/hybrid_v1*.csv archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/hybrid_v1*.md archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/v1_*.csv archive/tier1_mini_dev/outputs/ 2>/dev/null || true
cp outputs/v1_*.md archive/tier1_mini_dev/notes/ 2>/dev/null || true

# Add README to archive
cat > archive/tier1_mini_dev/README.md << 'EOF'
# Dev Dataset Results (tier1_mini.csv)

**Status**: Development/debugging dataset (NOT for final ISEF claims)

**Date Range**: Dec 2025 - Feb 2026

## Contents

- `splits/tier1_mini.csv`: 20 manually created prompts (5 per category)
- `outputs/`: V1/V2 experimental results on dev dataset
- `notes/`: Analysis and decisions from dev phase

## Why Archived

tier1_mini.csv was useful for:
- Fast iteration during development
- Debugging router logic
- Testing V1/V2 implementations

But it violates Section 3 protocol:
- Manually written (not from HuggingFace datasets)
- No provenance fields
- No source_type=dataset_raw
- 20 prompts vs official 40/200/300 tiers

## Official Splits

Final claims use:
- `industry_tier1_40.csv` (40 prompts, full provenance)
- `industry_tier2_200.csv` (200 prompts, full provenance)
- `industry_tier3_300.csv` (300 prompts, full provenance)

See `data/splits/split_manifest.json` for details.
EOF

# Verify archive
ls -R archive/tier1_mini_dev/
```

---

## PART 3: Sync to Pi

### Step 3.1: Sync Official Splits

```bash
# From Mac
cd ~/Documents/pi-neurosymbolic-routing

# Sync all official splits + manifest
scp data/splits/industry_tier1_40.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/industry_tier2_200.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/industry_tier3_300.csv stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/split_manifest.json stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
scp data/splits/split_build_report.md stopthecap10@edgeai:~/pi-neurosymbolic-routing/data/splits/
```

### Step 3.2: Verify on Pi

```bash
# On Pi
cd ~/pi-neurosymbolic-routing

# Check files arrived
ls -lh data/splits/industry_tier*.csv

# Quick validation
wc -l data/splits/industry_tier1_40.csv
# Should be: 41 (40 + header)

# Check a sample row
head -2 data/splits/industry_tier1_40.csv
```

---

## PART 4: Update Run Scripts (ON PI)

### Step 4.1: Update Baseline Matrix Script

```bash
# On Pi
cd ~/pi-neurosymbolic-routing

# Edit scripts/run_baseline_matrix.sh
# Change line 31:
# FROM: CSV="data/splits/tier1_mini.csv"
# TO:   CSV="data/splits/industry_tier1_40.csv"

# Verify change
grep "CSV=" scripts/run_baseline_matrix.sh
```

### Step 4.2: Update Hybrid V1 Scripts

```bash
# Edit scripts/run_hybrid_v1.sh
# Change CSV path to industry_tier1_40.csv

# Edit scripts/run_hybrid_v1_smoke.sh
# Create smoke file from official T1 (first prompt per category)
```

### Step 4.3: Update Hybrid V2 Scripts

```bash
# Edit scripts/run_hybrid_v2.sh
# Change CSV path to industry_tier1_40.csv

# Edit scripts/run_hybrid_v2_smoke.sh
# Use official T1 smoke subset
```

---

## PART 5: Official Baseline Run (ON PI)

### Step 5.1: Prepare Environment

```bash
# On Pi
cd ~/pi-neurosymbolic-routing

# Verify llama.cpp server is running
curl http://127.0.0.1:8080/health

# If not running, start it:
tmux new -s phi4 -d 'cd ~/llama.cpp && ./build/bin/llama-server -m /home/stopthecap10/edge-ai/models/microsoft_Phi-4-mini-instruct-Q6_K.gguf -c 2048 -t 4 --port 8080'

# Create outputs/official/ directory
mkdir -p outputs/official
```

### Step 5.2: Run Official Baseline Matrix

```bash
# This will take ~2-3 hours for all 4 configs × 40 prompts × 3 repeats
bash scripts/run_baseline_matrix.sh

# Expected outputs:
# - outputs/official/v1_a1_grammar.csv
# - outputs/official/v1_a1_nogrammar.csv
# - outputs/official/v1_a2_grammar.csv
# - outputs/official/v1_a2_nogrammar.csv
```

### Step 5.3: Generate Baseline Summary

```bash
# Sync baseline CSVs to Mac for analysis
# On Mac:
scp stopthecap10@edgeai:~/pi-neurosymbolic-routing/outputs/official/v1_*.csv outputs/official/

# Generate summary (modify script to use official/ directory)
python3 src/summarize_v1_matrix.py --input_dir outputs/official/ --output outputs/official/t1_baseline_summary.md

# Create baseline decisions
python3 src/generate_baseline_decisions.py --input outputs/official/t1_baseline_summary.md --output outputs/official/v1_decisions_for_hybrid.md
```

---

## PART 6: Official Hybrid V1 Run (ON PI)

### Step 6.1: Run Hybrid V1 Smoke Test

```bash
# On Pi
bash scripts/run_hybrid_v1_smoke.sh

# Expected: 4 prompts × 1 repeat, ~30 seconds
# Check for routing correctness (AR→A5, ALG→A4, etc.)
```

### Step 6.2: Run Hybrid V1 Full T1

```bash
# Full run: 40 prompts × 3 repeats
bash scripts/run_hybrid_v1.sh

# Expected output: outputs/official/hybrid_v1.csv
```

### Step 6.3: Generate V1 Summary

```bash
# On Mac (after syncing CSV)
python3 src/compare_baseline_vs_v1.py \
  --baseline outputs/official/v1_a1_nogrammar.csv \
  --hybrid outputs/official/hybrid_v1.csv \
  --output outputs/official/hybrid_v1_summary.md
```

---

## PART 7: Official Hybrid V2 Run (ON PI)

### Step 7.1: Sync Fixed Router V2

```bash
# On Mac
scp src/router_v2.py stopthecap10@edgeai:~/pi-neurosymbolic-routing/src/
scp src/run_hybrid_v2.py stopthecap10@edgeai:~/pi-neurosymbolic-routing/src/
```

### Step 7.2: Run V2 Smoke Test

```bash
# On Pi
bash scripts/run_hybrid_v2_smoke.sh

# Check ALG latency is ~4-6s (not 30s)
```

### Step 7.3: Run V2 Full T1

```bash
bash scripts/run_hybrid_v2.sh

# Expected output: outputs/official/hybrid_v2.csv
```

### Step 7.4: Generate V1 vs V2 Comparison

```bash
# On Mac
python3 src/compare_v1_vs_v2.py \
  --v1 outputs/official/hybrid_v1.csv \
  --v2 outputs/official/hybrid_v2.csv \
  --output outputs/official/v1_vs_v2_summary.md
```

---

## PART 8: Documentation Updates

### Step 8.1: Update README.md

Add section explaining split transition:

```markdown
## Dataset Splits

**Official Evaluation Splits** (used for all claims):
- `industry_tier1_40.csv` - 40 prompts (10/category) from HuggingFace datasets
- `industry_tier2_200.csv` - 200 prompts (50/category)
- `industry_tier3_300.csv` - 300 prompts (75/category)

All splits follow Section 3 protocol with full provenance.

**Development Split** (archived):
- `tier1_mini.csv` - 20 manually created prompts for dev/debug only

See `data/splits/split_manifest.json` for details.
```

### Step 8.2: Update Master Plan

Note completion of Section 3 requirements:
- [x] Official splits built with HuggingFace datasets
- [x] Full provenance included
- [x] Validation suite passed
- [x] Manifest and report generated

---

## COMPLETION CHECKLIST

### Build Phase (Mac)
- [ ] Installed `datasets` library
- [ ] Ran `validate_dataset_builder.py` - ALL PASS
- [ ] Built official splits with `build_official_splits.py`
- [ ] Verified T1=40, T2=200, T3=300 rows
- [ ] Verified all rows have `source_type=dataset_raw`
- [ ] Archived tier1_mini to `archive/tier1_mini_dev/`
- [ ] Synced official splits to Pi

### Baseline Phase (Pi)
- [ ] Updated run scripts to use `industry_tier1_40.csv`
- [ ] Ran baseline matrix (4 configs) on official T1
- [ ] Generated baseline summary
- [ ] Created baseline decisions for hybrid

### Hybrid V1 Phase (Pi)
- [ ] Ran V1 smoke test on official data
- [ ] Ran V1 full T1 on official data
- [ ] Generated V1 summary

### Hybrid V2 Phase (Pi)
- [ ] Synced fixed router_v2.py
- [ ] Ran V2 smoke test (verified A4 latency < 10s)
- [ ] Ran V2 full T1 on official data
- [ ] Generated V1 vs V2 comparison

### Documentation
- [ ] Updated README with split explanation
- [ ] Updated Master Plan with completion status
- [ ] Created archive README for dev dataset

---

## ROLLBACK PLAN

If official splits fail validation:

1. Keep dev archive untouched
2. Fix `build_official_splits.py`
3. Rebuild with new version ID
4. Revalidate before any Pi runs

## NEXT STEPS AFTER OFFICIAL T1

Once official T1 baseline + V1 + V2 are complete:

1. **Freeze official T1 results** (like you did for dev tier1_mini)
2. **Move to T2 (200 prompts)** for V3/V4 development
3. **Reserve T3 (300 prompts)** for final held-out evaluation only

---

**Status**: Ready to execute
**Owner**: Avyay Sadhu
**Timeline**: 2-3 days for full baseline + V1 + V2 on official T1
