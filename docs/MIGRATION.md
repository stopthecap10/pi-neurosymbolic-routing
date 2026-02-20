# Migration Guide: Refactored Code Structure

## Overview

The codebase has been refactored into a standard Python package structure for better organization, maintainability, and extensibility.

## What Changed

### Directory Structure

**Before:**
```
pi-neurosymbolic-routing/
├── src/
│   ├── run_baseline_phi2_server.py
│   ├── run_phi2_server_runner_*.py (multiple variants)
│   ├── run_hybrid_v1.py
│   ├── run_hybrid_v2.py
│   ├── build_benchmark.py
│   ├── *.bak files (15+ backup files)
│   └── ... (30+ files)
├── data/
│   ├── industry_tier*.csv
│   ├── *_idmap.csv
│   └── legacy/ (old files)
├── grammars/
│   └── grammar*.gbnf (33 files)
└── test_*.py (at root)
```

**After:**
```
pi-neurosymbolic-routing/
├── src/
│   └── pi_neuro_routing/           # Proper Python package
│       ├── core/                   # Core logic
│       │   ├── config.py
│       │   ├── extraction.py
│       │   └── phi2_runner.py
│       ├── runners/                # Specific implementations
│       ├── benchmarks/             # Benchmark generation
│       ├── analysis/               # Analysis tools
│       ├── utils/                  # Utilities
│       └── cli/                    # CLI interface
│           └── run_experiment.py
├── data/
│   └── benchmarks/                 # Organized benchmark data
│       ├── industry_tier*.csv
│       └── idmaps/
├── grammars/
│   ├── final/                      # Production grammars
│   └── experimental/               # Experimental variants
├── tests/                          # Proper test directory
├── archive/                        # Archived files (not deleted!)
│   ├── backup_scripts/
│   ├── legacy_data/
│   └── deprecated/
├── setup.py
└── pyproject.toml
```

### File Locations

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `src/run_baseline_phi2_server.py` | `src/pi_neuro_routing/core/phi2_runner.py` | Refactored into base class |
| `src/run_phi2_server_runner_*.py` | Use new CLI | Consolidated into unified interface |
| `src/build_benchmark.py` | `src/pi_neuro_routing/benchmarks/build_benchmark.py` | Copied |
| `src/validate_prompt_csv.py` | `src/pi_neuro_routing/benchmarks/validate.py` | Copied |
| `src/make_industry_prompts.py` | `src/pi_neuro_routing/benchmarks/generators.py` | Copied |
| `src/find_case_studies.py` | `src/pi_neuro_routing/analysis/case_studies.py` | Copied |
| `src/make_sheet_table.py` | `src/pi_neuro_routing/analysis/table_maker.py` | Copied |
| `src/summarize_results*.py` | `src/pi_neuro_routing/analysis/summarize.py` | Merged |
| `src/log_run_energy.py` | `src/pi_neuro_routing/utils/energy.py` | Copied |
| `src/debug_failures.py` | `src/pi_neuro_routing/utils/debug.py` | Copied |
| `test_*.py` (root) | `tests/test_*.py` | Moved |
| `data/industry_tier*.csv` | `data/benchmarks/` | Organized |
| `data/*_idmap.csv` | `data/benchmarks/idmaps/` | Organized |
| `grammars/grammar_phi2_*_final.gbnf` | `grammars/final/` | Renamed, organized |
| All `.bak` files | `archive/backup_scripts/` | Archived (not deleted) |

## How to Use the Refactored Code

### Installation

The package can now be installed in development mode:

```bash
pip install -e .
```

### Running Experiments

**Old way** (still works with original files in archive):
```bash
python src/run_baseline_phi2_server.py --csv data/industry_tier2_400.csv ...
python src/run_phi2_server_runner_clean_nogrammar.py --csv data/industry_tier2_400.csv ...
```

**New way** (unified CLI):
```bash
# Baseline with grammar
python -m pi_neuro_routing.cli.run_experiment --mode baseline --grammar \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_grammar.csv

# Baseline without grammar
python -m pi_neuro_routing.cli.run_experiment --mode baseline --no-grammar \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_nogrammar.csv

# Both variants
python -m pi_neuro_routing.cli.run_experiment --mode baseline --both \
    --csv data/benchmarks/industry_tier2_400.csv \
    --out results/baseline_both.csv \
    --trials_out results/baseline_both_trials.csv
```

### Accessing Functions Programmatically

**Old way:**
```python
from src.run_baseline_phi2_server import extract_final_answer, error_taxonomy
```

**New way:**
```python
from pi_neuro_routing.core import extract_final_answer, error_taxonomy
from pi_neuro_routing.core import Phi2Runner, create_baseline_config

# Create and run experiments programmatically
config = create_baseline_config(use_grammar=True, verbose=True)
runner = Phi2Runner(config)
summaries, trials = runner.run_batch(prompts)
```

### Grammar Files

**Old location:**
```
grammars/grammar_phi2_answer_int_strict_final.gbnf
grammars/grammar_phi2_answer_yesno_strict_final.gbnf
```

**New location:**
```
grammars/final/int_strict_final.gbnf
grammars/final/yesno_strict_final.gbnf
```

The new CLI automatically uses these paths. If you need custom grammars:
```bash
python -m pi_neuro_routing.cli.run_experiment --mode baseline --grammar \
    --num_grammar_file grammars/experimental/custom_int.gbnf \
    --yesno_grammar_file grammars/experimental/custom_yesno.gbnf \
    ...
```

### Data Files

Benchmark CSVs have moved:

**Old:** `data/industry_tier2_400.csv`
**New:** `data/benchmarks/industry_tier2_400.csv`

Update any scripts or documentation that reference the old paths.

## What Was Archived (Not Deleted)

All deprecated files were moved to `archive/` for reference:

- **`archive/backup_scripts/`**: 15 backup scripts (.bak, .BROKEN files)
- **`archive/legacy_data/`**: 3 legacy CSV files
- **`archive/old_runs/`**: Old run outputs
- **`archive/deprecated/`**: Original runner scripts (will be moved here after testing)

Nothing was permanently deleted. You can always access old files for reference.

## What Still Uses Old Paths

These files/scripts may still reference old paths and need updating:

- `scripts/run_all.sh` - Shell runner script
- `smoke_test_*.sh` - Smoke test scripts
- Any custom scripts you've written
- Documentation in README.md or other places

## Updating Your Scripts

If you have custom scripts that import from the old structure:

**Before:**
```python
import sys
sys.path.append("src")
from run_baseline_phi2_server import error_taxonomy
```

**After:**
```python
from pi_neuro_routing.core import error_taxonomy
```

## Testing

After refactoring, run tests to ensure everything works:

```bash
# Run smoke tests (need to update paths)
./smoke_test_e8_fix.sh
./smoke_test_nogrammar.sh

# Run Python tests
pytest tests/

# Test CLI
python -m pi_neuro_routing.cli.run_experiment --help
```

## Benefits of the New Structure

1. **Standard Python Package**: Follows best practices, can be installed with pip
2. **Better Organization**: Clear separation of core logic, runners, benchmarks, analysis
3. **Unified CLI**: Single entry point instead of multiple scripts
4. **Easier Testing**: Proper tests/ directory
5. **Cleaner Codebase**: Backup and legacy files archived
6. **Import Simplicity**: Clean import paths
7. **Extensibility**: Easy to add new runners or analysis tools
8. **ISEF Ready**: Professional structure suitable for presentation

## Rollback Plan

If you need to use the old structure:

1. Original files are in `archive/deprecated/`
2. Copy them back to `src/`
3. Use old commands

The refactoring is non-destructive - all original files are preserved.

## Questions?

If you encounter issues with the refactored structure:

1. Check if paths need updating (data/, grammars/)
2. Verify you're using the new CLI interface
3. Check `archive/` for original files
4. Consult this migration guide

## Future Work

Not yet implemented in refactored structure:

- Hybrid v1 runner (use original `src/run_hybrid_v1.py`)
- Hybrid v2 runner (use original `src/run_hybrid_v2.py`)
- Some utility scripts may need path updates

These will be integrated in future iterations.
