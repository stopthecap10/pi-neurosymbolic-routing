#!/usr/bin/env python3
"""
Official Split Builder - Protocol Section 3 Compliant

Builds industry_tier1_40.csv, industry_tier2_200.csv, industry_tier3_300.csv
with full provenance and validation.

NON-NEGOTIABLE RULES (Section 3.1):
1. Pull records directly from source datasets
2. Do not generate questions
3. Do not paraphrase/rewrite source text
4. Do not use model-generated labels
5. Only deterministic whitespace/newline normalization
6. Every split row must include full provenance
7. Hard-fail if any validation check fails
8. Official run mode must reject rows that are not source_type=dataset_raw

Dataset Sources (Section 3.2):
- AR:  DeepMind Mathematics v1.0 (arithmetic modules) - raw tarball from Google Storage
- ALG: DeepMind Mathematics v1.0 (algebra modules)   - raw tarball from Google Storage
- WP:  GSM8K (openai/gsm8k on HuggingFace)
- LOG: RuleTaker (tasksource/ruletaker on HuggingFace)
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not available")
    print("Install with: pip3 install datasets")
    sys.exit(1)


# =============================================================================
# FROZEN CONFIGURATION (Section 3.2 / 3.5.1)
# =============================================================================

# DeepMind Mathematics v1.0 - primary source (Google Storage)
DEEPMIND_MATH_URL = (
    "https://storage.googleapis.com/mathematics-dataset/"
    "mathematics_dataset-v1.0.tar.gz"
)
DEEPMIND_MATH_VERSION = "v1.0"

# Frozen module lists (Section 3.2)
ARITHMETIC_MODULES = [
    "arithmetic__add_or_sub",
    "arithmetic__add_sub_multiple",
    "arithmetic__mul",
    "arithmetic__div",
    "arithmetic__mixed",
]

ALGEBRA_MODULES = [
    "algebra__linear_1d",
    "algebra__linear_2d",
    "algebra__polynomial_roots",
]

# HuggingFace dataset paths
GSM8K_PATH = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"

RULETAKER_PATH = "tasksource/ruletaker"
RULETAKER_SPLIT = "train"


class SplitBuilder:
    """Official split builder following Section 3 protocol"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_seed = config.get('seed', 42)
        self.field_map_version = "fm_v1.0"
        self.schema_version = "schema_v1.0"
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))

        # Category-specific seeds (Section 3.9)
        self.category_seeds = {
            'AR': self.global_seed,
            'ALG': self.global_seed + 1,
            'WP': self.global_seed,
            'LOG': self.global_seed,
        }

        # Stats tracking
        self.stats = {
            'loaded': defaultdict(int),
            'filtered': defaultdict(lambda: defaultdict(int)),
            'sampled': defaultdict(lambda: defaultdict(int)),
        }
        self.loaded_modules = defaultdict(list)

    def compute_record_hash(self, record: Dict) -> str:
        """Compute deterministic hash for record deduplication"""
        canonical = json.dumps(record, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def normalize_text(self, text: str) -> str:
        """
        ONLY allowed transformation: whitespace normalization (Section 3.6)

        Allowed:  strip, normalize line breaks, collapse repeated spaces
        Forbidden: paraphrasing, LLM transforms, manual edits
        """
        text = text.strip()
        text = text.replace('\r\n', '\n')
        text = re.sub(r' +', ' ', text)
        return text

    # =========================================================================
    # AR / ALG: DeepMind Mathematics v1.0 (Section 3.5.1)
    # =========================================================================

    def _download_deepmind_math(self) -> Path:
        """Download and cache DeepMind Mathematics v1.0 tarball"""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tarball_path = self.cache_dir / "mathematics_dataset-v1.0.tar.gz"

        if tarball_path.exists():
            print(f"  Using cached tarball: {tarball_path}")
            return tarball_path

        print(f"  Downloading DeepMind Mathematics v1.0...")
        print(f"  URL: {DEEPMIND_MATH_URL}")
        print(f"  (This may take a few minutes, ~2.2GB)")

        urllib.request.urlretrieve(DEEPMIND_MATH_URL, str(tarball_path))
        print(f"  ✓ Downloaded to {tarball_path}")
        return tarball_path

    def _parse_deepmind_module(self, tarball_path: Path, module: str,
                                difficulty: str = "train-easy") -> List[Dict]:
        """
        Parse a single DeepMind Math module from the tarball.

        Raw format: alternating question/answer lines in text files.
        Path pattern: mathematics_dataset-v1.0/{difficulty}/{module}.txt
        """
        target_path = f"mathematics_dataset-v1.0/{difficulty}/{module}.txt"
        records = []

        try:
            with tarfile.open(str(tarball_path), 'r:gz') as tar:
                member = tar.getmember(target_path)
                f = tar.extractfile(member)
                if f is None:
                    print(f"    ERROR: Cannot read {target_path}")
                    return []

                content = f.read().decode('utf-8')
                lines = content.strip().split('\n')

                # Alternating question/answer pairs
                idx = 0
                pair_idx = 0
                while idx + 1 < len(lines):
                    question = lines[idx].strip()
                    answer = lines[idx + 1].strip()
                    idx += 2

                    if not question or not answer:
                        self.stats['filtered'][module]['empty'] += 1
                        continue

                    records.append({
                        'question': question,
                        'answer': answer,
                        'pair_idx': pair_idx,
                    })
                    pair_idx += 1

        except KeyError:
            print(f"    WARNING: {target_path} not found in tarball")
        except Exception as e:
            print(f"    ERROR reading {target_path}: {e}")

        return records

    def load_deepmind_math(self, category: str, modules: List[str]) -> List[Dict]:
        """
        Load DeepMind Mathematics v1.0 (Section 3.5.1)

        - Downloads raw tarball from Google Storage (primary source)
        - Parses train-easy difficulty for edge-model compatibility
        - prompt_text = original problem text (no alteration)
        - ground_truth = deterministic numeric answer
        - Records module in source_module
        """
        print(f"\n  [{category}] DeepMind Mathematics v1.0")

        tarball_path = self._download_deepmind_math()
        all_records = []

        for module in modules:
            print(f"    Parsing {module}...")
            raw_pairs = self._parse_deepmind_module(tarball_path, module, "train-easy")

            module_count = 0
            for pair in raw_pairs:
                question = pair['question']
                answer = pair['answer']

                # Deterministic filter: numeric answers only (Section 3.7)
                # Reject multi-value answers (e.g. polynomial roots "-1, 1")
                # BEFORE stripping commas, to avoid corrupting them into "-11"
                if ',' in answer:
                    self.stats['filtered'][category]['non_numeric'] += 1
                    continue
                answer_clean = answer.replace(' ', '')
                if not answer_clean.lstrip('-').isdigit():
                    self.stats['filtered'][category]['non_numeric'] += 1
                    continue

                # Build record with FULL provenance (Section 3.4)
                record = {
                    'dataset_name': 'deepmind_math',
                    'dataset_source': DEEPMIND_MATH_URL,
                    'dataset_version': DEEPMIND_MATH_VERSION,
                    'source_split': 'train-easy',
                    'source_record_id': f"{module}_easy_{pair['pair_idx']}",
                    'source_type': 'dataset_raw',
                    'field_map_version': self.field_map_version,
                    'prompt_text': self.normalize_text(question),
                    'ground_truth': answer_clean,
                    'source_answer_raw': answer,
                    'source_module': module,
                    'source_meta_json': json.dumps({
                        'pair_idx': pair['pair_idx'],
                        'module': module,
                        'difficulty': 'train-easy',
                    }),
                }

                all_records.append(record)
                module_count += 1
                self.stats['loaded'][category] += 1

            self.loaded_modules[category].append(module)
            print(f"      ✓ {module}: {module_count} numeric records")

        print(f"    Total {category}: {len(all_records)} records from {len(modules)} modules")
        return all_records

    # =========================================================================
    # WP: GSM8K (Section 3.5.2)
    # =========================================================================

    def load_gsm8k(self) -> List[Dict]:
        """
        Load GSM8K dataset (Section 3.5.2)

        - prompt_text = original question field
        - ground_truth = deterministic final numeric answer after ####
        - Store full answer rationale in source_answer_raw for audit
        """
        print(f"\n  [WP] GSM8K ({GSM8K_PATH}, {GSM8K_SPLIT} split)")

        try:
            ds = load_dataset(GSM8K_PATH, GSM8K_CONFIG, split=GSM8K_SPLIT)
        except Exception as e:
            print(f"    ERROR loading GSM8K: {e}")
            return []

        self.loaded_modules['WP'].append(f'{GSM8K_PATH}/{GSM8K_CONFIG}/{GSM8K_SPLIT}')

        records = []
        for idx, example in enumerate(ds):
            question = example.get('question', '')
            answer = example.get('answer', '')

            if not question or not answer:
                self.stats['filtered']['WP']['empty'] += 1
                continue

            if '####' not in answer:
                self.stats['filtered']['WP']['no_separator'] += 1
                continue

            # Deterministic extraction: final number after ####
            answer_final = answer.split('####')[-1].strip().replace(',', '')
            if not answer_final.lstrip('-').isdigit():
                self.stats['filtered']['WP']['non_numeric'] += 1
                continue

            record = {
                'dataset_name': 'gsm8k',
                'dataset_source': f'{GSM8K_PATH}/{GSM8K_CONFIG}',
                'dataset_version': 'unknown',
                'source_split': GSM8K_SPLIT,
                'source_record_id': f"gsm8k_{GSM8K_SPLIT}_{idx}",
                'source_type': 'dataset_raw',
                'field_map_version': self.field_map_version,
                'prompt_text': self.normalize_text(question),
                'ground_truth': answer_final,
                'source_answer_raw': answer,
                'source_module': '',
                'source_meta_json': json.dumps({'idx': idx}),
            }

            records.append(record)
            self.stats['loaded']['WP'] += 1

        print(f"    ✓ GSM8K: {len(records)} numeric records")
        return records

    # =========================================================================
    # LOG: RuleTaker (Section 3.5.3)
    # =========================================================================

    def load_ruletaker(self) -> List[Dict]:
        """
        Load RuleTaker dataset (Section 3.5.3)

        Using tasksource/ruletaker on HuggingFace.
        Fields: context, question, label, config
        Labels: "entailment" / "not entailment"

        Deterministic mapping:
        - "entailment" → "Yes"
        - "not entailment" → "No"

        Prompt template (fixed, versioned in field_map_version):
            {context}\n\nQuestion: {question}
        """
        print(f"\n  [LOG] RuleTaker ({RULETAKER_PATH}, {RULETAKER_SPLIT} split)")

        try:
            ds = load_dataset(RULETAKER_PATH, split=RULETAKER_SPLIT)
        except Exception as e:
            print(f"    ERROR loading RuleTaker: {e}")
            return []

        self.loaded_modules['LOG'].append(f'{RULETAKER_PATH}/{RULETAKER_SPLIT}')

        # Deterministic label mapping (frozen)
        LABEL_MAP = {
            'entailment': 'Yes',
            'not entailment': 'No',
            'not_entailment': 'No',
            'non-entailment': 'No',
        }

        records = []
        for idx, example in enumerate(ds):
            context = example.get('context', '')
            question = example.get('question', '')
            label = example.get('label', '')

            if not context or not question:
                self.stats['filtered']['LOG']['empty'] += 1
                continue

            # Deterministic Yes/No mapping
            if isinstance(label, int):
                # Handle integer labels (0=entailment, 1=not entailment)
                answer_norm = 'Yes' if label == 0 else 'No'
                original_label = str(label)
            elif isinstance(label, str):
                label_lower = label.lower().strip()
                answer_norm = LABEL_MAP.get(label_lower)
                original_label = label
            else:
                answer_norm = None
                original_label = str(label)

            if answer_norm is None:
                self.stats['filtered']['LOG']['invalid_label'] += 1
                continue

            # Fixed prompt template (versioned in field_map_version fm_v1.1)
            # Full prompt stored in prompt_text including instruction suffix,
            # so runner sends it as-is without appending.
            prompt_composed = (
                f"{self.normalize_text(context)}\n\n"
                f"Question: {self.normalize_text(question)}\n"
                f"Answer with only Yes or No.\nAnswer:"
            )

            record = {
                'dataset_name': 'ruletaker',
                'dataset_source': RULETAKER_PATH,
                'dataset_version': 'unknown',
                'source_split': RULETAKER_SPLIT,
                'source_record_id': f"ruletaker_{RULETAKER_SPLIT}_{idx}",
                'source_type': 'dataset_raw',
                'field_map_version': self.field_map_version,
                'prompt_text': prompt_composed,
                'ground_truth': answer_norm,
                'source_answer_raw': original_label,
                'source_module': example.get('config', ''),
                'source_meta_json': json.dumps({
                    'idx': idx,
                    'original_label': original_label,
                    'depth_config': example.get('config', ''),
                }),
            }

            records.append(record)
            self.stats['loaded']['LOG'] += 1

        print(f"    ✓ RuleTaker: {len(records)} records")
        return records

    # =========================================================================
    # Deduplication (Section 3.8)
    # =========================================================================

    def deduplicate(self, records: List[Dict], category: str) -> List[Dict]:
        """
        Deduplicate records (Section 3.8)
        1. Primary key: (dataset_source, source_record_id)
        2. Fallback: text hash
        """
        seen_keys = set()
        seen_hashes = set()
        deduped = []

        for rec in records:
            key = (rec['dataset_source'], rec['source_record_id'])
            if key in seen_keys:
                self.stats['filtered'][category]['duplicate_key'] += 1
                continue

            text_hash = self.compute_record_hash({
                'prompt': rec['prompt_text'],
                'answer': rec['ground_truth'],
            })
            if text_hash in seen_hashes:
                self.stats['filtered'][category]['duplicate_text'] += 1
                continue

            seen_keys.add(key)
            seen_hashes.add(text_hash)
            deduped.append(rec)

        removed = len(records) - len(deduped)
        if removed > 0:
            print(f"    Deduped {category}: {len(records)} → {len(deduped)} (removed {removed})")
        else:
            print(f"    Deduped {category}: {len(deduped)} (no duplicates)")
        return deduped

    # =========================================================================
    # Sampling (Section 3.9)
    # =========================================================================

    def sample_tier(self, pool: List[Dict], n: int, category: str, tier: str) -> List[Dict]:
        """
        Sample n records with deterministic seed (Section 3.9)
        Uses stable hash for tier offset (not Python's hash() which varies).
        """
        seed = self.category_seeds[category]
        tier_offset = int(hashlib.md5(tier.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed + tier_offset)

        if len(pool) < n:
            print(f"    WARNING: {category} pool has only {len(pool)}, need {n}")
            print(f"    Using all {len(pool)} available records")
            sampled = pool[:]
        else:
            sampled = rng.sample(pool, n)

        self.stats['sampled'][tier][category] = len(sampled)
        return sampled

    def assign_prompt_ids(self, records: List[Dict], category: str,
                          start_idx: int = 1) -> List[Dict]:
        """Assign deterministic prompt_id (Section 3.4)"""
        for i, rec in enumerate(records, start=start_idx):
            rec['prompt_id'] = f"{category}_{i:06d}"
            rec['category'] = category
        return records

    # =========================================================================
    # Sanity Check (10 rows per category before full build)
    # =========================================================================

    def sanity_check(self, records: List[Dict], category: str) -> bool:
        """
        Run 10-row sanity check per category before full build.
        Confirms:
        - prompt text is source text (no rewriting)
        - ground truth format valid
        - provenance columns fully populated
        """
        print(f"\n  Sanity check: {category} (first 10 rows)...")

        sample = records[:10] if len(records) >= 10 else records
        errors = []

        for i, rec in enumerate(sample):
            # Check provenance populated
            for field in ['dataset_name', 'dataset_source', 'source_split',
                          'source_record_id', 'source_type', 'field_map_version']:
                if not rec.get(field):
                    errors.append(f"  Row {i}: missing {field}")

            # Check source_type
            if rec.get('source_type') != 'dataset_raw':
                errors.append(f"  Row {i}: source_type={rec.get('source_type')}")

            # Check ground_truth format
            gt = rec.get('ground_truth', '')
            if category in ['AR', 'ALG', 'WP']:
                if not str(gt).lstrip('-').isdigit():
                    errors.append(f"  Row {i}: non-numeric gt='{gt}'")
            elif category == 'LOG':
                if gt not in ['Yes', 'No']:
                    errors.append(f"  Row {i}: invalid LOG gt='{gt}'")

            # Check prompt non-empty
            if not rec.get('prompt_text', '').strip():
                errors.append(f"  Row {i}: empty prompt_text")

        if errors:
            print("  SANITY CHECK FAILED:")
            for e in errors:
                print(f"    ✗ {e}")
            return False

        # Show samples
        for rec in sample[:3]:
            q = rec['prompt_text'][:60].replace('\n', ' ')
            print(f"    [{rec.get('source_record_id','')}] Q: {q}... A: {rec['ground_truth']}")

        print(f"  ✓ Sanity check passed ({len(sample)} rows)")
        return True

    # =========================================================================
    # Validation (Section 3.11)
    # =========================================================================

    def validate_schema(self, records: List[Dict], tier: str) -> bool:
        """Validation suite - hard-fail (Section 3.11)"""
        print(f"\n  Validating {tier} ({len(records)} records)...")

        required_fields = [
            'prompt_id', 'category', 'dataset_name', 'dataset_source',
            'dataset_version', 'source_split', 'source_record_id',
            'source_type', 'field_map_version', 'prompt_text', 'ground_truth',
        ]

        errors = []

        # 1+2. Schema + provenance completeness
        for rec in records:
            missing = [f for f in required_fields
                       if f not in rec or not str(rec.get(f, '')).strip()]
            if missing:
                errors.append(f"Missing/blank in {rec.get('prompt_id', '?')}: {missing}")

        # 3. Source type constraint
        for rec in records:
            if rec.get('source_type') != 'dataset_raw':
                errors.append(f"Bad source_type in {rec.get('prompt_id', '?')}")

        # 4. Label validity
        for rec in records:
            cat = rec.get('category', '')
            gt = rec.get('ground_truth', '')
            if cat in ['AR', 'ALG', 'WP']:
                if not str(gt).lstrip('-').isdigit():
                    errors.append(f"Non-numeric gt in {rec['prompt_id']}: '{gt}'")
            elif cat == 'LOG':
                if gt not in ['Yes', 'No']:
                    errors.append(f"Invalid LOG gt in {rec['prompt_id']}: '{gt}'")

        # 5. Uniqueness
        ids = [rec['prompt_id'] for rec in records]
        if len(ids) != len(set(ids)):
            dupes = [pid for pid in set(ids) if ids.count(pid) > 1]
            errors.append(f"Duplicate prompt_ids in {tier}: {dupes}")

        source_keys = [(rec['dataset_source'], rec['source_record_id']) for rec in records]
        if len(source_keys) != len(set(source_keys)):
            errors.append(f"Duplicate source keys in {tier}")

        if errors:
            print(f"  VALIDATION FAILED for {tier}:")
            for err in errors[:20]:
                print(f"    ✗ {err}")
            if len(errors) > 20:
                print(f"    ... and {len(errors) - 20} more")
            return False

        cat_counts = defaultdict(int)
        for rec in records:
            cat_counts[rec['category']] += 1
        for cat in sorted(cat_counts):
            print(f"    {cat}: {cat_counts[cat]}")
        print(f"  ✓ {tier} validation PASSED ({len(records)} records)")
        return True

    def validate_cross_tier(self, t1, t2, t3) -> bool:
        """Section 3.11 check 7: Cross-tier disjointness"""
        print("\n  Checking cross-tier disjointness...")

        def get_keys(records):
            keys = set()
            hashes = set()
            for rec in records:
                keys.add((rec['dataset_source'], rec['source_record_id']))
                hashes.add(self.compute_record_hash({
                    'prompt': rec['prompt_text'],
                    'answer': rec['ground_truth'],
                }))
            return keys, hashes

        t1k, t1h = get_keys(t1)
        t2k, t2h = get_keys(t2)
        t3k, t3h = get_keys(t3)

        errors = []
        for name, a, b in [('T1-T2', t1k, t2k), ('T1-T3', t1k, t3k), ('T2-T3', t2k, t3k)]:
            overlap = a & b
            if overlap:
                errors.append(f"{name} key overlap: {len(overlap)}")
        for name, a, b in [('T1-T2', t1h, t2h), ('T1-T3', t1h, t3h), ('T2-T3', t2h, t3h)]:
            overlap = a & b
            if overlap:
                errors.append(f"{name} text overlap: {len(overlap)}")

        if errors:
            for e in errors:
                print(f"    ✗ {e}")
            return False

        print("  ✓ Cross-tier disjointness verified (zero overlap)")
        return True

    # =========================================================================
    # Main Build Pipeline
    # =========================================================================

    def build_all_tiers(self, out_dir: Path, dry_run: bool = False):
        """
        Main pipeline (Section 3).
        Sampling order: T3 first → T2 → T1 (Section 3.9)
        """
        print("=" * 70)
        print("OFFICIAL SPLIT BUILDER - Section 3 Protocol")
        print("=" * 70)

        # ---- Step 1: Load ----
        print("\nStep 1: Loading datasets from primary sources...")

        ar_pool = self.load_deepmind_math('AR', ARITHMETIC_MODULES)
        alg_pool = self.load_deepmind_math('ALG', ALGEBRA_MODULES)
        wp_pool = self.load_gsm8k()
        log_pool = self.load_ruletaker()

        # ---- Step 1b: Sanity checks (10 rows each) ----
        print("\nStep 1b: Sanity checks (10 rows per category)...")
        all_sane = True
        for cat, pool in [('AR', ar_pool), ('ALG', alg_pool),
                           ('WP', wp_pool), ('LOG', log_pool)]:
            if not self.sanity_check(pool, cat):
                all_sane = False
        if not all_sane:
            print("\n❌ SANITY CHECKS FAILED - Aborting")
            sys.exit(1)

        # ---- Step 2: Deduplicate ----
        print("\nStep 2: Deduplicating...")
        ar_pool = self.deduplicate(ar_pool, 'AR')
        alg_pool = self.deduplicate(alg_pool, 'ALG')
        wp_pool = self.deduplicate(wp_pool, 'WP')
        log_pool = self.deduplicate(log_pool, 'LOG')

        # Check pool sizes
        required = 75 + 50 + 10  # T3 + T2 + T1
        for cat, pool in [('AR', ar_pool), ('ALG', alg_pool),
                           ('WP', wp_pool), ('LOG', log_pool)]:
            if len(pool) < required:
                print(f"\nFATAL: {cat} has {len(pool)} records, need {required}")
                sys.exit(1)

        # ---- Step 3: Sample T3 → T2 → T1 ----
        print("\nStep 3: Sampling tiers (T3 → T2 → T1)...")

        # T3: 75 each
        ar_t3 = self.sample_tier(ar_pool, 75, 'AR', 'T3')
        alg_t3 = self.sample_tier(alg_pool, 75, 'ALG', 'T3')
        wp_t3 = self.sample_tier(wp_pool, 75, 'WP', 'T3')
        log_t3 = self.sample_tier(log_pool, 75, 'LOG', 'T3')

        # Remove T3 from pools by source_record_id
        t3_ids = {r['source_record_id'] for r in ar_t3 + alg_t3 + wp_t3 + log_t3}
        ar_pool = [r for r in ar_pool if r['source_record_id'] not in t3_ids]
        alg_pool = [r for r in alg_pool if r['source_record_id'] not in t3_ids]
        wp_pool = [r for r in wp_pool if r['source_record_id'] not in t3_ids]
        log_pool = [r for r in log_pool if r['source_record_id'] not in t3_ids]

        # T2: 50 each
        ar_t2 = self.sample_tier(ar_pool, 50, 'AR', 'T2')
        alg_t2 = self.sample_tier(alg_pool, 50, 'ALG', 'T2')
        wp_t2 = self.sample_tier(wp_pool, 50, 'WP', 'T2')
        log_t2 = self.sample_tier(log_pool, 50, 'LOG', 'T2')

        t2_ids = {r['source_record_id'] for r in ar_t2 + alg_t2 + wp_t2 + log_t2}
        ar_pool = [r for r in ar_pool if r['source_record_id'] not in t2_ids]
        alg_pool = [r for r in alg_pool if r['source_record_id'] not in t2_ids]
        wp_pool = [r for r in wp_pool if r['source_record_id'] not in t2_ids]
        log_pool = [r for r in log_pool if r['source_record_id'] not in t2_ids]

        # T1: 10 each
        ar_t1 = self.sample_tier(ar_pool, 10, 'AR', 'T1')
        alg_t1 = self.sample_tier(alg_pool, 10, 'ALG', 'T1')
        wp_t1 = self.sample_tier(wp_pool, 10, 'WP', 'T1')
        log_t1 = self.sample_tier(log_pool, 10, 'LOG', 'T1')

        # Assign non-overlapping prompt IDs
        ar_t1 = self.assign_prompt_ids(ar_t1, 'AR', 1)
        ar_t2 = self.assign_prompt_ids(ar_t2, 'AR', 101)
        ar_t3 = self.assign_prompt_ids(ar_t3, 'AR', 201)

        alg_t1 = self.assign_prompt_ids(alg_t1, 'ALG', 1)
        alg_t2 = self.assign_prompt_ids(alg_t2, 'ALG', 101)
        alg_t3 = self.assign_prompt_ids(alg_t3, 'ALG', 201)

        wp_t1 = self.assign_prompt_ids(wp_t1, 'WP', 1)
        wp_t2 = self.assign_prompt_ids(wp_t2, 'WP', 101)
        wp_t3 = self.assign_prompt_ids(wp_t3, 'WP', 201)

        log_t1 = self.assign_prompt_ids(log_t1, 'LOG', 1)
        log_t2 = self.assign_prompt_ids(log_t2, 'LOG', 101)
        log_t3 = self.assign_prompt_ids(log_t3, 'LOG', 201)

        t1 = ar_t1 + alg_t1 + wp_t1 + log_t1
        t2 = ar_t2 + alg_t2 + wp_t2 + log_t2
        t3 = ar_t3 + alg_t3 + wp_t3 + log_t3

        # ---- Step 4: Validate ----
        print("\nStep 4: Validation (hard-fail)...")
        ok = True
        if not self.validate_schema(t1, 'T1'):
            ok = False
        if not self.validate_schema(t2, 'T2'):
            ok = False
        if not self.validate_schema(t3, 'T3'):
            ok = False
        if not self.validate_cross_tier(t1, t2, t3):
            ok = False

        if not ok:
            print("\n❌ VALIDATION FAILED - Aborting build")
            sys.exit(1)

        print("\n✅ ALL VALIDATION PASSED")

        if dry_run:
            print("\nDRY RUN - Not writing files")
            return

        # ---- Step 5: Write outputs ----
        print("\nStep 5: Writing output files...")
        out_dir.mkdir(parents=True, exist_ok=True)

        self._write_csv(t1, out_dir / "industry_tier1_40.csv")
        self._write_csv(t2, out_dir / "industry_tier2_200.csv")
        self._write_csv(t3, out_dir / "industry_tier3_300.csv")
        self._write_manifest(out_dir / "split_manifest.json", t1, t2, t3)
        self._write_report(out_dir / "split_build_report.md", t1, t2, t3)

        print("\n" + "=" * 70)
        print("BUILD COMPLETE")
        print("=" * 70)
        print(f"\n  ✓ {out_dir / 'industry_tier1_40.csv'} ({len(t1)} rows)")
        print(f"  ✓ {out_dir / 'industry_tier2_200.csv'} ({len(t2)} rows)")
        print(f"  ✓ {out_dir / 'industry_tier3_300.csv'} ({len(t3)} rows)")
        print(f"  ✓ {out_dir / 'split_manifest.json'}")
        print(f"  ✓ {out_dir / 'split_build_report.md'}")

    # =========================================================================
    # Output Writers
    # =========================================================================

    def _write_csv(self, records: List[Dict], path: Path):
        """Write split CSV with full schema (Section 3.4)"""
        fieldnames = [
            'prompt_id', 'category', 'dataset_name', 'dataset_source',
            'dataset_version', 'source_split', 'source_record_id',
            'source_type', 'field_map_version',
            'prompt_text', 'ground_truth',
            'source_answer_raw', 'source_module', 'source_meta_json',
        ]

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        print(f"  ✓ Wrote {len(records)} records to {path}")

    def _write_manifest(self, path: Path, t1, t2, t3):
        """Write split_manifest.json (Section 3.10.1)"""
        filter_stats = {}
        for cat, reasons in self.stats['filtered'].items():
            filter_stats[cat] = dict(reasons)

        sampled_stats = {}
        for tier, cats in self.stats['sampled'].items():
            sampled_stats[tier] = dict(cats)

        manifest = {
            'schema_version': self.schema_version,
            'build_timestamp': datetime.now().isoformat(),
            'builder_script': 'src/build_official_splits.py',
            'builder_version': 'v1.1',
            'global_seed': self.global_seed,
            'category_seeds': self.category_seeds,
            'field_map_version': self.field_map_version,
            'dataset_sources': {
                'AR': {
                    'name': 'DeepMind Mathematics v1.0',
                    'primary_url': DEEPMIND_MATH_URL,
                    'version': DEEPMIND_MATH_VERSION,
                    'modules': list(ARITHMETIC_MODULES),
                    'difficulty': 'train-easy',
                    'loader': 'raw_tarball',
                    'citation': 'Saxton et al., 2019 (arXiv:1904.01557)',
                },
                'ALG': {
                    'name': 'DeepMind Mathematics v1.0',
                    'primary_url': DEEPMIND_MATH_URL,
                    'version': DEEPMIND_MATH_VERSION,
                    'modules': list(ALGEBRA_MODULES),
                    'difficulty': 'train-easy',
                    'loader': 'raw_tarball',
                    'citation': 'Saxton et al., 2019 (arXiv:1904.01557)',
                },
                'WP': {
                    'name': 'GSM8K',
                    'hf_path': GSM8K_PATH,
                    'hf_config': GSM8K_CONFIG,
                    'split': GSM8K_SPLIT,
                    'loader': 'huggingface_datasets',
                    'citation': 'Cobbe et al., 2021 (arXiv:2110.14168)',
                },
                'LOG': {
                    'name': 'RuleTaker',
                    'hf_path': RULETAKER_PATH,
                    'split': RULETAKER_SPLIT,
                    'label_mapping': {
                        'entailment': 'Yes',
                        'not entailment': 'No',
                    },
                    'prompt_template': '{context}\\n\\nQuestion: {question}',
                    'loader': 'huggingface_datasets',
                    'citation': 'Clark et al., 2020 (arXiv:2002.05867)',
                },
            },
            'tier_counts': {
                'T1': {'total': len(t1), 'per_category': sampled_stats.get('T1', {})},
                'T2': {'total': len(t2), 'per_category': sampled_stats.get('T2', {})},
                'T3': {'total': len(t3), 'per_category': sampled_stats.get('T3', {})},
            },
            'filter_exclusion_counts': filter_stats,
            'loaded_counts': dict(self.stats['loaded']),
            'loaded_modules': dict(self.loaded_modules),
            'overlap_check': 'PASSED',
            'output_files': [
                'industry_tier1_40.csv',
                'industry_tier2_200.csv',
                'industry_tier3_300.csv',
            ],
        }

        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  ✓ Wrote manifest to {path}")

    def _write_report(self, path: Path, t1, t2, t3):
        """Write split_build_report.md (Section 3.10.2)"""
        lines = [
            "# Split Build Report",
            "",
            f"**Build Date**: {datetime.now().isoformat()}",
            f"**Builder**: src/build_official_splits.py v1.1",
            f"**Global Seed**: {self.global_seed}",
            f"**Category Seeds**: AR={self.category_seeds['AR']}, "
            f"ALG={self.category_seeds['ALG']}, "
            f"WP={self.category_seeds['WP']}, "
            f"LOG={self.category_seeds['LOG']}",
            "",
            "---",
            "",
            "## Dataset Sources",
            "",
            "| Category | Dataset | Source | Loader |",
            "|----------|---------|--------|--------|",
            f"| AR | DeepMind Math v1.0 | Google Storage (raw tarball) | tarball |",
            f"| ALG | DeepMind Math v1.0 | Google Storage (raw tarball) | tarball |",
            f"| WP | GSM8K | `{GSM8K_PATH}` (HuggingFace) | datasets lib |",
            f"| LOG | RuleTaker | `{RULETAKER_PATH}` (HuggingFace) | datasets lib |",
            "",
            "## Load Summary",
            "",
            "| Category | Records Loaded | Modules |",
            "|----------|----------------|---------|",
        ]

        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            loaded = self.stats['loaded'].get(cat, 0)
            mods = ', '.join(self.loaded_modules.get(cat, []))
            lines.append(f"| {cat} | {loaded:,} | {mods} |")

        # Exclusions
        lines.extend(["", "## Exclusions by Reason", ""])
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            reasons = dict(self.stats['filtered'].get(cat, {}))
            if reasons:
                lines.append(f"**{cat}**:")
                for reason, count in sorted(reasons.items()):
                    lines.append(f"- {reason}: {count:,}")
                lines.append("")

        # Tier counts
        lines.extend([
            "## Tier Counts",
            "",
            "| Tier | AR | ALG | WP | LOG | Total |",
            "|------|-----|-----|-----|-----|-------|",
        ])
        for name, recs in [('T1', t1), ('T2', t2), ('T3', t3)]:
            c = defaultdict(int)
            for r in recs:
                c[r['category']] += 1
            lines.append(f"| {name} | {c['AR']} | {c['ALG']} | {c['WP']} | {c['LOG']} | {len(recs)} |")

        # Validation
        lines.extend([
            "",
            "## Validation Results",
            "",
            "| Check | Result |",
            "|-------|--------|",
            "| Schema completeness | PASSED |",
            "| Provenance completeness | PASSED |",
            "| source_type = dataset_raw | PASSED |",
            "| Category quotas exact | PASSED |",
            "| Label validity | PASSED |",
            "| No duplicates within tiers | PASSED |",
            "| Cross-tier disjointness | PASSED |",
            "| 10-row sanity check | PASSED |",
        ])

        # Spot check
        lines.extend(["", "## Spot Check (T1 Samples)", ""])
        for cat in ['AR', 'ALG', 'WP', 'LOG']:
            sample = next((r for r in t1 if r['category'] == cat), None)
            if sample:
                q = sample['prompt_text'][:120].replace('\n', ' ')
                lines.extend([
                    f"### {cat} - {sample['prompt_id']}",
                    f"- **Source**: `{sample['dataset_source']}`",
                    f"- **Record ID**: `{sample['source_record_id']}`",
                    f"- **Prompt**: {q}...",
                    f"- **Answer**: `{sample['ground_truth']}`",
                    f"- **source_type**: `{sample['source_type']}`",
                    "",
                ])

        lines.extend([
            "## Overlap Verification",
            "",
            "Cross-tier disjointness verified by source key and text hash.",
            "Sampling order: T3 first, T2 from remaining, T1 from remaining.",
            "Zero overlap across all tier pairs.",
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"  ✓ Wrote report to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build official evaluation splits (Section 3 Protocol)"
    )
    parser.add_argument('--config', help='Config YAML (optional)')
    parser.add_argument('--out_dir', default='data/splits', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--cache_dir', default='data/cache',
                        help='Cache directory for downloaded datasets')
    parser.add_argument('--dry_run', action='store_true', help='Validate only')
    parser.add_argument('--validate_only', action='store_true', help='Same as --dry_run')

    args = parser.parse_args()

    config = {
        'seed': args.seed,
        'cache_dir': args.cache_dir,
    }

    builder = SplitBuilder(config)
    builder.build_all_tiers(
        out_dir=Path(args.out_dir),
        dry_run=args.dry_run or args.validate_only,
    )


if __name__ == "__main__":
    main()
