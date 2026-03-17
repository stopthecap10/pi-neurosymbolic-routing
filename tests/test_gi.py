#!/usr/bin/env python3
"""
Tests for grammatical inference modules:
  - token_classifier
  - dfa
  - rpni
  - lstar
"""

import csv
import json
import os
import tempfile
import pytest

from src.token_classifier import tokenize, ALPHABET, tokens_to_str, str_to_tokens
from src.dfa import DFA, MultiClassDFA
from src.rpni import RPNI, learn_one_vs_rest as rpni_learn
from src.lstar import LStar, learn_one_vs_rest as lstar_learn, \
    ground_truth_membership_oracle, ground_truth_equivalence_oracle


# ===================== Token Classifier Tests =====================

class TestTokenClassifier:

    def test_arithmetic_simple(self):
        toks = tokenize("Calculate 4 - (5 - (3 + 2)).")
        assert toks[0] == "CMD"
        assert "NUM" in toks
        assert "OP" in toks
        assert "PAREN" in toks

    def test_arithmetic_division(self):
        toks = tokenize("What is 710 divided by 142?")
        assert toks[0] == "CMD"  # "What is" -> CMD
        assert "NUM" in toks
        assert "QMARK" in toks

    def test_algebra_has_var(self):
        toks = tokenize("Solve 5*m - 7 - 18 = 0 for m.")
        assert toks[0] == "CMD"  # "Solve"
        assert "VAR" in toks

    def test_word_problem_has_narrative(self):
        toks = tokenize("A bus travels 60 miles per hour for 5 hours.")
        assert "NARR" in toks
        assert "UNIT" in toks
        assert "NUM" in toks

    def test_logic_has_entities_and_props(self):
        toks = tokenize("The bald eagle is green. The bald eagle is young.")
        assert "ENT" in toks or "NARR" in toks
        assert "PROP" in toks
        assert "DOT" in toks

    def test_logic_has_rule(self):
        toks = tokenize("If something is green then it likes the dog.")
        assert "RULE" in toks

    def test_narr_collapse(self):
        toks = tokenize("the big brown fox jumped over the lazy dog")
        # "the" is NARR, "big" is PROP, "brown" is NARR, etc.
        # consecutive NARR tokens should collapse
        narr_runs = 0
        for i in range(len(toks) - 1):
            if toks[i] == "NARR" and toks[i + 1] == "NARR":
                narr_runs += 1
        assert narr_runs == 0, "Consecutive NARR tokens should be collapsed"

    def test_max_length(self):
        long_text = "Calculate " + " + ".join(str(i) for i in range(100))
        toks = tokenize(long_text)
        assert len(toks) <= 15

    def test_empty_string(self):
        toks = tokenize("")
        assert toks == ()

    def test_tokens_to_str_roundtrip(self):
        toks = ("CMD", "NUM", "OP", "NUM")
        s = tokens_to_str(toks)
        assert s == "CMD NUM OP NUM"
        assert str_to_tokens(s) == toks


# ===================== DFA Tests =====================

class TestDFA:

    def _simple_dfa(self):
        """DFA that accepts sequences starting with CMD followed by NUM."""
        return DFA(
            states={0, 1, 2},
            alphabet=("CMD", "NUM", "OP"),
            transitions={
                (0, "CMD"): 1,
                (1, "NUM"): 2,
                (2, "OP"): 1,  # loop: OP goes back to expect NUM
                (2, "NUM"): 2,
            },
            start_state=0,
            accept_states={2},
        )

    def test_accept(self):
        dfa = self._simple_dfa()
        assert dfa.run(("CMD", "NUM")) is True
        assert dfa.run(("CMD", "NUM", "OP", "NUM")) is True

    def test_reject(self):
        dfa = self._simple_dfa()
        assert dfa.run(("NUM",)) is False
        assert dfa.run(("CMD",)) is False
        assert dfa.run(()) is False

    def test_no_transition(self):
        dfa = self._simple_dfa()
        assert dfa.run(("CMD", "OP")) is False

    def test_serialization_roundtrip(self):
        dfa = self._simple_dfa()
        d = dfa.to_dict()
        dfa2 = DFA.from_dict(d)
        assert dfa2.run(("CMD", "NUM")) is True
        assert dfa2.run(("NUM",)) is False

    def test_json_file_roundtrip(self):
        dfa = self._simple_dfa()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            dfa.to_json(path)
            dfa2 = DFA.from_json(path)
            assert dfa2.run(("CMD", "NUM")) is True
            assert dfa2.num_states == 3
        finally:
            os.unlink(path)

    def test_multiclass_dfa(self):
        dfa_a = DFA(
            states={0, 1}, alphabet=("X", "Y"),
            transitions={(0, "X"): 1}, start_state=0, accept_states={1}
        )
        dfa_b = DFA(
            states={0, 1}, alphabet=("X", "Y"),
            transitions={(0, "Y"): 1}, start_state=0, accept_states={1}
        )
        mc = MultiClassDFA({"A": dfa_a, "B": dfa_b}, priority=("A", "B"))
        assert mc.classify(("X",)) == "A"
        assert mc.classify(("Y",)) == "B"
        assert mc.classify(("X", "Y")) is None


# ===================== RPNI Tests =====================

class TestRPNI:

    def test_simple_language(self):
        """Learn a language that accepts strings starting with 'a'."""
        alphabet = ("a", "b")
        positive = [("a",), ("a", "b"), ("a", "a"), ("a", "b", "a")]
        negative = [("b",), ("b", "a"), ("b", "b"), ()]

        rpni = RPNI(alphabet)
        dfa = rpni.learn(positive, negative)

        for p in positive:
            assert dfa.run(p), f"Should accept {p}"
        for n in negative:
            assert not dfa.run(n), f"Should reject {n}"

    def test_exact_string(self):
        """Learn a language that accepts exactly one string."""
        alphabet = ("x", "y")
        positive = [("x", "y")]
        negative = [("x",), ("y",), ("y", "x"), ("x", "x"), ()]

        rpni = RPNI(alphabet)
        dfa = rpni.learn(positive, negative)

        assert dfa.run(("x", "y")) is True
        for n in negative:
            assert not dfa.run(n), f"Should reject {n}"

    def test_one_vs_rest(self):
        """Test the one-vs-rest wrapper."""
        data = [
            (("CMD", "NUM"), "AR"),
            (("CMD", "VAR"), "ALG"),
            (("ENT", "NARR"), "WP"),
            (("ENT", "PROP"), "LOG"),
        ]
        dfas = rpni_learn(
            alphabet=("CMD", "NUM", "VAR", "ENT", "NARR", "PROP"),
            labeled_data=data,
        )
        assert "AR" in dfas
        assert "ALG" in dfas
        assert dfas["AR"].run(("CMD", "NUM"))
        assert not dfas["AR"].run(("CMD", "VAR"))


# ===================== L* Tests =====================

class TestLStar:

    def test_learn_simple(self):
        """Learn a language: strings starting with 'a'."""
        alphabet = ("a", "b")
        target_set = {("a",), ("a", "b"), ("a", "a"), ("a", "b", "a"),
                      ("a", "a", "b"), ("a", "b", "b")}
        all_strings = list(target_set) + [
            ("b",), ("b", "a"), ("b", "b"), (),
            ("b", "a", "b"), ("b", "b", "a"),
        ]

        def mem_oracle(w):
            # Accept any string starting with 'a'
            return len(w) > 0 and w[0] == "a"

        def eq_oracle(hypothesis):
            for s in all_strings:
                expected = len(s) > 0 and s[0] == "a"
                if hypothesis.run(s) != expected:
                    return s
            return None

        learner = LStar(alphabet, mem_oracle, eq_oracle)
        dfa = learner.learn()

        assert dfa.run(("a",)) is True
        assert dfa.run(("a", "b")) is True
        assert dfa.run(("b",)) is False
        assert dfa.run(()) is False

    def test_learn_one_vs_rest(self):
        """Test the one-vs-rest wrapper with ground-truth oracles."""
        data = [
            (("CMD", "NUM", "OP", "NUM"), "AR"),
            (("CMD", "NUM", "OP", "NUM", "DOT"), "AR"),
            (("CMD", "NUM", "OP", "VAR"), "ALG"),
            (("CMD", "VAR", "OP", "NUM"), "ALG"),
            (("ENT", "NARR", "NUM", "UNIT"), "WP"),
            (("NUM", "NARR", "UNIT"), "WP"),
            (("ENT", "PROP", "DOT"), "LOG"),
            (("ENT", "NARR", "PROP", "DOT"), "LOG"),
        ]

        dfas = lstar_learn(
            alphabet=("CMD", "NUM", "VAR", "OP", "ENT", "NARR", "PROP", "UNIT", "DOT"),
            labeled_data=data,
        )

        assert "AR" in dfas
        assert "ALG" in dfas
        assert "WP" in dfas
        assert "LOG" in dfas


# ===================== Integration Test: Tier-1 Data =====================

class TestTier1Integration:
    """Test tokenizer + RPNI/L* on the actual 40 tier-1 prompts."""

    @pytest.fixture
    def tier1_data(self):
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "splits", "industry_tier1_40_v2.csv"
        )
        if not os.path.exists(csv_path):
            pytest.skip("Tier-1 CSV not found")

        data = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                toks = tokenize(row["prompt_text"])
                data.append((toks, row["category"]))
        return data

    def test_tokenizer_produces_valid_tokens(self, tier1_data):
        for toks, cat in tier1_data:
            assert len(toks) > 0, f"Empty token sequence for {cat}"
            assert len(toks) <= 15
            for t in toks:
                assert t in ALPHABET, f"Unknown token {t}"

    def test_categories_distinguishable(self, tier1_data):
        """Check that AR has no VAR, ALG has VAR, LOG has ENT+DOT pattern."""
        ar_has_var = any("VAR" in toks for toks, cat in tier1_data if cat == "AR")
        alg_has_var = all("VAR" in toks for toks, cat in tier1_data if cat == "ALG")
        log_has_ent = all("ENT" in toks and "DOT" in toks
                          for toks, cat in tier1_data if cat == "LOG")

        assert not ar_has_var, "AR should not contain VAR tokens"
        assert alg_has_var, "All ALG should contain VAR tokens"
        assert log_has_ent, "All LOG should contain ENT and DOT tokens"

    def test_rpni_on_tier1(self, tier1_data):
        """Train RPNI on tier-1 data and check classification accuracy."""
        dfas = rpni_learn(
            alphabet=ALPHABET,
            labeled_data=tier1_data,
        )
        mc = MultiClassDFA(dfas, priority=("LOG", "ALG", "AR", "WP"))

        correct = 0
        for toks, cat in tier1_data:
            pred = mc.classify(toks)
            if pred == cat:
                correct += 1

        accuracy = correct / len(tier1_data)
        assert accuracy >= 0.8, f"RPNI accuracy {accuracy:.1%} too low on training data"

    def test_lstar_on_tier1(self, tier1_data):
        """Train L* on tier-1 data and check classification accuracy."""
        dfas = lstar_learn(
            alphabet=ALPHABET,
            labeled_data=tier1_data,
        )
        mc = MultiClassDFA(dfas, priority=("LOG", "ALG", "AR", "WP"))

        correct = 0
        for toks, cat in tier1_data:
            pred = mc.classify(toks)
            if pred == cat:
                correct += 1

        accuracy = correct / len(tier1_data)
        assert accuracy >= 0.8, f"L* accuracy {accuracy:.1%} too low on training data"
