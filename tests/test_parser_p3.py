#!/usr/bin/env python3
"""
Tests for P3_numeric_context parser.
Run: python3 -m pytest tests/test_parser_p3.py -v
"""
import sys
from pathlib import Path

# Add src/ to path so we can import the parser
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_v1_baseline_matrix import parse_numeric_robust, extract_yesno


# ── Rule 1: Direct numeric output ────────────────────────

class TestDirectNumeric:
    def test_integer(self):
        assert parse_numeric_robust("201") == "201"

    def test_negative(self):
        assert parse_numeric_robust("-5") == "-5"

    def test_negative_integer(self):
        assert parse_numeric_robust("-10") == "-10"

    def test_decimal(self):
        assert parse_numeric_robust("-4.0048") == "-4.0048"

    def test_fraction_integral(self):
        # 6/2 = 3.0 -> normalized to "3"
        assert parse_numeric_robust("6/2") == "3"

    def test_fraction_non_integral(self):
        assert parse_numeric_robust("2/3") == str(2/3)

    def test_with_trailing_period(self):
        # "5." should be treated as direct number 5
        assert parse_numeric_robust("5.") == "5"

    def test_positive_sign(self):
        assert parse_numeric_robust("+7") == "7"

    def test_zero(self):
        assert parse_numeric_robust("0") == "0"


# ── Rule 2: Cue phrase extraction ────────────────────────

class TestCuePhrase:
    def test_final_answer_is(self):
        assert parse_numeric_robust("The final numeric answer for d is 6.") == "6"

    def test_var_equals(self):
        assert parse_numeric_robust("o = -6") == "-6"

    def test_var_equals_with_period(self):
        assert parse_numeric_robust("m = 5.") == "5"

    def test_var_equals_negative(self):
        assert parse_numeric_robust("c = -5") == "-5"

    def test_var_equals_simple(self):
        assert parse_numeric_robust("b = 5") == "5"

    def test_var_equals_large(self):
        assert parse_numeric_robust("t = 36") == "36"

    def test_var_equals_negative_large(self):
        assert parse_numeric_robust("z = 8") == "8"

    def test_final_answer_for(self):
        assert parse_numeric_robust("The final numeric answer for m is 10.") == "10"

    def test_var_equals_negative_4(self):
        assert parse_numeric_robust("f = -4") == "-4"

    def test_chained_equals(self):
        # "o = 8/94 = 4/47" -> should take last value after =
        result = parse_numeric_robust("o = 8/94 = 4/47")
        assert result == str(4/47)


# ── Rule 3: Expression rescue ────────────────────────────

class TestExpressionRescue:
    def test_addition(self):
        assert parse_numeric_robust("-7.5 + 2320.5 =") == "2313"

    def test_subtraction(self):
        assert parse_numeric_robust("100 - 37 =") == "63"

    def test_multiplication(self):
        assert parse_numeric_robust("12 * 5 =") == "60"

    def test_division(self):
        assert parse_numeric_robust("100 / 4 =") == "25"

    def test_parentheses(self):
        assert parse_numeric_robust("(10 + 5) * 2 =") == "30"

    def test_negative_result(self):
        assert parse_numeric_robust("-8 + 3 =") == "-5"

    def test_with_text_rejected(self):
        # Contains "miles" and "hours" -> should NOT evaluate, should return E8
        assert parse_numeric_robust("60 miles/hour * 5 hours =") == ""

    def test_with_units_rejected(self):
        # Contains text -> E8
        assert parse_numeric_robust("The bus traveled 60 miles/hour * 5 hours =") == ""


# ── Rule 4: Single standalone number ─────────────────────

class TestSingleNumber:
    def test_value_is_number(self):
        assert parse_numeric_robust("The value is 10.") == "10"

    def test_single_negative(self):
        assert parse_numeric_robust("-0.3333333333333333") == str(-1/3)

    def test_just_a_number_with_text(self):
        assert parse_numeric_robust("1") == "1"


# ── Rule 5: Ambiguity -> E8 ─────────────────────────────

class TestAmbiguityE8:
    def test_bus_with_units(self):
        assert parse_numeric_robust("The bus traveled 60 miles/hour * 5 hours =") == ""

    def test_money_with_fraction(self):
        assert parse_numeric_robust("Jack had $100. Sophia gave him 1/5") == ""

    def test_reasoning_with_units(self):
        assert parse_numeric_robust("Ruiz makes 120 pounds of chocolates in 2 hours") == ""

    def test_reasoning_counting(self):
        assert parse_numeric_robust("Each adult gets 6 chocolate bars, so 4 adults") == ""

    def test_reasoning_with_minutes(self):
        assert parse_numeric_robust("Grandma spends 40 minutes walking on the beach, which") == ""

    def test_unfinished_expression_with_text(self):
        assert parse_numeric_robust("The value is -8 - (-3 + 0)") == ""

    def test_reasoning_with_x(self):
        assert parse_numeric_robust("Let's say Rory retrieved x tennis balls in the second set.") == ""

    def test_money_paid(self):
        # "Jean paid $45000. " has text + one clear number -> should extract
        # Actually this has text and numbers, but "$45000" is cue-less
        # With 2 tokens (45000) and text, Rule 5 might apply
        # Let's check: cleaned = "Jean paid $45000. " -> tokens = ["45000"]
        # Single token -> Rule 4 -> "45000"
        # Actually $ gets stripped by comma removal? No. $ is not a comma.
        # NUM_TOKEN_RE finds "45000" -> 1 token -> Rule 4 returns "45000"
        result = parse_numeric_robust("Jean paid $45000. ")
        assert result == "45000"


# ── Edge cases ───────────────────────────────────────────

class TestEdgeCases:
    def test_empty(self):
        assert parse_numeric_robust("") == ""

    def test_none_text(self):
        assert parse_numeric_robust("no numbers here") == ""

    def test_only_text(self):
        assert parse_numeric_robust("The answer is unclear.") == ""

    def test_comma_separated(self):
        # "1,234" -> cleaned to "1234"
        assert parse_numeric_robust("1,234") == "1234"

    def test_negative_fraction_result(self):
        result = parse_numeric_robust("(-2)/(-3) = 2/3")
        # This has "= 2/3" -> Rule 2 (var_eq or chained equals)
        # 2/3 = 0.666...
        assert result == str(2/3)


# ── LOG parser unchanged ─────────────────────────────────

class TestYesNoUnchanged:
    def test_yes(self):
        assert extract_yesno("Yes.") == "Yes"

    def test_no(self):
        assert extract_yesno("No.") == "No"

    def test_yes_lower(self):
        assert extract_yesno("yes") == "Yes"

    def test_no_lower(self):
        assert extract_yesno("no") == "No"

    def test_empty(self):
        assert extract_yesno("") == ""

    def test_no_match(self):
        assert extract_yesno("maybe") == ""

    def test_last_yes_wins(self):
        assert extract_yesno("No, actually yes") == "Yes"

    def test_last_no_wins(self):
        assert extract_yesno("Yes, but actually no") == "No"
