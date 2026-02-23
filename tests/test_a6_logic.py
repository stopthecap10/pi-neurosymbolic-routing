#!/usr/bin/env python3
"""Unit tests for A6 logic parser and engine."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from a6_logic_parser import (
    parse_fact_sentence, parse_rule_sentence, parse_query,
    parse_prompt, Fact, normalize_entity, normalize_verb,
)
from a6_logic_engine import solve_logic


# ─── Parser tests ───────────────────────────────────────────────────

class TestFactParsing:
    def test_property(self):
        f = parse_fact_sentence("The dog is green")
        assert f == Fact("prop", ("dog", "green"))

    def test_property_name(self):
        f = parse_fact_sentence("Anne is kind")
        assert f == Fact("prop", ("anne", "kind"))

    def test_neg_property(self):
        f = parse_fact_sentence("The bear is not rough")
        assert f == Fact("negprop", ("bear", "rough"))

    def test_relation(self):
        f = parse_fact_sentence("The cat chases the mouse")
        assert f == Fact("rel", ("cat", "chase", "mouse"))

    def test_relation_likes(self):
        f = parse_fact_sentence("The squirrel likes the cat")
        assert f == Fact("rel", ("squirrel", "like", "cat"))

    def test_skip_rule(self):
        f = parse_fact_sentence("If something is big then it is green")
        assert f is None

    def test_skip_all(self):
        f = parse_fact_sentence("All blue people are white")
        assert f is None


class TestRuleParsing:
    def test_if_then_property(self):
        r = parse_rule_sentence("If something is big then it is green")
        assert r is not None
        assert len(r.conditions) == 1
        assert r.conditions[0]["prop"] == "big"
        assert r.conclusion["prop"] == "green"

    def test_if_then_negation(self):
        r = parse_rule_sentence("If someone is blue and not kind then they are quiet")
        assert r is not None
        assert len(r.conditions) == 2
        assert r.conditions[0]["prop"] == "blue"
        assert r.conditions[1]["type"] == "negprop"
        assert r.conditions[1]["prop"] == "kind"

    def test_if_then_relation(self):
        r = parse_rule_sentence("If something eats the squirrel then it sees the squirrel")
        assert r is not None
        assert r.conditions[0]["verb"] == "eat"
        assert r.conclusion["verb"] == "see"

    def test_all_are(self):
        r = parse_rule_sentence("All blue people are white")
        assert r is not None
        assert len(r.conditions) == 1
        assert r.conditions[0]["prop"] == "blue"
        assert r.conclusion["prop"] == "white"

    def test_comma_adj_are(self):
        r = parse_rule_sentence("Green, big things are kind")
        assert r is not None
        assert len(r.conditions) == 2

    def test_specific_entity_rule(self):
        r = parse_rule_sentence("If Anne is nice then Anne is smart")
        assert r is not None


class TestQueryParsing:
    def test_positive_prop(self):
        q = parse_query("Bob is smart")
        assert q is not None
        assert not q.negated
        assert q.args == ("bob", "smart")

    def test_negated_prop(self):
        q = parse_query("The dog is not green")
        assert q is not None
        assert q.negated
        assert q.args == ("dog", "green")

    def test_positive_rel(self):
        q = parse_query("The cat chases the squirrel")
        assert q is not None
        assert not q.negated
        assert q.args == ("cat", "chase", "squirrel")

    def test_negated_rel(self):
        q = parse_query("The lion does not visit the lion")
        assert q is not None
        assert q.negated
        assert q.args == ("lion", "visit", "lion")


# ─── Engine integration tests (all 10 LOG T1 prompts) ──────────────

LOG_PROMPTS = {
    "LOG_001": (
        "The bald eagle is green. The bald eagle is young. The cat likes the bald eagle. "
        "The dog is kind. The squirrel is big. The squirrel is green. The squirrel likes the cat. "
        "If something eats the squirrel then it sees the squirrel. "
        "If something is green and it sees the cat then the cat is big. "
        "If something is big then it is green. If something is green then it likes the dog. "
        "If something is kind then it eats the cat. Green, big things are kind. "
        "If the bald eagle likes the cat then the cat likes the bald eagle. "
        "If something is big then it sees the cat. If something eats the dog then it is green.\n\n"
        "Question: The dog is not green.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_002": (
        "The dog chases the mouse. The dog likes the mouse. The dog sees the mouse. "
        "The dog sees the rabbit. The mouse chases the rabbit. The mouse is blue. "
        "The mouse is nice. The mouse likes the dog. The mouse likes the rabbit. "
        "The mouse sees the rabbit. The rabbit is round. The rabbit likes the dog. "
        "The rabbit likes the mouse. The rabbit sees the dog. The rabbit sees the mouse. "
        "If something is round then it sees the mouse. If something is blue then it chases the mouse. "
        "If something is blue then it likes the mouse.\n\n"
        "Question: The dog chases the mouse.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_003": (
        "The cat chases the squirrel. The cat eats the squirrel. The cat visits the squirrel. "
        "The squirrel chases the cat. The squirrel eats the cat. The squirrel is blue. "
        "The squirrel is kind. If something eats the cat then it visits the cat.\n\n"
        "Question: The cat chases the squirrel.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_004": (
        "Anne is kind. Anne is quiet. Bob is blue. Dave is white. Gary is blue. "
        "Gary is kind. Gary is young. If someone is blue and not kind then they are quiet. "
        "If someone is quiet and not kind then they are smart.\n\n"
        "Question: Bob is smart.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_005": (
        "Anne is blue. Anne is nice. Anne is young. Charlie is nice. Charlie is red. "
        "Charlie is not round. Charlie is young. Fiona is cold. Harry is nice. Harry is round. "
        "If something is nice and not furry then it is blue.\n\n"
        "Question: Charlie is cold.\nAnswer with only Yes or No.\nAnswer:",
        "No",
    ),
    "LOG_006": (
        "The bear eats the squirrel. The bear is big. The bear is kind. The bear is red. "
        "The bear is not rough. The bear is not round. The bear needs the squirrel. "
        "The bear visits the squirrel. The squirrel eats the bear. The squirrel is big. "
        "The squirrel is kind. The squirrel is red. The squirrel is rough. "
        "The squirrel needs the bear. The squirrel visits the bear. "
        "If someone visits the squirrel and they are not red then they do not need the squirrel.\n\n"
        "Question: The bear is kind.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_007": (
        "The bear is rough. The bear needs the cat. The bear visits the lion. "
        "The cat is rough. The cat likes the lion. The cat needs the bear. "
        "The cat needs the lion. The cat visits the bear. The cat visits the lion. "
        "The lion likes the bear. The lion likes the cat. The lion visits the bear. "
        "If someone likes the bear and the bear visits the lion then they are red. "
        "If someone is rough and they like the bear then the bear needs the cat. "
        "If someone visits the bear and they like the cat then they visit the lion. "
        "If someone visits the lion and they need the cat then they like the bear. "
        "If someone is rough and they visit the lion then the lion needs the bear. "
        "Red people are big.\n\n"
        "Question: The lion does not visit the lion.\nAnswer with only Yes or No.\nAnswer:",
        "No",
    ),
    "LOG_008": (
        "Anne is nice. Charlie is blue. Charlie is kind. Erin is kind. Erin is white. "
        "Fiona is kind. Fiona is smart. If someone is kind then they are smart. "
        "If Anne is nice then Anne is smart. If someone is white then they are smart. "
        "All blue people are white. All kind people are blue. "
        "If someone is smart then they are kind. All white, nice people are quiet.\n\n"
        "Question: Anne is not quiet.\nAnswer with only Yes or No.\nAnswer:",
        "No",
    ),
    "LOG_009": (
        "Alan is kind. Bob is red. Bob is big. Eric is young. Eric is round. "
        "Eric is nice. Eric is green. Eric is cold. Gary is round. Gary is red. "
        "Gary is cold. Gary is blue. "
        "If someone is green and young and kind then they are round. "
        "If someone is young then they are red. "
        "If someone is red and nice then they are big. "
        "If someone is nice and blue and kind then they are green. "
        "If someone is nice and big then they are cold. "
        "If someone is young and rough and red then they are kind. "
        "If someone is red and big then they are kind.\n\n"
        "Question: Eric is big.\nAnswer with only Yes or No.\nAnswer:",
        "Yes",
    ),
    "LOG_010": (
        "Alan is kind. Bob is blue. Charlie is young. Charlie is round. Charlie is nice. "
        "Charlie is kind. Harry is red. "
        "If someone is kind and young then they are red. "
        "If someone is nice and red and young then they are rough. "
        "If someone is rough and kind and nice then they are green. "
        "If someone is blue and rough and red then they are round.\n\n"
        "Question: Bob is rough.\nAnswer with only Yes or No.\nAnswer:",
        "No",
    ),
}


@pytest.mark.parametrize("pid,prompt_gt", list(LOG_PROMPTS.items()))
def test_log_prompt(pid, prompt_gt):
    prompt_text, expected = prompt_gt
    result = solve_logic(prompt_text)
    assert result["parse_success"], f"{pid}: parse failed"
    assert result["answer"] == expected, (
        f"{pid}: expected {expected}, got {result['answer']}\n"
        f"Trace: {result['trace']}"
    )


# ─── Verify the 4 previously-failing prompts specifically ──────────

def test_log001_dog_not_green():
    """LOG_001: depth-5, dog is kind but never becomes green."""
    result = solve_logic(LOG_PROMPTS["LOG_001"][0])
    assert result["answer"] == "Yes"

def test_log004_bob_is_smart():
    """LOG_004: depth-2, closed-world negation-as-failure."""
    result = solve_logic(LOG_PROMPTS["LOG_004"][0])
    assert result["answer"] == "Yes"

def test_log008_anne_not_quiet():
    """LOG_008: depth-5, long chain anne→smart→kind→blue→white→quiet."""
    result = solve_logic(LOG_PROMPTS["LOG_008"][0])
    assert result["answer"] == "No"

def test_log009_eric_is_big():
    """LOG_009: depth-3, eric young→red, red+nice→big."""
    result = solve_logic(LOG_PROMPTS["LOG_009"][0])
    assert result["answer"] == "Yes"
