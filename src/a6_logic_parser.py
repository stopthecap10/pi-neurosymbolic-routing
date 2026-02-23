#!/usr/bin/env python3
"""
A6 Logic Parser — RuleTaker-style natural language to structured logic.

Parses facts, rules, and queries from RuleTaker prompts into a structured
representation for forward-chaining inference.

Supported patterns:
  Facts:
    - "X is P" (property)           → Prop(entity, property)
    - "X is not P"                  → NegProp(entity, property)
    - "X verb Y" (relation)         → Rel(subject, verb, object)

  Rules:
    - "If something/someone [conditions] then [conclusion]"
    - "All P things are Q"
    - "P, Q things are R"

  Queries:
    - "X is P" / "X is not P"
    - "X verb Y" / "X does not verb Y"
"""

import re
from typing import List, Tuple, Optional, Dict, Any

# ─── Data structures ────────────────────────────────────────────────

class Fact:
    """A ground fact: entity has property or entity relates to entity."""
    def __init__(self, kind: str, args: tuple):
        self.kind = kind   # "prop" or "rel"
        self.args = args   # ("dog", "green") or ("cat", "chases", "mouse")

    def __repr__(self):
        return f"Fact({self.kind}, {self.args})"

    def __eq__(self, other):
        return isinstance(other, Fact) and self.kind == other.kind and self.args == other.args

    def __hash__(self):
        return hash((self.kind, self.args))


class Rule:
    """An if-then rule with conditions and a conclusion."""
    def __init__(self, conditions: list, conclusion: dict, raw: str = ""):
        self.conditions = conditions   # list of condition dicts
        self.conclusion = conclusion   # conclusion dict
        self.raw = raw

    def __repr__(self):
        return f"Rule({self.conditions} => {self.conclusion})"


class Query:
    """A yes/no query."""
    def __init__(self, negated: bool, kind: str, args: tuple):
        self.negated = negated  # True if "X is not P" or "X does not verb Y"
        self.kind = kind        # "prop" or "rel"
        self.args = args

    def __repr__(self):
        return f"Query(negated={self.negated}, {self.kind}, {self.args})"


# ─── Normalization ──────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase and strip text."""
    return text.strip().lower()


def normalize_entity(name: str) -> str:
    """Normalize entity names: 'The dog' -> 'dog', 'Anne' -> 'anne'."""
    name = name.strip().lower()
    # Remove leading articles
    for art in ("the ", "a ", "an "):
        if name.startswith(art):
            name = name[len(art):]
    return name.strip()


def normalize_verb(verb: str) -> str:
    """Normalize verbs to base form for matching."""
    verb = verb.strip().lower()
    # Common irregular
    irregulars = {
        "eats": "eat", "sees": "see", "likes": "like",
        "visits": "visit", "chases": "chase", "needs": "need",
    }
    if verb in irregulars:
        return irregulars[verb]
    # Regular -s/-es
    if verb.endswith("es"):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss"):
        return verb[:-1]
    return verb


# ─── Sentence splitting ────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling RuleTaker format."""
    # Split on period followed by space or end
    parts = re.split(r'\.\s+|\.\s*$', text)
    return [p.strip() for p in parts if p.strip()]


# ─── Fact parsing ───────────────────────────────────────────────────

# Pattern: "X is [not] P"
RE_IS_NOT = re.compile(
    r'^(.+?)\s+is\s+not\s+(\w+)$', re.IGNORECASE
)
RE_IS = re.compile(
    r'^(.+?)\s+is\s+(\w+)$', re.IGNORECASE
)

# Pattern: "X verb Y" (relation)
KNOWN_VERBS = [
    "chases", "chase", "eats", "eat", "sees", "see",
    "likes", "like", "visits", "visit", "needs", "need",
]

RE_RELATION = re.compile(
    r'^(.+?)\s+(' + '|'.join(KNOWN_VERBS) + r')\s+(.+)$', re.IGNORECASE
)


def parse_fact_sentence(sentence: str) -> Optional[Fact]:
    """Parse a single fact sentence into a Fact, or None."""
    s = sentence.strip()
    if not s:
        return None

    # Skip rule sentences
    if s.lower().startswith("if ") or s.lower().startswith("all "):
        return None

    # Try "X is not P"
    m = RE_IS_NOT.match(s)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return Fact("negprop", (entity, prop))

    # Try "X does not verb Y" (negated relation fact)
    for vb in KNOWN_VERBS:
        base = normalize_verb(vb)
        pattern = re.compile(
            r'^(.+?)\s+does\s+not\s+' + base + r'\s+(.+)$', re.IGNORECASE
        )
        m = pattern.match(s)
        if m:
            subj = normalize_entity(m.group(1))
            obj = normalize_entity(m.group(2))
            return Fact("negrel", (subj, base, obj))

    # Try "X verb Y" (before "X is P" to avoid "X is P" matching relations)
    m = RE_RELATION.match(s)
    if m:
        subj = normalize_entity(m.group(1))
        verb = normalize_verb(m.group(2))
        obj = normalize_entity(m.group(3))
        return Fact("rel", (subj, verb, obj))

    # Try "X is P"
    m = RE_IS.match(s)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return Fact("prop", (entity, prop))

    return None


# ─── Rule parsing ───────────────────────────────────────────────────

def parse_rule_sentence(sentence: str) -> Optional[Rule]:
    """Parse an if-then or all-are rule into a Rule, or None."""
    s = sentence.strip()
    if not s:
        return None

    # Pattern 1: "If something/someone [conditions] then [conclusion]"
    rule = _parse_if_then(s)
    if rule:
        return rule

    # Pattern 2: "All P things/people are Q"
    rule = _parse_all_are(s)
    if rule:
        return rule

    # Pattern 3: "P, Q things are R" (comma-separated adjective list)
    rule = _parse_adj_list_are(s)
    if rule:
        return rule

    return None


def _parse_if_then(s: str) -> Optional[Rule]:
    """Parse 'If something/someone [conds] then [conclusion]'."""
    # Match if...then...
    m = re.match(r'^If\s+(.+?)\s+then\s+(.+)$', s, re.IGNORECASE)
    if not m:
        return None

    cond_text = m.group(1).strip()
    conc_text = m.group(2).strip()

    conditions = _parse_conditions(cond_text)
    conclusion = _parse_conclusion(conc_text)

    if conditions is not None and conclusion is not None:
        return Rule(conditions, conclusion, raw=s)
    return None


def _parse_conditions(text: str) -> Optional[list]:
    """Parse the condition part of an if-then rule."""
    conditions = []

    # Check for universal variable: "something", "someone"
    # e.g., "something is green and it sees the cat"
    # e.g., "someone is blue and not kind"
    # e.g., "the bald eagle likes the cat"

    # Detect variable binding
    var_match = re.match(
        r'^(something|someone)\s+(.+)$', text, re.IGNORECASE
    )

    if var_match:
        # Universal rule with variable
        var = "?x"
        rest = var_match.group(2)
        # Split on " and " (but not inside "and not")
        # We need to handle "and" as conjunction but "and not" as part of condition
        clauses = _split_and(rest)
        for clause in clauses:
            c = _parse_single_condition(clause.strip(), var)
            if c:
                conditions.append(c)
        if conditions:
            return conditions
        return None

    # Specific entity rule — may have multiple conditions joined by "and"
    # e.g. "Anne is nice", "Gary is smart and Gary is not rough",
    #       "Erin is quiet and Erin is cold"
    clauses = _split_and(text)
    for clause in clauses:
        clause = clause.strip()
        # "Entity is not P"
        m_neg = re.match(r'^(.+?)\s+is\s+not\s+(\w+)$', clause, re.IGNORECASE)
        if m_neg:
            entity = normalize_entity(m_neg.group(1))
            prop = normalize(m_neg.group(2))
            conditions.append({"type": "negprop", "entity": entity, "prop": prop, "var": None})
            continue
        # "Entity is P"
        m_named = re.match(r'^(.+?)\s+is\s+(\w+)$', clause, re.IGNORECASE)
        if m_named:
            entity = normalize_entity(m_named.group(1))
            prop = normalize(m_named.group(2))
            conditions.append({"type": "prop", "entity": entity, "prop": prop, "var": None})
            continue
        # "Entity does not verb Object"
        for vb in KNOWN_VERBS:
            base = normalize_verb(vb)
            m_negrel = re.match(
                r'^(.+?)\s+does\s+not\s+' + base + r'\s+(.+)$', clause, re.IGNORECASE
            )
            if m_negrel:
                subj = normalize_entity(m_negrel.group(1))
                obj = normalize_entity(m_negrel.group(2))
                conditions.append({"type": "negrel", "subj": subj, "verb": base, "obj": obj, "var": None})
                break
        else:
            # "Entity verb Object"
            m_rel = RE_RELATION.match(clause)
            if m_rel:
                subj = normalize_entity(m_rel.group(1))
                verb = normalize_verb(m_rel.group(2))
                obj = normalize_entity(m_rel.group(3))
                conditions.append({"type": "rel", "subj": subj, "verb": verb, "obj": obj, "var": None})
                continue
            # Unparseable clause
            return None

    return conditions if conditions else None


def _split_and(text: str) -> List[str]:
    """Split conditions on 'and' while keeping 'and not' together."""
    # Replace " and they " / " and it " as conjunction markers
    # but handle "and not" as negation within a clause
    parts = []
    # Split on " and " but be careful
    tokens = re.split(r'\s+and\s+', text)
    result = []
    for t in tokens:
        result.append(t.strip())
    return result


def _parse_single_condition(clause: str, var: str = "?x") -> Optional[dict]:
    """Parse one condition clause with a variable binding."""
    clause = clause.strip()

    # Remove leading pronoun references
    for pron in ("it ", "they "):
        if clause.startswith(pron):
            clause = clause[len(pron):]
            break
    # Also handle "it is" -> "is"
    if clause.startswith("it is ") or clause.startswith("they are "):
        clause = "is " + clause.split(" ", 2)[-1]

    # "is not P"
    m = re.match(r'^is\s+not\s+(\w+)$', clause, re.IGNORECASE)
    if m:
        return {"type": "negprop", "var": var, "prop": normalize(m.group(1))}

    # "is P"
    m = re.match(r'^is\s+(\w+)$', clause, re.IGNORECASE)
    if m:
        return {"type": "prop", "var": var, "prop": normalize(m.group(1))}

    # "not P" (shorthand for "is not P")
    m = re.match(r'^not\s+(\w+)$', clause, re.IGNORECASE)
    if m:
        return {"type": "negprop", "var": var, "prop": normalize(m.group(1))}

    # "do not verb Y" / "does not verb Y" (negated relation for ?x)
    for vb in KNOWN_VERBS:
        base = normalize_verb(vb)
        pattern = re.compile(r'^do(?:es)?\s+not\s+' + base + r'\s+(.+)$', re.IGNORECASE)
        m = pattern.match(clause)
        if m:
            obj = normalize_entity(m.group(1))
            return {"type": "negrel", "var": var, "verb": base, "obj": obj}

    # "verb Y" (e.g., "sees the cat", "eats the squirrel")
    for vb in KNOWN_VERBS:
        pattern = re.compile(r'^' + vb + r'\s+(.+)$', re.IGNORECASE)
        m = pattern.match(clause)
        if m:
            obj = normalize_entity(m.group(1))
            return {"type": "rel", "var": var, "verb": normalize_verb(vb), "obj": obj}

    # "the X does not verb Y" where subject is another specific entity
    for vb in KNOWN_VERBS:
        base = normalize_verb(vb)
        pattern = re.compile(r'^(.+?)\s+does\s+not\s+' + base + r'\s+(.+)$', re.IGNORECASE)
        m = pattern.match(clause)
        if m:
            subj = normalize_entity(m.group(1))
            obj = normalize_entity(m.group(2))
            return {"type": "negrel", "subj": subj, "verb": base, "obj": obj}

    # "the X verb Y" where subject is another entity (not variable)
    m = RE_RELATION.match(clause)
    if m:
        subj = normalize_entity(m.group(1))
        verb = normalize_verb(m.group(2))
        obj = normalize_entity(m.group(3))
        return {"type": "rel_specific", "subj": subj, "verb": verb, "obj": obj}

    # "P" bare adjective (property of ?x)
    m = re.match(r'^(\w+)$', clause)
    if m:
        return {"type": "prop", "var": var, "prop": normalize(m.group(1))}

    return None


def _parse_conclusion(text: str) -> Optional[dict]:
    """Parse the conclusion of a rule."""
    text = text.strip()

    # "it/they is/are [not] P"
    m = re.match(r'^(?:it|they)\s+(?:is|are)\s+not\s+(\w+)$', text, re.IGNORECASE)
    if m:
        return {"type": "negprop", "var": "?x", "prop": normalize(m.group(1))}

    m = re.match(r'^(?:it|they)\s+(?:is|are)\s+(\w+)$', text, re.IGNORECASE)
    if m:
        return {"type": "prop", "var": "?x", "prop": normalize(m.group(1))}

    # "it/they verb Y"
    for vb in KNOWN_VERBS:
        pattern = re.compile(
            r'^(?:it|they)\s+' + vb + r'\s+(.+)$', re.IGNORECASE
        )
        m = pattern.match(text)
        if m:
            obj = normalize_entity(m.group(1))
            return {"type": "rel", "var": "?x", "verb": normalize_verb(vb), "obj": obj}

    # "it/they do not verb Y"
    for vb in KNOWN_VERBS:
        base = normalize_verb(vb)
        pattern = re.compile(
            r'^(?:it|they)\s+do(?:es)?\s+not\s+' + base + r'\s+(.+)$', re.IGNORECASE
        )
        m = pattern.match(text)
        if m:
            obj = normalize_entity(m.group(1))
            return {"type": "negrel", "var": "?x", "verb": base, "obj": obj}

    # Specific entity conclusion: "the cat is big", "Anne is smart"
    m = re.match(r'^(.+?)\s+is\s+not\s+(\w+)$', text, re.IGNORECASE)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return {"type": "negprop", "entity": entity, "prop": prop, "var": None}

    m = re.match(r'^(.+?)\s+is\s+(\w+)$', text, re.IGNORECASE)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return {"type": "prop", "entity": entity, "prop": prop, "var": None}

    # Specific entity relation conclusion: "the cat likes the bald eagle"
    m = RE_RELATION.match(text)
    if m:
        subj = normalize_entity(m.group(1))
        verb = normalize_verb(m.group(2))
        obj = normalize_entity(m.group(3))
        return {"type": "rel", "subj": subj, "verb": verb, "obj": obj, "var": None}

    return None


def _parse_all_are(s: str) -> Optional[Rule]:
    """Parse 'All P things/people are Q'."""
    # "All blue people are white"
    # "All kind people are blue"
    m = re.match(
        r'^All\s+(.+?)\s+(?:things|people)\s+are\s+(\w+)$',
        s, re.IGNORECASE
    )
    if not m:
        return None

    adj_text = m.group(1).strip()
    result_prop = normalize(m.group(2))

    # Parse comma/and-separated adjectives
    adjs = _parse_adj_list(adj_text)
    conditions = [{"type": "prop", "var": "?x", "prop": normalize(a)} for a in adjs]
    conclusion = {"type": "prop", "var": "?x", "prop": result_prop}

    return Rule(conditions, conclusion, raw=s)


def _parse_adj_list_are(s: str) -> Optional[Rule]:
    """Parse 'P, Q things are [not] R' or 'P things are [not] R' (no 'All' prefix)."""
    # Try with "not" in conclusion
    m = re.match(
        r'^(.+?)\s+(?:things|people)\s+are\s+not\s+(\w+)$',
        s, re.IGNORECASE
    )
    conc_negated = bool(m)
    if not m:
        m = re.match(
            r'^(.+?)\s+(?:things|people)\s+are\s+(\w+)$',
            s, re.IGNORECASE
        )
    if not m:
        return None

    adj_text = m.group(1).strip()
    result_prop = normalize(m.group(2))

    adjs = _parse_adj_list(adj_text)
    if len(adjs) < 1:
        return None

    conditions = [{"type": "prop", "var": "?x", "prop": normalize(a)} for a in adjs]
    conc_type = "negprop" if conc_negated else "prop"
    conclusion = {"type": conc_type, "var": "?x", "prop": result_prop}

    return Rule(conditions, conclusion, raw=s)


def _parse_adj_list(text: str) -> List[str]:
    """Parse comma/and-separated adjective list."""
    # "blue" -> ["blue"]
    # "green, big" -> ["green", "big"]
    # "white, nice" -> ["white", "nice"]
    # "nice and blue and kind" -> ["nice", "blue", "kind"]
    text = text.replace(',', ' and ')
    parts = re.split(r'\s+and\s+', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


# ─── Query parsing ──────────────────────────────────────────────────

def parse_query(text: str) -> Optional[Query]:
    """Parse a question/query from the prompt."""
    text = text.strip().rstrip('.')

    # "X is not P"
    m = re.match(r'^(.+?)\s+is\s+not\s+(\w+)$', text, re.IGNORECASE)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return Query(negated=True, kind="prop", args=(entity, prop))

    # "X does not verb Y"
    for vb in KNOWN_VERBS:
        base = normalize_verb(vb)
        pattern = re.compile(
            r'^(.+?)\s+does\s+not\s+' + base + r'\s+(.+)$', re.IGNORECASE
        )
        m = pattern.match(text)
        if m:
            subj = normalize_entity(m.group(1))
            obj = normalize_entity(m.group(2))
            return Query(negated=True, kind="rel", args=(subj, base, obj))

    # "X verb Y"
    m = RE_RELATION.match(text)
    if m:
        subj = normalize_entity(m.group(1))
        verb = normalize_verb(m.group(2))
        obj = normalize_entity(m.group(3))
        return Query(negated=False, kind="rel", args=(subj, verb, obj))

    # "X is P"
    m = re.match(r'^(.+?)\s+is\s+(\w+)$', text, re.IGNORECASE)
    if m:
        entity = normalize_entity(m.group(1))
        prop = normalize(m.group(2))
        return Query(negated=False, kind="prop", args=(entity, prop))

    return None


# ─── Full prompt parsing ────────────────────────────────────────────

def parse_prompt(prompt_text: str) -> Dict[str, Any]:
    """
    Parse a full RuleTaker-style prompt into structured components.

    Returns dict with:
        facts: List[Fact]
        rules: List[Rule]
        query: Optional[Query]
        entities: set of entity names
        parse_success: bool
        unparsed: List[str] (sentences that couldn't be parsed)
    """
    # Extract the question from "Question: ..." line
    query = None
    question_match = re.search(r'Question:\s*(.+?)(?:\n|$)', prompt_text)
    if question_match:
        query_text = question_match.group(1).strip()
        query = parse_query(query_text)

    # Get the fact/rule block (everything before "Question:")
    body = prompt_text
    if question_match:
        body = prompt_text[:question_match.start()]

    # Remove "Answer with only Yes or No" and "Answer:" lines
    body = re.sub(r'Answer\s+with\s+only\s+Yes\s+or\s+No\.?', '', body, flags=re.IGNORECASE)
    body = re.sub(r'Answer:\s*$', '', body, flags=re.IGNORECASE)

    sentences = split_sentences(body)

    facts = []
    rules = []
    unparsed = []
    entities = set()

    for sent in sentences:
        # Try as rule first (rules start with "If" or "All" or have comma-adj pattern)
        rule = parse_rule_sentence(sent)
        if rule:
            rules.append(rule)
            continue

        # Try as fact
        fact = parse_fact_sentence(sent)
        if fact:
            facts.append(fact)
            # Track entities
            if fact.kind == "prop" or fact.kind == "negprop":
                entities.add(fact.args[0])
            elif fact.kind in ("rel", "negrel"):
                entities.add(fact.args[0])
                entities.add(fact.args[2])
            continue

        # Unparsed
        if sent.strip():
            unparsed.append(sent)

    return {
        "facts": facts,
        "rules": rules,
        "query": query,
        "entities": entities,
        "parse_success": query is not None and (len(facts) > 0 or len(rules) > 0),
        "unparsed": unparsed,
    }
