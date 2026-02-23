#!/usr/bin/env python3
"""
A6 Logic Engine — Forward-chaining inference for RuleTaker-style problems.

Uses closed-world assumption: anything not derivable is assumed false.
Runs forward chaining until fixed point, then evaluates the query.
"""

import time
from typing import Dict, Any, Set, List, Optional

from a6_logic_parser import Fact, Rule, Query, parse_prompt


# Maximum forward-chaining iterations (safety bound)
MAX_ITERATIONS = 50


def forward_chain(facts: List[Fact], rules: List[Rule],
                  entities: Set[str]) -> Set[Fact]:
    """
    Forward-chain rules over facts until fixed point.

    Uses eager/immediate application: each new fact is added to known
    immediately so that subsequent rule checks in the same iteration
    see it. This prevents spurious derivations where a positive rule
    and a negation-checking rule race in the same batch.

    Returns the complete set of derived facts (including originals).
    """
    known = set(facts)
    iteration = 0

    # Sort rules: positive-conclusion rules first, then rules with
    # negation conditions. This ensures facts are derived before
    # negation checks run.
    def _rule_priority(rule):
        has_neg_cond = any(c["type"] in ("negprop", "negrel") for c in rule.conditions)
        return 1 if has_neg_cond else 0
    sorted_rules = sorted(rules, key=_rule_priority)

    while iteration < MAX_ITERATIONS:
        changed = False
        for rule in sorted_rules:
            derived = _apply_rule(rule, known, entities)
            for f in derived:
                if f not in known:
                    known.add(f)
                    changed = True

        if not changed:
            break  # Fixed point reached

        iteration += 1

    return known


def _apply_rule(rule: Rule, known: Set[Fact], entities: Set[str]) -> Set[Fact]:
    """
    Apply a single rule against known facts.
    Returns set of new facts derived.
    """
    conditions = rule.conditions
    conclusion = rule.conclusion

    # Check if this is a universal rule (has variable ?x) or specific
    has_var = any(c.get("var") == "?x" for c in conditions)
    conc_has_var = conclusion.get("var") == "?x"

    if has_var or conc_has_var:
        # Universal: try binding ?x to each entity
        results = set()
        for entity in entities:
            if _check_conditions(conditions, known, entity, entities):
                fact = _instantiate_conclusion(conclusion, entity)
                if fact:
                    results.add(fact)
        return results
    else:
        # Specific: conditions and conclusion reference specific entities
        if _check_specific_conditions(conditions, known):
            fact = _instantiate_specific_conclusion(conclusion)
            if fact:
                return {fact}
        return set()


def _check_conditions(conditions: list, known: Set[Fact],
                      binding: str, entities: Set[str]) -> bool:
    """Check if all conditions are satisfied for a given ?x binding."""
    for cond in conditions:
        ctype = cond["type"]

        if ctype == "prop":
            var = cond.get("var")
            if var == "?x":
                target = Fact("prop", (binding, cond["prop"]))
            else:
                entity = cond.get("entity", binding)
                target = Fact("prop", (entity, cond["prop"]))
            if target not in known:
                return False

        elif ctype == "negprop":
            # Closed-world: "not P" means P is not in known facts
            var = cond.get("var")
            if var == "?x":
                target = Fact("prop", (binding, cond["prop"]))
            else:
                entity = cond.get("entity", binding)
                target = Fact("prop", (entity, cond["prop"]))
            if target in known:
                return False  # P IS known, so "not P" fails

        elif ctype == "rel":
            var = cond.get("var")
            if var == "?x":
                target = Fact("rel", (binding, cond["verb"], cond["obj"]))
            else:
                subj = cond.get("subj", binding)
                target = Fact("rel", (subj, cond["verb"], cond["obj"]))
            if target not in known:
                return False

        elif ctype == "rel_specific":
            # A relation between two specific entities (no variable)
            target = Fact("rel", (cond["subj"], cond["verb"], cond["obj"]))
            if target not in known:
                return False

        elif ctype == "negrel":
            var = cond.get("var")
            if var == "?x":
                target = Fact("rel", (binding, cond["verb"], cond["obj"]))
            else:
                subj = cond.get("subj", binding)
                target = Fact("rel", (subj, cond["verb"], cond["obj"]))
            if target in known:
                return False

    return True


def _check_specific_conditions(conditions: list, known: Set[Fact]) -> bool:
    """Check conditions that reference specific entities (no variable)."""
    for cond in conditions:
        ctype = cond["type"]
        if ctype == "prop":
            entity = cond.get("entity", "")
            target = Fact("prop", (entity, cond["prop"]))
            if target not in known:
                return False
        elif ctype == "negprop":
            entity = cond.get("entity", "")
            target = Fact("prop", (entity, cond["prop"]))
            if target in known:
                return False
        elif ctype == "negrel":
            subj = cond.get("subj", "")
            target = Fact("rel", (subj, cond["verb"], cond["obj"]))
            if target in known:
                return False  # Relation IS known, so "not rel" fails
        elif ctype == "rel":
            subj = cond.get("subj", "")
            target = Fact("rel", (subj, cond["verb"], cond["obj"]))
            if target not in known:
                return False
    return True


def _instantiate_conclusion(conclusion: dict, binding: str) -> Optional[Fact]:
    """Create a fact from a conclusion template with ?x bound to entity."""
    ctype = conclusion["type"]

    if ctype == "prop":
        var = conclusion.get("var")
        if var == "?x":
            return Fact("prop", (binding, conclusion["prop"]))
        elif conclusion.get("entity"):
            return Fact("prop", (conclusion["entity"], conclusion["prop"]))

    elif ctype == "negprop":
        # Derive an explicit negation fact
        var = conclusion.get("var")
        if var == "?x":
            return Fact("negprop", (binding, conclusion["prop"]))
        elif conclusion.get("entity"):
            return Fact("negprop", (conclusion["entity"], conclusion["prop"]))

    elif ctype == "rel":
        var = conclusion.get("var")
        if var == "?x":
            return Fact("rel", (binding, conclusion["verb"], conclusion["obj"]))
        elif conclusion.get("subj"):
            return Fact("rel", (conclusion["subj"], conclusion["verb"], conclusion["obj"]))

    elif ctype == "negrel":
        var = conclusion.get("var")
        if var == "?x":
            return Fact("negrel", (binding, conclusion["verb"], conclusion["obj"]))
        elif conclusion.get("subj"):
            return Fact("negrel", (conclusion["subj"], conclusion["verb"], conclusion["obj"]))

    return None


def _instantiate_specific_conclusion(conclusion: dict) -> Optional[Fact]:
    """Create a fact from a conclusion with specific entities."""
    ctype = conclusion["type"]

    if ctype == "prop" and conclusion.get("entity"):
        return Fact("prop", (conclusion["entity"], conclusion["prop"]))
    elif ctype == "negprop" and conclusion.get("entity"):
        return Fact("negprop", (conclusion["entity"], conclusion["prop"]))
    elif ctype == "rel" and conclusion.get("subj"):
        return Fact("rel", (conclusion["subj"], conclusion["verb"], conclusion["obj"]))
    elif ctype == "negrel" and conclusion.get("subj"):
        return Fact("negrel", (conclusion["subj"], conclusion["verb"], conclusion["obj"]))

    return None


def evaluate_query(query: Query, known: Set[Fact]) -> str:
    """
    Evaluate a query against known facts.

    Returns "Yes", "No", or "Unknown".

    Under closed-world assumption:
    - "X is P" → Yes if Prop(X,P) in known, No otherwise
    - "X is not P" → Yes if Prop(X,P) NOT in known, No if Prop(X,P) in known
    - "X verb Y" → Yes if Rel(X,verb,Y) in known, No otherwise
    - "X does not verb Y" → Yes if Rel(X,verb,Y) NOT in known, No otherwise
    """
    if query.kind == "prop":
        entity, prop = query.args
        positive_fact = Fact("prop", (entity, prop))
        explicit_neg = Fact("negprop", (entity, prop))

        if query.negated:
            # "X is not P" — true if P not derivable (or explicitly negated)
            if explicit_neg in known:
                return "Yes"
            if positive_fact not in known:
                return "Yes"  # Closed-world: not derivable = false
            return "No"  # P is derivable, so "not P" is false
        else:
            # "X is P" — true if derivable
            if positive_fact in known:
                return "Yes"
            return "No"

    elif query.kind == "rel":
        subj, verb, obj = query.args
        positive_fact = Fact("rel", (subj, verb, obj))
        explicit_neg = Fact("negrel", (subj, verb, obj))

        if query.negated:
            if explicit_neg in known:
                return "Yes"
            if positive_fact not in known:
                return "Yes"
            return "No"
        else:
            if positive_fact in known:
                return "Yes"
            return "No"

    return "Unknown"


# ─── Main A6 entry point ───────────────────────────────────────────

def solve_logic(prompt_text: str) -> Dict[str, Any]:
    """
    Full A6 pipeline: parse → forward chain → evaluate query.

    Returns:
        answer: "Yes" / "No" / "Unknown"
        parse_success: bool
        n_facts: int (initial facts)
        n_rules: int
        n_derived: int (total facts after chaining)
        iterations: int
        query_str: str
        trace: str (human-readable trace)
        latency_ms: float
    """
    t0 = time.time()

    # Parse
    parsed = parse_prompt(prompt_text)

    if not parsed["parse_success"]:
        return {
            "answer": "Unknown",
            "parse_success": False,
            "n_facts": 0,
            "n_rules": 0,
            "n_derived": 0,
            "iterations": 0,
            "query_str": "",
            "trace": f"Parse failed. Unparsed: {parsed['unparsed'][:3]}",
            "rule_fired": "none",
            "pattern": "unparsable",
            "latency_ms": (time.time() - t0) * 1000,
        }

    facts = parsed["facts"]
    rules = parsed["rules"]
    query = parsed["query"]
    entities = parsed["entities"]

    n_initial = len(facts)

    # Forward chain
    known = forward_chain(facts, rules, entities)
    n_derived = len(known)

    # Evaluate query
    answer = evaluate_query(query, known)

    # Build trace
    new_facts = known - set(facts)
    trace_lines = []
    trace_lines.append(f"Initial: {n_initial} facts, {len(rules)} rules, {len(entities)} entities")
    trace_lines.append(f"Derived: {n_derived - n_initial} new facts")
    if new_facts:
        for nf in sorted(new_facts, key=str)[:20]:
            trace_lines.append(f"  + {nf}")
    trace_lines.append(f"Query: {query} → {answer}")

    latency_ms = (time.time() - t0) * 1000

    return {
        "answer": answer,
        "parse_success": True,
        "n_facts": n_initial,
        "n_rules": len(rules),
        "n_derived": n_derived,
        "iterations": 0,  # TODO: track actual iteration count
        "query_str": str(query),
        "trace": "\n".join(trace_lines),
        "rule_fired": "forward_chain",
        "pattern": "ruletaker",
        "latency_ms": latency_ms,
    }


if __name__ == "__main__":
    # Quick self-test
    test_prompt = """Anne is kind. Anne is quiet. Bob is blue. Dave is white. Gary is blue. Gary is kind. Gary is young. If someone is blue and not kind then they are quiet. If someone is quiet and not kind then they are smart.

Question: Bob is smart.
Answer with only Yes or No.
Answer:"""

    result = solve_logic(test_prompt)
    print(f"Answer: {result['answer']}")
    print(f"Parse: {result['parse_success']}")
    print(f"Trace:\n{result['trace']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
