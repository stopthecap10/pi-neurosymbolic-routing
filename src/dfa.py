#!/usr/bin/env python3
"""
Minimal DFA (Deterministic Finite Automaton) implementation.

Used by L* and RPNI to represent learned routing classifiers.
Supports simulation, serialization, and conversion from observation tables.
"""

import json
from typing import Dict, FrozenSet, Optional, Set, Tuple


class DFA:
    """A deterministic finite automaton over a finite alphabet."""

    def __init__(
        self,
        states: Set[int],
        alphabet: Tuple[str, ...],
        transitions: Dict[Tuple[int, str], int],
        start_state: int,
        accept_states: Set[int],
    ):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # (state, symbol) -> state
        self.start_state = start_state
        self.accept_states = accept_states

    def run(self, sequence: Tuple[str, ...]) -> bool:
        """Simulate the DFA on a token sequence. Returns True if accepted."""
        state = self.start_state
        for symbol in sequence:
            key = (state, symbol)
            if key not in self.transitions:
                return False  # no transition = reject (implicit dead state)
            state = self.transitions[key]
        return state in self.accept_states

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict."""
        return {
            "states": sorted(self.states),
            "alphabet": list(self.alphabet),
            "transitions": {
                f"{s},{sym}": t for (s, sym), t in self.transitions.items()
            },
            "start_state": self.start_state,
            "accept_states": sorted(self.accept_states),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DFA":
        """Deserialize from a dict."""
        transitions = {}
        for key_str, target in d["transitions"].items():
            s_str, sym = key_str.split(",", 1)
            transitions[(int(s_str), sym)] = target
        return cls(
            states=set(d["states"]),
            alphabet=tuple(d["alphabet"]),
            transitions=transitions,
            start_state=d["start_state"],
            accept_states=set(d["accept_states"]),
        )

    def to_json(self, path: str) -> None:
        """Save DFA to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "DFA":
        """Load DFA from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return (
            f"DFA(states={self.num_states}, transitions={self.num_transitions}, "
            f"accept={len(self.accept_states)})"
        )


class MultiClassDFA:
    """Wrapper that holds one binary DFA per category for one-vs-rest classification."""

    def __init__(self, dfas: Dict[str, DFA], priority: Tuple[str, ...] = ()):
        self.dfas = dfas  # category_name -> DFA
        self.priority = priority or tuple(dfas.keys())

    def classify(self, sequence: Tuple[str, ...]) -> Optional[str]:
        """Classify a token sequence. Returns category or None if no DFA accepts."""
        for cat in self.priority:
            if cat in self.dfas and self.dfas[cat].run(sequence):
                return cat
        return None

    def classify_all(self, sequence: Tuple[str, ...]) -> list:
        """Return all categories whose DFA accepts the sequence."""
        return [cat for cat, dfa in self.dfas.items() if dfa.run(sequence)]

    def to_dict(self) -> dict:
        return {
            "priority": list(self.priority),
            "dfas": {cat: dfa.to_dict() for cat, dfa in self.dfas.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultiClassDFA":
        dfas = {cat: DFA.from_dict(dfa_d) for cat, dfa_d in d["dfas"].items()}
        return cls(dfas=dfas, priority=tuple(d["priority"]))

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "MultiClassDFA":
        with open(path) as f:
            return cls.from_dict(json.load(f))
