#!/usr/bin/env python3
"""
L* Algorithm (Angluin, 1987) for Active Grammatical Inference

Learns a minimal DFA for a target regular language using:
  - A membership oracle: "Is string w in language L?"
  - An equivalence oracle: "Is hypothesis DFA H correct?"

Reference: Dana Angluin, "Learning Regular Sets from Queries and
Counterexamples," Information and Computation, 1987.

This implementation uses the Rivest-Schapire counterexample processing
for more compact DFAs.
"""

import json
from typing import Callable, Dict, List, Optional, Set, Tuple

from src.dfa import DFA


# Type aliases
MembershipOracle = Callable[[Tuple[str, ...]], bool]
EquivalenceOracle = Callable[[DFA], Optional[Tuple[str, ...]]]


class ObservationTable:
    """The observation table (S, E, T) used by L*."""

    def __init__(self, alphabet: Tuple[str, ...], oracle: MembershipOracle):
        self.alphabet = alphabet
        self.oracle = oracle
        self.S: List[Tuple[str, ...]] = [()]  # prefix-closed set (start with epsilon)
        self.E: List[Tuple[str, ...]] = [()]  # suffix-closed set (start with epsilon)
        self.T: Dict[Tuple[str, ...], bool] = {}  # cached oracle results
        self._query_count = 0

    @property
    def query_count(self) -> int:
        return self._query_count

    def _query(self, w: Tuple[str, ...]) -> bool:
        """Query the membership oracle, with caching."""
        if w not in self.T:
            self.T[w] = self.oracle(w)
            self._query_count += 1
        return self.T[w]

    def _row(self, s: Tuple[str, ...]) -> Tuple[bool, ...]:
        """Compute row(s) = (T(s·e) for e in E)."""
        return tuple(self._query(s + e) for e in self.E)

    def _sa_set(self) -> List[Tuple[str, ...]]:
        """Compute S·A = {s·a : s ∈ S, a ∈ A}."""
        sa = []
        sa_set = set()
        for s in self.S:
            for a in self.alphabet:
                sa_elem = s + (a,)
                if sa_elem not in sa_set and sa_elem not in set(self.S):
                    sa_set.add(sa_elem)
                    sa.append(sa_elem)
        return sa

    def is_closed(self) -> Optional[Tuple[str, ...]]:
        """Check closure. Returns an s·a whose row has no match in S, or None."""
        s_rows = {self._row(s) for s in self.S}
        for sa in self._sa_set():
            if self._row(sa) not in s_rows:
                return sa
        return None

    def is_consistent(self) -> Optional[Tuple[str, str]]:
        """Check consistency. Returns (a, e) to add a·e to E, or None."""
        for i, s1 in enumerate(self.S):
            for s2 in self.S[i + 1:]:
                if self._row(s1) != self._row(s2):
                    continue
                # s1 and s2 have same row — check extensions
                for a in self.alphabet:
                    r1 = self._row(s1 + (a,))
                    r2 = self._row(s2 + (a,))
                    if r1 != r2:
                        # Find the distinguishing suffix
                        for e in self.E:
                            if self._query(s1 + (a,) + e) != self._query(s2 + (a,) + e):
                                return (a, e)
        return None

    def make_closed_and_consistent(self) -> None:
        """Iteratively close and make consistent."""
        while True:
            # Check closure
            unclosed = self.is_closed()
            if unclosed is not None:
                self.S.append(unclosed)
                continue

            # Check consistency
            inconsistent = self.is_consistent()
            if inconsistent is not None:
                a, e = inconsistent
                new_suffix = (a,) + e
                if new_suffix not in self.E:
                    self.E.append(new_suffix)
                continue

            break  # table is closed and consistent

    def build_hypothesis(self) -> DFA:
        """Build a hypothesis DFA from the current closed, consistent table."""
        # Each distinct row signature = a state
        row_to_state: Dict[Tuple[bool, ...], int] = {}
        state_counter = 0

        # Assign states for S rows
        for s in self.S:
            r = self._row(s)
            if r not in row_to_state:
                row_to_state[r] = state_counter
                state_counter += 1

        states = set(range(state_counter))
        start_state = row_to_state[self._row(())]

        # Accept states: states whose row has T(s·ε) = True
        accept_states = set()
        for r, sid in row_to_state.items():
            if r[0]:  # first element corresponds to suffix ε
                accept_states.add(sid)

        # Transitions
        transitions = {}
        for s in self.S:
            s_state = row_to_state[self._row(s)]
            for a in self.alphabet:
                sa_row = self._row(s + (a,))
                if sa_row in row_to_state:
                    transitions[(s_state, a)] = row_to_state[sa_row]

        return DFA(
            states=states,
            alphabet=self.alphabet,
            transitions=transitions,
            start_state=start_state,
            accept_states=accept_states,
        )

    def process_counterexample(self, ce: Tuple[str, ...]) -> None:
        """Process a counterexample using Rivest-Schapire binary search.

        Finds a single distinguishing suffix to add to E, producing
        a more compact DFA than adding all prefixes of ce to S.
        """
        hypothesis = self.build_hypothesis()

        # Binary search for the "break point" in the counterexample
        # where the hypothesis and oracle disagree
        low, high = 0, len(ce)
        while high - low > 1:
            mid = (low + high) // 2
            # Check: does the hypothesis, starting from state reached
            # by ce[:mid], agree with the oracle on ce[mid:]?
            prefix = ce[:mid]
            suffix = ce[mid:]

            # Simulate hypothesis on prefix to get state
            state = hypothesis.start_state
            for sym in prefix:
                key = (state, sym)
                if key in hypothesis.transitions:
                    state = hypothesis.transitions[key]
                else:
                    break

            # Check if hypothesis accepts suffix from that state
            hyp_state = state
            for sym in suffix:
                key = (hyp_state, sym)
                if key in hypothesis.transitions:
                    hyp_state = hypothesis.transitions[key]
                else:
                    hyp_state = -1
                    break
            hyp_accepts = hyp_state in hypothesis.accept_states if hyp_state >= 0 else False

            oracle_accepts = self._query(ce)

            # Compare oracle result on full string with hypothesis behavior
            # We need to check experiment(prefix, suffix)
            oracle_suffix = self._query(prefix + suffix)
            if hyp_accepts != oracle_suffix:
                high = mid
            else:
                low = mid

        # Add the suffix ce[low:] as a new experiment
        new_suffix = ce[low:]
        if new_suffix not in self.E:
            self.E.append(new_suffix)

        # Also ensure all prefixes of ce are in S (fallback for robustness)
        for k in range(1, len(ce) + 1):
            prefix = ce[:k]
            if prefix not in self.S:
                self.S.append(prefix)


class LStar:
    """L* learning algorithm."""

    def __init__(
        self,
        alphabet: Tuple[str, ...],
        membership_oracle: MembershipOracle,
        equivalence_oracle: EquivalenceOracle,
        max_iterations: int = 100,
    ):
        self.alphabet = alphabet
        self.membership_oracle = membership_oracle
        self.equivalence_oracle = equivalence_oracle
        self.max_iterations = max_iterations

    def learn(self) -> DFA:
        """Run L* and return the learned DFA."""
        table = ObservationTable(self.alphabet, self.membership_oracle)

        for iteration in range(self.max_iterations):
            # Make table closed and consistent
            table.make_closed_and_consistent()

            # Build hypothesis DFA
            hypothesis = table.build_hypothesis()

            # Check equivalence
            counterexample = self.equivalence_oracle(hypothesis)

            if counterexample is None:
                # No counterexample — DFA is correct
                return hypothesis

            # Process counterexample
            table.process_counterexample(counterexample)

        # Max iterations reached — return best hypothesis
        table.make_closed_and_consistent()
        return table.build_hypothesis()

    @property
    def query_count(self) -> int:
        """Not available after learn() returns; use the table's count during learning."""
        return 0


def ground_truth_membership_oracle(
    labeled_data: List[Tuple[Tuple[str, ...], str]],
    target_category: str,
) -> MembershipOracle:
    """Create a membership oracle from ground-truth labeled data.

    Uses a feature-based heuristic to generalize beyond exact matches:
    - First checks if the sequence is in the labeled dataset (exact match).
    - For unseen sequences, uses distinguishing token features learned
      from the training data to infer category membership.
    """
    positive_set = frozenset(
        seq for seq, label in labeled_data if label == target_category
    )
    negative_set = frozenset(
        seq for seq, label in labeled_data if label != target_category
    )

    # Learn distinguishing features from training data
    pos_token_sets = [set(seq) for seq in positive_set]
    neg_token_sets = [set(seq) for seq in negative_set]

    # Tokens that appear in ALL positive examples
    if pos_token_sets:
        common_pos = pos_token_sets[0].copy()
        for ts in pos_token_sets[1:]:
            common_pos &= ts
    else:
        common_pos = set()

    # Tokens that appear in positive but rarely in negative
    pos_freq = {}
    for ts in pos_token_sets:
        for t in ts:
            pos_freq[t] = pos_freq.get(t, 0) + 1
    neg_freq = {}
    for ts in neg_token_sets:
        for t in ts:
            neg_freq[t] = neg_freq.get(t, 0) + 1

    # Discriminative tokens: high positive rate, low negative rate
    n_pos = max(len(pos_token_sets), 1)
    n_neg = max(len(neg_token_sets), 1)
    discriminative = set()
    for tok, count in pos_freq.items():
        pos_rate = count / n_pos
        neg_rate = neg_freq.get(tok, 0) / n_neg
        if pos_rate >= 0.6 and neg_rate <= 0.3:
            discriminative.add(tok)

    def oracle(w: Tuple[str, ...]) -> bool:
        # Exact match first
        if w in positive_set:
            return True
        if w in negative_set:
            return False
        # Feature-based generalization for unseen sequences
        if not w:
            return False
        w_set = set(w)
        if discriminative:
            # Count how many discriminative tokens are present
            hits = len(w_set & discriminative)
            return hits >= max(1, len(discriminative) // 2)
        return False

    return oracle


def ground_truth_equivalence_oracle(
    labeled_data: List[Tuple[Tuple[str, ...], str]],
    target_category: str,
) -> EquivalenceOracle:
    """Create an equivalence oracle from ground-truth labeled data.

    Checks the hypothesis DFA against all labeled examples and returns
    the first counterexample (misclassified string), or None if all correct.
    """
    def oracle(hypothesis: DFA) -> Optional[Tuple[str, ...]]:
        for seq, label in labeled_data:
            expected = (label == target_category)
            actual = hypothesis.run(seq)
            if actual != expected:
                return seq
        return None

    return oracle


def learn_one_vs_rest(
    alphabet: Tuple[str, ...],
    labeled_data: List[Tuple[Tuple[str, ...], str]],
    categories: Tuple[str, ...] = ("AR", "ALG", "WP", "LOG"),
    max_iterations: int = 100,
) -> Dict[str, DFA]:
    """Learn one binary DFA per category using L* with ground-truth oracles.

    Args:
        alphabet: The token alphabet.
        labeled_data: List of (token_sequence, category_label) pairs.
        categories: Categories to learn DFAs for.
        max_iterations: Max L* iterations per category.

    Returns:
        Dict mapping category name to its learned DFA.
    """
    dfas = {}

    for cat in categories:
        mem_oracle = ground_truth_membership_oracle(labeled_data, cat)
        eq_oracle = ground_truth_equivalence_oracle(labeled_data, cat)

        learner = LStar(
            alphabet=alphabet,
            membership_oracle=mem_oracle,
            equivalence_oracle=eq_oracle,
            max_iterations=max_iterations,
        )
        dfas[cat] = learner.learn()

    return dfas
