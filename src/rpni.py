#!/usr/bin/env python3
"""
RPNI (Regular Positive and Negative Inference) Algorithm

Learns a DFA from labeled positive and negative examples.
Used as a passive grammatical inference baseline.

Reference: Oncina & García (1992), "Inferring Regular Languages in
Polynomial Update Time."
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from src.dfa import DFA


class PrefixTreeState:
    """A state in the prefix tree acceptor."""
    __slots__ = ("id", "transitions", "is_accept")

    def __init__(self, state_id: int, is_accept: bool = False):
        self.id = state_id
        self.transitions: Dict[str, int] = {}  # symbol -> state_id
        self.is_accept = is_accept


class RPNI:
    """RPNI algorithm for learning DFAs from positive and negative examples."""

    def __init__(self, alphabet: Tuple[str, ...]):
        self.alphabet = alphabet

    def learn(
        self,
        positive: List[Tuple[str, ...]],
        negative: List[Tuple[str, ...]],
    ) -> DFA:
        """Learn a DFA that accepts all positive and rejects all negative examples.

        Args:
            positive: List of token sequences IN the target language.
            negative: List of token sequences NOT in the target language.

        Returns:
            A DFA consistent with the training data.
        """
        # Step 1: Build prefix tree acceptor (PTA) from positive examples
        states, next_id = self._build_pta(positive)

        # Step 2: Get states in BFS order (canonical ordering for merging)
        ordered = self._bfs_order(states)

        # Step 3: Try merging states in order
        for i in range(1, len(ordered)):
            target_id = ordered[i]
            if target_id not in states:
                continue  # already merged away

            # Try merging target into each earlier state
            for j in range(i):
                candidate_id = ordered[j]
                if candidate_id not in states:
                    continue

                # Attempt merge
                merged = self._try_merge(states, candidate_id, target_id)
                if merged is not None:
                    # Check if any negative example is accepted
                    if not self._accepts_any_negative(merged, negative):
                        states = merged
                        break  # merge accepted, move to next state

        # Step 4: Convert to DFA
        return self._states_to_dfa(states)

    def _build_pta(
        self, positive: List[Tuple[str, ...]]
    ) -> Tuple[Dict[int, PrefixTreeState], int]:
        """Build a prefix tree acceptor from positive examples."""
        states: Dict[int, PrefixTreeState] = {}
        root = PrefixTreeState(0, is_accept=(() in positive))
        states[0] = root
        next_id = 1

        for seq in positive:
            current = 0
            for sym in seq:
                if sym not in states[current].transitions:
                    new_state = PrefixTreeState(next_id)
                    states[next_id] = new_state
                    states[current].transitions[sym] = next_id
                    next_id += 1
                current = states[current].transitions[sym]
            states[current].is_accept = True

        return states, next_id

    def _bfs_order(self, states: Dict[int, PrefixTreeState]) -> List[int]:
        """Return state IDs in BFS order from root."""
        order = []
        visited = set()
        queue = deque([0])
        while queue:
            sid = queue.popleft()
            if sid in visited or sid not in states:
                continue
            visited.add(sid)
            order.append(sid)
            for sym in sorted(states[sid].transitions.keys()):
                child = states[sid].transitions[sym]
                if child not in visited:
                    queue.append(child)
        return order

    def _try_merge(
        self,
        states: Dict[int, PrefixTreeState],
        keep_id: int,
        remove_id: int,
    ) -> Optional[Dict[int, PrefixTreeState]]:
        """Try to merge remove_id into keep_id. Returns new state dict or None on conflict."""
        # Deep copy states
        new_states: Dict[int, PrefixTreeState] = {}
        for sid, st in states.items():
            ns = PrefixTreeState(st.id, st.is_accept)
            ns.transitions = dict(st.transitions)
            new_states[sid] = ns

        # Perform the merge using a worklist
        merge_queue = deque([(keep_id, remove_id)])
        while merge_queue:
            k, r = merge_queue.popleft()
            if k == r:
                continue
            if k not in new_states or r not in new_states:
                continue

            keep_st = new_states[k]
            remove_st = new_states[r]

            # Union accept status
            if remove_st.is_accept:
                keep_st.is_accept = True

            # Merge transitions
            for sym, target in remove_st.transitions.items():
                if sym in keep_st.transitions:
                    # Both have transition on sym — need recursive merge
                    existing_target = keep_st.transitions[sym]
                    if existing_target != target:
                        merge_queue.append((existing_target, target))
                else:
                    keep_st.transitions[sym] = target

            # Redirect all references to remove_id → keep_id
            for sid, st in new_states.items():
                for sym, target in list(st.transitions.items()):
                    if target == r:
                        st.transitions[sym] = k

            # Remove the merged state
            del new_states[r]

        return new_states

    def _accepts_any_negative(
        self,
        states: Dict[int, PrefixTreeState],
        negative: List[Tuple[str, ...]],
    ) -> bool:
        """Check if the automaton accepts any negative example."""
        for seq in negative:
            if self._simulate(states, seq):
                return True
        return False

    def _simulate(
        self,
        states: Dict[int, PrefixTreeState],
        seq: Tuple[str, ...],
    ) -> bool:
        """Simulate the automaton on a sequence. Returns True if accepted."""
        current = 0
        for sym in seq:
            if current not in states:
                return False
            if sym not in states[current].transitions:
                return False
            current = states[current].transitions[sym]
        return current in states and states[current].is_accept

    def _states_to_dfa(self, states: Dict[int, PrefixTreeState]) -> DFA:
        """Convert internal state representation to a DFA object."""
        # Remap state IDs to 0..n-1
        old_ids = sorted(states.keys())
        id_map = {old: new for new, old in enumerate(old_ids)}

        dfa_states = set(range(len(old_ids)))
        transitions = {}
        accept_states = set()

        for old_id, st in states.items():
            new_id = id_map[old_id]
            if st.is_accept:
                accept_states.add(new_id)
            for sym, target in st.transitions.items():
                if target in id_map:
                    transitions[(new_id, sym)] = id_map[target]

        return DFA(
            states=dfa_states,
            alphabet=self.alphabet,
            transitions=transitions,
            start_state=id_map[0],
            accept_states=accept_states,
        )


def learn_one_vs_rest(
    alphabet: Tuple[str, ...],
    labeled_data: List[Tuple[Tuple[str, ...], str]],
    categories: Tuple[str, ...] = ("AR", "ALG", "WP", "LOG"),
) -> Dict[str, DFA]:
    """Learn one binary DFA per category using RPNI.

    Args:
        alphabet: The token alphabet.
        labeled_data: List of (token_sequence, category_label) pairs.
        categories: Categories to learn DFAs for.

    Returns:
        Dict mapping category name to its learned DFA.
    """
    rpni = RPNI(alphabet)
    dfas = {}

    for cat in categories:
        positive = [seq for seq, label in labeled_data if label == cat]
        negative = [seq for seq, label in labeled_data if label != cat]
        dfas[cat] = rpni.learn(positive, negative)

    return dfas
