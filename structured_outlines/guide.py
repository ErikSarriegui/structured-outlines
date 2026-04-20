from __future__ import annotations

from collections import defaultdict

from .regex_parser import DFA, regex_to_dfa

STATE_DONE = -2


class RegexGuide:
    """Maps (DFA state, token_id) → next DFA state for constrained decoding."""

    def __init__(self, pattern: str, tokenizer, *, verbose: bool = False, max_states: int | None = None):
        kwargs = {}
        if max_states is not None:
            kwargs["max_states"] = max_states
        self.dfa: DFA = regex_to_dfa(pattern, **kwargs)
        self.eos_token_id: int | None = getattr(tokenizer, "eos_token_id", None)
        self._transitions: dict[int, dict[int, int]] = {}
        self._precompute(tokenizer, verbose=verbose)

    # ------------------------------------------------------------------
    # Precomputation
    # ------------------------------------------------------------------

    def _precompute(self, tokenizer, *, verbose: bool = False):
        vocab_size: int = tokenizer.vocab_size
        special_ids: set[int] = set(getattr(tokenizer, "all_special_ids", []))

        # 1. Decode every non-special token once.
        token_strs: dict[int, str] = {}
        for tid in range(vocab_size):
            if tid in special_ids:
                continue
            try:
                s = tokenizer.decode([tid])
            except Exception:
                continue
            if s:
                token_strs[tid] = s

        # 2. Group tokens by their first character for faster filtering.
        by_first: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for tid, s in token_strs.items():
            by_first[s[0]].append((tid, s))

        n_states = self.dfa.num_states
        for state in range(n_states):
            if verbose and n_states > 20 and state % max(1, n_states // 10) == 0:
                print(f"  [RegexGuide] precomputing state {state}/{n_states} …")

            mapping: dict[int, int] = {}

            lit_trans = self.dfa.transitions.get(state, {})
            has_default = state in self.dfa.default_trans

            if has_default:
                # A wildcard transition exists — any token's first char could
                # be accepted, so we cannot prune by first char.
                candidates = token_strs.items()
            else:
                # Only tokens whose first char has a literal transition can
                # possibly be accepted from this state.
                candidates = [
                    (tid, s)
                    for first_ch in lit_trans
                    for tid, s in by_first.get(first_ch, [])
                ]

            for tid, tok_str in candidates:
                ns = self.dfa.walk(state, tok_str)
                if ns != DFA.DEAD:
                    mapping[tid] = ns

            # EOS is valid only when the DFA is in an accept state.
            if self.dfa.is_accept(state) and self.eos_token_id is not None:
                mapping[self.eos_token_id] = STATE_DONE

            self._transitions[state] = mapping

        if verbose:
            print(f"  [RegexGuide] precomputation done — {n_states} states.")

    # ------------------------------------------------------------------
    # Runtime queries
    # ------------------------------------------------------------------

    def get_valid_tokens(self, state: int) -> list[int]:
        """Return token IDs that are legal in *state*."""
        return list(self._transitions.get(state, {}).keys())

    def advance(self, state: int, token_id: int) -> int:
        """Return the DFA state after consuming *token_id* in *state*."""
        return self._transitions.get(state, {}).get(token_id, DFA.DEAD)

    @property
    def start_state(self) -> int:
        return self.dfa.start_state