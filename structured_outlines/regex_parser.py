"""Minimal regex engine: regex string -> NFA (Thompson's) -> DFA (subset construction).

Supports the regex subset needed for JSON schema patterns:
  - Literals with escaping (\\n, \\t, \\\\, \\", etc.)
  - Character classes: [abc], [a-z], [^"\\\\]
  - Quantifiers: *, +, ?
  - Alternation: |
  - Grouping: (...)
  - Dot: . (any char except newline)
"""

from collections import deque

# ---------------------------------------------------------------------------
# Character universe (printable ASCII + common whitespace)
# ---------------------------------------------------------------------------

ALL_CHARS = frozenset(chr(i) for i in range(32, 127)) | frozenset("\t\n\r")

# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

class Epsilon:
    """Matches the empty string."""

class Literal:
    __slots__ = ("char",)
    def __init__(self, char: str):
        self.char = char

class CharClass:
    __slots__ = ("chars", "negated")
    def __init__(self, chars: set, negated: bool = False):
        self.chars = chars
        self.negated = negated
    def matching_chars(self) -> frozenset:
        return ALL_CHARS - self.chars if self.negated else frozenset(self.chars)

class Dot:
    def matching_chars(self) -> frozenset:
        return ALL_CHARS - frozenset("\n")

class Concat:
    __slots__ = ("left", "right")
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Alt:
    __slots__ = ("left", "right")
    def __init__(self, left, right):
        self.left = left
        self.right = right

class Star:
    __slots__ = ("child",)
    def __init__(self, child):
        self.child = child

class Plus:
    __slots__ = ("child",)
    def __init__(self, child):
        self.child = child

class Question:
    __slots__ = ("child",)
    def __init__(self, child):
        self.child = child

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_ESCAPE_MAP = {"n": "\n", "t": "\t", "r": "\r"}

def tokenize(pattern: str) -> list:
    """Tokenize a regex pattern into (type, value) pairs."""
    tokens: list = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            # Escaped character
            i += 1
            if i >= len(pattern):
                raise ValueError("Trailing backslash in pattern")
            tokens.append(("LIT", _ESCAPE_MAP.get(pattern[i], pattern[i])))
        elif c == "[":
            # Character class
            i += 1
            negated = False
            if i < len(pattern) and pattern[i] == "^":
                negated = True
                i += 1
            chars: set = set()
            while i < len(pattern) and pattern[i] != "]":
                if pattern[i] == "\\":
                    i += 1
                    if i >= len(pattern):
                        raise ValueError("Unterminated character class")
                    chars.add(_ESCAPE_MAP.get(pattern[i], pattern[i]))
                elif (
                    i + 2 < len(pattern)
                    and pattern[i + 1] == "-"
                    and pattern[i + 2] != "]"
                ):
                    for code in range(ord(pattern[i]), ord(pattern[i + 2]) + 1):
                        chars.add(chr(code))
                    i += 2
                else:
                    chars.add(pattern[i])
                i += 1
            if i >= len(pattern):
                raise ValueError("Unterminated character class")
            # i now points at ']'
            tokens.append(("CC", (chars, negated)))
        elif c == "(":
            tokens.append(("LPAR", None))
        elif c == ")":
            tokens.append(("RPAR", None))
        elif c == "|":
            tokens.append(("ALT", None))
        elif c == "*":
            tokens.append(("STAR", None))
        elif c == "+":
            tokens.append(("PLUS", None))
        elif c == "?":
            tokens.append(("QMRK", None))
        elif c == ".":
            tokens.append(("DOT", None))
        else:
            tokens.append(("LIT", c))
        i += 1
    return tokens

# ---------------------------------------------------------------------------
# Recursive-descent parser  (precedence: atom < quant < concat < alt)
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _eat(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    # ── public ──

    def parse(self):
        node = self._alt()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self._peek()}")
        return node

    # ── grammar rules ──

    def _alt(self):
        left = self._concat()
        while self._peek() and self._peek()[0] == "ALT":
            self._eat()
            right = self._concat()
            left = Alt(left, right)
        return left

    def _concat(self):
        nodes: list = []
        while self._peek() and self._peek()[0] not in ("ALT", "RPAR"):
            nodes.append(self._quant())
        if not nodes:
            return Epsilon()
        result = nodes[0]
        for n in nodes[1:]:
            result = Concat(result, n)
        return result

    def _quant(self):
        node = self._atom()
        tok = self._peek()
        if tok and tok[0] == "STAR":
            self._eat()
            return Star(node)
        if tok and tok[0] == "PLUS":
            self._eat()
            return Plus(node)
        if tok and tok[0] == "QMRK":
            self._eat()
            return Question(node)
        return node

    def _atom(self):
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of pattern")
        if tok[0] == "LIT":
            self._eat()
            return Literal(tok[1])
        if tok[0] == "CC":
            self._eat()
            chars, neg = tok[1]
            return CharClass(chars, neg)
        if tok[0] == "DOT":
            self._eat()
            return Dot()
        if tok[0] == "LPAR":
            self._eat()
            node = self._alt()
            if not self._peek() or self._peek()[0] != "RPAR":
                raise ValueError("Unmatched '('")
            self._eat()
            return node
        raise ValueError(f"Unexpected token: {tok}")

# ---------------------------------------------------------------------------
# NFA  (Thompson's construction)
# ---------------------------------------------------------------------------

class _NFA:
    def __init__(self):
        self.size = 0
        self.trans: dict[int, list[tuple[str | None, int]]] = {}

    def new(self) -> int:
        s = self.size
        self.size += 1
        self.trans[s] = []
        return s

    def add(self, src: int, ch: str | None, dst: int):
        self.trans[src].append((ch, dst))


def _build(nfa: _NFA, node) -> tuple[int, int]:
    """Return (start, accept) states for *node*."""

    if isinstance(node, Epsilon):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, None, a)
        return s, a

    if isinstance(node, Literal):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, node.char, a)
        return s, a

    if isinstance(node, (CharClass, Dot)):
        s, a = nfa.new(), nfa.new()
        for ch in node.matching_chars():
            nfa.add(s, ch, a)
        return s, a

    if isinstance(node, Concat):
        s1, a1 = _build(nfa, node.left)
        s2, a2 = _build(nfa, node.right)
        nfa.add(a1, None, s2)
        return s1, a2

    if isinstance(node, Alt):
        s, a = nfa.new(), nfa.new()
        s1, a1 = _build(nfa, node.left)
        s2, a2 = _build(nfa, node.right)
        nfa.add(s, None, s1)
        nfa.add(s, None, s2)
        nfa.add(a1, None, a)
        nfa.add(a2, None, a)
        return s, a

    if isinstance(node, Star):
        s, a = nfa.new(), nfa.new()
        s1, a1 = _build(nfa, node.child)
        nfa.add(s, None, s1)
        nfa.add(s, None, a)
        nfa.add(a1, None, s1)
        nfa.add(a1, None, a)
        return s, a

    if isinstance(node, Plus):
        s, a = nfa.new(), nfa.new()
        s1, a1 = _build(nfa, node.child)
        nfa.add(s, None, s1)
        nfa.add(a1, None, s1)
        nfa.add(a1, None, a)
        return s, a

    if isinstance(node, Question):
        s, a = nfa.new(), nfa.new()
        s1, a1 = _build(nfa, node.child)
        nfa.add(s, None, s1)
        nfa.add(s, None, a)
        nfa.add(a1, None, a)
        return s, a

    raise TypeError(f"Unknown AST node: {type(node).__name__}")

# ---------------------------------------------------------------------------
# DFA  (subset / powerset construction)
# ---------------------------------------------------------------------------

class DFA:
    """Deterministic finite automaton with integer state IDs."""

    DEAD = -1  # convention for dead / invalid state

    def __init__(self, start: int, trans: dict, accept: frozenset, n_states: int):
        self.start_state = start
        self.transitions = trans      # dict[int, dict[str, int]]
        self.accept_states = accept   # frozenset[int]
        self.num_states = n_states

    def next_state(self, state: int, ch: str) -> int:
        if state == self.DEAD:
            return self.DEAD
        return self.transitions.get(state, {}).get(ch, self.DEAD)

    def is_accept(self, state: int) -> bool:
        return state in self.accept_states

    def walk(self, state: int, string: str) -> int:
        """Walk the DFA from *state* through each char in *string*."""
        for ch in string:
            state = self.next_state(state, ch)
            if state == self.DEAD:
                return self.DEAD
        return state


def _epsilon_closure(nfa: _NFA, states: frozenset) -> frozenset:
    stack = list(states)
    closure = set(states)
    while stack:
        s = stack.pop()
        for ch, ns in nfa.trans.get(s, []):
            if ch is None and ns not in closure:
                closure.add(ns)
                stack.append(ns)
    return frozenset(closure)


def _nfa_to_dfa(nfa: _NFA, nfa_accept: int) -> DFA:
    # Gather alphabet from NFA transitions
    alphabet: set[str] = set()
    for arcs in nfa.trans.values():
        for ch, _ in arcs:
            if ch is not None:
                alphabet.add(ch)

    start = _epsilon_closure(nfa, frozenset({nfa.trans and 0}))
    # nfa start is always state 0 by construction in regex_to_dfa
    start = _epsilon_closure(nfa, frozenset({0}))

    state_map: dict[frozenset, int] = {start: 0}
    dfa_trans: dict[int, dict[str, int]] = {}
    dfa_accept: set[int] = set()
    queue: deque[frozenset] = deque([start])
    next_id = 1

    while queue:
        current = queue.popleft()
        cid = state_map[current]
        dfa_trans[cid] = {}

        if nfa_accept in current:
            dfa_accept.add(cid)

        for ch in alphabet:
            move: set[int] = set()
            for s in current:
                for c, ns in nfa.trans.get(s, []):
                    if c == ch:
                        move.add(ns)
            if not move:
                continue
            nxt = _epsilon_closure(nfa, frozenset(move))
            if nxt not in state_map:
                state_map[nxt] = next_id
                next_id += 1
                queue.append(nxt)
            dfa_trans[cid][ch] = state_map[nxt]

    return DFA(0, dfa_trans, frozenset(dfa_accept), next_id)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_regex(pattern: str):
    """Parse a regex pattern string into an AST."""
    return _Parser(tokenize(pattern)).parse()


def regex_to_dfa(pattern: str) -> DFA:
    """Compile a regex pattern into a DFA."""
    ast = parse_regex(pattern)
    nfa = _NFA()
    start, accept = _build(nfa, ast)
    # Renumber so that start == 0 (needed by _nfa_to_dfa)
    # Thompson's construction already makes start the first allocated state
    # when the root is built first, but let's be safe:
    if start != 0:
        # Swap state 0 and start in the transition table
        nfa.trans[0], nfa.trans[start] = nfa.trans[start], nfa.trans[0]
        # Update references
        for s in nfa.trans:
            nfa.trans[s] = [
                (ch, (0 if ns == start else start if ns == 0 else ns))
                for ch, ns in nfa.trans[s]
            ]
        if accept == 0:
            accept = start
        elif accept == start:
            accept = 0
    return _nfa_to_dfa(nfa, accept)
