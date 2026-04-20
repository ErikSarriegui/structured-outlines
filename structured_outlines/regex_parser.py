"""Minimal regex engine: regex -> NFA (Thompson's) -> DFA (subset construction).

Supports the regex subset needed for JSON schema patterns:
  - Literals with escaping (\\n, \\t, \\\\, \\", etc.)
  - Character classes: [abc], [a-z], [^"\\\\]
  - Quantifiers: *, +, ?
  - Alternation: |
  - Grouping: (...)
  - Dot: . (any char except newline)

Transitions are represented symbolically: NFA edges carry (charset, negated)
labels, and each DFA state has both explicit per-char transitions and an
optional "default" transition covering any other char. This keeps the DFA
small even when negated classes span Unicode.
"""

from __future__ import annotations

from collections import deque

# ---------------------------------------------------------------------------
# Character universe (kept for backwards compatibility with AST helpers)
# ---------------------------------------------------------------------------

ALL_CHARS = frozenset(chr(i) for i in range(32, 127)) | frozenset("\t\n\r")

# Default cap on the number of DFA states — protects against exponential
# blow-up from adversarial regex/schema inputs.
DEFAULT_MAX_STATES = 10_000

# ---------------------------------------------------------------------------
# Symbolic matchers
# ---------------------------------------------------------------------------
# A matcher is ``(chars: frozenset[str], negated: bool)``.
# ``_m_matches(m, ch)`` is true iff ``(ch in chars) XOR negated``.
# ``None`` on an NFA edge represents an epsilon transition.

def _m_lit(ch: str):
    return (frozenset((ch,)), False)

def _m_cls(chars, negated: bool):
    return (frozenset(chars), negated)

def _m_dot():
    # "." matches any char except newline.
    return (frozenset(("\n",)), True)

def _m_matches(matcher, ch: str) -> bool:
    chars, negated = matcher
    return (ch in chars) != negated

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
    def __init__(self, chars, negated: bool = False):
        self.chars = frozenset(chars)
        self.negated = negated
    def matching_chars(self) -> frozenset:
        # Retained for backwards compatibility. The DFA builder no longer
        # enumerates characters — it keeps classes symbolic.
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
            i += 1
            if i >= len(pattern):
                raise ValueError("Trailing backslash in pattern")
            tokens.append(("LIT", _ESCAPE_MAP.get(pattern[i], pattern[i])))
        elif c == "[":
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
                    start_ch, end_ch = pattern[i], pattern[i + 2]
                    if ord(start_ch) > ord(end_ch):
                        raise ValueError(
                            f"Invalid character range '{start_ch}-{end_ch}': "
                            "start code point is greater than end"
                        )
                    for code in range(ord(start_ch), ord(end_ch) + 1):
                        chars.add(chr(code))
                    i += 2
                else:
                    chars.add(pattern[i])
                i += 1
            if i >= len(pattern):
                raise ValueError("Unterminated character class")
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

    def parse(self):
        node = self._alt()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected token at position {self.pos}: {self._peek()}")
        return node

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
# NFA (Thompson's construction) — edges carry symbolic matchers
# ---------------------------------------------------------------------------

class _NFA:
    def __init__(self):
        self.size = 0
        self.trans: dict[int, list[tuple]] = {}

    def new(self) -> int:
        s = self.size
        self.size += 1
        self.trans[s] = []
        return s

    def add(self, src: int, matcher, dst: int):
        # `matcher` is None (epsilon) or a (frozenset, bool) tuple.
        self.trans[src].append((matcher, dst))


def _build(nfa: _NFA, node) -> tuple[int, int]:
    """Return (start, accept) states for *node*."""

    if isinstance(node, Epsilon):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, None, a)
        return s, a

    if isinstance(node, Literal):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, _m_lit(node.char), a)
        return s, a

    if isinstance(node, CharClass):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, _m_cls(node.chars, node.negated), a)
        return s, a

    if isinstance(node, Dot):
        s, a = nfa.new(), nfa.new()
        nfa.add(s, _m_dot(), a)
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
# DFA (subset / powerset construction)
# ---------------------------------------------------------------------------

class DFA:
    """Deterministic finite automaton with integer state IDs.

    ``transitions[state]`` maps characters to next states (explicit, literal
    transitions). ``default_trans[state]`` — if present — is the state taken
    for *any* character not listed explicitly (wildcard from negated classes).
    """

    DEAD = -1

    def __init__(self, start: int, trans: dict, accept: frozenset, n_states: int, default_trans: dict | None = None):
        self.start_state = start
        self.transitions = trans                  # dict[int, dict[str, int]]
        self.default_trans = default_trans or {}  # dict[int, int]
        self.accept_states = accept               # frozenset[int]
        self.num_states = n_states

    def next_state(self, state: int, ch: str) -> int:
        if state == self.DEAD:
            return self.DEAD
        t = self.transitions.get(state)
        if t is not None and ch in t:
            return t[ch]
        default = self.default_trans.get(state)
        return self.DEAD if default is None else default

    def is_accept(self, state: int) -> bool:
        return state in self.accept_states

    def walk(self, state: int, string: str) -> int:
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
        for matcher, ns in nfa.trans.get(s, []):
            if matcher is None and ns not in closure:
                closure.add(ns)
                stack.append(ns)
    return frozenset(closure)


def _nfa_to_dfa(nfa: _NFA, nfa_accept: int, max_states: int) -> DFA:
    start = _epsilon_closure(nfa, frozenset({0}))

    state_map: dict[frozenset, int] = {start: 0}
    dfa_trans: dict[int, dict[str, int]] = {}
    dfa_default: dict[int, int] = {}
    dfa_accept: set[int] = set()
    queue: deque[frozenset] = deque([start])
    next_id = 1

    def _intern(nfa_states: set) -> int:
        nonlocal next_id
        nxt = _epsilon_closure(nfa, frozenset(nfa_states))
        sid = state_map.get(nxt)
        if sid is None:
            if next_id >= max_states:
                raise ValueError(
                    f"DFA state explosion: exceeded max_states={max_states}. "
                    "The schema/regex produces too many DFA states — simplify "
                    "it or pass a larger ``max_states`` limit."
                )
            sid = next_id
            state_map[nxt] = sid
            next_id += 1
            queue.append(nxt)
        return sid

    while queue:
        current = queue.popleft()
        cid = state_map[current]
        dfa_trans[cid] = {}

        if nfa_accept in current:
            dfa_accept.add(cid)

        # Collect all non-epsilon outgoing matchers from NFA states in `current`.
        outgoing: list = []
        for s in current:
            for matcher, ns in nfa.trans.get(s, []):
                if matcher is not None:
                    outgoing.append((matcher, ns))

        if not outgoing:
            continue

        # Every char that appears in some matcher gets an explicit transition.
        # Any other char falls through to the default (wildcard) transition if
        # at least one matcher is negated.
        referenced: set = set()
        has_negated = False
        for (chars, negated), _ in outgoing:
            referenced |= chars
            if negated:
                has_negated = True

        for ch in referenced:
            targets = {ns for (chars, neg), ns in outgoing if (ch in chars) != neg}
            if targets:
                dfa_trans[cid][ch] = _intern(targets)
            else:
                # `ch` is referenced (appears in some matcher's charset) but no
                # matcher accepts it from this DFA state. Mark it explicitly
                # DEAD so the default wildcard transition does not fire on it.
                dfa_trans[cid][ch] = DFA.DEAD

        if has_negated:
            # A negated matcher with excluded set ⊆ `referenced` matches every
            # char outside `referenced`. Take the union of all such targets.
            targets = {ns for (_, neg), ns in outgoing if neg}
            if targets:
                dfa_default[cid] = _intern(targets)

    return DFA(0, dfa_trans, frozenset(dfa_accept), next_id, default_trans=dfa_default)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_regex(pattern: str):
    """Parse a regex pattern string into an AST."""
    return _Parser(tokenize(pattern)).parse()


def regex_to_dfa(pattern: str, max_states: int = DEFAULT_MAX_STATES) -> DFA:
    """Compile a regex pattern into a DFA.

    Raises ``ValueError`` if the DFA would exceed ``max_states`` states.
    """
    ast = parse_regex(pattern)
    nfa = _NFA()
    start, accept = _build(nfa, ast)
    # Thompson's construction always allocates the start state first, so it is
    # guaranteed to be state 0. No renumbering is needed.
    assert start == 0, "Internal error: Thompson NFA start is not state 0"
    return _nfa_to_dfa(nfa, accept, max_states=max_states)