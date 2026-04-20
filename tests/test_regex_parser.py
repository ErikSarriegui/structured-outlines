"""Unit tests for the custom regex engine."""
import unittest

from structured_outlines.regex_parser import DFA, regex_to_dfa


def _matches(pattern: str, s: str) -> bool:
    dfa = regex_to_dfa(pattern)
    state = dfa.walk(dfa.start_state, s)
    return state != DFA.DEAD and dfa.is_accept(state)


class TestRegexBasics(unittest.TestCase):
    def test_literal(self):
        self.assertTrue(_matches("abc", "abc"))
        self.assertFalse(_matches("abc", "ab"))
        self.assertFalse(_matches("abc", "abcd"))

    def test_alternation(self):
        self.assertTrue(_matches("a|b", "a"))
        self.assertTrue(_matches("a|b", "b"))
        self.assertFalse(_matches("a|b", "c"))

    def test_star(self):
        self.assertTrue(_matches("a*", ""))
        self.assertTrue(_matches("a*", "aaaa"))
        self.assertFalse(_matches("a*", "ab"))

    def test_plus(self):
        self.assertFalse(_matches("a+", ""))
        self.assertTrue(_matches("a+", "a"))
        self.assertTrue(_matches("a+", "aaa"))

    def test_question(self):
        self.assertTrue(_matches("a?b", "b"))
        self.assertTrue(_matches("a?b", "ab"))
        self.assertFalse(_matches("a?b", "aab"))

    def test_grouping(self):
        self.assertTrue(_matches("(ab)+", "ababab"))
        self.assertFalse(_matches("(ab)+", "aba"))


class TestCharClasses(unittest.TestCase):
    def test_positive_class(self):
        self.assertTrue(_matches("[abc]", "b"))
        self.assertFalse(_matches("[abc]", "d"))

    def test_negated_class_ascii(self):
        self.assertTrue(_matches("[^abc]", "d"))
        self.assertFalse(_matches("[^abc]", "a"))

    def test_negated_class_unicode(self):
        # The symbolic DFA must accept non-ASCII via the default transition.
        self.assertTrue(_matches("[^abc]", "é"))
        self.assertTrue(_matches("[^abc]", "日"))
        self.assertTrue(_matches("[^abc]", "🌟"))

    def test_range(self):
        self.assertTrue(_matches("[a-z]", "m"))
        self.assertFalse(_matches("[a-z]", "A"))

    def test_invalid_range_raises(self):
        with self.assertRaises(ValueError):
            regex_to_dfa("[z-a]")

    def test_trailing_backslash_raises(self):
        with self.assertRaises(ValueError):
            regex_to_dfa("\\")


class TestDot(unittest.TestCase):
    def test_dot_matches_non_newline(self):
        self.assertTrue(_matches(".", "a"))
        self.assertTrue(_matches(".", "é"))
        self.assertFalse(_matches(".", "\n"))


class TestStateLimit(unittest.TestCase):
    def test_max_states_raises_value_error(self):
        # A regex with many alternatives needs more than 2 DFA states.
        with self.assertRaises(ValueError):
            regex_to_dfa("a|b|c|d|e|f|g", max_states=2)

    def test_max_states_respects_large_limit(self):
        dfa = regex_to_dfa("a|b|c|d", max_states=50)
        self.assertGreater(dfa.num_states, 1)


class TestEscapes(unittest.TestCase):
    def test_escaped_special_literal(self):
        self.assertTrue(_matches(r"\.", "."))
        self.assertFalse(_matches(r"\.", "a"))

    def test_escape_newline(self):
        self.assertTrue(_matches(r"\n", "\n"))


if __name__ == "__main__":
    unittest.main()
