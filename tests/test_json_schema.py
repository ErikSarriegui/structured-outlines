"""Unit tests for Pydantic → regex conversion."""
import unittest
from typing import List, Optional

from pydantic import BaseModel

from structured_outlines.json_schema import STRING, model_to_regex
from structured_outlines.regex_parser import DFA, regex_to_dfa


def _accepts(pattern: str, s: str) -> bool:
    dfa = regex_to_dfa(pattern)
    state = dfa.walk(dfa.start_state, s)
    return state != DFA.DEAD and dfa.is_accept(state)


class TestSimpleModel(unittest.TestCase):
    def test_required_fields(self):
        class User(BaseModel):
            name: str
            age: int

        pat = model_to_regex(User)
        self.assertTrue(_accepts(pat, '{"name":"Alex","age":22}'))
        self.assertTrue(_accepts(pat, '{ "name" : "Alex" , "age" : 22 }'))
        self.assertFalse(_accepts(pat, '{"name":"Alex"}'))  # missing age
        self.assertFalse(_accepts(pat, '{"age":22,"name":"Alex"}'))  # wrong order

    def test_unicode_in_string(self):
        class Msg(BaseModel):
            text: str

        pat = model_to_regex(Msg)
        self.assertTrue(_accepts(pat, '{"text":"café"}'))
        self.assertTrue(_accepts(pat, '{"text":"日本語"}'))


class TestOptionalFields(unittest.TestCase):
    def test_optional_after_required(self):
        class Prof(BaseModel):
            name: str
            nickname: Optional[str] = None

        pat = model_to_regex(Prof)
        self.assertTrue(_accepts(pat, '{"name":"A"}'))
        self.assertTrue(_accepts(pat, '{"name":"A","nickname":"B"}'))
        # Trailing comma is never valid JSON.
        self.assertFalse(_accepts(pat, '{"name":"A",}'))

    def test_optional_before_required(self):
        class Prof(BaseModel):
            nickname: Optional[str] = None
            name: str

        pat = model_to_regex(Prof)
        # Absent optional: name alone is fine.
        self.assertTrue(_accepts(pat, '{"name":"A"}'))
        # Present optional: order is preserved.
        self.assertTrue(_accepts(pat, '{"nickname":"B","name":"A"}'))
        # No dangling leading comma when optional is missing.
        self.assertFalse(_accepts(pat, '{,"name":"A"}'))

    def test_all_optional(self):
        class Opt(BaseModel):
            a: Optional[str] = None
            b: Optional[str] = None

        pat = model_to_regex(Opt)
        self.assertTrue(_accepts(pat, '{}'))
        self.assertTrue(_accepts(pat, '{"a":"x"}'))
        self.assertTrue(_accepts(pat, '{"b":"y"}'))
        self.assertTrue(_accepts(pat, '{"a":"x","b":"y"}'))
        # "b" alone must not force a leading comma.
        self.assertFalse(_accepts(pat, '{,"b":"y"}'))


class TestStringEscapes(unittest.TestCase):
    def test_valid_escapes_accepted(self):
        self.assertTrue(_accepts(STRING, r'"hello"'))
        self.assertTrue(_accepts(STRING, r'"he\"llo"'))
        self.assertTrue(_accepts(STRING, r'"a\\b"'))
        self.assertTrue(_accepts(STRING, r'"\u00e9"'))
        self.assertTrue(_accepts(STRING, r'"\n\t\r\b\f"'))

    def test_invalid_escapes_rejected(self):
        # \q and \x are not valid JSON escapes.
        self.assertFalse(_accepts(STRING, r'"\q"'))
        self.assertFalse(_accepts(STRING, r'"\x41"'))
        # Unicode escape with too few hex digits.
        self.assertFalse(_accepts(STRING, r'"\u123"'))
        # Unicode escape with a non-hex char.
        self.assertFalse(_accepts(STRING, r'"\u00zz"'))


class TestRecursiveSchemaRejected(unittest.TestCase):
    def test_self_referencing_model(self):
        class Node(BaseModel):
            val: int
            children: Optional[List["Node"]] = None

        Node.model_rebuild()

        with self.assertRaises(ValueError) as cm:
            model_to_regex(Node)
        self.assertIn("Recursive", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
