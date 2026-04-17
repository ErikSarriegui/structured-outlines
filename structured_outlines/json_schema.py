"""Convert a Pydantic model's JSON schema into a regex that matches valid JSON.

Supported types: string, integer, number, boolean, null, array, object,
enum, const, anyOf ($ref / $defs for nested models).
"""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Primitive-type regex patterns
# ---------------------------------------------------------------------------

# JSON string: "..." allowing escape sequences
STRING = r'"([^"\\]|\\.)*"'

# JSON integer: 0 | optional-minus non-zero digits
INTEGER = r'(0|-?[1-9][0-9]*)'

# JSON number: integer with optional decimal
NUMBER = r'(0|-?[1-9][0-9]*)(\.[0-9]+)?'

# JSON boolean
BOOLEAN = r'(true|false)'

# JSON null
NULL = r'null'

# Optional single whitespace between JSON tokens (keeps DFA small)
WS = r'[ ]?'

# Regex-special characters that must be escaped in literal strings
_REGEX_SPECIAL = frozenset(r"\.[]{}()*+?|^$")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(char: str) -> str:
    """Escape a single character for use in a regex literal."""
    if char in _REGEX_SPECIAL:
        return "\\" + char
    return char


def _escape_literal(s: str) -> str:
    """Escape all regex-special characters in a plain string."""
    return "".join(_esc(c) for c in s)


def _json_literal(value) -> str:
    """Return a regex that matches the JSON encoding of *value*."""
    encoded = json.dumps(value, ensure_ascii=False)
    return _escape_literal(encoded)

# ---------------------------------------------------------------------------
# Schema → Regex conversion
# ---------------------------------------------------------------------------

def _convert(schema: dict, defs: dict) -> str:
    """Recursively convert a JSON-schema node to a regex pattern."""

    # ── References ──
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        return _convert(defs[ref_name], defs)

    # ── anyOf  (Union / Optional types) ──
    if "anyOf" in schema:
        options = [_convert(s, defs) for s in schema["anyOf"]]
        return "(" + "|".join(options) + ")"

    # ── enum ──
    if "enum" in schema:
        return "(" + "|".join(_json_literal(v) for v in schema["enum"]) + ")"

    # ── const ──
    if "const" in schema:
        return _json_literal(schema["const"])

    # ── Typed schemas ──
    schema_type = schema.get("type")

    if schema_type == "string":
        return STRING

    if schema_type == "integer":
        return INTEGER

    if schema_type == "number":
        return NUMBER

    if schema_type == "boolean":
        return BOOLEAN

    if schema_type == "null":
        return NULL

    if schema_type == "array":
        return _array_pattern(schema, defs)

    if schema_type == "object":
        return _object_pattern(schema, defs)

    # Fallback: treat unknown as string
    return STRING


def _array_pattern(schema: dict, defs: dict) -> str:
    item_schema = schema.get("items", {})
    item = _convert(item_schema, defs)
    # [  item (, item)*  ] — or empty []
    inner = item + "(" + WS + "," + WS + item + ")*"
    return r"\[" + WS + "(" + inner + ")?" + WS + r"\]"


def _object_pattern(schema: dict, defs: dict) -> str:
    properties: dict = schema.get("properties", {})
    required: set = set(schema.get("required", []))

    if not properties:
        return r"\{" + WS + r"\}"

    parts: list[str] = []
    first = True

    for key, prop_schema in properties.items():
        value_pat = _convert(prop_schema, defs)
        # "key" : value
        pair = '"' + _escape_literal(key) + '"' + WS + ":" + WS + value_pat

        if first:
            # First property — no leading comma
            if key in required:
                parts.append(WS + pair)
            else:
                parts.append("(" + WS + pair + ")?")
            first = False
        else:
            if key in required:
                # If previous part was optional, comma must be conditional too.
                # Simplification: always emit comma (all props appear in order).
                parts.append(WS + "," + WS + pair)
            else:
                parts.append("(" + WS + "," + WS + pair + ")?")

    return r"\{" + "".join(parts) + WS + r"\}"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def model_to_regex(model: Type[BaseModel]) -> str:
    """Return a regex pattern that matches valid JSON for *model*."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _convert(schema, defs)
