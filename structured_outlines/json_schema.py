"""Convert a Pydantic model's JSON schema into a regex that matches valid JSON.

Supported types: string, integer, number, boolean, null, array, object,
enum, const, anyOf ($ref / $defs for nested models).

Recursive schemas ($ref cycles) are rejected by design: regular languages
cannot encode recursive grammars. Restructure the schema to be finite if you
hit the ``Recursive $ref`` error.
"""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Primitive-type regex patterns
# ---------------------------------------------------------------------------

# JSON string per RFC 8259.
#   Body: any char except `"`, `\`, and control chars (< 0x20).
#   Escape: `\` followed by one of `" \ / b f n r t` or `uXXXX` (4 hex digits).
_HEX = r"[0-9a-fA-F]"
STRING = (
    r'"([^"\\' + '\x00-\x1f' + r']|\\(["\\/bfnrt]|u' + _HEX + _HEX + _HEX + _HEX + r'))*"'
)

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

def _convert(schema: dict, defs: dict, seen: frozenset = frozenset()) -> str:
    """Recursively convert a JSON-schema node to a regex pattern.

    ``seen`` carries the set of ``$ref`` names already entered on the current
    path from the root, so cycles can be rejected with a clear error.
    """

    # ── References ──
    if "$ref" in schema:
        ref_name = schema["$ref"].rsplit("/", 1)[-1]
        if ref_name in seen:
            raise ValueError(
                f"Recursive $ref detected: '{ref_name}'. Regular expressions "
                "cannot describe recursive grammars; restructure the schema "
                "to be finite (e.g., inline the type or bound the depth)."
            )
        return _convert(defs[ref_name], defs, seen | {ref_name})

    # ── anyOf  (Union / Optional types) ──
    if "anyOf" in schema:
        options = [_convert(s, defs, seen) for s in schema["anyOf"]]
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
        return _array_pattern(schema, defs, seen)

    if schema_type == "object":
        return _object_pattern(schema, defs, seen)

    raise ValueError(f"Unsupported schema node: {schema}")


def _array_pattern(schema: dict, defs: dict, seen: frozenset) -> str:
    item_schema = schema.get("items", {})
    item = _convert(item_schema, defs, seen)
    # [  item (, item)*  ] — or empty []
    inner = item + "(" + WS + "," + WS + item + ")*"
    return r"\[" + WS + "(" + inner + ")?" + WS + r"\]"


def _object_pattern(schema: dict, defs: dict, seen: frozenset) -> str:
    properties: dict = schema.get("properties", {})
    required: set = set(schema.get("required", []))

    if not properties:
        return r"\{" + WS + r"\}"

    # Build (pair_pattern, is_required) for each property in declared order.
    items: list[tuple[str, bool]] = []
    for key, prop_schema in properties.items():
        value_pat = _convert(prop_schema, defs, seen)
        pair = '"' + _escape_literal(key) + '"' + WS + ":" + WS + value_pat
        items.append((pair, key in required))

    first_req = next((i for i, (_, r) in enumerate(items) if r), None)

    if first_req is None:
        # All properties are optional. Any of them may be the first one
        # emitted (no leading comma); later ones carry a comma.
        alternatives: list[str] = []
        for i in range(len(items)):
            pat = items[i][0]
            for j in range(i + 1, len(items)):
                pat += "(" + WS + "," + WS + items[j][0] + ")?"
            alternatives.append(pat)
        body = "(" + "|".join(alternatives) + ")?"
        return r"\{" + WS + body + WS + r"\}"

    # At least one required property exists — anchor the regex on the first
    # required one. Optional properties before it emit "(pair, )?"; optional
    # properties after it emit "(, pair)?"; required ones after emit ", pair".
    parts: list[str] = []
    for i in range(first_req):
        pair, _ = items[i]
        parts.append("(" + pair + WS + "," + WS + ")?")
    parts.append(items[first_req][0])
    for i in range(first_req + 1, len(items)):
        pair, req = items[i]
        if req:
            parts.append(WS + "," + WS + pair)
        else:
            parts.append("(" + WS + "," + WS + pair + ")?")

    return r"\{" + WS + "".join(parts) + WS + r"\}"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def model_to_regex(model: Type[BaseModel]) -> str:
    """Return a regex pattern that matches valid JSON for *model*."""
    schema = model.model_json_schema()
    defs = schema.get("$defs", {})
    return _convert(schema, defs)