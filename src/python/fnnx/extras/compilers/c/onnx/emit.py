"""Primitives for emitting C source: identifiers, literals, and initializer lists."""

from __future__ import annotations

import math
import string
from collections.abc import Iterable, Iterator
from typing import Any

from onnx import TensorProto

C_KEYWORDS = frozenset(
    {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "_Bool",
        "_Complex",
        "_Imaginary",
    }
)

# The nonzero status an entrypoint returns for an argument it cannot serve; the enumerator
# is the artifact's prefix followed by this, and a kernel that validates an operand at run
# time returns it by that name.
INVALID_ARGUMENT_STATUS = "ERROR_INVALID_ARGUMENT"

_IDENTIFIER_CHARS = frozenset(string.ascii_letters + string.digits + "_")
_STRING_ESCAPES = {'"': '\\"', "\\": "\\\\", "?": "\\?"}
_UNSIGNED_TYPES = frozenset({TensorProto.UINT8, TensorProto.UINT16, TensorProto.UINT32})

# Decimal precision that reads back as the exact same value, per IEEE-754 binary32/binary64.
_FLOAT_DIGITS = 9
_DOUBLE_DIGITS = 17

_LINE_WIDTH = 88


def sanitize_identifier(name: str, *, fallback: str) -> str:
    """Turn an arbitrary ONNX name into a valid C identifier, deterministically."""
    cleaned = "".join(char if char in _IDENTIFIER_CHARS else "_" for char in name)
    if not any(char.isalnum() for char in cleaned):
        return fallback
    if not cleaned[0].isalpha():
        # C identifiers cannot start with a digit, and a leading underscore is reserved
        # for the implementation at file scope.
        cleaned = f"v_{cleaned}"
    if cleaned in C_KEYWORDS:
        cleaned = f"{cleaned}_"
    return cleaned


class UniqueNames:
    """Hands out distinct C identifiers, disambiguating collisions deterministically.

    Uniqueness is case-insensitive so that the uppercase macro names derived from these
    identifiers stay distinct too.
    """

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def is_taken(self, name: str) -> bool:
        return name.upper() in self._taken

    def assign(self, name: str, *, fallback: str) -> str:
        base = sanitize_identifier(name, fallback=fallback)
        candidate = base
        suffix = 2
        while candidate.upper() in self._taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self._taken.add(candidate.upper())
        return candidate


def scalar_literal(value: Any, elem_type: int) -> str:
    """A C literal that reads back as exactly `value` at element type `elem_type`."""
    if elem_type == TensorProto.FLOAT:
        return _float_literal(float(value), digits=_FLOAT_DIGITS, suffix="f")
    if elem_type == TensorProto.DOUBLE:
        return _float_literal(float(value), digits=_DOUBLE_DIGITS, suffix="")
    if elem_type == TensorProto.BOOL:
        return "1" if value else "0"
    return _integer_literal(int(value), elem_type)


def string_literal(value: str) -> str:
    """A C string literal holding exactly `value`'s UTF-8 bytes.

    Everything outside printable ASCII goes in as an octal escape rather than a hex one:
    `\\x` escapes are greedy, so a `\\xff` followed by a literal `a` would read as a single
    out-of-range character, while an octal escape stops after three digits. `?` is escaped
    too, since a run of them forms a trigraph the C99 preprocessor still rewrites.
    """
    pieces = []
    for byte in value.encode("utf-8"):
        char = chr(byte)
        if char in _STRING_ESCAPES:
            pieces.append(_STRING_ESCAPES[char])
        elif 0x20 <= byte < 0x7F:
            pieces.append(char)
        else:
            pieces.append(f"\\{byte:03o}")
    return '"' + "".join(pieces) + '"'


def initializer_lines(
    literals: Iterable[str], *, indent: str = "    "
) -> Iterator[str]:
    """Wrap literals into `{ ... }`-body lines, one comma-separated run per line."""
    line = indent
    for literal in literals:
        piece = f"{literal},"
        if line != indent and len(line) + len(piece) > _LINE_WIDTH:
            yield line.rstrip()
            line = indent
        line += f"{piece} "
    if line.strip():
        yield line.rstrip()


def comment_safe(text: str) -> str:
    """Neutralize block-comment delimiters so arbitrary names can go in comments.

    A nested `/*` is not a syntax error but is a `-Wall` diagnostic, which the artifact's
    `-Werror` build contract turns into a failure, so it is broken up like `*/` is.
    """
    return text.replace("*/", "* /").replace("/*", "/ *")


def _float_literal(value: float, *, digits: int, suffix: str) -> str:
    # NaN and the infinities have no decimal form; <math.h>'s macros are constant
    # expressions usable in the static initializers weights are emitted as.
    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "INFINITY" if value > 0 else "-INFINITY"
    text = f"{value:.{digits}g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + suffix


def _integer_literal(value: int, elem_type: int) -> str:
    if elem_type == TensorProto.INT64:
        if value == -(2**63):
            # `-9223372036854775808` is negation applied to a constant too large to be
            # signed; the minimum has to be built from the maximum.
            return "(-INT64_C(9223372036854775807) - INT64_C(1))"
        return f"INT64_C({value})"
    if elem_type == TensorProto.UINT64:
        return f"UINT64_C({value})"
    if elem_type == TensorProto.INT32 and value == -(2**31):
        return "(-2147483647 - 1)"
    if elem_type in _UNSIGNED_TYPES:
        return f"{value}u"
    return str(value)
