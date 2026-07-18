"""Strict, dependency-free JSON decoding for untrusted artifact boundaries."""

from __future__ import annotations

import json
from typing import NoReturn


class StrictJSONError(ValueError):
    """A valid-looking JSON value used semantics TicketTune does not permit."""


class DuplicateJSONKeyError(StrictJSONError):
    """A JSON object repeated a member name at any nesting depth."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"duplicate JSON object key {key!r}")


class NonFiniteJSONConstantError(StrictJSONError):
    """A JSON payload used Python's non-standard NaN or infinity constants."""

    def __init__(self, constant: str) -> None:
        self.constant = constant
        super().__init__(f"non-finite JSON constant is not permitted: {constant}")


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJSONKeyError(key)
        value[key] = item
    return value


def _reject_non_finite_constant(value: str) -> NoReturn:
    raise NonFiniteJSONConstantError(value)


def strict_json_decoder() -> json.JSONDecoder:
    """Return a decoder that rejects ambiguous and non-standard JSON values."""

    return json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_non_finite_constant,
    )


def loads_strict(payload: str | bytes | bytearray) -> object:
    """Decode JSON, rejecting duplicate keys recursively and non-finite constants."""

    return json.loads(
        payload,
        object_pairs_hook=_reject_duplicate_object_pairs,
        parse_constant=_reject_non_finite_constant,
    )
