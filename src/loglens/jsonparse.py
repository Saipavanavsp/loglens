"""Turn raw model text into a validated TriageLabel.

Small models often wrap JSON in code fences or add a sentence before it, so we
look for the first balanced {...} object instead of calling json.loads on the
whole string.
"""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from loglens.schema import TriageLabel


@dataclass
class ParseResult:
    label: TriageLabel | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.label is not None


def _balanced_objects(text: str):
    """Yield every balanced {...} substring, left to right."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield text[start : i + 1]
                    break
        start = text.find("{", start + 1)


def extract_json_object(text: str) -> str | None:
    """Return the first balanced {...} that parses as a JSON object, or None."""
    for candidate in _balanced_objects(text):
        try:
            if isinstance(json.loads(candidate), dict):
                return candidate
        except json.JSONDecodeError:
            continue
    return None


def parse_label(text: str) -> ParseResult:
    """Parse and validate a model response. Never raises."""
    text = text or ""
    candidate = extract_json_object(text)
    if candidate is None:
        return ParseResult(None, "invalid_json" if "{" in text else "no_json_object")
    data = json.loads(candidate)
    if isinstance(data, dict):
        # Be lenient on case for enum fields; strict on everything else.
        for key in ("severity", "category"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip().lower()
    try:
        return ParseResult(TriageLabel.model_validate(data))
    except ValidationError as exc:
        return ParseResult(None, f"schema:{exc.errors()[0]['loc']}")
