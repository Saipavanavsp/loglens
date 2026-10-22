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


def extract_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in text, or None."""
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
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_label(text: str) -> ParseResult:
    """Parse and validate a model response. Never raises."""
    candidate = extract_json_object(text or "")
    if candidate is None:
        return ParseResult(None, "no_json_object")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return ParseResult(None, "invalid_json")
    if isinstance(data, dict):
        # Be lenient on case for enum fields; strict on everything else.
        for key in ("severity", "category"):
            if isinstance(data.get(key), str):
                data[key] = data[key].strip().lower()
    try:
        return ParseResult(TriageLabel.model_validate(data))
    except ValidationError as exc:
        return ParseResult(None, f"schema:{exc.errors()[0]['loc']}")
