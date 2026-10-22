"""Mask variable parts of a log message so near-identical lines share a key.

"Invalid user admin from 1.2.3.4 port 5555" and
"Invalid user admin from 5.6.7.8 port 4242" both become
"invalid user admin from <ip> port <num>". Usernames and words are kept on
purpose; the fuzzy check in split.py handles lines that differ by a word.
"""

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b"), "<email>"),
    (re.compile(r"blk_-?\d+"), "<blk>"),
    (re.compile(r"\b\d{1,3}(\.\d{1,3}){3}(:\d+)?\b"), "<ip>"),
    (re.compile(r"\b(?:[0-9a-f]{1,4}:){2,7}[0-9a-f]{1,4}\b", re.I), "<ip6>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.I), "<hex>"),
    # 6+ hex chars with at least one digit: register dumps, addresses, hashes.
    (re.compile(r"\b(?=[0-9a-f]*\d)[0-9a-f]{6,}\b", re.I), "<hex>"),
    (re.compile(r"(/[\w.@-]+){2,}/?"), "<path>"),
    (re.compile(r"\b\d+\b"), "<num>"),
]
_WS = re.compile(r"\s+")


def mask(message: str) -> str:
    out = message
    for pattern, token in _PATTERNS:
        out = pattern.sub(token, out)
    return _WS.sub(" ", out).strip().lower()
