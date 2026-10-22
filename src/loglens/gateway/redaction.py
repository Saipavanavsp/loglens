"""Redact PII and secrets before a log line leaves our network.

Applied to anything sent to an external provider (Claude). The local model sees
the original text.

Each distinct value is replaced by a numbered placeholder (<IP_1>, <IP_2>, ...),
so the model still sees that two fields were the same address or user. The
mapping is returned to the caller and never sent anywhere; it can be used to
put real values back into the answer for the user.

Order matters: secrets first (a token can look like a hostname), then emails
(they contain hostnames), then addresses, then names.
"""

import ipaddress
import re
from dataclasses import dataclass, field

# key=value / key: value secrets. Value runs until whitespace, quote, comma or semicolon.
_SECRET_KV = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|auth|session[_-]?id|"
    r"client[_-]?secret)(\s*[=:]\s*)([^\s\"',;]+)"
)
_BEARER = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S
)
# Well-known token shapes: AWS access key ids, GitHub, Slack, Anthropic/OpenAI style, JWTs.
_KNOWN_TOKENS = re.compile(
    r"\b(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|xox[abpr]-[A-Za-z0-9-]{10,}|"
    r"sk-[A-Za-z0-9_-]{20,}|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Not part of a longer dotted number (version strings), but a trailing full stop is fine.
_IPV4 = re.compile(
    r"(?<!\d)(?<!\d\.)(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d|\.\d)"
)
# Loose candidate; ipaddress decides. Rejects times like 06:55:46.
_IPV6_CANDIDATE = re.compile(r"(?<![\w:.])[0-9A-Fa-f:]{2,39}(?![\w:])")
_MAC = re.compile(r"(?i)\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b")
# Hostnames: at least two dot-separated labels and a letter-only TLD (avoids versions like 1.4.1).
_HOST = re.compile(
    r"(?i)\b(?=[a-z0-9-]*[a-z])[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\.[a-z]{2,24}\b"
)
# Usernames in common log phrasings.
_USER_PATTERNS = [
    re.compile(r"(?i)(\b(?:invalid|illegal)\s+user\s+)([^\s\[\]]+)"),
    re.compile(r"(?i)(\bfor\s+(?:invalid\s+user\s+)?)([A-Za-z_][\w.-]{0,31})(?=\s+from\b)"),
    re.compile(r"(?i)(\b(?:user|ruser|logname|uid)=)([^\s;,]+)"),
    re.compile(r"(?i)(\bsession\s+(?:opened|closed)\s+for\s+user\s+)([^\s()]+)"),
    re.compile(r"(?i)(\bby\s+user\s+)([^\s()]+)"),
]
# Accounts that identify a role rather than a person; keeping them helps triage.
_KEEP_USERS = {"root", "nobody", "unknown", "0", ""}


@dataclass
class Redaction:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original

    @property
    def count(self) -> int:
        return len(self.mapping)


class _Replacer:
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self._by_value: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def placeholder(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._by_value:
            self._counts[kind] = self._counts.get(kind, 0) + 1
            ph = f"<{kind}_{self._counts[kind]}>"
            self._by_value[key] = ph
            self.mapping[ph] = value
        return self._by_value[key]


_CODE_PREFIXES = ("java.", "javax.", "org.apache.", "com.sun.", "sun.", "org.springframework.")


def _looks_like_code(name: str) -> bool:
    """Java/Python dotted names (java.io.IOException) are not hostnames."""
    return bool(re.search(r"[a-z][A-Z]", name)) or name.startswith(_CODE_PREFIXES)


def _is_ipv6(s: str) -> bool:
    if s.count(":") < 2:
        return False
    try:
        ipaddress.IPv6Address(s)
    except ValueError:
        return False
    return True


def redact(text: str) -> Redaction:
    r = _Replacer()
    out = _PRIVATE_KEY.sub(lambda m: r.placeholder("PRIVATE_KEY", m.group(0)), text)
    out = _SECRET_KV.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{r.placeholder('SECRET', m.group(3))}", out
    )
    out = _BEARER.sub(lambda m: f"{m.group(1)} {r.placeholder('SECRET', m.group(0))}", out)
    out = _KNOWN_TOKENS.sub(lambda m: r.placeholder("SECRET", m.group(0)), out)
    out = _EMAIL.sub(lambda m: r.placeholder("EMAIL", m.group(0)), out)
    out = _MAC.sub(lambda m: r.placeholder("MAC", m.group(0)), out)
    out = _IPV4.sub(lambda m: r.placeholder("IP", m.group(0)), out)
    out = _IPV6_CANDIDATE.sub(
        lambda m: r.placeholder("IP", m.group(0)) if _is_ipv6(m.group(0)) else m.group(0), out
    )
    out = _HOST.sub(
        lambda m: m.group(0) if _looks_like_code(m.group(0)) else r.placeholder("HOST", m.group(0)),
        out,
    )

    def _user(m: re.Match) -> str:
        name = m.group(2)
        if name.lower() in _KEEP_USERS or name.startswith("<"):
            return m.group(0)
        return m.group(1) + r.placeholder("USER", name)

    for pattern in _USER_PATTERNS:
        out = pattern.sub(_user, out)
    return Redaction(out, r.mapping)


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put original values back into a model answer (for the end user only)."""
    for ph in sorted(mapping, key=len, reverse=True):
        text = text.replace(ph, mapping[ph])
    return text
