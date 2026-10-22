"""Parse raw Loghub files into data/interim/parsed.jsonl.

Each source has its own line format, so each gets a small regex. Lines that do
not match are counted and skipped rather than guessed at.

Usage: python -m loglens.data.parse
"""

import argparse
import hashlib
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from loglens.data.io import write_jsonl
from loglens.data.normalize import mask
from loglens.data.paths import PARSED, RAW
from loglens.schema import LogRecord

# Jun  9 06:06:20 combo sshd(pam_unix)[6737]: message
SYSLOG = re.compile(
    r"^(?P<ts>[A-Z][a-z]{2}\s+\d{1,2} \d{2}:\d{2}:\d{2}) (?P<host>\S+) "
    r"(?P<component>[^:\[]+(?:\([^)]*\))?)(?:\[(?P<pid>\d+)\])?: ?(?P<msg>.*)$"
)
# [Thu Jun 09 06:07:04 2005] [notice] message
APACHE = re.compile(r"^\[(?P<ts>[^\]]+)\] \[(?P<level>\w+)\] (?P<msg>.*)$")
# - 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO msg
BGL = re.compile(
    r"^(?P<alert>\S+) \d+ \S+ (?P<node>\S+) (?P<ts>\S+) \S+ \S+ (?P<component>\S+) (?P<level>\S+) ?(?P<msg>.*)$"
)


def _record_id(source: str, raw: str) -> str:
    return hashlib.sha1(f"{source}\x00{raw}".encode(), usedforsecurity=False).hexdigest()[:16]


def parse_line(source: str, raw: str) -> LogRecord | None:
    raw = raw.rstrip("\r\n")
    if not raw.strip():
        return None
    if source in {"linux", "openssh"}:
        m = SYSLOG.match(raw)
        if not m:
            return None
        return LogRecord(
            id=_record_id(source, raw),
            source=source,
            raw=raw,
            message=m["msg"].strip(),
            timestamp=m["ts"],
            host=m["host"],
            component=m["component"].strip(),
        )
    if source == "apache":
        m = APACHE.match(raw)
        if not m:
            return None
        return LogRecord(
            id=_record_id(source, raw),
            source=source,
            raw=raw,
            message=f"[{m['level']}] {m['msg'].strip()}",
            timestamp=m["ts"],
            level=m["level"],
        )
    if source == "bgl":
        m = BGL.match(raw)
        if not m or not m["msg"].strip():
            return None
        return LogRecord(
            id=_record_id(source, raw),
            source=source,
            raw=raw,
            message=f"{m['component']} {m['level']} {m['msg'].strip()}",
            timestamp=m["ts"],
            host=m["node"],
            component=m["component"],
            level=m["level"],
            alert=m["alert"] != "-",
        )
    raise ValueError(f"unknown source {source!r}")


def iter_source(source: str, path: Path, stats: Counter, max_per_key: int) -> Iterator[dict]:
    """Yield parsed rows, keeping at most max_per_key lines per masked message."""
    seen: Counter = Counter()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_line(source, line)
            if rec is None:
                stats[f"{source}:skipped"] += 1
                continue
            stats[f"{source}:parsed"] += 1
            key = mask(rec.message)
            seen[key] += 1
            if seen[key] > max_per_key:
                continue
            row = rec.model_dump()
            row["key"] = key
            stats[f"{source}:kept"] += 1
            yield row
    stats[f"{source}:unique_keys"] = len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw Loghub logs to JSONL")
    parser.add_argument("--out", type=Path, default=PARSED)
    parser.add_argument(
        "--max-per-key", type=int, default=3, help="copies kept per masked message (default 3)"
    )
    args = parser.parse_args()

    stats: Counter = Counter()

    def all_rows() -> Iterator[dict]:
        for src_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
            log = src_dir / f"{src_dir.name}.log"
            if log.exists():
                yield from iter_source(src_dir.name, log, stats, args.max_per_key)

    n = write_jsonl(args.out, all_rows())
    for k, v in sorted(stats.items()):
        print(f"{k:24s} {v:>9,}")
    print(f"wrote {n:,} records to {args.out}")


if __name__ == "__main__":
    main()
