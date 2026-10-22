from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CACHE = DATA / "cache"
REPORTS = ROOT / "reports"

PARSED = INTERIM / "parsed.jsonl"
SAMPLE = INTERIM / "to_label.jsonl"
TEACHER_LABELS = INTERIM / "teacher_labels.jsonl"
REVIEWED = INTERIM / "reviewed.jsonl"
LABEL_CACHE = CACHE / "teacher_cache.jsonl"
