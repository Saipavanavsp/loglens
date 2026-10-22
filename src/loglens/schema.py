"""Label schema shared by the data pipeline, training, evals and the gateway."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Category(StrEnum):
    AUTH = "auth"
    NETWORK = "network"
    DISK = "disk"
    MEMORY = "memory"
    SERVICE = "service"
    SECURITY = "security"
    OTHER = "other"


SEVERITIES = [s.value for s in Severity]
CATEGORIES = [c.value for c in Category]


class TriageLabel(BaseModel):
    """What a model must return for one log line."""

    model_config = ConfigDict(extra="forbid")

    severity: Severity
    category: Category
    root_cause: str = Field(min_length=3, max_length=400)
    suggested_action: str = Field(min_length=3, max_length=400)


class LogRecord(BaseModel):
    """One parsed log line."""

    id: str
    source: str
    raw: str
    message: str
    timestamp: str | None = None
    host: str | None = None
    component: str | None = None
    level: str | None = None
    alert: bool | None = None  # BGL only: Loghub's ground-truth alert flag


class LabeledRecord(LogRecord):
    label: TriageLabel
    label_source: str  # e.g. "teacher:claude-opus-5-5:v1" or "human"
    reviewed: bool = False


def triage_json_schema() -> dict:
    """JSON schema for a single label, in the subset accepted by structured outputs."""
    return {
        "type": "object",
        "properties": {
            "severity": {"type": "string", "enum": SEVERITIES},
            "category": {"type": "string", "enum": CATEGORIES},
            "root_cause": {"type": "string"},
            "suggested_action": {"type": "string"},
        },
        "required": ["severity", "category", "root_cause", "suggested_action"],
        "additionalProperties": False,
    }
