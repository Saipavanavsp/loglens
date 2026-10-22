"""Load versioned prompt files from prompts/<name>/<version>.yaml.

Prompts are plain files under version control so a prompt change is a reviewed
diff, and every prediction we store records which prompt version produced it.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    system: str
    user_template: str
    item_template: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.name}:{self.version}"

    def render_user(self, **kwargs: str) -> str:
        return self.user_template.format(**kwargs)


@lru_cache
def load_prompt(name: str, version: str, prompts_dir: Path = PROMPTS_DIR) -> Prompt:
    path = prompts_dir / name / f"{version}.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != version:
        raise ValueError(f"{path} declares version {data.get('version')!r}, expected {version!r}")
    return Prompt(
        name=name,
        version=version,
        system=data["system"].strip(),
        user_template=data["user_template"],
        item_template=data.get("item_template"),
    )


def triage_messages(prompt: Prompt, source: str, message: str) -> list[dict[str, str]]:
    """Chat messages for one log line (system + user), as used by the student model."""
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.render_user(source=source, message=message).strip()},
    ]
