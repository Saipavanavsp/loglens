"""Assemble the Hugging Face Space folder and upload it.

    $env:HF_TOKEN = "<token with write access>"     # never commit it
    python deploy/hf_space/push_space.py --space your-name/loglens [--gguf-repo your-name/loglens-qwen2.5-1.5b]

Creates the Space (Docker SDK) if needed, uploads the app, and sets the
LOGLENS_GGUF_REPO variable. If the model repo is private, also add HF_TOKEN as
a Space *secret* in the Space settings (read access is enough).
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent


def assemble(dest: Path) -> None:
    shutil.copy(HERE / "Dockerfile", dest / "Dockerfile")
    shutil.copy(HERE / "start.sh", dest / "start.sh")
    shutil.copy(HERE / "SPACE_README.md", dest / "README.md")
    shutil.copy(ROOT / "pyproject.toml", dest / "pyproject.toml")
    shutil.copy(ROOT / "infra" / "ollama" / "Modelfile", dest / "Modelfile")
    for folder in ("src", "prompts", "configs"):
        shutil.copytree(
            ROOT / folder, dest / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )


def main() -> None:
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--space", required=True, help="e.g. your-name/loglens")
    p.add_argument("--gguf-repo", help="HF model repo holding loglens-q4_k_m.gguf")
    p.add_argument("--dry-run", action="store_true", help="assemble locally and list files only")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        assemble(dest)
        files = sorted(str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file())
        print(f"{len(files)} files, e.g. {files[:6]}")
        if args.dry_run:
            return
        api = HfApi()
        api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(dest), repo_id=args.space, repo_type="space")
        if args.gguf_repo:
            api.add_space_variable(args.space, "LOGLENS_GGUF_REPO", args.gguf_repo)
    print(f"https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
