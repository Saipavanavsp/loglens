"""Merge the LoRA adapter into the base model and export it for Ollama.

Steps (run on Colab after training, CPU RAM ~8 GB is enough for 1.5B):

1. merge:  load the base model in fp16, apply the adapter, merge_and_unload()
2. gguf:   convert with llama.cpp's convert_hf_to_gguf.py and quantise to Q4_K_M
3. push:   optionally upload adapter + GGUF to a private Hugging Face repo

    python merge_and_export.py --adapter outputs/run1/adapter --out outputs/run1 \
        --llama-cpp /content/llama.cpp [--hf-repo your-name/loglens-qwen2.5-1.5b]

Why Q4_K_M: about 1 GB on disk for a 1.5B model, small quality loss vs fp16, and
fast enough on a laptop CPU. Q8_0 is the safer choice if quality drops.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def merge(base_model: str, adapter: Path, out_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged_dir = out_dir / "merged"
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float16)
    model = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(merged_dir))
    print(f"merged model saved to {merged_dir}")
    return merged_dir


def to_gguf(merged_dir: Path, llama_cpp: Path, out_dir: Path, quant: str) -> Path:
    f16 = out_dir / "loglens-f16.gguf"
    final = out_dir / f"loglens-{quant.lower()}.gguf"
    subprocess.run(
        [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"), str(merged_dir),
         "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )  # fmt: skip
    quantize = next(
        (
            p
            for p in [llama_cpp / "build/bin/llama-quantize", llama_cpp / "llama-quantize"]
            if p.exists()
        ),
        None,
    )
    if quantize is None:
        raise SystemExit("llama-quantize not found; build llama.cpp first (see FINETUNING.md)")
    subprocess.run([str(quantize), str(f16), str(final), quant], check=True)
    f16.unlink()
    print(f"GGUF written to {final} ({final.stat().st_size / 1e9:.2f} GB)")
    return final


def push(repo: str, adapter: Path, gguf: Path | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()  # uses HF_TOKEN from the environment / Colab secrets
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(adapter), repo_id=repo, path_in_repo="adapter")
    if gguf is not None:
        api.upload_file(path_or_fileobj=str(gguf), path_in_repo=gguf.name, repo_id=repo)
    print(f"uploaded to https://huggingface.co/{repo} (private)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--llama-cpp", type=Path, help="path to a llama.cpp checkout (for GGUF)")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--hf-repo", help="private HF repo id to upload to, e.g. you/loglens-qwen")
    args = p.parse_args()

    merged = merge(args.base_model, args.adapter, args.out)
    gguf = to_gguf(merged, args.llama_cpp, args.out, args.quant) if args.llama_cpp else None
    if args.hf_repo:
        push(args.hf_repo, args.adapter, gguf)


if __name__ == "__main__":
    main()
