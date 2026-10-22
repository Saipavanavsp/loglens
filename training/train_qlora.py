"""QLoRA fine-tuning of a small instruct model for log triage.

Runs on a single free Colab / Kaggle GPU (T4 16 GB or P100). Self-contained on
purpose: it only needs the files in this folder plus train_chat.jsonl and
val_chat.jsonl produced by `python -m loglens.data.split`.

    python train_qlora.py --data-dir data --output-dir outputs/run1

Writes:
  <output-dir>/adapter/              LoRA adapter + tokenizer
  <output-dir>/training_summary.json config, loss curve, timing, sanity metrics

Library versions this was written against (checked October 2026):
transformers 5.20, trl 1.16, peft 0.21, bitsandbytes 0.50. See requirements.txt.
"""

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    # --- model ---
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- LoRA ---
    # Rank of the low-rank update matrices. Higher = more capacity and more
    # trainable parameters. 16 is plenty for a narrow classification + short
    # explanation task; 8 also works, 64 is overkill for ~1k examples.
    lora_r: int = 16
    # Scaling of the LoRA update (effective scale = alpha / r). alpha = 2 * r is
    # the usual starting point.
    lora_alpha: int = 32
    # Dropout on the LoRA path; a little regularisation for a small dataset.
    lora_dropout: float = 0.05
    # Apply LoRA to every linear projection in attention and MLP blocks. Targeting
    # only q/v is cheaper but measurably weaker in the QLoRA paper.
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # --- optimisation ---
    # 2e-4 is the standard QLoRA learning rate; full fine-tuning would use ~1e-5.
    learning_rate: float = 2e-4
    # ~1k examples: 3 epochs is enough to learn the format and labels; watch val
    # loss for overfitting after epoch 2.
    num_epochs: int = 3
    per_device_batch_size: int = 8
    # Effective batch = batch * grad_accum = 16.
    grad_accum_steps: int = 2
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    # Prompt (system + log line) + JSON answer fit well under 512 tokens.
    max_length: int = 512
    seed: int = 42

    # --- post-training sanity check ---
    sanity_samples: int = 100
    max_new_tokens: int = 200


def load_chat_jsonl(path: Path) -> list[dict]:
    """Read chat rows and turn them into TRL prompt/completion format.

    With prompt/completion data, TRL computes the loss only on the completion
    (the assistant's JSON), not on the system prompt and log line.
    """
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                msgs = json.loads(line)["messages"]
                rows.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
    return rows


def extract_json(text: str) -> dict | None:
    """First balanced {...} in text, parsed. Mirrors loglens.jsonparse."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
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
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def sanity_eval(model, tokenizer, rows: list[dict], cfg: TrainConfig) -> dict:
    """Greedy-decode a few validation rows: valid-JSON rate and label accuracy.

    This is a quick check that training worked. The real evaluation is the
    golden-set eval harness (Phase 3).
    """
    import torch

    model.eval()
    valid = sev_ok = cat_ok = 0
    rows = rows[: cfg.sanity_samples]
    for row in rows:
        inputs = tokenizer.apply_chat_template(
            row["prompt"], add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = extract_json(text)
        gold = json.loads(row["completion"][0]["content"])
        if pred and {"severity", "category", "root_cause", "suggested_action"} <= pred.keys():
            valid += 1
            sev_ok += pred.get("severity") == gold["severity"]
            cat_ok += pred.get("category") == gold["category"]
    n = max(1, len(rows))
    return {
        "sanity_n": len(rows),
        "sanity_valid_json_rate": valid / n,
        "sanity_severity_acc": sev_ok / n,
        "sanity_category_acc": cat_ok / n,
    }


def train(cfg: TrainConfig, data_dir: Path, output_dir: Path) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("No GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16

    train_rows = load_chat_jsonl(data_dir / "train_chat.jsonl")
    val_rows = load_chat_jsonl(data_dir / "val_chat.jsonl")
    print(f"train {len(train_rows)} | val {len(val_rows)} | bf16 {bf16}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit NF4 weights with double quantisation; matmuls run in fp16/bf16.
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_length=cfg.max_length,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        seed=cfg.seed,
        model_init_kwargs={"dtype": compute_dtype},
    )
    trainer = SFTTrainer(
        model=cfg.base_model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        processing_class=tokenizer,
        quantization_config=bnb,
        peft_config=lora,
    )
    trainable, total = trainer.model.get_nb_trainable_parameters()
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    started = time.time()
    trainer.train()
    train_seconds = time.time() - started

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    sanity = sanity_eval(trainer.model, tokenizer, val_rows, cfg)
    summary = {
        "config": asdict(cfg),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(train_seconds, 1),
        "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "python": platform.python_version(),
        "log_history": trainer.state.log_history,
        "best_eval_loss": trainer.state.best_metric,
        **sanity,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))
    _maybe_log_mlflow(summary, adapter_dir)
    return summary


def _maybe_log_mlflow(summary: dict, adapter_dir: Path) -> None:
    """Log to MLflow only if a tracking URI is configured (e.g. a hosted server)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return
    import mlflow

    mlflow.set_experiment("loglens-finetune")
    with mlflow.start_run():
        mlflow.log_params({f"cfg.{k}": v for k, v in summary["config"].items()})
        for entry in summary["log_history"]:
            step = entry.get("step", 0)
            for k, v in entry.items():
                if isinstance(v, int | float) and k != "step":
                    mlflow.log_metric(k, v, step=step)
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")


def main() -> None:
    p = argparse.ArgumentParser(description="QLoRA fine-tuning for LogLens")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/run1"))
    for name, value in asdict(TrainConfig()).items():
        if isinstance(value, list):
            continue
        p.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    ns = p.parse_args()
    overrides = {k: getattr(ns, k) for k in asdict(TrainConfig()) if hasattr(ns, k)}
    cfg = TrainConfig(**overrides)
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    train(cfg, ns.data_dir, ns.output_dir)


if __name__ == "__main__":
    main()
