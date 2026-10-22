# Mini transformer from scratch

A small character-level GPT written by hand in PyTorch to understand what is inside the model that
LogLens fine-tunes. It is separate from the main project and nothing else imports it.

| File | What it shows |
|---|---|
| `model.py` | sinusoidal positional encoding, scaled dot-product attention with a causal mask, multi-head attention, pre-norm residual blocks, weight tying |
| `train.py` | character tokenizer, next-token batches, AdamW training loop, sampling |

```bash
pip install torch
python learning/mini_transformer/train.py --steps 2000
```

Default size: 4 layers, 4 heads, 128-dim embeddings, 64-character context (about 0.8M parameters).
A unit test (`tests/test_mini_transformer.py`) checks the attention mask and that loss goes down.

How it maps to Qwen2.5-1.5B: same decoder-only structure, but with 28 layers, a 1,536-dim hidden
size, rotary position embeddings, RMSNorm, SwiGLU feed-forward and grouped-query attention, trained
on trillions of tokens.
