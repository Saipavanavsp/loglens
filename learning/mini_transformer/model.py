"""A tiny decoder-only transformer, written out by hand.

This is a learning exercise kept separate from LogLens. It implements the same
building blocks as Qwen/Llama-style models at toy scale:

    token embedding + positional encoding
    -> N x [LayerNorm -> causal multi-head self-attention -> residual
            LayerNorm -> feed-forward MLP              -> residual]
    -> LayerNorm -> linear projection to vocabulary logits

Real models differ in details (RoPE instead of sinusoidal positions, RMSNorm,
SwiGLU MLPs, grouped-query attention), but the data flow is the same.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class SinusoidalPositionalEncoding(nn.Module):
    """Adds a fixed position signal to each token embedding.

    Attention itself is order-blind: it sees a set of vectors. Adding sin/cos
    waves of different frequencies gives every position a unique pattern, and
    relative offsets become linear functions of these patterns, which attention
    can learn to use.
    """

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)  # (T, 1)
        freq = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * freq)  # even dims: sine
        pe[:, 1::2] = torch.cos(pos * freq)  # odd dims: cosine
        self.register_buffer("pe", pe)  # not a parameter: never trained

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask: torch.Tensor
) -> torch.Tensor:
    """softmax(Q K^T / sqrt(d)) V, written out explicitly.

    q, k, v: (batch, heads, T, d_head). Each query position scores every key
    position; dividing by sqrt(d_head) keeps scores from growing with the
    dimension (which would make softmax saturate). The causal mask sets scores
    for future positions to -inf so a token can only look backwards.
    """
    d_head = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_head)  # (B, H, T, T)
    scores = scores.masked_fill(causal_mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)  # each row sums to 1
    return weights @ v  # weighted average of value vectors


class MultiHeadAttention(nn.Module):
    """Runs several attention 'heads' in parallel on slices of the embedding.

    Each head can specialise (e.g. one tracks the previous token, another the
    start of a word). Their outputs are concatenated and mixed by a final
    linear layer.
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)  # one matmul for Q, K and V
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, T, D) -> (B, H, T, d_head)
        q, k, v = (z.view(b, t, self.n_heads, self.d_head).transpose(1, 2) for z in (q, k, v))
        y = scaled_dot_product_attention(q, k, v, self.mask[:t, :t])
        y = y.transpose(1, 2).contiguous().view(b, t, d)  # concatenate heads
        return self.out(y)


class Block(nn.Module):
    """Pre-norm transformer block: attention then MLP, each with a residual."""

    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        # Position-wise MLP: expands 4x, applies a non-linearity, projects back.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections let gradients flow straight through deep stacks.
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class MiniTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, max_len, dropout) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying: fewer params, often better
        # Small init (as in GPT-2). PyTorch's default N(0, 1) embedding init makes the
        # tied output layer produce huge logits, so the first loss is far above ln(vocab).
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.pos(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))  # (B, T, vocab)
        loss = None
        if targets is not None:
            # Next-token prediction: position t predicts token t+1.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.max_len :])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
