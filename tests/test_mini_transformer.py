import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learning" / "mini_transformer"))
from model import MiniTransformer, scaled_dot_product_attention  # noqa: E402
from train import train  # noqa: E402


def test_causal_mask_blocks_future_positions():
    torch.manual_seed(0)
    q = k = torch.randn(1, 1, 4, 8)
    v = torch.eye(4).view(1, 1, 4, 4)  # value i = one-hot i, so output shows attention weights
    out = scaled_dot_product_attention(q, k, v, torch.tril(torch.ones(4, 4)))
    weights = out[0, 0]
    assert torch.allclose(weights.sum(-1), torch.ones(4))
    assert torch.all(torch.triu(weights, diagonal=1) == 0)  # nothing above the diagonal


def test_output_shapes():
    model = MiniTransformer(vocab_size=11, d_model=32, n_heads=4, n_layers=2, max_len=16)
    logits, loss = model(torch.randint(0, 11, (2, 16)), torch.randint(0, 11, (2, 16)))
    assert logits.shape == (2, 16, 11)
    assert loss.ndim == 0


def test_loss_decreases_on_repetitive_text():
    result = train("abcabcabc " * 200, steps=60, block=16, batch=16)
    first, last = result["history"][0], result["history"][-1]
    assert last[1] < first[1] * 0.7
