"""Train the mini transformer on a small text, character by character.

    python learning/mini_transformer/train.py --steps 2000

Downloads Tiny Shakespeare (~1 MB) on first run. On a laptop CPU, 2,000 steps
take a few minutes and produce recognisably Shakespeare-shaped gibberish.
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import MiniTransformer  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE = Path(__file__).parent / "input.txt"


class CharTokenizer:
    """Maps each distinct character to an integer id."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def get_batch(data: torch.Tensor, block: int, batch: int, gen: torch.Generator):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i : i + block] for i in ix])
    y = torch.stack([data[i + 1 : i + block + 1] for i in ix])  # shifted by one
    return x, y


def train(text: str, steps: int, block: int = 64, batch: int = 32, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    model = MiniTransformer(tok.vocab_size, max_len=block)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []
    for step in range(steps + 1):
        x, y = get_batch(train_data, block, batch, gen)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 10) == 0:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, block, batch, gen)
                _, vloss = model(vx, vy)
            model.train()
            history.append((step, loss.item(), vloss.item()))
            print(f"step {step:5d} | train {loss.item():.3f} | val {vloss.item():.3f}")
    model.eval()
    start = torch.tensor([tok.encode(text[:1])], dtype=torch.long)
    sample = tok.decode(model.generate(start, 200, temperature=0.8)[0].tolist())
    return {
        "history": history,
        "sample": sample,
        "params": sum(p.numel() for p in model.parameters()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    if not CACHE.exists():
        urllib.request.urlretrieve(URL, CACHE)  # noqa: S310 - fixed https URL
    result = train(CACHE.read_text(encoding="utf-8"), args.steps)
    print(f"\nparameters: {result['params']:,}\n--- sample ---\n{result['sample']}")


if __name__ == "__main__":
    main()
