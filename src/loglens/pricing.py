"""Model prices and rough cost estimation.

Prices are USD per million tokens, taken from Anthropic's published price list
as of 2026-10-06. Check https://www.anthropic.com/pricing before relying on them.
The local model has no per-token price; its cost is the hardware it runs on.
"""

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


def price_for(model: str) -> tuple[float, float]:
    """(input, output) price per million tokens. Unknown models return (0, 0).

    Use `budget_price_for` when checking spend limits.
    """
    return PRICES_PER_MTOK.get(model, (0.0, 0.0))


def budget_price_for(model: str) -> tuple[float, float]:
    """Price for budget checks: unknown models are assumed to cost the most we know of."""
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    return max(PRICES_PER_MTOK.values(), key=lambda p: p[1])


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English and logs).

    Only used for pre-run estimates; real usage is read from API responses.
    """
    return max(1, len(text) // 4)
